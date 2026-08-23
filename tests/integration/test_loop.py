"""The whole cycle, end to end (backlog T18).

Three generations of a deliberately tiny 4x4 configuration: play, store, train,
repeat. This does not check that the agent gets *good* -- that is day 6's gate,
and it needs opponents to measure against. What it checks is that the loop closes:
games become shards, shards become batches, batches become weight updates, and
every artifact that should exist afterwards does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from reversi.config import Config
from reversi.data.shards import Manifest
from reversi.obs.metrics import MetricsHub, read_jsonl
from reversi.obs.runmeta import RunPaths
from reversi.train.loop import run_training


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return RunPaths(run_id="test-loop", root=tmp_path / "run")


def test_three_generations_complete(smoke_config: Config, paths: RunPaths) -> None:
    reports = run_training(smoke_config, paths, generations=3)

    assert [r.generation for r in reports] == [1, 2, 3]
    for report in reports:
        assert report.games == smoke_config.selfplay.games_per_generation
        assert report.positions > 0
        assert report.seconds > 0
        assert "total_loss" in report.training


def test_the_buffer_grows_as_generations_accumulate(smoke_config: Config, paths: RunPaths) -> None:
    reports = run_training(smoke_config, paths, generations=3)
    sizes = [r.buffer_size for r in reports]

    assert sizes == sorted(sizes), f"the window should not shrink while below capacity: {sizes}"
    assert sizes[-1] > sizes[0]


def test_every_generation_leaves_a_shard_and_a_checkpoint(
    smoke_config: Config, paths: RunPaths
) -> None:
    run_training(smoke_config, paths, generations=3)

    shards = sorted(p.name for p in paths.replay.glob("*.npz"))
    assert shards == ["gen_00001_w00.npz", "gen_00002_w00.npz", "gen_00003_w00.npz"]

    checkpoints = sorted(p.name for p in paths.checkpoints.glob("*.pt"))
    assert checkpoints == ["gen_00001.pt", "gen_00002.pt", "gen_00003.pt", "latest.pt"]


def test_the_manifest_agrees_with_what_is_on_disk(smoke_config: Config, paths: RunPaths) -> None:
    reports = run_training(smoke_config, paths, generations=3)

    manifest = Manifest.load(paths.replay)
    assert len(manifest.shards) == 3
    assert manifest.total_positions == sum(r.positions for r in reports)
    # Nothing was dropped, which means every checksum still matches.
    assert manifest.verify() == []


def test_the_saved_checkpoint_can_be_loaded_back(smoke_config: Config, paths: RunPaths) -> None:
    run_training(smoke_config, paths, generations=2)

    payload = torch.load(paths.checkpoints / "latest.pt", weights_only=False)

    assert payload["generation"] == 2
    assert payload["arch"]["board_size"] == smoke_config.game.board_size
    assert payload["config_sha256"] == smoke_config.sha256
    assert payload["model_state_dict"], "weights should not be empty"
    # The per-generation file and `latest` describe the same generation.
    same = torch.load(paths.checkpoints / "gen_00002.pt", weights_only=False)
    assert same["generation"] == payload["generation"]


def test_training_steps_accumulate_across_generations(
    smoke_config: Config, paths: RunPaths
) -> None:
    """The step counter is what the learning-rate schedule reads.

    If it reset every generation, every generation would repeat the warmup and the
    run would never reach its configured rate.
    """
    run_training(smoke_config, paths, generations=3)

    first = torch.load(paths.checkpoints / "gen_00001.pt", weights_only=False)
    last = torch.load(paths.checkpoints / "gen_00003.pt", weights_only=False)

    per_generation = smoke_config.train.steps_per_generation
    assert first["global_step"] == per_generation
    assert last["global_step"] == 3 * per_generation


def test_the_weights_actually_change(smoke_config: Config, paths: RunPaths) -> None:
    """A loop that runs without training anything would pass every test above."""
    run_training(smoke_config, paths, generations=2)

    first = torch.load(paths.checkpoints / "gen_00001.pt", weights_only=False)
    second = torch.load(paths.checkpoints / "gen_00002.pt", weights_only=False)

    changed = [
        key
        for key, value in first["model_state_dict"].items()
        if not torch.equal(value, second["model_state_dict"][key])
    ]
    assert changed, "no parameter changed between generations"


def test_the_loop_stops_when_asked(smoke_config: Config, paths: RunPaths) -> None:
    """The hook day 7 wires a signal handler to."""
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # allow two generations, refuse the third

    reports = run_training(smoke_config, paths, generations=5, should_stop=should_stop)

    assert [r.generation for r in reports] == [1, 2]


def test_old_shards_are_pruned(smoke_config: Config, paths: RunPaths) -> None:
    """Replay shards are the bulk of a run's disk use; stale ones are dead weight."""
    config = smoke_config.model_copy(
        update={"replay": smoke_config.replay.model_copy(update={"retain_shards": 2})}
    )

    run_training(config, paths, generations=4)

    on_disk = sorted(p.name for p in paths.replay.glob("*.npz"))
    assert on_disk == ["gen_00003_w00.npz", "gen_00004_w00.npz"]
    assert len(Manifest.load(paths.replay).shards) == 2


def test_metrics_are_written_for_every_generation(smoke_config: Config, paths: RunPaths) -> None:
    """JSONL is the source of truth for every figure in the final report."""
    paths.ensure()
    with MetricsHub(paths.metrics, run_id=paths.run_id) as metrics:
        run_training(smoke_config, paths, metrics=metrics, generations=2)

    selfplay = read_jsonl(paths.metrics / "selfplay.jsonl")
    train = read_jsonl(paths.metrics / "train.jsonl")
    replay = read_jsonl(paths.metrics / "replay.jsonl")

    assert [row["generation"] for row in selfplay] == [1, 2]
    assert [row["generation"] for row in train] == [1, 2]
    assert [row["generation"] for row in replay] == [1, 2]

    assert selfplay[0]["games"] == smoke_config.selfplay.games_per_generation
    assert selfplay[0]["games_per_s"] > 0
    assert train[0]["policy_loss"] > 0
    assert replay[0]["buffer_size"] > 0
    assert 0.0 < replay[0]["unique_positions_fraction"] <= 1.0


def test_a_run_can_be_extended_by_replaying_the_shards(
    smoke_config: Config, paths: RunPaths
) -> None:
    """Starting a fresh loop over an existing run directory picks the games back up.

    Not full resume -- the weights and optimiser state still restart, and that is
    day 7's job. What this proves is that the shards and manifest written by one
    process are readable by the next one, which is the half of resume that lives
    in this package.
    """
    run_training(smoke_config, paths, generations=2)
    before = Manifest.load(paths.replay).total_positions

    reports = run_training(smoke_config, paths, generations=1)

    # The new run saw the earlier generations' positions in its window from the start.
    assert reports[0].buffer_size > reports[0].positions
    assert Manifest.load(paths.replay).total_positions >= before
