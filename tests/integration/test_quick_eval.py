"""The loop rates itself: a quick match every few generations, and ``best.pt`` follows it.

Until this existed a run produced a hundred checkpoints and the question "which
one is best" was answered afterwards, by hand, or not at all. These tests pin the
plumbing: the rating lands in the sidecar and the metrics, ``best`` tracks the
highest estimate across a resume, and the whole thing can be switched off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.ckpt.manager import CheckpointManager
from reversi.ckpt.meta import CheckpointMeta
from reversi.config import Config
from reversi.obs.metrics import MetricsHub, read_jsonl
from reversi.obs.runmeta import RunPaths
from reversi.train.evaluate import quick_evaluate
from reversi.train.loop import run_training


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return RunPaths(run_id="test-quick", root=tmp_path / "run")


def test_every_rated_generation_carries_its_estimate(smoke_config: Config, paths: RunPaths) -> None:
    """``every_n_generations`` is 1 in the test config, so every generation is rated."""
    paths.ensure()
    with MetricsHub(paths.metrics, run_id=paths.run_id) as metrics:
        reports = run_training(smoke_config, paths, metrics=metrics, generations=2)

    assert all(r.elo_estimate is not None for r in reports)

    for generation in (1, 2):
        meta = CheckpointMeta.read(paths.checkpoints / f"gen_{generation:05d}.json")
        assert meta.elo_estimate is not None
        assert meta.elo_estimate == pytest.approx(reports[generation - 1].elo_estimate)

    # latest.json describes generation 2 and must carry the same number.
    latest = CheckpointMeta.read(paths.checkpoints / "latest.json")
    assert latest.generation == 2
    assert latest.elo_estimate == pytest.approx(reports[1].elo_estimate)

    arena = read_jsonl(paths.metrics / "arena.jsonl")
    assert [row["generation"] for row in arena] == [1, 2]
    assert "elo_estimate" in arena[0]
    assert "score_vs_random" in arena[0]
    assert 0.0 <= arena[0]["score_vs_random"] <= 1.0


def test_best_points_at_the_highest_estimate(smoke_config: Config, paths: RunPaths) -> None:
    reports = run_training(smoke_config, paths, generations=3)

    best = CheckpointMeta.read(paths.checkpoints / "best.json")
    highest = max(reports, key=lambda r: r.elo_estimate or float("-inf"))
    assert best.generation == highest.generation
    assert best.elo_estimate == pytest.approx(highest.elo_estimate)
    assert (paths.checkpoints / "best.pt").is_file()


def test_best_survives_a_resume(smoke_config: Config, paths: RunPaths) -> None:
    """A resumed run must not crown its first new generation just because it is new."""
    first = run_training(smoke_config, paths, generations=2)
    manager = CheckpointManager(
        paths.checkpoints, run_id=paths.run_id, config_sha256=smoke_config.sha256
    )
    remembered = manager.best_estimate()
    assert remembered is not None
    best_generation, best_elo = remembered
    assert best_generation in {1, 2}
    assert best_elo == pytest.approx(max(r.elo_estimate or 0.0 for r in first))

    second = run_training(smoke_config, paths, generations=3)
    after = manager.best_estimate()
    assert after is not None
    if (second[0].elo_estimate or 0.0) > best_elo:
        assert after == (3, pytest.approx(second[0].elo_estimate))
    else:
        assert after == (best_generation, pytest.approx(best_elo))


def test_the_quick_evaluation_can_be_switched_off(smoke_config: Config, paths: RunPaths) -> None:
    payload = smoke_config.model_dump()
    payload["arena"]["every_n_generations"] = 0
    off = Config.model_validate(payload)

    reports = run_training(off, paths, generations=2)

    assert all(r.elo_estimate is None for r in reports)
    assert not (paths.checkpoints / "best.pt").exists()
    assert not (paths.metrics / "arena.jsonl").exists()
    meta = CheckpointMeta.read(paths.checkpoints / "gen_00002.json")
    assert meta.elo_estimate is None


def test_the_schedule_is_honoured(smoke_config: Config, paths: RunPaths) -> None:
    payload = smoke_config.model_dump()
    payload["arena"]["every_n_generations"] = 2
    every_other = Config.model_validate(payload)

    reports = run_training(every_other, paths, generations=4)

    rated = [r.generation for r in reports if r.elo_estimate is not None]
    assert rated == [2, 4]


def test_quick_evaluate_rates_a_checkpoint_on_its_own(
    smoke_config: Config, paths: RunPaths
) -> None:
    reports = run_training(smoke_config, paths, generations=1)
    checkpoint = reports[0].checkpoint
    assert checkpoint is not None

    result = quick_evaluate(checkpoint, config=smoke_config, generation=1)

    assert result.generation == 1
    assert set(result.scores) == {"random", "greedy"}
    assert result.games == smoke_config.arena.quick_games
    assert result.simulations == smoke_config.arena.quick_simulations
    # Random is the anchor at 0, so any finite number is a valid rating.
    assert result.elo_estimate == result.elo_estimate  # not NaN
    metrics = result.as_metrics()
    assert metrics["elo_estimate"] == result.elo_estimate
    assert "score_vs_greedy" in metrics


def test_quick_games_must_be_even(smoke_config: Config) -> None:
    payload = smoke_config.model_dump()
    payload["arena"]["quick_games"] = 5
    with pytest.raises(ValueError, match="quick_games must be even"):
        Config.model_validate(payload)
