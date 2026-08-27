"""Self-play across worker processes (test matrix T23).

The design being tested is a deliberate rejection of the usual one: no queues, no
shared memory, no central inference server. Each worker gets a weights file and a
range of game indices, writes its own shard, and exits. The parent learns what
happened by reading files.

That makes the interesting tests structural -- how the work is divided, what
happens when a worker dies -- rather than about message passing, because there are
no messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.ckpt import CheckpointManager
from reversi.config import Config, load_config
from reversi.data.shards import read_shard
from reversi.errors import WorkerError
from reversi.nn.model import build
from reversi.selfplay.runner import WorkerResult, merge_summaries, run_workers, split_games

# ===========================================================================
# Dividing the work
# ===========================================================================


def test_games_are_split_into_contiguous_ranges() -> None:
    assert split_games(12, 4) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert split_games(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert split_games(5, 5) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_every_game_is_assigned_exactly_once() -> None:
    for total, workers in ((100, 6), (7, 7), (1000, 13)):
        ranges = split_games(total, workers)
        covered = [index for first, last in ranges for index in range(first, last)]
        assert covered == list(range(total))


def test_the_split_is_as_even_as_it_can_be() -> None:
    """A worker with twice the games ends the generation twice as late, and the
    generation ends with its slowest worker."""
    sizes = [last - first for first, last in split_games(100, 6)]
    assert max(sizes) - min(sizes) <= 1


def test_asking_for_more_workers_than_games_is_refused() -> None:
    """Silently idling a worker would make the reported worker count a lie."""
    with pytest.raises(WorkerError, match="cannot split"):
        split_games(3, 6)


def test_zero_workers_is_refused() -> None:
    with pytest.raises(WorkerError, match="at least one worker"):
        split_games(10, 0)


# ===========================================================================
# Merging what the workers report
# ===========================================================================


def make_result(worker_id: int, **counters: object) -> WorkerResult:
    from reversi.data.shards import ShardInfo

    base = {
        "games": 0,
        "positions": 0,
        "plies": [],
        "passes": 0,
        "forced_moves": 0,
        "black_wins": 0,
        "white_wins": 0,
        "draws": 0,
        "branching_total": 0,
        "branching_count": 0,
    }
    base.update(counters)
    return WorkerResult(
        worker_id=worker_id,
        shard=ShardInfo(filename="x.npz", n_positions=1, generation=1, sha256="0" * 64),
        summary=base,
        seconds=1.0,
    )


def test_summaries_add_up() -> None:
    merged = merge_summaries(
        [
            make_result(0, games=10, positions=90, plies=[8, 9], black_wins=6, white_wins=4),
            make_result(1, games=5, positions=40, plies=[7], black_wins=2, draws=3),
        ]
    )

    assert merged.games == 15
    assert merged.positions == 130
    assert merged.plies == [8, 9, 7]
    assert merged.black_wins == 8
    assert merged.draws == 3


def test_merging_sums_counts_rather_than_averaging_averages() -> None:
    """With unequal game counts, the average of averages is simply wrong.

    One worker playing 90 games at a 0.9 win rate and another playing 10 at 0.1
    is 0.82 overall, not 0.5 -- and the wrong number would look perfectly
    plausible in a metrics file.
    """
    merged = merge_summaries(
        [
            make_result(0, games=90, black_wins=81, white_wins=9),
            make_result(1, games=10, black_wins=1, white_wins=9),
        ]
    )

    metrics = merged.as_metrics()
    assert metrics["games"] == 100
    assert metrics["first_player_win_rate"] == pytest.approx(82 / 100)


# ===========================================================================
# Actually running them (slow: real processes)
# ===========================================================================


@pytest.fixture
def tiny_config() -> Config:
    return load_config(
        Path("configs/smoke4x4.yaml"),
        overrides=[
            "selfplay.games_per_generation=12",
            "selfplay.games_in_flight=3",
            "mcts.n_simulations=6",
            "net.n_blocks=1",
            "net.channels=8",
        ],
    )


def weights_for(tmp_path: Path, config: Config) -> Path:
    model = build(config.net, config.game.board_size, seed=1)
    manager = CheckpointManager(
        tmp_path / "checkpoints", run_id="runner-test", config_sha256=config.sha256
    )
    manager.save(model=model, generation=0, global_step=0)
    return tmp_path / "checkpoints" / "latest.pt"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_workers_produce_one_shard_each(tiny_config: Config, tmp_path: Path) -> None:
    weights = weights_for(tmp_path, tiny_config)
    replay = tmp_path / "replay"

    results = run_workers(
        weights_path=weights,
        config=tiny_config,
        replay_dir=replay,
        generation=4,
        n_workers=3,
        device="cpu",
    )

    assert [r.worker_id for r in results] == [0, 1, 2]
    assert sorted(r.shard.filename for r in results) == [
        "gen_00004_w00.npz",
        "gen_00004_w01.npz",
        "gen_00004_w02.npz",
    ]
    assert merge_summaries(results).games == 12

    # Every shard is real, readable, and matches the checksum the worker reported.
    for result in results:
        arrays = read_shard(
            replay / result.shard.filename,
            board_size=4,
            expect_sha256=result.shard.sha256,
        )
        assert len(arrays["black"]) == result.shard.n_positions
        assert set(arrays["generation"]) == {4}

    # The summary files are cleaned up -- they are a handover, not an artifact.
    assert not list(replay.glob("*.summary.json"))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_workers_do_not_play_each_others_games(tiny_config: Config, tmp_path: Path) -> None:
    """Seeds derive from the worker id as well as the game index.

    Two workers producing identical games would halve the real size of a
    generation while the counts still looked right.
    """
    weights = weights_for(tmp_path, tiny_config)
    replay = tmp_path / "replay"

    results = run_workers(
        weights_path=weights,
        config=tiny_config,
        replay_dir=replay,
        generation=1,
        n_workers=2,
        device="cpu",
    )

    positions = []
    for result in results:
        arrays = read_shard(replay / result.shard.filename, board_size=4)
        positions.append({(int(b), int(w)) for b, w in zip(arrays["black"], arrays["white"])})

    assert positions[0] != positions[1], "two workers produced the same games"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_missing_weights_file_fails_loudly(tiny_config: Config, tmp_path: Path) -> None:
    """Every worker dies the same way, and the parent says so rather than
    producing an empty generation."""
    with pytest.raises(WorkerError, match="failed twice"):
        run_workers(
            weights_path=tmp_path / "not-a-checkpoint.pt",
            config=tiny_config,
            replay_dir=tmp_path / "replay",
            generation=1,
            n_workers=2,
            device="cpu",
        )
