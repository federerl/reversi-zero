"""Training games against a fixed opponent (research/step3-design.md).

Three things have to hold for the opponent-diversity experiment to mean what
it says. The share of external games must be exactly what the config states
and the same on every re-run, so the groups differ in one thing. Only the
network's own decisions may become training examples, so an external game
teaches the network about *its* moves and nothing about the opponent's. And
the batched scheduler must play exactly the games the one-at-a-time path
plays, as it already must for self-play.
"""

from __future__ import annotations

from collections import Counter

import pytest

from reversi.agents import GreedyAgent, RandomAgent
from reversi.config import Config
from reversi.errors import ConfigError, WorkerError
from reversi.game import rules
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.seeding import rng as make_rng
from reversi.selfplay.game_batch import BatchedSelfPlay
from reversi.selfplay.opponents import OpponentSchedule, build_opponents
from reversi.selfplay.worker import SelfPlaySummary, play_game, play_games
from reversi.types import Player

BOARD = 4
SEARCH = SearchConfig(n_simulations=12, temp_moves=3, dirichlet_eps=0.25, dirichlet_alpha=2.0)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("selfplay", "external"), [(480, 240), (10, 3), (7, 7), (5, 0), (0, 4)])
def test_exactly_the_stated_number_of_games_are_external(selfplay: int, external: int) -> None:
    schedule = OpponentSchedule(selfplay, external, ("greedy", "minimax-d2"))
    picked = schedule.external_indices()
    assert len(picked) == external
    assert all(0 <= i < schedule.total for i in picked)
    assert sum(schedule.ordinal(i) is None for i in range(schedule.total)) == selfplay


def test_external_games_are_spread_evenly_not_bunched() -> None:
    """Workers take contiguous index ranges, so bunching would hand one worker
    every external game and the others none."""
    schedule = OpponentSchedule(480, 240, ("greedy",))
    gaps = {b - a for a, b in zip(schedule.external_indices(), schedule.external_indices()[1:])}
    assert gaps <= {2, 3}


def test_opponents_take_turns_and_colours_alternate() -> None:
    schedule = OpponentSchedule(100, 40, ("greedy", "minimax-d2"))
    assignments = [schedule.assignment(i) for i in schedule.external_indices()]
    assert all(a is not None for a in assignments)
    counts = Counter((a.opponent, a.network) for a in assignments if a is not None)
    assert counts == {
        ("greedy", Player.BLACK): 10,
        ("greedy", Player.WHITE): 10,
        ("minimax-d2", Player.BLACK): 10,
        ("minimax-d2", Player.WHITE): 10,
    }


def test_the_schedule_is_a_pure_function_of_the_index() -> None:
    first = OpponentSchedule(30, 12, ("greedy",))
    again = OpponentSchedule(30, 12, ("greedy",))
    assert [first.assignment(i) for i in range(42)] == [again.assignment(i) for i in range(42)]


def test_a_schedule_without_opponents_cannot_ask_for_external_games() -> None:
    with pytest.raises(ConfigError, match="no opponent named"):
        OpponentSchedule(10, 2, ())
    assert OpponentSchedule(10, 0, ()).assignment(3) is None


def test_an_index_outside_the_generation_is_refused() -> None:
    with pytest.raises(ConfigError, match="outside this generation"):
        OpponentSchedule(10, 2, ("greedy",)).ordinal(12)


def test_opponents_are_built_by_their_arena_names() -> None:
    built = build_opponents(("greedy", "minimax2"))
    assert set(built) == {"greedy", "minimax2"}
    assert built["minimax2"].name == "minimax-d2"
    with pytest.raises(ConfigError, match="same opponent"):
        build_opponents(("minimax2", "minimax-d2"))
    with pytest.raises(ConfigError, match="is a network"):
        build_opponents(("gen60=pyproject.toml@8",))


# ---------------------------------------------------------------------------
# One external game
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", [Player.BLACK, Player.WHITE])
def test_only_the_networks_decisions_are_recorded(network: Player) -> None:
    record, branching = play_game(
        StubEvaluator(),
        SEARCH,
        board_size=BOARD,
        rng=make_rng(3),
        opponent=GreedyAgent(),
        opponent_name="greedy",
        network=network,
    )
    assert record.samples, "a game with choices in it"
    assert all(sample.to_move is network for sample in record.samples)
    assert record.opponent == "greedy"
    assert record.network is network
    assert len(branching) == len(record.samples)


def test_every_recorded_z_is_the_games_result_for_the_network() -> None:
    record, _ = play_game(
        StubEvaluator(),
        SEARCH,
        board_size=BOARD,
        rng=make_rng(5),
        opponent=RandomAgent(),
        opponent_name="random",
        network=Player.WHITE,
    )
    expected = -record.result_for_black  # white's view
    assert all(sample.z == expected for sample in record.samples)
    assert record.black_discs + record.white_discs <= BOARD * BOARD


def test_a_self_play_game_records_no_opponent() -> None:
    record, _ = play_game(StubEvaluator(), SEARCH, board_size=BOARD, rng=make_rng(1))
    assert record.opponent is None
    assert record.network is None
    assert {sample.to_move for sample in record.samples} == {Player.BLACK, Player.WHITE}


def test_the_summary_counts_external_games_from_the_networks_side() -> None:
    summary = SelfPlaySummary()
    wipeouts = 0
    for seed in range(6):
        record, branching = play_game(
            StubEvaluator(),
            SEARCH,
            board_size=BOARD,
            rng=make_rng(seed),
            opponent=GreedyAgent(),
            opponent_name="greedy",
            network=Player.BLACK if seed % 2 == 0 else Player.WHITE,
        )
        summary.observe(record, branching)
        discs = record.black_discs if record.network is Player.BLACK else record.white_discs
        wipeouts += discs == 0
    assert summary.opponent_games == 6
    assert summary.network_wipeouts == wipeouts
    assert summary.opponent_wins + summary.opponent_losses + summary.opponent_draws == 6
    metrics = summary.as_metrics()
    assert metrics["opponent_games"] == 6
    assert 0.0 <= metrics["opponent_score"] <= 1.0
    assert 0.0 <= metrics["opponent_wipeout_rate"] <= 1.0


def test_pure_self_play_reports_zero_external_games() -> None:
    summary = SelfPlaySummary()
    record, branching = play_game(StubEvaluator(), SEARCH, board_size=BOARD, rng=make_rng(1))
    summary.observe(record, branching)
    assert summary.as_metrics()["opponent_games"] == 0
    assert summary.as_metrics()["opponent_score"] == 0.0


# ---------------------------------------------------------------------------
# Batched and unbatched play the same games
# ---------------------------------------------------------------------------


def fingerprint(records) -> dict:
    out = {}
    for record, branching in records:
        out[record.game_index] = (
            record.opponent,
            record.network,
            record.plies,
            record.passes,
            record.result_for_black,
            tuple(branching),
            tuple(
                (s.black, s.white, int(s.to_move), s.move_no, s.z, tuple(s.pi))
                for s in record.samples
            ),
        )
    return out


def test_batching_plays_exactly_the_same_external_games() -> None:
    schedule = OpponentSchedule(6, 4, ("greedy", "random"))
    opponents = build_opponents(schedule.opponents)
    solo = play_games(
        StubEvaluator(),
        SEARCH,
        board_size=BOARD,
        n_games=schedule.total,
        root_seed=7,
        generation=1,
        schedule=schedule,
        opponents=opponents,
    )
    batched = BatchedSelfPlay(
        StubEvaluator(),
        SEARCH,
        board_size=BOARD,
        games_in_flight=4,
        root_seed=7,
        generation=1,
        schedule=schedule,
        opponents=build_opponents(schedule.opponents),
    )
    a, b = fingerprint(solo), fingerprint(batched.play(schedule.total))
    assert a == b
    assert sum(1 for v in a.values() if v[0] is not None) == 4


def test_the_worker_offset_selects_the_generations_games_not_the_workers() -> None:
    """Game 5 of the generation is external whether worker 0 or worker 1 plays it."""
    schedule = OpponentSchedule(6, 4, ("greedy",))
    external = schedule.external_indices()
    first_worker = list(
        play_games(
            StubEvaluator(),
            SEARCH,
            board_size=BOARD,
            n_games=5,
            root_seed=7,
            generation=1,
            schedule=schedule,
            opponents=build_opponents(("greedy",)),
            first_game=0,
        )
    )
    second_worker = list(
        play_games(
            StubEvaluator(),
            SEARCH,
            board_size=BOARD,
            n_games=5,
            root_seed=7,
            generation=1,
            worker_id=1,
            schedule=schedule,
            opponents=build_opponents(("greedy",)),
            first_game=5,
        )
    )
    played = [r.opponent is not None for r, _ in first_worker] + [
        r.opponent is not None for r, _ in second_worker
    ]
    assert played == [i in external for i in range(10)]


def test_a_schedule_naming_an_unbuilt_opponent_fails_loudly() -> None:
    schedule = OpponentSchedule(2, 2, ("greedy",))
    with pytest.raises(WorkerError, match="none was built"):
        list(
            play_games(
                StubEvaluator(),
                SEARCH,
                board_size=BOARD,
                n_games=4,
                root_seed=1,
                generation=1,
                schedule=schedule,
                opponents={},
            )
        )


def test_external_games_finish_legally_against_a_searching_opponent() -> None:
    """Minimax searches; its moves must still be legal in every position it is given."""
    schedule = OpponentSchedule(0, 3, ("minimax-d2",))
    for record, _ in play_games(
        StubEvaluator(),
        SEARCH,
        board_size=BOARD,
        n_games=3,
        root_seed=11,
        generation=2,
        schedule=schedule,
        opponents=build_opponents(schedule.opponents),
    ):
        assert record.opponent == "minimax-d2"
        # The record holds only the network's positions, so the game cannot be
        # replayed; the counts must at least describe a finished game whose
        # result matches them. ``apply`` would have raised on an illegal move.
        assert record.black_discs + record.white_discs > rules.initial_state(BOARD).disc_count()
        margin = record.black_discs - record.white_discs
        assert record.result_for_black == (margin > 0) - (margin < 0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def smoke(**selfplay: object) -> dict:
    return {
        "name": "t",
        "seed": 7,
        "game": {"board_size": 4},
        "net": {"n_blocks": 1, "channels": 8, "value_hidden": 16},
        "mcts": {"n_simulations": 8, "temp_moves": 2, "dirichlet_alpha": 2.0},
        "selfplay": {
            "games_per_generation": 4,
            "n_workers": 1,
            "games_in_flight": 2,
            "max_generations": 1,
            **selfplay,
        },
        "replay": {"window": 512, "retain_shards": 4},
        "train": {"steps_per_generation": 2, "batch_size": 8},
    }


def test_the_default_config_is_pure_self_play() -> None:
    config = Config.model_validate(smoke())
    assert config.selfplay.opponents == ()
    assert config.selfplay.opponent_games == 0
    assert config.selfplay.total_games == 4
    assert config.selfplay.schedule() is None


def test_opponent_games_add_to_the_generation() -> None:
    config = Config.model_validate(smoke(opponents=["greedy", "minimax-d2"], opponent_games=2))
    assert config.selfplay.total_games == 6
    schedule = config.selfplay.schedule()
    assert schedule is not None
    assert schedule.opponents == ("greedy", "minimax-d2")
    assert len(schedule.external_indices()) == 2


def test_opponent_games_need_an_opponent() -> None:
    with pytest.raises(ValueError, match="name at least one fixed opponent"):
        Config.model_validate(smoke(opponent_games=2))


def test_duplicate_opponents_are_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        Config.model_validate(smoke(opponents=["greedy", "greedy"], opponent_games=2))
