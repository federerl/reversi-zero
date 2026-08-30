"""Measuring the difficulty ladder (criterion S15).

Two things are checked here, and they are different in kind.

The first is that the *agent* plays the level it says it does -- that a rung with
a guardrail actually applies it, and that the easiest rung is genuinely different
from the hardest rather than differing only in a config file.

The second is that the *judgement* is right: that a ladder which does not
separate is reported as failing. That matters more than it sounds. A calibration
that reports success regardless is worse than none at all, because it puts a
number on something nobody measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.arena.elo import Rating, RatingTable
from reversi.arena.stats import Interval
from reversi.difficulty.calibrate import (
    MIN_ADJACENT_GAP,
    DifficultyAgent,
    _judge,
    check_guardrail,
)
from reversi.difficulty.levels import LEVELS, DifficultyLevel, level_by_name
from reversi.game import rules
from reversi.search.evaluator import StubEvaluator

BOARD = 8


# ===========================================================================
# The agent plays the level it advertises
# ===========================================================================


def opinionated_stub() -> StubEvaluator:
    """A stand-in with a clear favourite, so a guardrail has something to bite on.

    A uniform evaluator would make every move equally good, and a guardrail that
    never has to reject anything proves nothing.
    """

    def policy(state):
        width = state.size * state.size + 1
        weights = np.zeros(width, dtype=np.float32)
        for offset, action in enumerate(rules.legal_actions(state)):
            weights[action] = 3.0 - offset  # first legal move strongly preferred
        return weights

    return StubEvaluator(policy=policy, value=lambda s: 0.0)


def test_the_agent_only_ever_plays_a_legal_move() -> None:
    """Contract C5 again, at the level the interface actually uses."""
    rng = np.random.default_rng(0)
    agent = DifficultyAgent(opinionated_stub(), level_by_name("casual"))

    state = rules.initial_state(BOARD)
    played = 0
    while not rules.is_terminal(state) and played < 80:
        action = agent.select(state, rng)
        assert action in rules.legal_actions(state)
        state = rules.apply(state, action)
        played += 1

    assert played > 20


def test_a_level_with_no_temperature_is_deterministic() -> None:
    """The top rungs play their best move, so two runs must agree.

    If they did not, a rated match against them would be measuring the sampling
    rather than the level.
    """
    agent = DifficultyAgent(opinionated_stub(), level_by_name("max"))
    state = rules.initial_state(BOARD)

    first = [agent.select(state, np.random.default_rng(1)) for _ in range(3)]
    assert len(set(first)) == 1


def test_the_easiest_level_does_not_always_play_the_best_move() -> None:
    """Otherwise it is the hardest level with a smaller budget, not an easier one.

    This is the property that makes the ladder a ladder rather than four labels.
    """
    agent = DifficultyAgent(opinionated_stub(), level_by_name("casual"))
    state = rules.initial_state(BOARD)

    chosen = {agent.select(state, np.random.default_rng(seed)) for seed in range(25)}
    assert len(chosen) > 1, "casual played the same move every time"


def test_a_forced_move_skips_the_search_entirely() -> None:
    """Nothing to choose between, so there is nothing to spend simulations on."""
    from reversi.game import reference as ref
    from reversi.game.state import State
    from reversi.types import Player, pass_action

    position = ref.from_ascii("BBW.\nBBWW\nWBWW\nBBBB", Player.WHITE)
    black, white = position.bitboards()
    state = State(black=black, white=white, to_move=Player.WHITE, size=4)

    stub = opinionated_stub()
    agent = DifficultyAgent(stub, level_by_name("max"))
    assert agent.select(state, np.random.default_rng(0)) == pass_action(4)
    assert stub.calls == 0, "the network was consulted about a position with one legal move"


# ===========================================================================
# The guardrail (S15e)
# ===========================================================================


def test_the_guardrail_holds_for_the_level_that_has_one() -> None:
    """Casual samples among moves, but never one far below the best available.

    This is what makes an easy opponent *weak* rather than *stupid* -- it plays a
    second-best move instead of giving away a corner.
    """
    report = check_guardrail(
        opinionated_stub(), level_by_name("casual"), board_size=BOARD, samples=60, seed=3
    )

    assert report.samples == 60
    assert report.violations == 0, report.describe()
    assert report.worst_drop <= level_by_name("casual").guard + 1e-9


def test_a_loosened_guardrail_is_reported_as_violated() -> None:
    """The check has to be able to fail, or it is decoration.

    A level is built here whose guardrail is far tighter than its sampling, so
    moves outside the rule are chosen and must be counted.
    """
    reckless = DifficultyLevel(
        name="reckless",
        label="Reckless",
        simulations=8,
        temperature=1.5,
        top_k=None,
        guard=0.0,  # only the very best move is permitted...
        description="samples widely while claiming to allow nothing",
    )

    def lopsided_value(state):
        # Values that differ per position, so sampled moves land at different Qs.
        return float(np.sin(state.black % 97) * 0.9)

    stub = StubEvaluator(
        policy=lambda s: np.asarray(
            [1.0 if a in rules.legal_actions(s) else 0.0 for a in range(s.size * s.size + 1)],
            dtype=np.float32,
        ),
        value=lopsided_value,
    )

    report = check_guardrail(stub, reckless, board_size=BOARD, samples=80, seed=5)
    assert report.violations > 0, "a guardrail of 0 with wide sampling should be violated"
    assert not report.ok


# ===========================================================================
# The judgement (S15a-d)
# ===========================================================================


def table(entries: dict[str, tuple[float, float, float]]) -> RatingTable:
    """Build a rating table from name -> (elo, ci_low, ci_high)."""
    return RatingTable(
        ratings=[
            Rating(name=name, elo=elo, interval=Interval(point=elo, low=low, high=high), games=600)
            for name, (elo, low, high) in entries.items()
        ],
        anchor="random",
        results={},
    )


class FakeTournament:
    """Just enough of a tournament for the judgement to read."""

    def __init__(self, ratings: RatingTable) -> None:
        self.ratings = ratings


def four_levels() -> list[DifficultyLevel]:
    return list(LEVELS)


def guard_ok():
    from reversi.difficulty.calibrate import GuardrailReport

    return GuardrailReport(level="casual", guard=0.35, samples=500, worst_drop=0.2, violations=0)


def test_a_well_separated_ladder_passes() -> None:
    ratings = table(
        {
            "casual": (100.0, 70.0, 130.0),
            "club": (220.0, 190.0, 250.0),
            "strong": (340.0, 310.0, 370.0),
            "max": (460.0, 430.0, 490.0),
        }
    )
    ratings.results[("casual", "random")] = (480.0, 600)

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert all(c["passed"] for c in checks.values()), checks


def test_a_ladder_out_of_order_fails_the_monotonic_check() -> None:
    # The failure this exists to catch: more search producing a *worse* opponent
    # would mean something is wrong with the search, not with the labels.
    ratings = table(
        {
            "casual": (300.0, 270.0, 330.0),
            "club": (220.0, 190.0, 250.0),
            "strong": (340.0, 310.0, 370.0),
            "max": (460.0, 430.0, 490.0),
        }
    )
    ratings.results[("casual", "random")] = (480.0, 600)

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert not checks["monotonic"]["passed"]


def test_rungs_that_are_too_close_fail_even_when_ordered() -> None:
    """Ordered but indistinguishable is the likely real-world failure.

    Four levels 20 Elo apart are in the right order and still, to a person
    playing a few games against each, the same opponent.
    """
    ratings = table(
        {
            "casual": (100.0, 90.0, 110.0),
            "club": (120.0, 110.0, 130.0),
            "strong": (140.0, 130.0, 150.0),
            "max": (160.0, 150.0, 170.0),
        }
    )
    ratings.results[("casual", "random")] = (480.0, 600)

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert checks["monotonic"]["passed"]
    assert not checks["adjacent_gap"]["passed"]
    assert all(gap < MIN_ADJACENT_GAP for gap in checks["adjacent_gap"]["gaps"])


def test_overlapping_intervals_fail_however_far_apart_the_points_are() -> None:
    """The strict reading, and the same one criterion S14 uses for generations.

    Two point estimates in the right order are not evidence when the intervals
    overlap. Holding the ladder to a weaker standard than the training result
    would be having it both ways.
    """
    ratings = table(
        {
            "casual": (100.0, 0.0, 400.0),
            "club": (200.0, 100.0, 500.0),
            "strong": (300.0, 200.0, 600.0),
            "max": (400.0, 300.0, 700.0),
        }
    )
    ratings.results[("casual", "random")] = (480.0, 600)

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert checks["adjacent_gap"]["passed"]
    assert not checks["intervals_disjoint"]["passed"]


def test_an_easiest_rung_that_loses_to_random_fails() -> None:
    """An opponent that cannot beat random play is not an easy rung; it is broken."""
    ratings = table(
        {
            "casual": (100.0, 70.0, 130.0),
            "club": (220.0, 190.0, 250.0),
            "strong": (340.0, 310.0, 370.0),
            "max": (460.0, 430.0, 490.0),
        }
    )
    ratings.results[("casual", "random")] = (300.0, 600)  # 50%

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert not checks["easiest_beats_random"]["passed"]


def test_the_pairing_is_found_whichever_way_round_it_was_stored() -> None:
    """Pairings are recorded once, in entrant order. Reading only one direction
    would silently skip the check rather than fail it."""
    ratings = table(
        {
            "casual": (100.0, 70.0, 130.0),
            "club": (220.0, 190.0, 250.0),
            "strong": (340.0, 310.0, 370.0),
            "max": (460.0, 430.0, 490.0),
        }
    )
    # Stored as (random, casual): random scored 120 of 600, so casual scored 480.
    ratings.results[("random", "casual")] = (120.0, 600)

    checks = _judge(FakeTournament(ratings), four_levels(), guard_ok(), include_baselines=True)
    assert checks["easiest_beats_random"]["passed"], checks["easiest_beats_random"]["detail"]


def test_a_violated_guardrail_fails_the_whole_calibration() -> None:
    from reversi.difficulty.calibrate import GuardrailReport

    ratings = table(
        {
            "casual": (100.0, 70.0, 130.0),
            "club": (220.0, 190.0, 250.0),
            "strong": (340.0, 310.0, 370.0),
            "max": (460.0, 430.0, 490.0),
        }
    )
    ratings.results[("casual", "random")] = (480.0, 600)
    bad = GuardrailReport(level="casual", guard=0.35, samples=500, worst_drop=0.9, violations=7)

    checks = _judge(FakeTournament(ratings), four_levels(), bad, include_baselines=True)
    assert not checks["guardrail"]["passed"]
    assert "7 violation" in checks["guardrail"]["detail"]


@pytest.mark.parametrize("level", LEVELS)
def test_every_shipped_level_can_be_built_and_asserts_no_noise(level: DifficultyLevel) -> None:
    """Contract C7. Exploration noise belongs to self-play; a rated match with it
    on would be measuring something other than how the level plays."""
    agent = DifficultyAgent(opinionated_stub(), level)
    assert agent.name == level.name
    assert agent.mcts.config.dirichlet_eps == 0.0
    assert agent.mcts.config.n_simulations == level.simulations
