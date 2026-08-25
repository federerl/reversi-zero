"""4x4 Reversi, solved exactly.

The 4x4 board is small enough to search to the end of every line -- 3,306 distinct
positions -- so its game-theoretic value is not a matter of opinion. That makes it
useful twice over.

**As an independent check on the rules engine.** This test never looks at a
single flip or a single legal-move list. It plays every possible game to the end
and asks who wins with perfect play. Almost any rules bug -- a flip direction
missed, a pass handled wrongly, a terminal condition off by one -- would change the
answer. The differential test proves the two engines agree with each other; this
proves the thing they agree on is Reversi.

**As the explanation for a result that otherwise looks like a bug.** The trained
4x4 agent scores near 100% as white and noticeably less as black. That asymmetry
is correct and expected: **white wins 4x4 with perfect play**, so black is playing
a theoretically lost game and can only win when the opponent errs. An agent that
scored *equally* with both colours on this board would be the surprising one.
"""

from __future__ import annotations

import sys
from functools import cache

import pytest

from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.types import Outcome, Player


@cache
def solve(black: int, white: int, to_move: int) -> Outcome:
    """The result with perfect play, from the point of view of the player to move.

    The same sign convention as everything else in the project (contract C2): the
    value flips once per level, because a position that is winning for me is
    losing for whoever chose the move that got here.
    """
    state = State(black=black, white=white, to_move=Player(to_move), size=4)
    if rules.is_terminal(state):
        return scoring.result(state)

    best = -2
    for action in rules.legal_actions(state):
        child = rules.apply(state, action)
        best = max(best, -solve(child.black, child.white, int(child.to_move)))
        if best == 1:
            break  # cannot do better than a win
    return best


@pytest.fixture(autouse=True)
def _deep_recursion() -> None:
    # A 4x4 game is at most ~16 plies, but the default limit is shared with
    # pytest's own stack, which is already deep by the time a test runs.
    sys.setrecursionlimit(10_000)


def test_white_wins_four_by_four_with_perfect_play() -> None:
    """The whole engine, checked against a fact about the game itself."""
    start = rules.initial_state(4)
    value_for_black = solve(start.black, start.white, int(start.to_move))

    assert value_for_black == -1, (
        "4x4 Reversi is a win for white with perfect play. A different answer here "
        "means the rules engine is not playing Reversi, however well its two "
        "implementations agree with each other."
    )


def test_the_solver_agrees_with_the_engine_on_finished_games() -> None:
    """Sanity on the solver itself, so a failure above accuses the right code."""
    finished = State(black=0, white=0xFFFF, to_move=Player.BLACK, size=4)
    assert rules.is_terminal(finished)
    assert solve(finished.black, finished.white, int(finished.to_move)) == -1

    black_won = State(black=0xFFFF, white=0, to_move=Player.BLACK, size=4)
    assert solve(black_won.black, black_won.white, int(black_won.to_move)) == 1


def test_a_perfect_white_player_never_loses_to_any_black_play() -> None:
    """Stronger than the headline: white's win does not depend on black cooperating.

    Whatever black plays first, white still has a winning reply -- which is what
    "solved for white" actually means, and what the trained agent has to have
    picked up to score as it does with that colour.
    """
    start = rules.initial_state(4)
    for action in rules.legal_actions(start):
        after_black = rules.apply(start, action)
        value_for_white = solve(after_black.black, after_black.white, int(after_black.to_move))
        assert value_for_white == 1, f"black playing {action} escapes the loss"
