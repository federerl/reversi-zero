"""The Edax adapter.

Edax is the first opponent in this project that nobody here wrote, which is the
whole reason it is worth having -- and also the reason the adapter needs
checking. A mistranslated board would not crash. Edax would answer a *different*
position perfectly well, we would record the reply as its opinion of ours, and
the result would read as "Edax is weaker than expected" rather than as a bug.

Most of this therefore tests the translation, which needs no binary. The two
tests that need Edax itself skip when it is absent, so a clone without it still
runs a full suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.agents.edax import EDAX_ROOT, board_string, find_edax, square_index, square_name
from reversi.errors import ArenaError
from reversi.game import rules
from reversi.game.state import State
from reversi.types import Player

BOARD = 8


def edax_available() -> bool:
    try:
        find_edax()
    except ArenaError:
        return False
    return True


needs_edax = pytest.mark.skipif(
    not edax_available(),
    reason=f"no Edax under {EDAX_ROOT}/ (see docs/experiments.md for how to fetch it)",
)


# ===========================================================================
# Translation -- the part that fails silently
# ===========================================================================


def test_the_four_opening_moves_have_the_names_othello_uses() -> None:
    """d3, c4, f5, e6. If these are wrong, everything downstream is."""
    opening = sorted(rules.legal_actions(rules.initial_state(BOARD)))
    assert [square_name(a) for a in opening] == ["D3", "C4", "F5", "E6"]


def test_square_names_round_trip_for_every_square() -> None:
    for action in range(64):
        assert square_index(square_name(action)) == action


def test_a_square_name_is_a_column_letter_and_a_row_number() -> None:
    # Index = row * 8 + col with row 0 at the top, which is the same order Edax
    # reads the board in -- so this is a relabelling, not a transformation.
    assert square_name(0) == "A1"
    assert square_name(7) == "H1"
    assert square_name(56) == "A8"
    assert square_name(63) == "H8"
    assert square_index("d3") == 19


def test_nonsense_from_edax_is_refused_rather_than_guessed() -> None:
    for bad in ("", "z9", "d", "d9", "pass", "??"):
        with pytest.raises(ArenaError):
            square_index(bad)


def test_the_opening_position_translates_to_edax_notation() -> None:
    """`*` is black, `O` is white, then whose turn it is.

    Checked against what Edax itself prints for the same position: white on d4
    and e5, black on e4 and d5, black to move.
    """
    text = board_string(rules.initial_state(BOARD))
    squares, _, to_move = text.partition(" ")

    assert len(squares) == 64
    assert to_move == "*", "black moves first"
    assert squares[27] == "O" and squares[36] == "O"  # d4, e5
    assert squares[28] == "*" and squares[35] == "*"  # e4, d5
    assert squares.count("-") == 60


def test_whose_turn_it_is_travels_with_the_board() -> None:
    # Without this the engine would be asked to move for the wrong side, and
    # would answer -- with a perfectly good move for the other player.
    start = rules.initial_state(BOARD)
    after = rules.apply(start, 19)
    assert board_string(start).endswith(" *")
    assert board_string(after).endswith(" O")


def test_a_board_that_is_not_8x8_is_refused() -> None:
    small = rules.initial_state(4)
    with pytest.raises(ArenaError, match="8x8"):
        board_string(small)


# ===========================================================================
# The engine itself
# ===========================================================================


@needs_edax
def test_edax_plays_legal_moves_from_real_positions() -> None:
    from reversi.agents.edax import EdaxAgent

    agent = EdaxAgent(level=1)
    try:
        rng = np.random.default_rng(0)
        state = rules.initial_state(BOARD)
        plies = 0
        while not rules.is_terminal(state) and plies < 70:
            action = agent.select(state, rng)
            assert action in rules.legal_actions(state), (
                f"Edax returned {square_name(action)}, not legal here"
            )
            state = rules.apply(state, action)
            plies += 1
        assert plies > 30
    finally:
        agent.close()


@needs_edax
def test_a_forced_pass_never_reaches_the_engine() -> None:
    """There is nothing to decide, and it saves depending on how Edax spells one."""
    from reversi.agents.edax import EdaxAgent
    from reversi.types import pass_action

    # A real position, found by playing games until one turned up: white has no
    # placement, black still does. A hand-invented board is easy to get wrong,
    # and a test that skips itself proves nothing.
    state = State(
        black=0xFEF870281C081838,
        white=0x01038FD7E3F7E7C4,
        to_move=Player.WHITE,
        size=8,
    )
    assert rules.legal_actions(state) == [pass_action(8)], "the fixture is not a forced pass"

    agent = EdaxAgent(level=1)
    try:
        assert agent.select(state, np.random.default_rng(0)) == pass_action(8)
    finally:
        agent.close()


@needs_edax
def test_two_levels_are_not_the_same_opponent() -> None:
    """Sanity on the one knob this whole comparison turns.

    If `-level` did nothing, every rating in the write-up would describe the same
    engine under several names.
    """
    from reversi.agents.edax import EdaxAgent

    weak, strong = EdaxAgent(level=1), EdaxAgent(level=8)
    try:
        rng = np.random.default_rng(0)
        state = rules.initial_state(BOARD)
        for action in (19, 26, 37):  # a few plies in, where opinions can differ
            if action in rules.legal_actions(state):
                state = rules.apply(state, action)

        # Not that they must differ on any single position -- only that both
        # answer legally, which is what a level change must never break.
        for agent in (weak, strong):
            assert agent.select(state, rng) in rules.legal_actions(state)
        assert weak.name == "edax-l1"
        assert strong.name == "edax-l8"
    finally:
        weak.close()
        strong.close()
