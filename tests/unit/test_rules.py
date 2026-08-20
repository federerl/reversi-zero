"""Hand-checked Reversi rules (test matrix T1-T9).

Every position here is written out as a picture so it can be verified by eye:
``B`` is a black disc, ``W`` white, ``.`` empty, and row 0 is the top line.
That matters because these tests are the ground truth the whole project rests
on. A test you cannot check by reading it is not evidence of much.

Where a test says "black plays X", the board shows the position *before* the
move and the assertion names exactly which discs must flip.
"""

from __future__ import annotations

import pytest

from reversi.errors import IllegalMoveError
from reversi.game import reference as ref
from reversi.types import Player, pass_action

# ---------------------------------------------------------------------------
# T1 - flipping in each of the eight directions
#
# Each board is 6x6 with black to play at row 2, column 2 (index 14). A single
# white disc sits one step away in the direction under test, with a black disc
# one step beyond it to close the trap. Exactly one disc may flip.
# ---------------------------------------------------------------------------

PLAY_SQUARE = 14  # row 2, col 2 on a 6x6 board

DIRECTION_CASES = {
    "up_left": (
        """
        B.....
        .W....
        ......
        ......
        ......
        ......
        """,
        7,  # the white disc at row 1, col 1
    ),
    "up": (
        """
        ..B...
        ..W...
        ......
        ......
        ......
        ......
        """,
        8,  # row 1, col 2
    ),
    "up_right": (
        """
        ....B.
        ...W..
        ......
        ......
        ......
        ......
        """,
        9,  # row 1, col 3
    ),
    "left": (
        """
        ......
        ......
        BW....
        ......
        ......
        ......
        """,
        13,  # row 2, col 1
    ),
    "right": (
        """
        ......
        ......
        ..-WB.
        ......
        ......
        ......
        """,
        15,  # row 2, col 3
    ),
    "down_left": (
        """
        ......
        ......
        ......
        .W....
        B.....
        ......
        """,
        19,  # row 3, col 1
    ),
    "down": (
        """
        ......
        ......
        ......
        ..W...
        ..B...
        ......
        """,
        20,  # row 3, col 2
    ),
    "down_right": (
        """
        ......
        ......
        ......
        ...W..
        ....B.
        ......
        """,
        21,  # row 3, col 3
    ),
}


@pytest.mark.parametrize(("name", "case"), DIRECTION_CASES.items(), ids=list(DIRECTION_CASES))
def test_flips_in_each_of_the_eight_directions(name: str, case: tuple[str, int]) -> None:
    board, expected_flip = case
    state = ref.from_ascii(board.replace("-", "."), to_move=Player.BLACK)

    assert ref.flips(state, PLAY_SQUARE) == [expected_flip], f"{name} flipped the wrong discs"
    assert PLAY_SQUARE in ref.legal_placements(state), f"{name} should be a legal move"


@pytest.mark.parametrize(("name", "case"), DIRECTION_CASES.items(), ids=list(DIRECTION_CASES))
def test_playing_flips_exactly_that_disc(name: str, case: tuple[str, int]) -> None:
    board, expected_flip = case
    state = ref.from_ascii(board.replace("-", "."), to_move=Player.BLACK)
    after = ref.apply(state, PLAY_SQUARE)

    row, col = ref.to_coords(expected_flip, 6)
    assert after.cell(row, col) is Player.BLACK, f"{name}: trapped disc did not flip"
    assert after.cell(2, 2) is Player.BLACK, f"{name}: played disc was not placed"
    assert after.to_move is Player.WHITE


# ---------------------------------------------------------------------------
# T2 - one move flipping in several directions at once
# ---------------------------------------------------------------------------


def test_flips_in_three_directions_at_once() -> None:
    """Black plays row 2 col 2. White discs to its left, above, and above-left
    are each backed by a black disc, so all three lines flip together."""
    state = ref.from_ascii(
        """
        B.B...
        .WW...
        BW....
        ......
        ......
        ......
        """,
        to_move=Player.BLACK,
    )
    assert sorted(ref.flips(state, 14)) == [7, 8, 13]

    after = ref.apply(state, 14)
    black, white = ref.disc_counts(after)
    assert (black, white) == (7, 0), "3 existing + 3 flipped + 1 placed = 7 black, 0 white"


def test_flips_in_five_directions_at_once() -> None:
    """A white disc in five directions from the played square, each closed off
    by a black disc. All five lines flip from a single move."""
    state = ref.from_ascii(
        """
        B.B.B.
        .WWW..
        BW....
        ..W...
        ..B...
        ......
        """,
        to_move=Player.BLACK,
    )
    # up-left (7), up (8), up-right (9), left (13), down (20)
    assert sorted(ref.flips(state, 14)) == [7, 8, 9, 13, 20]


def test_a_long_run_flips_entirely() -> None:
    """Trapped runs flip in full, not just the disc nearest the new one."""
    state = ref.from_ascii(
        """
        ......
        ......
        .WWWWB
        ......
        ......
        ......
        """,
        to_move=Player.BLACK,
    )
    assert sorted(ref.flips(state, 12)) == [13, 14, 15, 16]


# ---------------------------------------------------------------------------
# T3 - a move that traps nothing is illegal
# ---------------------------------------------------------------------------


def test_move_next_to_own_disc_with_nothing_to_trap_is_illegal() -> None:
    state = ref.from_ascii(
        """
        ......
        ......
        ..B...
        ......
        ......
        ......
        """,
        to_move=Player.BLACK,
    )
    assert ref.flips(state, 13) == []
    assert 13 not in ref.legal_placements(state)


def test_unterminated_run_does_not_flip() -> None:
    """White discs running off the edge with no black disc to close the trap."""
    state = ref.from_ascii(
        """
        ......
        ......
        .WWWW.
        ......
        ......
        ......
        """,
        to_move=Player.BLACK,
    )
    assert ref.flips(state, 12) == []


def test_gap_breaks_the_run() -> None:
    """An empty square between the white run and the closing black disc means
    nothing is trapped."""
    state = ref.from_ascii(
        """
        ......
        ......
        .WW.B.
        ......
        ......
        ......
        """,
        to_move=Player.BLACK,
    )
    assert ref.flips(state, 12) == []


# ---------------------------------------------------------------------------
# T4 - forced pass (contract C3)
# ---------------------------------------------------------------------------


def test_player_with_no_move_must_pass() -> None:
    """White is to move and has no legal square, but black does. The only
    legal action is PASS, the game is not over, and no disc changes."""
    state = ref.from_ascii(
        """
        BBBB
        BBBB
        BBBW
        ....
        """,
        to_move=Player.WHITE,
    )
    assert ref.legal_placements(state) == []
    assert ref.legal_placements_for(state, Player.BLACK) != []
    assert ref.legal_actions(state) == [pass_action(4)]
    assert not ref.is_terminal(state)

    after = ref.apply(state, pass_action(4))
    assert after.grid == state.grid, "passing must not move any disc"
    assert after.to_move is Player.BLACK, "passing hands the turn over"


def test_cannot_pass_when_a_move_is_available() -> None:
    state = ref.initial_state(8)
    assert ref.legal_placements(state) != []
    with pytest.raises(IllegalMoveError, match="cannot pass"):
        ref.apply(state, pass_action(8))


# ---------------------------------------------------------------------------
# T5, T6 - the game ending (contract C3)
# ---------------------------------------------------------------------------


def test_terminal_when_neither_side_can_move_with_empties_left() -> None:
    """All discs are black, so white has nothing to flip and black has nothing
    to flip either. The game is over despite four empty squares."""
    state = ref.from_ascii(
        """
        BBBB
        BBBB
        BBBB
        ....
        """,
        to_move=Player.WHITE,
    )
    assert ref.is_terminal(state)
    assert ref.legal_actions(state) == []
    assert ref.disc_counts(state) == (12, 0)


def test_full_board_is_terminal_without_a_special_case() -> None:
    """Termination falls out of "neither player has a move" - there is no
    separate "is the board full" branch to get wrong."""
    state = ref.from_ascii(
        """
        BBBB
        BBBB
        WWWW
        WWWW
        """,
        to_move=Player.BLACK,
    )
    assert ref.is_terminal(state)
    assert ref.legal_actions(state) == []


def test_two_passes_in_a_row_is_not_a_reachable_state() -> None:
    """A position where the mover has no move is either a pass (opponent can
    move) or terminal. It can never require counting a second pass."""
    state = ref.from_ascii(
        """
        BBBB
        BBBB
        BBBB
        ....
        """,
        to_move=Player.WHITE,
    )
    assert ref.legal_actions(state) == []  # terminal, not a pass
    assert ref.is_terminal(state)


# ---------------------------------------------------------------------------
# T7 - illegal moves are rejected with a usable message
# ---------------------------------------------------------------------------


def test_occupied_square_is_rejected() -> None:
    state = ref.initial_state(8)
    with pytest.raises(IllegalMoveError, match="occupied"):
        ref.apply(state, 27)  # d4, one of the four starting discs


def test_out_of_range_action_is_rejected() -> None:
    state = ref.initial_state(8)
    with pytest.raises(IllegalMoveError, match="out of range"):
        ref.apply(state, 999)


def test_move_that_traps_nothing_is_rejected() -> None:
    state = ref.initial_state(8)
    with pytest.raises(IllegalMoveError, match="traps no opposing discs"):
        ref.apply(state, 0)  # a1, far from every disc


def test_any_action_at_a_finished_game_is_rejected() -> None:
    state = ref.from_ascii("BBBB\nBBBB\nBBBB\nBBBB", to_move=Player.WHITE)
    with pytest.raises(IllegalMoveError, match="already over"):
        ref.apply(state, 0)


def test_error_message_can_reconstruct_the_position() -> None:
    """An illegal move outside a test means another layer failed to filter it,
    so the message has to work as a standalone bug report."""
    state = ref.initial_state(8)
    with pytest.raises(IllegalMoveError) as excinfo:
        ref.apply(state, 0)

    message = str(excinfo.value)
    assert "to_move=black" in message
    assert "size=8" in message
    assert "black=0x" in message and "white=0x" in message
    assert "legal=" in message


# ---------------------------------------------------------------------------
# T8 - scoring
# ---------------------------------------------------------------------------


def test_disc_counts_and_result_for_a_win() -> None:
    state = ref.from_ascii(
        """
        BBBB
        BBBB
        BBBB
        WWWW
        """,
        to_move=Player.BLACK,
    )
    assert ref.disc_counts(state) == (12, 4)
    assert ref.result(state) == 1, "black to move and ahead 12-4"


def test_result_is_from_the_movers_point_of_view() -> None:
    """The same board scores +1 or -1 depending only on whose turn it is. One
    perspective convention, used everywhere, is what keeps sign errors out of
    the search later."""
    board = """
        BBBB
        BBBB
        BBBB
        WWWW
        """
    assert ref.result(ref.from_ascii(board, to_move=Player.BLACK)) == 1
    assert ref.result(ref.from_ascii(board, to_move=Player.WHITE)) == -1


def test_draw_is_zero() -> None:
    state = ref.from_ascii("BBBB\nBBBB\nWWWW\nWWWW", to_move=Player.BLACK)
    assert ref.disc_counts(state) == (8, 8)
    assert ref.result(state) == 0


def test_wipeout_when_a_player_has_no_discs_left() -> None:
    """Losing every disc ends the game - neither side can move, because a legal
    move requires trapping an opponent disc and there are none."""
    state = ref.from_ascii(
        """
        BBB.
        BBB.
        BBB.
        ....
        """,
        to_move=Player.WHITE,
    )
    assert ref.disc_counts(state) == (9, 0)
    assert ref.is_terminal(state)
    assert ref.result(state) == -1, "white has no discs, so white loses"


def test_empty_squares_count_for_nobody() -> None:
    state = ref.from_ascii("BBBB\nBBBB\nWWWW\nWWWW", to_move=Player.BLACK)
    black, white = ref.disc_counts(state)
    assert black + white == 16, "all squares filled here"

    partial = ref.from_ascii("BBBB\nBBBB\nWWWW\n....", to_move=Player.BLACK)
    assert ref.disc_counts(partial) == (8, 4), "the empty row belongs to neither side"


# ---------------------------------------------------------------------------
# T9 - positions never change underneath you
# ---------------------------------------------------------------------------


def test_apply_returns_a_new_state_and_leaves_the_original_alone() -> None:
    state = ref.initial_state(8)
    before = ref.to_ascii(state)

    for action in ref.legal_placements(state):
        ref.apply(state, action)

    assert ref.to_ascii(state) == before
    assert state.to_move is Player.BLACK


def test_a_thousand_applies_do_not_disturb_the_original() -> None:
    """Immutability removes the worst bug class in a search tree: a node quietly
    holding a board that something else has since modified."""
    state = ref.initial_state(8)
    snapshot = state.bitboards()

    for _ in range(1000):
        ref.apply(state, ref.legal_placements(state)[0])

    assert state.bitboards() == snapshot


def test_states_are_hashable_and_compare_by_value() -> None:
    a = ref.initial_state(8)
    b = ref.initial_state(8)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# Board setup and text round-tripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [4, 6, 8])
def test_opening_position(size: int) -> None:
    state = ref.initial_state(size)
    assert ref.disc_counts(state) == (2, 2)
    assert state.to_move is Player.BLACK, "black always moves first"
    assert len(ref.legal_placements(state)) == 4, "the opening always offers four moves"


def test_standard_8x8_opening_matches_othello() -> None:
    """d3, c4, f5, e6 - the four opening moves in every Othello book."""
    assert ref.legal_placements(ref.initial_state(8)) == [19, 26, 37, 44]


@pytest.mark.parametrize("size", [3, 5, 7, 2, 0])
def test_odd_or_tiny_boards_are_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="even and at least 4"):
        ref.initial_state(size)


def test_ascii_round_trip() -> None:
    state = ref.initial_state(8)
    assert ref.from_ascii(ref.to_ascii(state)).grid == state.grid


def test_ascii_rejects_ragged_or_unknown_characters() -> None:
    with pytest.raises(ValueError, match="square"):
        ref.from_ascii("BBB\nBB\nBBB")
    with pytest.raises(ValueError, match="only contain"):
        ref.from_ascii("BBXB\nBBBB\nBBBB\nBBBB")


def test_index_and_coordinates_agree_with_contract_c1() -> None:
    """index = row * size + col, row 0 at the top, column 0 on the left. Shared
    by the engine, the network encoding, the symmetries, the API, and the UI."""
    assert ref.to_index(0, 0, 8) == 0
    assert ref.to_index(0, 7, 8) == 7
    assert ref.to_index(1, 0, 8) == 8
    assert ref.to_index(7, 7, 8) == 63
    for index in range(64):
        row, col = ref.to_coords(index, 8)
        assert ref.to_index(row, col, 8) == index
