"""The rules of Reversi, computed with bit operations.

Same rules as ``reference.py``, same answers, about ten times faster (measured
on this laptop: 369 random games per second against the reference engine's 39).

The speed comes from one idea: a shift moves *every* disc on the board one step
in the same direction at once. So instead of looping over 64 squares and
walking outward from each, we work on all squares simultaneously.

Finding the legal moves in one direction, in four lines:

1. Shift my discs one step. Keep the results that landed on an opponent disc —
   those are opponent discs sitting directly next to one of mine.
2. Repeat, growing each of those into the full run of opponent discs behind it.
3. Shift once more. Anywhere that lands on an empty square is a legal move,
   because it means: empty square, then an unbroken run of opponent discs, then
   one of my discs. Which is the rule.
4. Do that for all eight directions and combine.

Correctness here is not obvious from reading, unlike ``reference.py``. That is
what ``tests/unit/test_differential.py`` is for: both engines play thousands of
games and are compared at every move.
"""

from __future__ import annotations

from reversi.errors import IllegalMoveError
from reversi.game.bitboard import geometry, indices, shift
from reversi.game.state import State, initial_state
from reversi.types import Action, Bitboard, Player, pass_action

__all__ = [
    "State",
    "apply",
    "flips",
    "initial_state",
    "is_terminal",
    "legal_actions",
    "legal_placements",
    "legal_placements_for",
    "must_pass",
]


# ===========================================================================
# Legal moves
# ===========================================================================


def _placements(mine: Bitboard, theirs: Bitboard, size: int) -> Bitboard:
    """Every square this player could legally play, as a bitboard."""
    geo = geometry(size)
    empty = geo.full & ~(mine | theirs)
    moves = 0

    for step in geo.shifts:
        # Opponent discs sitting directly next to one of mine, in this direction.
        run = shift(mine, step, geo.full) & theirs

        # Grow each into the whole unbroken run of opponent discs behind it. A
        # run can be at most size-2 long, and each pass extends it by one, so
        # size-2 passes is comfortably enough. Extra passes change nothing once
        # the run has stopped growing.
        for _ in range(size - 2):
            run |= shift(run, step, geo.full) & theirs

        # One step past the end of the run: if that square is empty, playing it
        # would trap the whole run.
        moves |= shift(run, step, geo.full) & empty

    return moves


def legal_placements(state: State) -> Bitboard:
    """Squares the player to move may play, as a bitboard."""
    return _placements(state.mine, state.theirs, state.size)


def legal_placements_for(state: State, player: Player) -> Bitboard:
    """Squares ``player`` could play, whether or not it is their turn."""
    mine, theirs = (
        (state.black, state.white) if player is Player.BLACK else (state.white, state.black)
    )
    return _placements(mine, theirs, state.size)


# ===========================================================================
# What a move flips
# ===========================================================================


def _flips_from(square: Bitboard, mine: Bitboard, theirs: Bitboard, size: int) -> Bitboard:
    """Discs flipped by playing ``square``, across all eight directions.

    Per direction: step outward collecting opponent discs, then check whether
    the next square holds one of mine. If it does, the collected run is trapped
    and flips; otherwise this direction contributes nothing. The loop runs at
    most size-1 times because that is the longest possible run.
    """
    geo = geometry(size)
    flipped = 0

    for step in geo.shifts:
        run = 0
        probe = shift(square, step, geo.full)
        while probe & theirs:
            run |= probe
            probe = shift(probe, step, geo.full)
        if run and (probe & mine):
            flipped |= run

    return flipped


def flips(state: State, action: Action) -> Bitboard:
    """Discs the player to move would flip by playing ``action``.

    Zero when the move is illegal, since trapping nothing is exactly what makes
    a move illegal.
    """
    geo = geometry(state.size)
    if not 0 <= action < geo.n_squares:
        return 0
    square = 1 << action
    if square & (state.black | state.white):
        return 0
    return _flips_from(square, state.mine, state.theirs, state.size)


# ===========================================================================
# Passing and ending — contract C3
#
#   if I have a placement:      play it;      not over
#   elif my opponent has one:   I must pass;  not over
#   else:                       game over
#
# There is no counting of consecutive passes anywhere. Two passes in a row would
# mean neither player can move, and that is caught here as the game being over.
# A full board is terminal by the same rule rather than by a special case.
# ===========================================================================


def legal_actions(state: State) -> list[Action]:
    """Everything the player to move may do, in ascending order.

    Squares to play, or exactly ``[PASS]`` when they have no move but the
    opponent does, or an empty list when the game is over.
    """
    placements = legal_placements(state)
    if placements:
        return indices(placements)
    if _placements(state.theirs, state.mine, state.size):
        return [pass_action(state.size)]
    return []


def is_terminal(state: State) -> bool:
    """True when neither player has a legal move."""
    return not legal_placements(state) and not _placements(state.theirs, state.mine, state.size)


def must_pass(state: State) -> bool:
    """True when the player to move has no square to play but the game goes on."""
    return not legal_placements(state) and bool(_placements(state.theirs, state.mine, state.size))


# ===========================================================================
# Playing a move
# ===========================================================================


def apply(state: State, action: Action) -> State:
    """Play ``action`` and return the resulting position.

    The given state is not modified.
    """
    geo = geometry(state.size)

    if action == pass_action(state.size):
        if legal_placements(state):
            raise IllegalMoveError(_explain(state, action, "cannot pass while a move exists"))
        if not _placements(state.theirs, state.mine, state.size):
            raise IllegalMoveError(_explain(state, action, "the game is already over"))
        return state.with_discs(state.mine, state.theirs)

    if not 0 <= action < geo.n_squares:
        raise IllegalMoveError(_explain(state, action, "action out of range"))

    square = 1 << action
    if square & (state.black | state.white):
        raise IllegalMoveError(_explain(state, action, "square is occupied"))

    flipped = _flips_from(square, state.mine, state.theirs, state.size)
    if not flipped:
        raise IllegalMoveError(_explain(state, action, "move traps no opposing discs"))

    # My discs gain the new square and everything it trapped; theirs lose what
    # was trapped. Then it is the other player's turn.
    return state.with_discs(
        mine=state.mine | flipped | square,
        theirs=state.theirs & ~flipped,
    )


def _explain(state: State, action: Action, reason: str) -> str:
    """Error text carrying enough detail to reconstruct the position.

    An illegal move reaching this point outside a test means some other layer
    failed to filter it (contract C5), so the message has to stand alone as a
    bug report.
    """
    return (
        f"illegal action {action}: {reason}. "
        f"to_move={state.to_move.label} size={state.size} "
        f"black=0x{state.black:016x} white=0x{state.white:016x} "
        f"legal={legal_actions(state)}\n{state}"
    )
