"""Storing a board in two integers, and moving around it with bit shifts.

An 8x8 board has 64 squares, and a 64-bit integer has 64 bits — so one integer
can record "which of these squares hold a black disc" with one bit per square.
Two integers describe the whole position. Bit ``i`` corresponds to square
``i = row * size + col`` (contract C1).

Why bother, when ``reference.py`` already works: the search asks "what are the
legal moves here?" tens of millions of times per training generation. The
reference engine answers by looping over every square and walking outward in
eight directions. This one answers all squares at once, because a shift moves
*every* disc on the board one step in the same direction simultaneously.

The cost is that correctness is no longer visible by reading the code. That is
paid for by ``tests/unit/test_differential.py``, which makes both engines play
thousands of games and compares them at every move.

The one genuinely tricky part is wraparound. Squares are numbered in reading
order, so the square to the right of column 7 is column 0 of the *next row* —
numerically adjacent, but not adjacent on the board. Shifting sideways without
care teleports discs across the edge. Every sideways shift therefore drops the
discs that are already against that edge before shifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from reversi.types import Action, Bitboard

# The eight directions as (row step, column step), same order as reference.py.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class Shift:
    """One direction, precomputed as "drop these discs, then move this far".

    ``delta`` is how far the square index changes: ``row_step * size +
    col_step``. Positive means shift left (towards higher indices).

    ``guard`` is the set of squares allowed to make this move. Moving right, a
    disc already in the last column has nowhere to go, and shifting it anyway
    would wrap it onto the next row. Clearing those discs first is what keeps
    the board's edges real.
    """

    d_row: int
    d_col: int
    delta: int
    guard: Bitboard


@dataclass(frozen=True, slots=True)
class Geometry:
    """Everything about a board size that can be worked out once, in advance."""

    size: int
    n_squares: int
    full: Bitboard
    shifts: tuple[Shift, ...]


@cache
def geometry(size: int) -> Geometry:
    """Masks and shifts for a board of the given size. Computed once per size."""
    if size < 4 or size % 2 != 0:
        msg = f"board size must be even and at least 4, got {size}"
        raise ValueError(msg)

    n_squares = size * size
    full = (1 << n_squares) - 1

    # Squares not in the last column, and squares not in the first column.
    not_last_col = full
    not_first_col = full
    for row in range(size):
        not_last_col &= ~(1 << (row * size + size - 1))
        not_first_col &= ~(1 << (row * size))

    shifts = []
    for d_row, d_col in DIRECTIONS:
        if d_col > 0:
            guard = not_last_col
        elif d_col < 0:
            guard = not_first_col
        else:
            guard = full
        shifts.append(Shift(d_row=d_row, d_col=d_col, delta=d_row * size + d_col, guard=guard))

    return Geometry(size=size, n_squares=n_squares, full=full, shifts=tuple(shifts))


def shift(bits: Bitboard, step: Shift, full: Bitboard) -> Bitboard:
    """Move every disc one square in one direction.

    Discs that would leave the board simply disappear: sideways ones are
    removed by the guard beforehand, and vertical ones fall off the ends when
    the result is masked back down to the board.
    """
    bits &= step.guard
    if step.delta > 0:
        return (bits << step.delta) & full
    return bits >> -step.delta


# ===========================================================================
# Small helpers
# ===========================================================================


def popcount(bits: Bitboard) -> int:
    """How many discs. ``int.bit_count`` is a single CPU instruction."""
    return bits.bit_count()


def bit(index: Action) -> Bitboard:
    """The bitboard holding exactly one square."""
    return 1 << index


def indices(bits: Bitboard) -> list[Action]:
    """Which squares are set, in ascending order.

    ``bits & -bits`` isolates the lowest set bit, and ``bit_length() - 1`` turns
    that into its index. The loop therefore runs once per disc rather than once
    per square, which matters because most boards are mostly empty early on.
    """
    out: list[Action] = []
    while bits:
        lowest = bits & -bits
        out.append(lowest.bit_length() - 1)
        bits ^= lowest
    return out


def to_index(row: int, col: int, size: int) -> Action:
    """Contract C1: index = row * size + col, row 0 top, column 0 left."""
    return row * size + col


def to_coords(index: Action, size: int) -> tuple[int, int]:
    return divmod(index, size)
