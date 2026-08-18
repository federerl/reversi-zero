"""The eight ways a Reversi board can be turned without changing the game.

Rotate the board 90, 180, or 270 degrees, or mirror it across either axis or
either diagonal, and you get eight arrangements in total. The rules do not care
which one you are looking at: the same moves are legal, the same discs flip, and
the same side wins. A position and its rotation are the same problem with the
same right answer.

That gives the training a free eightfold multiplier. Every position recorded
during self-play can be shown to the network in any of its eight orientations,
and the correct answer just gets turned the same way. Eight times the training
data for the cost of shuffling 64 numbers.

**The detail that is easy to get wrong:** a policy vector has 65 entries — 64
squares plus "pass". The 64 square entries get permuted. The pass entry does
not, because passing is not a square and no rotation moves it. Rotating all 65
entries would quietly scramble every pass probability in the training set.

One honest subtlety, worth knowing before someone asks. Four of the eight
turns map the standard starting position onto itself; the other four map it
onto the mirrored start. Positions from that second group are perfectly legal
Reversi positions, but they are not reachable from the standard opening. That
shifts the training distribution very slightly. We accept it, because the
correctness argument above depends only on the rules being symmetric, not on
which positions happen to be reachable — and it is what every AlphaZero-style
Othello implementation does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

from reversi.game.state import State
from reversi.types import Action

# Each transform, as a function from (row, col) to its new (row, col).
_TRANSFORMS: tuple[tuple[str, Callable[[int, int, int], tuple[int, int]]], ...] = (
    ("identity", lambda r, c, _n: (r, c)),
    ("rotate_90", lambda r, c, n: (c, n - 1 - r)),
    ("rotate_180", lambda r, c, n: (n - 1 - r, n - 1 - c)),
    ("rotate_270", lambda r, c, n: (n - 1 - c, r)),
    ("mirror_columns", lambda r, c, n: (r, n - 1 - c)),
    ("mirror_rows", lambda r, c, n: (n - 1 - r, c)),
    ("transpose", lambda r, c, _n: (c, r)),
    ("anti_transpose", lambda r, c, n: (n - 1 - c, n - 1 - r)),
)


@dataclass(frozen=True, slots=True)
class Symmetry:
    """One of the eight turns, precomputed as a lookup table.

    ``perm[i]`` is where square ``i`` ends up. ``inverse[j]`` is where square
    ``j`` came from, so applying one then the other gets you back.
    """

    name: str
    size: int
    perm: tuple[int, ...]
    inverse: tuple[int, ...]

    def action(self, action: Action) -> Action:
        """Where an action moves to. PASS is left alone (see the module docs)."""
        n_squares = self.size * self.size
        return self.perm[action] if action < n_squares else action

    def inverse_action(self, action: Action) -> Action:
        n_squares = self.size * self.size
        return self.inverse[action] if action < n_squares else action


@cache
def symmetries(size: int) -> tuple[Symmetry, ...]:
    """The eight turns for a board of this size. Computed once per size."""
    n_squares = size * size
    out = []

    for name, transform in _TRANSFORMS:
        perm = [0] * n_squares
        for row in range(size):
            for col in range(size):
                new_row, new_col = transform(row, col, size)
                perm[row * size + col] = new_row * size + new_col

        inverse = [0] * n_squares
        for source, destination in enumerate(perm):
            inverse[destination] = source

        out.append(Symmetry(name=name, size=size, perm=tuple(perm), inverse=tuple(inverse)))

    return tuple(out)


# ===========================================================================
# Applying a turn
# ===========================================================================


def transform_bits(bits: int, sym: Symmetry) -> int:
    """Turn one bitboard. Runs once per square that holds a disc."""
    out = 0
    remaining = bits
    while remaining:
        lowest = remaining & -remaining
        out |= 1 << sym.perm[lowest.bit_length() - 1]
        remaining ^= lowest
    return out


def transform_state(state: State, sym: Symmetry) -> State:
    """Turn a whole position. Whose turn it is does not change."""
    return State(
        black=transform_bits(state.black, sym),
        white=transform_bits(state.white, sym),
        to_move=state.to_move,
        size=state.size,
    )


def transform_policy(policy: Sequence[float], sym: Symmetry) -> list[float]:
    """Turn a policy vector the same way the board was turned.

    ``policy`` has one entry per square plus a final entry for PASS. The square
    entries move; the PASS entry stays put, because passing is not a square.
    """
    n_squares = sym.size * sym.size
    expected = n_squares + 1
    if len(policy) != expected:
        msg = (
            f"policy must have {expected} entries for a {sym.size}x{sym.size} board, "
            f"got {len(policy)}"
        )
        raise ValueError(msg)

    out = [0.0] * expected
    for index in range(n_squares):
        out[sym.perm[index]] = policy[index]
    out[n_squares] = policy[n_squares]
    return out
