"""A board position, as two integers and whose turn it is.

Positions never change. ``apply`` returns a new one rather than editing this
one. That is normally something you would think twice about, but a position
here is two machine words — copying it is cheaper than keeping an undo stack,
and it removes the worst bug class in a tree search, where a node quietly holds
a board that something else has since modified. Those are miserable to find, so
we design them out at the start.

Colours are stored as they are. ``black`` is always black's discs, never "the
current player's discs". The current-player view is produced later, only where
it is needed, by the neural network's input encoding — keeping that conversion
in exactly one place is what stops perspective bugs (contract C1).
"""

from __future__ import annotations

from dataclasses import dataclass

from reversi.game.bitboard import geometry, popcount
from reversi.types import Bitboard, Player


@dataclass(frozen=True, slots=True)
class State:
    """One Reversi position.

    ``black`` and ``white`` are bitboards: bit ``i`` is set when square
    ``i = row * size + col`` holds a disc of that colour. No square may be set
    in both.
    """

    black: Bitboard
    white: Bitboard
    to_move: Player
    size: int

    @property
    def mine(self) -> Bitboard:
        """Discs belonging to the player about to move."""
        return self.black if self.to_move is Player.BLACK else self.white

    @property
    def theirs(self) -> Bitboard:
        """Discs belonging to the player waiting."""
        return self.white if self.to_move is Player.BLACK else self.black

    @property
    def occupied(self) -> Bitboard:
        return self.black | self.white

    @property
    def empty(self) -> Bitboard:
        return geometry(self.size).full & ~self.occupied

    def with_discs(self, mine: Bitboard, theirs: Bitboard, *, switch: bool = True) -> State:
        """Build the next position from the mover's and waiter's disc sets.

        Working in "mine / theirs" terms rather than "black / white" keeps the
        rules code colour-blind: the same lines run whoever is to move. This
        function is the single place where that view is turned back into
        colours, so there is one place to get it wrong instead of many.
        """
        black, white = (mine, theirs) if self.to_move is Player.BLACK else (theirs, mine)
        return State(
            black=black,
            white=white,
            to_move=self.to_move.opponent if switch else self.to_move,
            size=self.size,
        )

    def disc_count(self) -> int:
        return popcount(self.occupied)

    def __str__(self) -> str:
        """A picture of the board, for debugging and error messages."""
        rows = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                mask = 1 << (row * self.size + col)
                if self.black & mask:
                    line.append("B")
                elif self.white & mask:
                    line.append("W")
                else:
                    line.append(".")
            rows.append("".join(line))
        return "\n".join(rows)


def initial_state(size: int = 8) -> State:
    """The standard opening: four discs crossed in the centre, black to move.

    On a full-size board that is White on d4 and e5, Black on d5 and e4.
    """
    geo = geometry(size)  # also validates the size
    mid = size // 2

    def square(row: int, col: int) -> Bitboard:
        return 1 << (row * geo.size + col)

    white = square(mid - 1, mid - 1) | square(mid, mid)
    black = square(mid - 1, mid) | square(mid, mid - 1)
    return State(black=black, white=white, to_move=Player.BLACK, size=size)
