"""One ingredient of the Minimax evaluation at a time, with no search.

``MinimaxAgent`` judges a position with three ideas added together: corners
and the squares around them are worth a fixed amount each, having more legal
moves than the opponent is good, and (late in the game) so is having more
discs. The two agents here each follow a single one of those ideas, one move
at a time, and know nothing else.

They are controls, not opponents worth beating. The research plan tests
whether an agent trained against Minimax gets better against opponents it has
*not* seen, and a test opponent that is one term of Minimax's own evaluation
would be the training opponent in disguise. These agents exist so that the
similarity check in ``research/held-out-opponents.md`` can show that, with a
number, rather than assert it.
"""

from __future__ import annotations

import numpy as np

from reversi.agents.greedy import _pick
from reversi.agents.minimax import square_values
from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.types import Action

__all__ = ["CornerAgent", "MobilityAgent"]


class CornerAgent:
    """Plays the square the positional table rates highest; ties at random.

    The table is Minimax's own (``minimax.square_values``): corners first,
    then edges, and the squares next to a corner last of all. Nothing is
    looked ahead; nothing is counted.
    """

    def __init__(self, name: str = "corner") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choices(self, state: State) -> list[Action]:
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(actions) == 1:
            return actions
        table = square_values(state.size)
        best = max(table[a] for a in actions)
        return [a for a in actions if table[a] == best]

    def select(self, state: State, rng: np.random.Generator) -> Action:
        return _pick(self.choices(state), rng)


class MobilityAgent:
    """Plays the move that leaves the opponent the fewest legal moves; ties at random.

    Mobility is the idea that running out of moves is how you lose in Othello:
    a player with few options is forced into bad squares. This is that idea on
    its own, one ply deep.
    """

    def __init__(self, name: str = "mobility") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choices(self, state: State) -> list[Action]:
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(actions) == 1:
            return actions
        # After the move it is the opponent's turn, so their legal placements are
        # simply the next position's.
        replies = [popcount(rules.legal_placements(rules.apply(state, a))) for a in actions]
        fewest = min(replies)
        return [a for a, n in zip(actions, replies, strict=True) if n == fewest]

    def select(self, state: State, rng: np.random.Generator) -> Action:
        return _pick(self.choices(state), rng)
