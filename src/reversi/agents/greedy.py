"""Takes whichever move flips the most discs right now.

A step up from random, and a useful one, because it is the obvious wrong strategy
in Reversi -- and beating it is genuinely informative.

**Why maximising flips is a bad idea.** Discs flip back. A move that turns over
eight of your opponent's pieces in the opening leaves you with a large, exposed
group and hands them the moves that flip it all back. Strong Reversi play does
close to the opposite: keep your disc count *low* in the early game, keep your
options open, and take the squares that cannot be flipped -- corners, and the
edges that lead to them. Greedy will happily give away a corner to gain three
discs in the middle.

So an agent that beats Greedy has learned something real: that the disc count on
move twenty does not predict the disc count on move sixty. That is not obvious
from the rules, and nothing in the training signal states it. It has to be
discovered from games.
"""

from __future__ import annotations

import numpy as np

from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.types import Action, pass_action

__all__ = ["GreedyAgent"]


class GreedyAgent:
    """Maximises discs flipped this move; ties broken uniformly at random."""

    def __init__(self, name: str = "greedy") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if actions == [pass_action(state.size)]:
            return actions[0]

        gains = [popcount(rules.flips(state, action)) for action in actions]
        best = max(gains)
        # Ties are common and must not be broken by board order: always taking
        # the lowest-numbered square would make Greedy prefer the top-left corner
        # of the board, which is a positional bias rather than a greedy one, and
        # it would quietly flatter any agent that learns to exploit it.
        tied = [action for action, gain in zip(actions, gains, strict=True) if gain == best]
        return tied[int(rng.integers(0, len(tied)))]
