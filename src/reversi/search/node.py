"""One position in the search tree, and the statistics hanging off it.

A node holds a position plus, for each legal move out of it, three running
numbers: how many times the search went that way, the total value that came
back, and how much the network liked the move to begin with. Everything the
search does is reading and updating those.

**The arrays only cover legal moves.** A node with 7 legal moves has arrays of
length 7, not 65. That is contract C5's second layer, and it is structural rather
than defensive: there is no entry for an illegal move, so no amount of searching
can visit one. A masking bug cannot exist here because there is nothing to mask.

**Plain Python lists, not numpy.** Reversi positions average 8-10 legal moves,
and at that size a numpy operation spends more time in call overhead than a
Python loop spends doing the work. This is the deliberately naive version; Day 8
benchmarks both and keeps whichever actually wins.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from reversi.game.state import State
from reversi.types import Action

__all__ = ["Node", "priors_from_logits"]


def priors_from_logits(
    logits: NDArray[np.float32],
    actions: Sequence[Action],
) -> list[float]:
    """Turn the network's raw output into probabilities over the legal moves only.

    The network emits a number for all 65 actions with no idea which are legal.
    We pull out the legal ones and softmax over just those, so the probabilities
    sum to 1 across the moves that actually exist.

    Subtracting the maximum first is not cosmetic: a logit of +90 overflows
    float32 when exponentiated, and the whole distribution becomes NaN.
    """
    picked = np.asarray([logits[a] for a in actions], dtype=np.float64)
    largest = picked.max()
    if not np.isfinite(largest):
        # Every legal move was -inf (or the network produced garbage). Refusing
        # to guess is better than returning NaN and poisoning the tree.
        return [1.0 / len(actions)] * len(actions)

    exp = np.exp(picked - largest)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return [1.0 / len(actions)] * len(actions)
    return [float(x) for x in exp / total]


class Node:
    """A position, plus what the search has learned about the moves out of it."""

    __slots__ = (
        "actions",
        "children",
        "expanded",
        "net_value",
        "prior",
        "state",
        "terminal_value",
        "value_sum",
        "visits",
    )

    def __init__(self, state: State) -> None:
        self.state = state

        # Filled in by expansion. Until then this node is a leaf.
        self.actions: tuple[Action, ...] = ()
        self.prior: list[float] = []
        self.visits: list[int] = []
        self.value_sum: list[float] = []
        self.children: list[Node | None] = []
        self.expanded = False

        # The network's own opinion of this position, from its mover's point of
        # view. Used as the starting point for untried moves (see fpu_reduction).
        self.net_value = 0.0

        # The exact result, for a finished position. Terminal nodes are never
        # sent to the network -- we know the answer, and a guess would only be
        # worse.
        self.terminal_value: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def total_visits(self) -> int:
        """Simulations that have passed through this node's children."""
        return sum(self.visits)

    def q(self, index: int) -> float:
        """Average value of the move at ``index``, from *this* node's mover's view.

        Zero for an untried move -- but the search never uses this for untried
        moves; it substitutes the first-play-urgency value instead.
        """
        visits = self.visits[index]
        return self.value_sum[index] / visits if visits else 0.0

    def mean_value(self) -> float:
        """How this node is going overall, from its own mover's point of view.

        Falls back to the network's estimate before any child has been tried,
        which is exactly the situation the first simulation from a fresh node
        finds itself in.
        """
        total = self.total_visits
        if total == 0:
            return self.net_value
        return sum(self.value_sum) / total

    def open_with(
        self,
        actions: Sequence[Action],
        priors: Sequence[float],
        net_value: float,
    ) -> None:
        """Attach the legal moves and the network's opinion; mark this node expanded."""
        if not actions:
            msg = "cannot expand a node with no legal actions -- it is terminal"
            raise ValueError(msg)
        if len(priors) != len(actions):
            msg = f"got {len(priors)} priors for {len(actions)} actions"
            raise ValueError(msg)

        self.actions = tuple(actions)
        self.prior = list(priors)
        self.visits = [0] * len(actions)
        self.value_sum = [0.0] * len(actions)
        self.children = [None] * len(actions)
        self.net_value = net_value
        self.expanded = True
