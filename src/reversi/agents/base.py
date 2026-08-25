"""The one thing every player has to do: look at a position and pick a move.

Deliberately tiny. An agent gets a position and a source of randomness, and
returns one legal action. No game history, no state carried between moves, no
notion of which colour it is -- a Reversi position already says whose turn it is,
and anything an agent needs to know is in it.

Keeping the interface this small is what lets the arena be a loop over a list.
Random, greedy, a depth-4 search, and a neural network with 458,696 parameters
are all the same shape from outside.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from reversi.game.state import State
from reversi.types import Action

__all__ = ["Agent"]


@runtime_checkable
class Agent(Protocol):
    """Something that can choose a move.

    ``select`` must return an action from ``rules.legal_actions(state)`` -- always,
    including PASS when passing is the only thing available. Returning anything
    else is a bug in the agent, and ``apply`` will refuse it loudly rather than
    quietly playing something else (contract C5).

    ``rng`` is passed in rather than held, so a match can be reproduced exactly
    from one seed no matter how many agents are involved. An agent that ignores
    it is simply deterministic, which is fine and expected -- a search with no
    exploration noise plays the same move every time by design.
    """

    @property
    def name(self) -> str:
        """Short label used in match reports and rating tables."""
        ...

    def select(self, state: State, rng: np.random.Generator) -> Action: ...
