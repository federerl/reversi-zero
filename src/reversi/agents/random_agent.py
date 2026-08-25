"""Plays a uniformly random legal move.

The floor. An agent that cannot beat this convincingly has learned nothing at
all, which is exactly why it is the first opponent the pipeline is measured
against -- and why the day-6 gate demands 90% rather than "better than half".

It is also the anchor for the whole rating scale: Random is defined as 0 Elo, and
every other number in the project is relative to it.
"""

from __future__ import annotations

import numpy as np

from reversi.game import rules
from reversi.game.state import State
from reversi.types import Action

__all__ = ["RandomAgent"]


class RandomAgent:
    """Uniform over the legal moves."""

    def __init__(self, name: str = "random") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        return actions[int(rng.integers(0, len(actions)))]
