"""The trained agent: a network, a tree search, and no randomness.

This is what everything else in the project exists to produce, and what every
strength number is measured on.

**Exploration is off, and that is checked at construction rather than trusted.**
During self-play the search deliberately randomises its opening preferences so
the training games vary. Leaving that on while measuring strength would mean
every result understates the agent by an unknown amount -- and nothing would fail,
no test would go red, the numbers would simply be quietly wrong. So building this
agent with a self-play search config raises immediately (contract C7).

**The move is the most-visited one, not the highest-scoring one.** A move can
score well on a couple of lucky simulations; it can only accumulate visits by
continuing to look good as the search examines it harder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import Evaluator
from reversi.search.mcts import MCTS
from reversi.types import Action

__all__ = ["AZAgent"]


class AZAgent:
    """A policy-value network plus PUCT search, playing its best move."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig,
        *,
        name: str = "az",
    ) -> None:
        config.assert_no_noise(f"the agent {name!r}")
        self.mcts = MCTS(evaluator, config)
        self._name = name

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        simulations: int,
        device: str = "cpu",
        name: str | None = None,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
    ) -> AZAgent:
        """Build an agent from a saved network.

        ``simulations`` is given explicitly rather than read from the training
        config, because how hard an agent thinks is a property of *this* match,
        not of how the network was produced. It is also the main dial the
        difficulty levels turn.
        """
        from reversi.nn.evaluator import TorchEvaluator
        from reversi.nn.loader import load_model

        model = load_model(path, device=device)
        return cls(
            TorchEvaluator(model, device=device),
            SearchConfig(
                n_simulations=simulations,
                c_puct=c_puct,
                fpu_reduction=fpu_reduction,
                dirichlet_eps=0.0,
                temp_moves=0,
            ),
            name=name if name is not None else f"az-s{simulations}",
        )

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        # rng is unused: with no exploration noise and no temperature this agent
        # is deterministic, which is what makes a tournament reproducible.
        _ = rng
        return self.mcts.run(state).best_action()
