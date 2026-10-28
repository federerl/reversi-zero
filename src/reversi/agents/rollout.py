"""A tree search with no knowledge at all: random playouts instead of a network.

The trained agent judges a position by asking its network. Minimax judges it
with a hand-written table. This agent judges it by **playing random moves to
the end of the game and seeing who won**, many times over, and otherwise runs
the same tree search the trained agent does.

**Why it exists.** The research plan trains the agent against Greedy and
Minimax and then asks whether it got better against opponents it never
trained against. That question needs test opponents whose judgement comes
from somewhere the training opponents' judgement does not. Greedy and Minimax
both encode human Othello knowledge in code; the network learned from
self-play; Edax learned from a database of games. Random playouts encode
nothing. Whatever this agent knows about a position it found by trying moves
and counting, so it cannot share a blind spot with any table.

**How a playout judges a position.** From the leaf the search reached, both
sides play uniformly random legal moves until the game ends. The leaf's value
is +1, 0 or -1 for the leaf's mover depending on who won that one random game.
One playout is a very noisy estimate; the search runs one per simulation and
the tree averages them, so a move's value settles down as it gets visited.

**Every move gets the same prior.** The search's selection rule normally
weights moves by how much the network likes them. There is no network here, so
every legal move starts equal and only the playout results pull the search
toward one or another. That makes it close to the classic UCT search, with
this project's selection formula rather than the textbook one.

**It is slow in Python.** A playout from the opening is about three
milliseconds, so a thousand playouts per move is a few seconds per move. That
is fine for a few hundred evaluation games and would be far too slow for
training, which is not what it is for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.mcts import MCTS
from reversi.types import Action, policy_size

__all__ = ["RolloutAgent", "RolloutEvaluator"]


class RolloutEvaluator:
    """Judges a position by one random playout; satisfies the search's ``Evaluator`` protocol.

    ``rng`` is set by whoever runs the search, before each search, so a match
    that seeds its games reproduces exactly. Playing random moves with an rng
    nobody chose would make every result unrepeatable.
    """

    def __init__(self, rng: np.random.Generator, *, playouts: int = 1) -> None:
        if playouts < 1:
            msg = f"playouts per leaf must be at least 1, got {playouts}"
            raise ValueError(msg)
        self.rng = rng
        self.playouts = playouts
        self.calls = 0
        self.positions = 0

    def evaluate(self, states: Sequence[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not states:
            msg = "evaluate requires at least one state"
            raise ValueError(msg)
        self.calls += 1
        self.positions += len(states)

        # All-zero logits are a uniform distribution once the search softmaxes
        # them over the legal moves: no move is preferred before it is tried.
        logits = np.zeros((len(states), policy_size(states[0].size)), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)
        for row, state in enumerate(states):
            total = 0.0
            for _ in range(self.playouts):
                total += self.playout(state)
            values[row] = total / self.playouts
        return logits, values

    def playout(self, state: State) -> float:
        """Play random moves to the end. Returns the result for ``state``'s mover."""
        mover = state.to_move
        rng = self.rng
        while not rules.is_terminal(state):
            actions = rules.legal_actions(state)
            state = rules.apply(state, actions[int(rng.integers(0, len(actions)))])
        return float(scoring.result_for(state, mover))


class RolloutAgent:
    """Random-playout tree search, ``rollouts`` playouts per move."""

    def __init__(
        self,
        rollouts: int,
        *,
        name: str | None = None,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
    ) -> None:
        if rollouts < 1:
            msg = f"rollouts per move must be at least 1, got {rollouts}"
            raise ValueError(msg)
        self.rollouts = rollouts
        self._name = name if name is not None else f"rollout-r{rollouts}"
        # A placeholder rng: ``select`` always swaps in the match's before searching.
        self._evaluator = RolloutEvaluator(np.random.default_rng(0))
        self.mcts = MCTS(
            self._evaluator,
            SearchConfig(
                n_simulations=rollouts,
                c_puct=c_puct,
                fpu_reduction=fpu_reduction,
                dirichlet_eps=0.0,
                temp_moves=0,
            ),
        )

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(actions) == 1:
            return actions[0]
        # The match's rng drives the playouts, so the same seed replays the
        # same game, exactly as it does for Greedy's tie-breaks.
        self._evaluator.rng = rng
        return self.mcts.run(state).best_action()
