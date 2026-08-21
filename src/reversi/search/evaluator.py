"""The seam between the search and the network.

The search needs exactly two things from a position: a preference over moves, and
a guess at who is winning. ``Evaluator`` says so and nothing more. The real
implementation wraps a torch model (``reversi.nn.evaluator``); the one here
returns fixed numbers and imports nothing heavier than numpy.

That seam is doing real work. Because the search only depends on this protocol:

* every search test runs in milliseconds against a stub, with no model to build,
  no weights to initialise, and no torch import at all;
* the trickiest bugs in the whole project -- the sign of the value as it comes
  back up the tree, whether illegal moves can be reached -- get tested against an
  evaluator whose answers we chose, so a failure can only be the search's fault;
* swapping in batched inference on Day 8 changes nothing above this line.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from reversi.game.state import State
from reversi.types import policy_size

__all__ = ["Evaluator", "StubEvaluator"]


@runtime_checkable
class Evaluator(Protocol):
    """Something that can score a batch of positions.

    ``evaluate`` takes positions and returns ``(logits, values)``:

    * ``logits`` has shape ``(len(states), board_size**2 + 1)`` -- one raw,
      unmasked number per action, including PASS. Raw is deliberate: masking
      belongs to the search, which knows the rules (contract C5).
    * ``values`` has shape ``(len(states),)``, each in ``[-1, +1]``, and each
      **from the point of view of the player to move in that position**. Never
      from black's point of view, never from the root's. This is the single
      convention the whole project uses (contract C2), and it is the one that
      silently ruins everything if it is broken in one place.

    Positions come in a batch because that is how the GPU wants them -- one call
    with 48 positions costs barely more than one call with a single position.
    """

    def evaluate(
        self, states: Sequence[State]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


class StubEvaluator:
    """A stand-in network that returns whatever you tell it to.

    Defaults to "no opinion": every move equally likely, every position a draw.
    That default is useful on its own -- a search driven by it has no information
    beyond what it discovers by playing moves out, so its behaviour is pure PUCT
    and can be predicted by hand.

    Counts its own calls, so tests can assert things like "terminal positions are
    never sent to the network" without reaching inside the search.
    """

    def __init__(
        self,
        *,
        policy: Callable[[State], NDArray[np.float32] | Sequence[float]] | None = None,
        value: Callable[[State], float] | None = None,
    ) -> None:
        self._policy = policy
        self._value = value
        self.calls = 0
        self.positions = 0

    def evaluate(self, states: Sequence[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not states:
            msg = "evaluate requires at least one state"
            raise ValueError(msg)

        self.calls += 1
        self.positions += len(states)

        width = policy_size(states[0].size)
        logits = np.zeros((len(states), width), dtype=np.float32)
        values = np.zeros(len(states), dtype=np.float32)

        for row, state in enumerate(states):
            if self._policy is not None:
                given = np.asarray(self._policy(state), dtype=np.float32)
                if given.shape != (width,):
                    msg = f"policy callback returned shape {given.shape}, expected ({width},)"
                    raise ValueError(msg)
                logits[row] = given
            if self._value is not None:
                values[row] = np.float32(self._value(state))

        return logits, values
