"""The real evaluator: the tree search's questions, answered by the network.

This is the only place where a torch model and the search meet. It encodes
positions, runs one forward pass, and hands back plain numpy arrays -- so
everything above it stays free of torch, and can be tested without one.

Two settings here matter more than they look:

* ``model.eval()`` -- batch normalisation behaves differently while training
  (it uses the statistics of the current batch) than when playing (it uses the
  averages it accumulated). Leaving the model in training mode means the answer
  for a position depends on which *other* positions happened to be in the same
  batch, so the same position gets different answers at different times, and the
  search is being fed noise. It fails nothing and produces a weaker agent, so it
  is asserted rather than trusted.
* ``torch.inference_mode()`` -- switches off the bookkeeping torch keeps for
  computing gradients. We are never going to differentiate a self-play forward
  pass, and that bookkeeping is pure cost: memory and time, on the one path where
  we have the least of both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from reversi.game.state import State
from reversi.nn import features
from reversi.nn.model import PolicyValueNet

__all__ = ["TorchEvaluator"]


class TorchEvaluator:
    """Runs a ``PolicyValueNet`` on batches of positions.

    Satisfies the ``reversi.search.evaluator.Evaluator`` protocol, which is
    checked structurally -- there is no base class to inherit, on purpose, so the
    search never has to import this module or torch.
    """

    def __init__(self, model: PolicyValueNet, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.board_size = model.board_size

        # Counters, matching StubEvaluator, so throughput reporting and tests can
        # ask the same questions of either.
        self.calls = 0
        self.positions = 0

    def evaluate(self, states: Sequence[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not states:
            msg = "evaluate requires at least one state"
            raise ValueError(msg)
        if self.model.training:
            msg = (
                "the model is in training mode during inference. Batch norm would "
                "then answer using the current batch's statistics, so the same "
                "position gets different answers depending on what it was batched "
                "with. Call model.eval() first."
            )
            raise RuntimeError(msg)

        self.calls += 1
        self.positions += len(states)

        planes = features.encode_batch(states)
        with torch.inference_mode():
            batch = torch.from_numpy(planes).to(self.device)
            logits, values = self.model(batch)

        return (
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        )
