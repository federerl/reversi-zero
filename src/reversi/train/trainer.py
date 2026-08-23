"""One optimisation step, and the bookkeeping around it.

Nothing here is unusual -- it is a standard supervised training step. The parts
worth knowing about:

**Gradient clipping.** Self-play data is noisy, and occasionally a batch arrives
whose gradient is enormous (a run of games that all ended the same way, a
generation where the search was unusually confident). One such batch can undo a
generation of progress. Clipping the total gradient size caps the damage any
single batch can do, at the cost of slowing down the rare batch that really did
deserve a big step.

**Weight decay is on the optimiser, not in the loss.** The plan writes the
objective as including an L2 penalty on the weights, and that is what
``weight_decay`` does -- torch applies it as part of the update. Writing it into
the loss as well would apply it twice.

**SGD for the long run, AdamW for the short ones.** AdamW converges faster from a
cold start, which is what you want when iterating on a ten-minute pipeline test.
SGD with momentum generalises slightly better and is what the AlphaZero lineage
uses, which matters more when the run is eight hours and the result is the
headline number. The config picks per profile.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from reversi.config import TrainConfig
from reversi.data.replay import Batch
from reversi.nn.model import PolicyValueNet
from reversi.train.loss import policy_entropy, policy_value_loss
from reversi.train.schedule import learning_rate

__all__ = ["Trainer"]

log = logging.getLogger(__name__)


class Trainer:
    """Owns the optimiser and the step counter for one run."""

    def __init__(
        self,
        model: PolicyValueNet,
        config: TrainConfig,
        *,
        total_steps: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.total_steps = max(1, total_steps)
        self.global_step = 0

        self.model.to(self.device)
        self.optimizer = _build_optimizer(model, config)

        if config.warmup_steps > self.total_steps // 2:
            log.warning(
                "train.warmup_steps is %d but this run is only %d steps; warmup will be "
                "capped at %d so the run actually reaches lr=%g rather than spending its "
                "whole life ramping up",
                config.warmup_steps,
                self.total_steps,
                max(1, self.total_steps // 2),
                config.lr,
            )

    def train_on(self, batch: Batch) -> dict[str, float]:
        """One step on one batch. Returns the numbers worth logging."""
        self.model.train()

        planes = torch.from_numpy(np.ascontiguousarray(batch.planes)).to(self.device)
        pi_target = torch.from_numpy(np.ascontiguousarray(batch.pi)).to(self.device)
        z_target = torch.from_numpy(np.ascontiguousarray(batch.z)).to(self.device)

        lr = learning_rate(
            self.global_step,
            base_lr=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.total_steps,
            floor_divisor=self.config.lr_floor_divisor,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        policy_logits, value_pred = self.model(planes)
        parts = policy_value_loss(
            policy_logits,
            value_pred,
            pi_target,
            z_target,
            value_weight=self.config.value_loss_weight,
        )

        self.optimizer.zero_grad(set_to_none=True)
        parts.total.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
        self.optimizer.step()
        self.global_step += 1

        with torch.no_grad():
            entropy = float(policy_entropy(policy_logits.detach()))
            value_mae = float((value_pred.detach().reshape(-1) - z_target).abs().mean())

        return {
            **parts.as_metrics(),
            "lr": lr,
            "grad_norm": grad_norm,
            "policy_entropy": entropy,
            "value_mae": value_mae,
            "batch_size": len(batch),
        }

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to carry on exactly where this left off.

        Used by the checkpoint manager on day 7. The optimiser state is the part
        people forget: AdamW keeps a running estimate per parameter, and resuming
        without it means the first few hundred steps after a restart are taken
        with the wrong step sizes.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "total_steps": self.total_steps,
        }


def _build_optimizer(model: PolicyValueNet, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.momentum > 0.0,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
