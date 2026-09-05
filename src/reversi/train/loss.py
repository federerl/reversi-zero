"""What the network is scored on.

Two terms, added, and an optional third:

    loss = cross_entropy(what the search concluded, what the policy head says)
         + weight * mean_squared_error(how the game ended, what the value head says)
         + ownership_weight * mean_squared_error(who owns each square, ownership head)

**The policy term** asks the network to imitate the search. Not to imitate a human,
and not to maximise anything -- just to predict, before searching, what searching
would have concluded. Every time it gets better at that, the next search starts
from a better place and concludes something better still.

**The value term** asks it to predict the eventual result from the position alone.
That is what lets the search skip playing games out to the end.

**Why one combined loss rather than two separate trainings.** The two heads share
a body, and "what matters on this board" is nearly the same question either way --
corners, mobility, who is running out of moves. Training them together means the
value signal teaches the shared body things the policy signal alone would not, and
vice versa. It also halves the work: one forward pass answers both.

The weight on the value term is 1.0 by default. It matters because the two terms
have different natural scales: cross-entropy over 65 actions starts around ln(65)
which is about 4.2, while a squared error on a number in [-1, 1] starts around 1.
Left unweighted the policy dominates early, which is roughly what we want.

**The ownership term**, when a network has the head for it, asks for the final
owner of every square: +1 mine, -1 theirs, 0 empty. The result ``z`` is the sum
of those 64 numbers, so this is the value target broken into its parts. The point
is the amount of signal: one position now teaches the shared body 64 things about
"who is winning where" instead of one, and it does so without touching the
policy term, which is what a heavier value weight (run 2) got wrong. The head is
never used at play time; it exists to shape the body. Positions read from shards
written before the target existed carry no ownership and are masked out of this
term only. The weight defaults to 0, which is exactly the two-term loss above.

**No entropy bonus, no advantage, no importance weights.** This is plain
supervised learning against targets that happen to have been produced by search.
If you are looking for the reinforcement learning, it is in where the data comes
from, not in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

__all__ = ["LossParts", "policy_value_loss"]


@dataclass(frozen=True, slots=True)
class LossParts:
    """The total and its pieces, kept apart so the metrics can show both.

    Watching them separately is how you tell what a run is doing. A policy loss
    that falls while the value loss sits flat means the network is learning which
    moves look reasonable but has no idea who is winning -- which shows up in play
    as an agent with decent instincts and no endgame.
    """

    total: Tensor
    policy: Tensor
    value: Tensor
    ownership: Tensor | None = None

    def as_metrics(self) -> dict[str, float]:
        metrics = {
            "total_loss": float(self.total.detach()),
            "policy_loss": float(self.policy.detach()),
            "value_loss": float(self.value.detach()),
        }
        if self.ownership is not None:
            metrics["ownership_loss"] = float(self.ownership.detach())
        return metrics


def policy_value_loss(
    policy_logits: Tensor,
    value_pred: Tensor,
    pi_target: Tensor,
    z_target: Tensor,
    *,
    value_weight: float = 1.0,
    ownership_pred: Tensor | None = None,
    own_target: Tensor | None = None,
    own_valid: Tensor | None = None,
    ownership_weight: float = 0.0,
) -> LossParts:
    """Score one batch.

    The ownership term is computed only when ``ownership_pred`` is given and
    ``ownership_weight`` is positive. ``own_valid`` marks the rows whose target is
    real; the others are excluded from the mean rather than counted as "all
    empty", and a batch with no valid row contributes zero.

    ``policy_logits`` is ``(batch, actions)`` raw and unmasked -- the network has
    no idea which moves are legal, and does not need to. The target is zero on
    illegal moves, so those terms contribute nothing to the sum and the gradient
    pushes their logits down as a side effect. Nothing relies on that happening
    (contract C5): the search cannot reach an illegal move regardless.

    ``value_pred`` may be ``(batch, 1)`` or ``(batch,)``; both are accepted
    because the model returns the former and it is a common place to get a silent
    broadcast wrong.
    """
    if policy_logits.shape != pi_target.shape:
        msg = (
            f"policy logits {tuple(policy_logits.shape)} and targets "
            f"{tuple(pi_target.shape)} must have the same shape"
        )
        raise ValueError(msg)

    value = value_pred.reshape(-1)
    target = z_target.reshape(-1)
    if value.shape != target.shape:
        msg = (
            f"value predictions {tuple(value.shape)} and targets "
            f"{tuple(target.shape)} must have the same length"
        )
        raise ValueError(msg)

    # Cross-entropy against a full distribution, not a single label: the target
    # says "62% of the search went here, 31% there", and we want the network to
    # reproduce those proportions rather than just pick the argmax.
    log_probabilities = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(pi_target * log_probabilities).sum(dim=1).mean()

    value_loss = F.mse_loss(value, target)

    total = policy_loss + value_weight * value_loss
    ownership_loss: Tensor | None = None
    if ownership_pred is not None and ownership_weight > 0.0:
        if own_target is None:
            msg = "an ownership prediction was given without an ownership target"
            raise ValueError(msg)
        if ownership_pred.shape != own_target.shape:
            msg = (
                f"ownership predictions {tuple(ownership_pred.shape)} and targets "
                f"{tuple(own_target.shape)} must have the same shape"
            )
            raise ValueError(msg)
        # Mean over the squares of each position, then over the positions whose
        # target is real. That keeps the term on the same scale as the value
        # term -- one number in [0, 4] per position -- so a weight of 1.0 means
        # "as important as the result".
        per_position = ((ownership_pred - own_target) ** 2).mean(dim=1)
        if own_valid is not None:
            mask = own_valid.to(per_position.dtype)
            valid = mask.sum()
            ownership_loss = (
                (per_position * mask).sum() / valid if valid > 0 else per_position.sum() * 0.0
            )
        else:
            ownership_loss = per_position.mean()
        total = total + ownership_weight * ownership_loss

    return LossParts(
        total=total,
        policy=policy_loss,
        value=value_loss,
        ownership=ownership_loss,
    )


def policy_entropy(policy_logits: Tensor) -> Tensor:
    """Average spread of the network's move preferences, in nats.

    A diagnostic, not part of the loss. Two failures show up here before they show
    up anywhere else: entropy collapsing toward zero means the network has become
    certain about everything and self-play has stopped exploring, while entropy
    stuck at its maximum (``ln(number of actions)``) means it has learned nothing
    at all.
    """
    log_probabilities = F.log_softmax(policy_logits, dim=1)
    probabilities = torch.exp(log_probabilities)
    return -(probabilities * log_probabilities).sum(dim=1).mean()
