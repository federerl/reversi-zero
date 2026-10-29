"""A value target with a material term in it.

The value head is normally trained toward ``z``, how the game ended, from the
mover's point of view. Step 2 of the research plan found a blind spot in that
target: positions where the mover holds one or two discs almost never occur in
self-play, and when they do the side holding them usually goes on to win, so the
network learns that holding almost no discs is a fine place to be. Against an
opponent that flips every disc it can, it is one move from the end.

The blend here is the cheapest intervention aimed at that blind spot and at
nothing else. With weight ``w`` the target becomes::

    (1 - w) * z + w * share

where ``share`` is a number in [-1, 1] that says how the discs are split. It is
near zero in a balanced position, so most targets barely move, and it is close
to -1 where the mover holds one disc against forty, which is exactly the region
where ``z`` alone misleads. Two choices of ``share``:

* ``"current"``: the split *in this position*, (mine - theirs) / (mine + theirs),
  read off the input planes. This is the one the research design uses: it pulls
  a one-disc position that self-play labelled +1 to about 0 at ``w = 0.5``.
* ``"final"``: the split *when the game ended*, the sum of the ownership target
  over the board divided by the number of squares. The usual "score" target in
  engines that use one. It does not change the sign of a one-disc position that
  went on to be won, so it is kept as the conventional comparison, not as the fix.

The blend is applied when a batch is drawn, not when a game is recorded, so it
covers every position in the replay window from the first gradient step,
including positions written before the setting existed. ``w = 0`` is exactly the
ordinary target.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = ["MaterialSource", "blend_value_target", "current_share", "final_share"]

MaterialSource = Literal["current", "final"]


def current_share(planes: Tensor) -> Tensor:
    """``(mine - theirs) / (mine + theirs)`` for each position, from the mover's view.

    ``planes`` is ``(batch, planes, size, size)`` as the network reads it: plane 0
    is the mover's discs and plane 1 the opponent's. A board with no discs cannot
    occur, but the denominator is clamped so that a hand-built batch cannot divide
    by zero.
    """
    mine = planes[:, 0].flatten(1).sum(dim=1)
    theirs = planes[:, 1].flatten(1).sum(dim=1)
    return (mine - theirs) / (mine + theirs).clamp_min(1.0)


def final_share(own: Tensor) -> Tensor:
    """The final disc margin as a share of the board, from the mover's view.

    ``own`` is ``(batch, squares)`` with +1 the mover's disc, -1 the opponent's,
    0 empty, so its sum is the final margin and dividing by the number of
    squares puts it in [-1, 1].
    """
    return own.sum(dim=1) / own.shape[1]


def blend_value_target(
    z: Tensor,
    *,
    weight: float,
    source: MaterialSource,
    planes: Tensor,
    own: Tensor | None = None,
    own_valid: Tensor | None = None,
) -> Tensor:
    """``(1 - weight) * z + weight * share``, with ``share`` chosen by ``source``.

    ``weight`` of zero returns ``z`` untouched. For ``"final"``, rows whose
    ownership target is not real (``own_valid`` false, positions from shards
    written before that target existed) keep ``z``: a missing target is not a
    zero margin.
    """
    if weight <= 0.0:
        return z
    if source == "current":
        share = current_share(planes)
        return (1.0 - weight) * z + weight * share
    if source == "final":
        if own is None or own_valid is None:
            msg = "material_target 'final' needs the ownership target and its validity mask"
            raise ValueError(msg)
        blended = (1.0 - weight) * z + weight * final_share(own)
        return torch.where(own_valid, blended, z)
    msg = f"unknown material target source {source!r}"  # pragma: no cover - config checks it
    raise ValueError(msg)
