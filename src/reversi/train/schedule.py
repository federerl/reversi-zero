"""How the learning rate changes over a run.

Two phases:

**Warmup.** Start near zero and ramp up over the first couple of hundred steps.
A freshly initialised network produces nonsense, so its first gradients are large
and point in arbitrary directions; taking full-sized steps along them can wreck
the weights before any signal has arrived. Ramping up costs a few seconds and
removes that whole failure mode.

**Cosine decay.** Ease down to a small fraction of the peak by the end. Early on,
big steps are what you want -- the network is far from anything good. Later, the
targets themselves are close to correct and big steps just bounce around the
answer. Cosine is the standard shape here; it spends more of the run at a useful
rate than a linear ramp down does, and it has no cliff for a resume to land on.

This is a plain function of the step number rather than a torch scheduler object,
for one reason: a scheduler is stateful, so resuming a run means saving and
restoring that state correctly and hoping it matches. A function of the step
cannot get out of sync -- give it the step number and it gives the same answer
every time.
"""

from __future__ import annotations

import math

__all__ = ["learning_rate"]


def learning_rate(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    floor_divisor: float = 20.0,
) -> float:
    """The learning rate to use at ``step`` (zero-based).

    Rises linearly to ``base_lr`` over ``warmup_steps``, then decays on a cosine
    curve to ``base_lr / floor_divisor`` at ``total_steps``. Beyond that it stays
    at the floor rather than turning around, so a run that goes longer than
    planned does not start taking big steps again.
    """
    if base_lr <= 0.0:
        msg = f"base_lr must be positive, got {base_lr}"
        raise ValueError(msg)
    if floor_divisor < 1.0:
        msg = f"floor_divisor must be at least 1, got {floor_divisor}"
        raise ValueError(msg)

    step = max(0, step)
    floor = base_lr / floor_divisor

    # A warmup longer than the run itself means the run never reaches the
    # learning rate it was configured with -- it just crawls up the ramp and
    # stops. That is a silent way to waste a training job, so warmup is capped at
    # half the run. The Trainer warns when this cap actually bites.
    warmup_steps = min(warmup_steps, max(1, total_steps // 2)) if warmup_steps > 0 else 0

    if warmup_steps > 0 and step < warmup_steps:
        # step+1 so the very first step is not exactly zero -- a zero-sized step
        # is a wasted batch.
        return base_lr * (step + 1) / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    if progress >= 1.0:
        return floor

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base_lr - floor) * cosine
