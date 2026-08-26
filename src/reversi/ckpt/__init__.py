"""Saving a run so it can be picked up again after being killed.

A training run is a sequence of interruptions. On a shared machine it is somebody
rebooting, an ssh session dropping, or you pressing Ctrl-C; on a scheduler it is
a wall-clock limit or a preemption. Either way the run has to survive being
stopped at an arbitrary moment and continue without losing more than one
generation.

That is what makes this package critical path rather than housekeeping: **a run
that cannot be resumed cannot use an overnight slot at all.** Eight hours of
work that vanishes when the session closes is eight hours wasted.

Two files:

* ``meta.py`` -- the description of a checkpoint, written next to it as plain
  JSON so the lineage of a run can be read without importing torch.
* ``manager.py`` -- writing them atomically, verifying them on the way back in,
  and choosing which one to resume from.
"""

from __future__ import annotations

from reversi.ckpt.manager import CheckpointManager
from reversi.ckpt.meta import FORMAT_VERSION, CheckpointMeta

__all__ = ["FORMAT_VERSION", "CheckpointManager", "CheckpointMeta"]
