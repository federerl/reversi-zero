"""Turning collected games back into a better network.

* ``loss.py`` -- the two things the network is scored on, and why they are added
  together rather than trained separately.
* ``schedule.py`` -- how the learning rate changes over a run.
* ``trainer.py`` -- one optimisation step.
* ``loop.py`` -- the generational cycle: play, store, train, repeat.

This is the cheap half of the project. One generation of training is a few
hundred steps on a small network -- seconds of work -- against tens of millions of
network calls to produce the games it trains on. Optimising anything in here is
almost always the wrong place to spend effort.
"""

from __future__ import annotations
