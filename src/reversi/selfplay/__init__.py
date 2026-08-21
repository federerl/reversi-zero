"""Playing games against yourself to produce training data.

This is where almost all the compute goes. One generation of the full 8x8 profile
is roughly 2,500 games x 58 moves x 300 simulations, which is around 44 million
network calls -- against a few seconds of actual training afterwards. The ratio is
about a hundred to one, so any effort spent making the trainer faster is effort
wasted; the thing to optimise is here.

Day 5 ships the simple version: one process, one game at a time, one position per
network call. It is correct and easy to reason about, and it is fast enough for
the 4x4 pipeline test. Day 8 adds the two things that make 8x8 practical --
advancing many games in lockstep so their network calls batch together, and
running several worker processes at once.
"""

from __future__ import annotations
