"""Measuring how strong something actually is, by making it play.

This package is the answer to "how do you know it learned?". Nothing in it looks
at a loss curve.

**It must never import from ``train`` or ``data``.** Arena games are evaluation,
not training data: if a single evaluation game leaked into the replay buffer the
agent would be training on its own test set, and every number afterwards would be
inflated by an unknowable amount. The layering rule is what makes that leak
impossible rather than merely unlikely.

Day 6 ships only the match runner -- enough to answer "does the 4x4 agent beat
Random and Greedy?". The opening book, confidence intervals, Bradley-Terry
ratings and the report schema arrive on days 9 and 10, and those are what turn a
win rate into a defensible claim.
"""

from __future__ import annotations

from reversi.arena.match import MatchResult, play_match

__all__ = ["MatchResult", "play_match"]
