"""Measuring how strong something actually is, by making it play.

This package is the answer to "how do you know it learned?". Nothing in it looks
at a loss curve.

**It must never import from ``train`` or ``data``.** Arena games are evaluation,
not training data: if a single evaluation game leaked into the replay buffer the
agent would be training on its own test set, and every number afterwards would be
inflated by an unknowable amount. The layering rule is what makes that leak
impossible rather than merely unlikely.

As of day 9 this holds the match runner, the seeded opening book, the frozen
Minimax-d4 baseline, and the report schema -- enough to run a fair matchup and
write down why it should be believed. Confidence intervals and Bradley-Terry
ratings arrive on day 10, and those are what turn a win rate into a claim with an
error bar on it.
"""

from __future__ import annotations

from reversi.arena.match import MatchResult, play_match
from reversi.arena.openings import Opening, apply_opening, random_openings
from reversi.arena.report import MatchReport, check_fairness, write_report

__all__ = [
    "MatchReport",
    "MatchResult",
    "Opening",
    "apply_opening",
    "check_fairness",
    "play_match",
    "random_openings",
    "write_report",
]
