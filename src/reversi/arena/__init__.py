"""Measuring how strong something actually is, by making it play.

This package is the answer to "how do you know it learned?". Nothing in it looks
at a loss curve.

**It must never import from ``train`` or ``data``.** Arena games are evaluation,
not training data: if a single evaluation game leaked into the replay buffer the
agent would be training on its own test set, and every number afterwards would be
inflated by an unknowable amount. The layering rule is what makes that leak
impossible rather than merely unlikely.

A win rate on its own is not evidence. "19 wins from 20" and "950 from 1000"
are both 95%, and only one of them means anything. So this package does three
things: it runs matches *fairly* (equal colours, varied openings), it attaches
*uncertainty* to every number (Wilson intervals per matchup, bootstrap intervals
per rating), and it *writes down why* a result should be believed.

``elo.py`` fits every agent onto one scale at once, anchored at Random = 0, so
"generation 60 is stronger than generation 20" becomes a number with an interval
rather than an eyeball comparison of two win rates.
"""

from __future__ import annotations

from reversi.arena.elo import Rating, RatingTable, fit_bradley_terry
from reversi.arena.entrants import EntrantSpec, build_agent, describe_entrant, parse_entrant
from reversi.arena.match import MatchResult, play_match
from reversi.arena.openings import Opening, apply_opening, random_openings
from reversi.arena.report import MatchReport, check_fairness, write_report
from reversi.arena.stats import Interval, bootstrap_interval, wilson_interval
from reversi.arena.tournament import (
    Entrant,
    TournamentResult,
    round_robin,
    round_robin_parallel,
    write_tournament,
)

__all__ = [
    "Entrant",
    "EntrantSpec",
    "Interval",
    "MatchReport",
    "MatchResult",
    "Opening",
    "Rating",
    "RatingTable",
    "TournamentResult",
    "apply_opening",
    "bootstrap_interval",
    "build_agent",
    "check_fairness",
    "describe_entrant",
    "fit_bradley_terry",
    "parse_entrant",
    "play_match",
    "random_openings",
    "round_robin",
    "round_robin_parallel",
    "wilson_interval",
    "write_report",
    "write_tournament",
]
