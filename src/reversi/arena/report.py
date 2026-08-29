"""Writing down a match so the result can be defended later.

A win rate on its own is not evidence. "Agent A scored 68% against Minimax-d4" is
only meaningful alongside: which checkpoint A was, how hard it was allowed to
think, which opponent weights were used, how many games, whether the colours were
even, which openings, what seed, and what code produced it.

All of that goes in the report, because the question that eventually gets asked
is not "what was the number" but **"is the number trustworthy"**, and answering
that six weeks later means having written it down at the time.

**The fairness checks run on every report.** They are cheap, and the things they
catch -- an odd number of games, a colour imbalance, an opening book that was
silently empty -- would each bias a result by an amount nobody could estimate
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reversi.arena.match import MatchResult
from reversi.atomicio import atomic_write_json
from reversi.errors import ArenaError
from reversi.obs.runmeta import git_info

__all__ = ["MatchReport", "check_fairness", "write_report"]

REPORT_VERSION = 1


@dataclass(slots=True)
class MatchReport:
    """One matchup, with everything needed to judge whether to believe it."""

    result: MatchResult
    board_size: int
    agent_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """What each agent actually was: checkpoint, simulations, search settings,
    evaluation weights. Two runs disagreeing about an agent's strength usually
    turn out to have been measuring different agents."""

    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        result = self.result
        return {
            "report_version": REPORT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "board_size": self.board_size,
            "agents": {"a": result.agent_a, "b": result.agent_b},
            "agent_specs": self.agent_specs,
            "games": result.games,
            "score_a": result.score,
            "record": {
                "wins": result.wins,
                "losses": result.losses,
                "draws": result.draws,
            },
            "by_colour": {
                "a_as_black": {
                    "games": result.games // 2,
                    "wins": result.wins_as_black,
                    "draws": result.draws_as_black,
                    "score": result.score_as_black,
                },
                "a_as_white": {
                    "games": result.games // 2,
                    "wins": result.wins_as_white,
                    "draws": result.draws_as_white,
                    "score": result.score_as_white,
                },
            },
            "protocol": {
                "seed": result.seed,
                "opening_plies": result.opening_plies,
                "openings_used": result.openings_used,
                "colour_balanced": True,
                "mean_plies": result.mean_plies,
            },
            "provenance": {"git": git_info()},
            "notes": self.notes,
        }


def check_fairness(result: MatchResult, *, require_openings: bool = True) -> None:
    """Refuse to report a match that was not run fairly (contract S13).

    Every one of these would bias the number rather than break it, which is why
    they are checked rather than assumed:

    * an odd game count gives one agent an extra game as black, and black moves
      first;
    * the colour totals not adding up means games were miscounted somewhere;
    * an empty opening book against deterministic agents means the "200 games"
      were one game played 200 times.
    """
    if result.games < 2 or result.games % 2 != 0:
        msg = f"a fair match needs an even number of games; this one had {result.games}"
        raise ArenaError(msg)

    counted = result.wins + result.losses + result.draws
    if counted != result.games:
        msg = f"the record sums to {counted} but {result.games} games were played"
        raise ArenaError(msg)

    if result.wins_as_black + result.wins_as_white != result.wins:
        msg = "wins by colour do not add up to the total"
        raise ArenaError(msg)
    if result.draws_as_black + result.draws_as_white != result.draws:
        msg = "draws by colour do not add up to the total"
        raise ArenaError(msg)

    if require_openings:
        if result.opening_plies <= 0:
            msg = (
                "this match started every game from the standard position. Against "
                "agents that play deterministically that is one game repeated, not "
                f"{result.games} games. Pass opening_plies, or say require_openings=False "
                "if both agents are genuinely random."
            )
            raise ArenaError(msg)
        if result.openings_used * 2 != result.games:
            msg = (
                f"{result.openings_used} openings for {result.games} games: each "
                "opening must be played exactly twice, once with each colour"
            )
            raise ArenaError(msg)


def write_report(
    path: Path,
    report: MatchReport,
    *,
    require_openings: bool = True,
) -> dict[str, Any]:
    """Check the match was fair, then write it. Returns what was written."""
    check_fairness(report.result, require_openings=require_openings)
    payload = report.to_json()
    atomic_write_json(path, payload)
    return payload
