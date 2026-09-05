"""Everyone plays everyone, then the whole result matrix is fitted at once.

This is what answers the question the project exists to answer: **is a later
checkpoint actually stronger than an earlier one, and by how much, and how sure
are we?**

Doing it as a round robin rather than a chain of pairwise comparisons matters.
"Generation 60 beat generation 50 by 55%" is one number with a wide interval.
Fitting every result together lets each agent's rating draw on all of its games
against everyone, so the comparison between two checkpoints is informed by how
each of them did against Random, Greedy and Minimax too.

**The baselines are in the tournament, not outside it.** They are what pin the
scale to something a reader can interpret: Random at 0 by definition, and
Minimax-d4 wherever it lands. A rating gap between two checkpoints means little
on its own; the same gap expressed as "crossed from below Greedy to above
Minimax-d4" means something.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from reversi.agents.base import Agent
from reversi.arena.elo import RatingTable, fit_bradley_terry
from reversi.arena.entrants import EntrantSpec, build_agent
from reversi.arena.match import MatchResult, play_match
from reversi.arena.report import check_fairness
from reversi.arena.stats import wilson_interval
from reversi.atomicio import atomic_write_json
from reversi.errors import ArenaError
from reversi.obs.runmeta import git_info
from reversi.seeding import derive_seed

__all__ = [
    "Entrant",
    "TournamentResult",
    "round_robin",
    "round_robin_parallel",
    "write_tournament",
]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Entrant:
    """One competitor: how to build it, and what it is.

    ``build`` is a factory rather than an instance so that a tournament over
    twenty checkpoints does not hold twenty networks in memory at once.
    """

    name: str
    build: Callable[[], Agent]
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TournamentResult:
    ratings: RatingTable
    matches: list[MatchResult]
    board_size: int
    games_per_pair: int
    opening_plies: int
    seed: int
    seconds: float

    def matchup_table(self) -> list[dict[str, Any]]:
        """Every pairing with its win rate and Wilson interval."""
        rows = []
        for match in self.matches:
            interval = wilson_interval(match.wins + 0.5 * match.draws, match.games)
            rows.append(
                {
                    "a": match.agent_a,
                    "b": match.agent_b,
                    "games": match.games,
                    "score_a": match.score,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "decisive": interval.excludes(0.5),
                    "record": f"{match.wins}W {match.losses}L {match.draws}D",
                    "score_as_black": match.score_as_black,
                    "score_as_white": match.score_as_white,
                }
            )
        return rows

    def describe(self) -> str:
        lines = [self.ratings.describe(), "", "matchups:"]
        for row in self.matchup_table():
            marker = " " if row["decisive"] else "?"
            lines.append(
                f" {marker} {row['a']:<16} vs {row['b']:<16} "
                f"{row['score_a']:6.1%} [{row['ci_low']:.1%}, {row['ci_high']:.1%}]  "
                f"{row['record']}"
            )
        lines.append("")
        lines.append("  ? = the interval spans 50%, so this pairing is not decisive")
        return "\n".join(lines)


def round_robin(
    entrants: list[Entrant],
    *,
    games_per_pair: int,
    board_size: int,
    seed: int,
    opening_plies: int = 4,
    anchor: str = "random",
    bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> TournamentResult:
    """Play every pairing, then fit ratings to the whole matrix.

    Every matchup uses the same number of games and the same opening depth, so no
    agent's rating is better determined than another's for reasons unrelated to
    how it played.
    """
    if len(entrants) < 2:
        msg = f"a tournament needs at least two entrants, got {len(entrants)}"
        raise ArenaError(msg)
    names = [e.name for e in entrants]
    if len(set(names)) != len(names):
        msg = f"entrant names must be unique: {names}"
        raise ArenaError(msg)

    started = time.perf_counter()
    matches: list[MatchResult] = []
    results: dict[tuple[str, str], tuple[float, int]] = {}

    for first, second in combinations(entrants, 2):
        a, b = first.build(), second.build()
        log.info("playing %s vs %s (%d games)", first.name, second.name, games_per_pair)
        match = play_match(
            a,
            b,
            games=games_per_pair,
            board_size=board_size,
            seed=seed,
            opening_plies=opening_plies,
        )
        check_fairness(match, require_openings=opening_plies > 0)
        matches.append(match)
        results[(first.name, second.name)] = (
            match.wins + 0.5 * match.draws,
            match.games,
        )

    ratings = fit_bradley_terry(results, anchor=anchor, bootstrap=bootstrap, rng=rng)

    return TournamentResult(
        ratings=ratings,
        matches=matches,
        board_size=board_size,
        games_per_pair=games_per_pair,
        opening_plies=opening_plies,
        seed=seed,
        seconds=time.perf_counter() - started,
    )


def write_tournament(
    path: Path,
    result: TournamentResult,
    *,
    specs: dict[str, dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Write the whole tournament, ratings and matchups together."""
    payload = {
        "report_version": 1,
        "kind": "round_robin",
        "board_size": result.board_size,
        "protocol": {
            "games_per_pair": result.games_per_pair,
            "opening_plies": result.opening_plies,
            "seed": result.seed,
            "anchor": result.ratings.anchor,
            "seconds": result.seconds,
        },
        "ratings": [
            {
                "name": r.name,
                "elo": r.elo,
                "ci_low": r.interval.low if r.interval else None,
                "ci_high": r.interval.high if r.interval else None,
                "games": r.games,
            }
            for r in result.ratings.sorted()
        ],
        "matchups": result.matchup_table(),
        "agent_specs": specs or {},
        "provenance": {"git": git_info()},
        "notes": notes,
    }
    atomic_write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# The same tournament, with the pairings spread across processes
# ---------------------------------------------------------------------------
#
# The pairings of a round robin are independent, so they parallelise perfectly.
# Each process builds its own agents from an ``EntrantSpec`` and loads its own
# copy of any network, so there is no shared state and nothing to deadlock on.
#
# On the CPU rather than the GPU, which is counterintuitive until measured: a
# tournament asks the network about one position at a time, and at batch size one
# a GPU is barely faster than a CPU core. Several CPU processes beat one GPU
# process by a wide margin and leave the GPU free for training.


@dataclass(frozen=True, slots=True)
class _PairJob:
    """One pairing, in a form that survives being sent to another process."""

    a: EntrantSpec
    b: EntrantSpec
    board_size: int
    games: int
    seed: int
    opening_plies: int
    device: str


def _play_pairing(job: _PairJob) -> MatchResult:
    """Play one pairing. Runs in its own process and loads its own networks."""
    if job.a.needs_network or job.b.needs_network:
        import torch

        # One thread per process: a tree search asks about one position at a
        # time, so there is nothing for intra-op threads to divide. Left at the
        # default, eight processes each try to use every core and spend their
        # time contending rather than searching.
        torch.set_num_threads(1)

    return play_match(
        build_agent(job.a, device=job.device),
        build_agent(job.b, device=job.device),
        games=job.games,
        board_size=job.board_size,
        seed=job.seed,
        opening_plies=job.opening_plies,
    )


def round_robin_parallel(
    entrants: list[EntrantSpec],
    *,
    games_per_pair: int,
    board_size: int,
    seed: int,
    workers: int = 1,
    opening_plies: int = 4,
    anchor: str = "random",
    bootstrap: int = 500,
    device: str = "cpu",
    scope: str = "arena",
) -> TournamentResult:
    """``round_robin`` over specs instead of live agents, with the pairings spread out.

    Identical protocol. Each pairing derives its seed from the tournament seed,
    ``scope`` and the two names, so a result does not depend on how many processes
    happened to run it, and adding an entrant does not renumber everyone else's
    games. ``workers=1`` plays in this process, which is what tests use.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if len(entrants) < 2:
        msg = f"a tournament needs at least two entrants, got {len(entrants)}"
        raise ArenaError(msg)
    names = [e.name for e in entrants]
    if len(set(names)) != len(names):
        msg = f"entrant names must be unique: {names}"
        raise ArenaError(msg)
    if anchor not in names:
        msg = f"the anchor {anchor!r} must be one of the entrants: {names}"
        raise ArenaError(msg)

    started = time.perf_counter()
    jobs = [
        _PairJob(
            a=a,
            b=b,
            board_size=board_size,
            games=games_per_pair,
            seed=derive_seed(seed, scope, a.name, b.name),
            opening_plies=opening_plies,
            device=device,
        )
        for a, b in combinations(entrants, 2)
    ]
    log.info(
        "playing %d pairings of %d games across %d process(es) on %s",
        len(jobs),
        games_per_pair,
        max(1, workers),
        device,
    )

    matches: list[MatchResult] = []
    if workers <= 1:
        for job in jobs:
            matches.append(_play_pairing(job))
            _log_match(matches[-1])
    else:
        # `spawn` everywhere: it is the only start method Windows has, and using
        # it on every platform means the code is exercised the same way wherever
        # it runs.
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for match in pool.map(_play_pairing, jobs):
                matches.append(match)
                _log_match(match)

    results: dict[tuple[str, str], tuple[float, int]] = {}
    for match in matches:
        check_fairness(match, require_openings=opening_plies > 0)
        results[(match.agent_a, match.agent_b)] = (match.wins + 0.5 * match.draws, match.games)

    return TournamentResult(
        ratings=fit_bradley_terry(results, anchor=anchor, bootstrap=bootstrap),
        matches=matches,
        board_size=board_size,
        games_per_pair=games_per_pair,
        opening_plies=opening_plies,
        seed=seed,
        seconds=time.perf_counter() - started,
    )


def _log_match(match: MatchResult) -> None:
    log.info(
        "%s vs %s: %.1f%% over %d games",
        match.agent_a,
        match.agent_b,
        100 * match.score,
        match.games,
    )
