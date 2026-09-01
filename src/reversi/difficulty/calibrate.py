"""Measuring the difficulty ladder, rather than asserting it.

Four levels exist and are *designed* to differ: each one searches more than the
one below it, samples less randomly, and tolerates less of a drop in value before
refusing a move. None of that is evidence. A ladder whose rungs are not actually
separated is four names for the same opponent, and a player who moves up and
notices nothing has been told something untrue.

So this plays them against each other and against the frozen baselines, fits the
whole result matrix at once, and says plainly whether the ladder holds up.

**Criterion S15**, which is what "done" means here:

    a  ratings strictly increase from Casual to Max
    b  adjacent levels differ by at least 80 Elo
    c  adjacent 95% intervals do not overlap
    d  Casual beats Random with a Wilson lower bound above 0.60
    e  Casual never plays a move whose value is worse than the best available
       by more than its guardrail, over 500 sampled moves

(a) to (c) are what make it a ladder. (d) is what stops the easy end being
*useless* rather than easy -- an opponent that loses to random play is not a
rung. (e) is the guardrail doing its job: Casual should play second-best moves,
not give away corners.

**Every level uses the same checkpoint.** That is the point. If the levels
separate, the separation comes from how much they search and how they choose,
which is a claim about the method rather than about four different networks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations, pairwise
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reversi.agents.base import Agent
from reversi.agents.greedy import GreedyAgent
from reversi.agents.minimax import MinimaxAgent
from reversi.agents.random_agent import RandomAgent
from reversi.arena.elo import fit_bradley_terry
from reversi.arena.match import MatchResult, play_match
from reversi.arena.report import check_fairness
from reversi.arena.stats import wilson_interval
from reversi.arena.tournament import TournamentResult
from reversi.difficulty.levels import LEVELS, DifficultyLevel, choose_move, level_by_name
from reversi.errors import CheckpointError, ConfigError
from reversi.game import rules
from reversi.search.config import SearchConfig
from reversi.search.mcts import MCTS
from reversi.seeding import derive_seed
from reversi.types import Action

if TYPE_CHECKING:
    from reversi.game.state import State
    from reversi.search.evaluator import Evaluator

__all__ = [
    "GUARD_SAMPLES",
    "MIN_ADJACENT_GAP",
    "CalibrationReport",
    "DifficultyAgent",
    "calibrate",
    "check_guardrail",
]

log = logging.getLogger(__name__)

# The gap the plan asks for between adjacent rungs. Below this, two levels are
# not distinguishable by a person playing a handful of games against each.
MIN_ADJACENT_GAP = 80.0

# How many of Casual's own moves to inspect for the guardrail check.
GUARD_SAMPLES = 500


class DifficultyAgent:
    """One network, played at one difficulty.

    ``AZAgent`` always plays its most-visited move, which is the right thing for
    an arena and the wrong thing here: the whole point of the easier rungs is
    that they *do not*. This routes the search result through the level's own
    choice rule instead -- guardrail first, then sample.
    """

    __slots__ = ("_name", "level", "mcts")

    def __init__(self, evaluator: Evaluator, level: DifficultyLevel) -> None:
        config = SearchConfig(
            n_simulations=level.simulations,
            dirichlet_eps=0.0,
            temp_moves=0,
        )
        # Contract C7. Exploration noise belongs to self-play; a rated match with
        # it switched on would be measuring something other than how this level
        # plays.
        config.assert_no_noise(f"difficulty calibration at level {level.name!r}")

        self.mcts = MCTS(evaluator, config)
        self.level = level
        self._name = level.name

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        legal = rules.legal_actions(state)
        if not legal:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(legal) == 1:
            return legal[0]
        return choose_move(self.mcts.run(state), self.level, rng)


# ---------------------------------------------------------------------------
# The guardrail (S15e)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    """How far the easiest level's moves fell below the best available."""

    level: str
    guard: float
    samples: int
    worst_drop: float
    violations: int

    @property
    def ok(self) -> bool:
        return self.violations == 0

    def describe(self) -> str:
        return (
            f"{self.level}: {self.samples} moves, worst drop {self.worst_drop:.3f} "
            f"against a guardrail of {self.guard:.2f}, {self.violations} violation(s)"
        )


def check_guardrail(
    evaluator: Evaluator,
    level: DifficultyLevel,
    *,
    board_size: int,
    samples: int = GUARD_SAMPLES,
    seed: int = 0,
) -> GuardrailReport:
    """Play games at ``level`` and check every move it chose against its own rule.

    This is the difference between an opponent that is *weak* and one that is
    *stupid*. A weak opponent plays the second-best move; a stupid one gives away
    a corner. The guardrail is what enforces that, and it is only worth having if
    it actually holds -- so this measures the drop in value for each move
    actually played, rather than trusting the filter that produced it.
    """
    rng = np.random.default_rng(seed)
    agent = DifficultyAgent(evaluator, level)

    worst = 0.0
    violations = 0
    seen = 0

    while seen < samples:
        state = rules.initial_state(board_size)
        while not rules.is_terminal(state) and seen < samples:
            legal = rules.legal_actions(state)
            if len(legal) < 2:
                state = rules.apply(state, legal[0])
                continue

            result = agent.mcts.run(state)
            action = choose_move(result, level, rng)

            # Judge against the best move the search actually *looked at*, which
            # is what the guardrail itself does.
            #
            # An unvisited move's Q is a placeholder 0.0 rather than an estimate.
            # Taking the maximum over every move therefore treats that 0.0 as an
            # opinion, and in a losing position -- where every move the search
            # examined scores, say, -0.8 -- the placeholder becomes the "best"
            # and every choice looks like a 0.8 drop. Measured that way this
            # reported 97 violations in 500 moves against code that was behaving
            # correctly.
            visited = [i for i in range(len(result.actions)) if result.visits[i] > 0]
            index = result.actions.index(action)
            if not visited or index not in visited:
                # Nothing was searched, so there is no opinion to fall short of.
                state = rules.apply(state, action)
                seen += 1
                continue

            best_q = max(result.q_values[i] for i in visited)
            drop = best_q - result.q_values[index]

            worst = max(worst, drop)
            # A small tolerance: the guardrail compares floats, and a move exactly
            # at the boundary is inside the rule rather than outside it.
            if drop > level.guard + 1e-9:
                violations += 1

            state = rules.apply(state, action)
            seen += 1

    return GuardrailReport(
        level=level.name,
        guard=level.guard,
        samples=seen,
        worst_drop=worst,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# The tournament
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CalibrationReport:
    """What the ladder actually measured, and whether it holds up."""

    tournament: TournamentResult
    guardrail: GuardrailReport
    checks: dict[str, dict[str, Any]]
    levels: list[str]
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks.values())

    def summary(self) -> str:
        lines = [f"S15: {'PASS' if self.passed else 'FAIL'}", ""]
        for name, check in self.checks.items():
            mark = "ok  " if check["passed"] else "FAIL"
            lines.append(f"  {mark} {name}: {check['detail']}")
        return "\n".join(lines)


def _rating(tournament: TournamentResult, name: str) -> Any:
    table = tournament.ratings.by_name()
    if name not in table:
        msg = f"{name} did not appear in the tournament ratings"
        raise ConfigError(msg)
    return table[name]


def _bounds(rating: Any) -> tuple[float, float]:
    """A rating's interval, falling back to the point estimate if it has none."""
    if rating.interval is None:
        return rating.elo, rating.elo
    return rating.interval.low, rating.interval.high


def _evaluator_for(model_path: Path, device: str) -> Evaluator:
    """Load either kind of file this project produces.

    A *training checkpoint* keeps its architecture at the top level; an
    *exported* model keeps it under ``meta``, alongside the provenance. Both are
    reasonable things to point a calibration at -- the export is what ships, and
    the checkpoint is what you have to hand mid-run -- so this accepts either
    rather than making the caller remember which is which.
    """
    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.export import load_export
    from reversi.nn.loader import load_model

    try:
        model = load_export(model_path, device=device).model
    except CheckpointError:
        model = load_model(model_path, device=device)

    return TorchEvaluator(model, device=device)


def calibrate(
    model_path: Path,
    *,
    board_size: int = 8,
    games_per_pair: int = 300,
    seed: int = 20260830,
    device: str = "cpu",
    guard_samples: int = GUARD_SAMPLES,
    include_baselines: bool = True,
    levels: list[DifficultyLevel] | None = None,
    workers: int = 1,
) -> CalibrationReport:
    """Play the ladder against itself and the baselines, then judge it.

    The baselines are included for a reason beyond completeness: rating the
    levels only against each other would fix their *spacing* but leave the whole
    ladder floating, with no way to say what any rung is worth. Anchored at
    random play, "Casual is +180" means something.
    """
    started = time.perf_counter()
    rungs = list(levels if levels is not None else LEVELS)
    if len(rungs) < 2:
        msg = f"calibration needs at least two levels, got {len(rungs)}"
        raise ConfigError(msg)

    specs = [
        (
            level.name,
            {
                "kind": "difficulty",
                "simulations": level.simulations,
                "temperature": level.temperature,
                "top_k": level.top_k,
                "guard": level.guard,
                "model": model_path.name,
            },
        )
        for level in rungs
    ]
    _ = specs

    names = [level.name for level in rungs]
    if include_baselines:
        names += ["random", "greedy", "minimax-d4"]

    tournament = round_robin_parallel(
        names,
        model_path=model_path,
        games_per_pair=games_per_pair,
        board_size=board_size,
        seed=seed,
        workers=workers,
        anchor="random" if include_baselines else rungs[0].name,
        device=device,
    )

    guardrail = check_guardrail(
        _evaluator_for(model_path, device),
        rungs[0],
        board_size=board_size,
        samples=guard_samples,
        seed=seed + 1,
    )

    checks = _judge(tournament, rungs, guardrail, include_baselines=include_baselines)

    return CalibrationReport(
        tournament=tournament,
        guardrail=guardrail,
        checks=checks,
        levels=[level.name for level in rungs],
        elapsed_s=time.perf_counter() - started,
    )


def _judge(
    tournament: Any,
    rungs: list[DifficultyLevel],
    guardrail: GuardrailReport,
    *,
    include_baselines: bool,
) -> dict[str, dict[str, Any]]:
    """Apply S15 to what was measured. Each check reports its own numbers.

    Takes anything carrying a ``ratings`` table rather than a whole
    ``TournamentResult``, so the judgement can be tested against hand-built
    ladders. Checking that a ladder which does not separate is *reported* as
    failing needs ladders that do not separate, and playing thousands of real
    games to manufacture one would be absurd.
    """
    rated = [_rating(tournament, level.name) for level in rungs]
    checks: dict[str, dict[str, Any]] = {}

    # (a) strictly increasing
    rising = all(a.elo < b.elo for a, b in pairwise(rated))
    checks["monotonic"] = {
        "passed": rising,
        "detail": " < ".join(f"{r.name} {r.elo:.0f}" for r in rated)
        + ("" if rising else "   -- not strictly increasing"),
    }

    # (b) adjacent gaps
    gaps = [b.elo - a.elo for a, b in pairwise(rated)]
    checks["adjacent_gap"] = {
        "passed": all(gap >= MIN_ADJACENT_GAP for gap in gaps),
        "detail": (
            ", ".join(
                f"{a.name}->{b.name} {gap:+.0f}"
                for (a, b), gap in zip(pairwise(rated), gaps, strict=True)
            )
            + f"  (need >= {MIN_ADJACENT_GAP:.0f})"
        ),
        "gaps": [round(g, 1) for g in gaps],
    }

    # (c) intervals do not overlap
    # `RatingTable.stronger` is the same strict reading criterion S14 uses --
    # the higher interval must lie entirely above the lower one. Reusing it means
    # the ladder is held to the same standard as the generations were.
    overlaps = [
        (a.name, b.name)
        for a, b in pairwise(rated)
        if not tournament.ratings.stronger(b.name, a.name)
    ]
    checks["intervals_disjoint"] = {
        "passed": not overlaps,
        "detail": (
            "every adjacent pair is separated"
            if not overlaps
            else "overlapping: " + ", ".join(f"{a}/{b}" for a, b in overlaps)
        ),
    }

    # (d) the easiest rung still beats random play
    if include_baselines:
        easiest = rungs[0].name
        pairings = tournament.ratings.results
        record = pairings.get((easiest, "random"))
        flipped = False
        if record is None:
            # Pairings are stored once, in the order the entrants were listed.
            other = pairings.get(("random", easiest))
            if other is not None:
                score, games = other
                record, flipped = (games - score, games), True
        if record is None:
            checks["easiest_beats_random"] = {
                "passed": False,
                "detail": f"no {easiest} vs random result in the tournament",
            }
        else:
            score, games = record
            _ = flipped
            interval = wilson_interval(score, games)
            checks["easiest_beats_random"] = {
                "passed": interval.low > 0.60,
                "detail": (
                    f"{easiest} scored {score:.1f}/{games} vs random "
                    f"= {score / games:.1%} (95% lower bound {interval.low:.1%}, need > 60%)"
                ),
            }

    # (e) the guardrail held
    checks["guardrail"] = {
        "passed": guardrail.ok,
        "detail": guardrail.describe(),
    }

    return checks


# ---------------------------------------------------------------------------
# Writing it down
# ---------------------------------------------------------------------------


def write_report(destination: Path, report: CalibrationReport, model_path: Path) -> dict[str, Any]:
    """Write the evidence, in the same shape as every other arena report."""
    payload = {
        "kind": "difficulty_calibration",
        "report_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model_path.name,
        "levels": report.levels,
        "passed": report.passed,
        "checks": report.checks,
        "guardrail": {
            "level": report.guardrail.level,
            "guard": report.guardrail.guard,
            "samples": report.guardrail.samples,
            "worst_drop": round(report.guardrail.worst_drop, 4),
            "violations": report.guardrail.violations,
        },
        "ratings": [
            {
                "name": r.name,
                "elo": round(r.elo, 1),
                "ci_low": round(_bounds(r)[0], 1),
                "ci_high": round(_bounds(r)[1], 1),
                "games": r.games,
            }
            for r in report.tournament.ratings.sorted()
        ],
        "matchups": report.tournament.matchup_table(),
        "protocol": {
            "board_size": report.tournament.board_size,
            "games_per_pair": report.tournament.games_per_pair,
            "opening_plies": report.tournament.opening_plies,
            "seed": report.tournament.seed,
            "anchor": report.tournament.ratings.anchor,
            "dirichlet_eps": 0.0,
        },
        "elapsed_s": round(report.elapsed_s, 1),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


def write_difficulty_config(
    destination: Path,
    report: CalibrationReport,
    rungs: list[DifficultyLevel],
    model_path: Path,
) -> None:
    """Write the settings *with the evidence that justifies them* attached.

    The rating of each level is recorded next to the numbers that produced it, so
    a reader can see what a rung is worth without going looking -- and so that a
    change to the settings without a re-run is visibly a change to a file that
    claims to be measured.
    """
    ratings = report.tournament.ratings.by_name()
    lines = [
        "# Difficulty levels, with the measurement that justifies them.",
        "#",
        "# GENERATED by `reversi calibrate`. Editing the numbers here without",
        "# re-running it leaves the ratings below describing settings that no",
        "# longer exist.",
        "#",
        f"# model:   {model_path.name}",
        f"# created: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# S15:     {'PASS' if report.passed else 'FAIL'}",
        "",
        "levels:",
    ]

    for level in rungs:
        rating = ratings.get(level.name)
        lines += [
            f"  - name: {level.name}",
            f"    label: {level.label!r}",
            f"    simulations: {level.simulations}",
            f"    temperature: {level.temperature}",
            f"    top_k: {level.top_k if level.top_k is not None else 'null'}",
            f"    guard: {level.guard}",
        ]
        if rating is not None:
            low, high = _bounds(rating)
            lines += [
                f"    elo: {rating.elo:.1f}",
                f"    elo_interval: [{low:.1f}, {high:.1f}]",
            ]
        lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _unused(agent: Agent) -> None:  # pragma: no cover - keeps the Agent import honest
    """`DifficultyAgent` satisfies the `Agent` protocol; this is where that is stated."""
    _ = agent.name


# ---------------------------------------------------------------------------
# Running the pairings in parallel
# ---------------------------------------------------------------------------
#
# A calibration is far more expensive than it looks. Measured on the trained
# network, one `max` move is 1.7 seconds -- 800 simulations at batch one -- so a
# single game where `max` plays costs about 52 seconds, and the plan's 300 games
# per pairing across 21 pairings comes to roughly 38 hours in one process. That
# is why this criterion went unmet for so long: it was never a missing function,
# it was a two-day job nobody could schedule.
#
# The pairings are independent, so they parallelise perfectly, and the arena
# already established the pattern that makes this safe -- separate processes,
# each loading its own copy of the network, no shared state and nothing to
# deadlock on.
#
# **On the CPU rather than the GPU**, which is counterintuitive until you look at
# the day-7 measurement: at batch size one the GPU is only 1.5x a CPU core (499
# positions per second against 329). An arena plays one position at a time, so
# there is nothing to batch, and six CPU processes beat one GPU process by about
# four times while leaving the GPU free.


@dataclass(frozen=True, slots=True)
class _PairJob:
    """One pairing, described in a form that survives being sent to a process."""

    a: str
    b: str
    model_path: str
    board_size: int
    games: int
    seed: int
    opening_plies: int
    device: str


def agent_named(name: str, evaluator: Evaluator | None) -> Agent:
    """Build an entrant from its name alone.

    By name rather than by closure, because a closure over a loaded network
    cannot be sent to another process -- and each process needs its own copy of
    the weights anyway.
    """
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name.startswith("minimax-d"):
        return MinimaxAgent(int(name.removeprefix("minimax-d")), name=name)

    if evaluator is None:
        msg = f"{name} is a difficulty level and needs a network, but none was loaded"
        raise ConfigError(msg)
    return DifficultyAgent(evaluator, level_by_name(name))


def _needs_network(name: str) -> bool:
    return not (name == "random" or name == "greedy" or name.startswith("minimax-d"))


def _play_pairing(job: _PairJob) -> MatchResult:
    """Play one pairing. Runs in its own process; loads its own network."""
    import torch

    # One thread per worker, for the same reason the self-play workers do it: a
    # tree search asks the network about one position at a time, so there is
    # nothing for intra-op threads to divide up. Left at the default, eight
    # worker processes each try to use every core and spend their time
    # contending rather than searching -- measured here as a calibration that
    # managed three pairings in two hours.
    torch.set_num_threads(1)

    evaluator = None
    if _needs_network(job.a) or _needs_network(job.b):
        evaluator = _evaluator_for(Path(job.model_path), job.device)

    return play_match(
        agent_named(job.a, evaluator),
        agent_named(job.b, evaluator),
        games=job.games,
        board_size=job.board_size,
        seed=job.seed,
        opening_plies=job.opening_plies,
    )


def round_robin_parallel(
    names: list[str],
    *,
    model_path: Path,
    games_per_pair: int,
    board_size: int,
    seed: int,
    workers: int,
    opening_plies: int = 4,
    anchor: str = "random",
    bootstrap: int = 500,
    device: str = "cpu",
) -> TournamentResult:
    """The same tournament as ``arena.round_robin``, with the pairings spread out.

    Identical protocol, identical seeding: each pairing derives its seed from the
    tournament seed and the names, so a result does not depend on how many
    processes happened to run it. That is what makes the parallel and sequential
    versions comparable rather than merely similar.
    """
    from concurrent.futures import ProcessPoolExecutor

    started = time.perf_counter()
    jobs = [
        _PairJob(
            a=a,
            b=b,
            model_path=str(model_path),
            board_size=board_size,
            games=games_per_pair,
            # Derived, not sequential: the pairing's seed depends on who is in it,
            # so adding an entrant does not renumber everybody else's games.
            seed=derive_seed(seed, "calibration", a, b),
            opening_plies=opening_plies,
            device=device,
        )
        for a, b in combinations(names, 2)
    ]

    log.info(
        "playing %d pairings of %d games across %d process(es) on %s",
        len(jobs),
        games_per_pair,
        workers,
        device,
    )

    matches: list[MatchResult] = []
    if workers <= 1:
        matches = [_play_pairing(job) for job in jobs]
    else:
        # `spawn` everywhere: it is the only start method Windows has, and using
        # it on every platform means the code is exercised the same way wherever
        # it runs.
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for match in pool.map(_play_pairing, jobs):
                matches.append(match)
                log.info(
                    "%s vs %s: %.1f%% over %d games",
                    match.agent_a,
                    match.agent_b,
                    100 * match.score,
                    match.games,
                )

    results: dict[tuple[str, str], tuple[float, int]] = {}
    for match in matches:
        check_fairness(match, require_openings=opening_plies > 0)
        results[(match.agent_a, match.agent_b)] = (
            match.wins + 0.5 * match.draws,
            match.games,
        )

    return TournamentResult(
        ratings=fit_bradley_terry(results, anchor=anchor, bootstrap=bootstrap),
        matches=matches,
        board_size=board_size,
        games_per_pair=games_per_pair,
        opening_plies=opening_plies,
        seed=seed,
        seconds=time.perf_counter() - started,
    )
