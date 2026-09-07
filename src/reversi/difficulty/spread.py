"""How *consistently* a difficulty level plays, as opposed to how strongly.

The calibration in ``calibrate.py`` answers a different question. It shows that
the four levels are separated -- Casual really is weaker than Club, and by how
much. That is necessary and it is not sufficient. A level can sit at a stable
average strength and still be unpleasant to play against, because an average is
not an experience: an opponent that plays two good moves and then throws away a
corner has the same rating as one that plays three mediocre moves, and only one
of them is worth playing twice.

This measures the second thing, in two ways, because "inconsistent" turns out to
mean two separable things.

**Within a game: how far below its own best move does it play?** After the
search finishes, the level's choice rule may pick something other than the most
promising move. The gap in value between what it played and the best move the
search actually examined is the *drop*, and the guardrail is the ceiling on it.
The calibration already checks that the ceiling holds. What it does not report is
the shape of the distribution underneath, and that shape is the answer to "it
feels erratic": a level whose drops are almost always tiny but occasionally near
its guard will feel like two different opponents.

**Between games: do the results vary more than chance allows?** A level that
scores 60% against a fixed opponent will not score exactly 60% in every block of
games -- coin flips do not either. The question is whether it varies *more* than
coin flips would. Blocks of games are compared against the variance the same
results would show if every game were an independent draw at the observed rate.
The ratio of the two is the dispersion: 1.0 is chance, above 1.0 is a level whose
form comes and goes.

Two design notes that matter for reading the numbers.

*The unit is an opening pair, not a game.* A match plays each opening twice with
the colours swapped, so the two games in a pair are strongly anti-correlated --
whoever benefits from the opening tends to win the first and lose the second.
Treating those as independent draws would understate the chance variance and
manufacture overdispersion out of the protocol. Pairs are independent of each
other; games within a pair are not.

*Two of the four levels do not sample at all.* Strong and Max play their most
visited move at temperature 0, so given a position they always answer the same
way. For them the only sources of variation are the opening book and the
opponent, and a dispersion near 1.0 says nothing about the level itself. They are
measured anyway, as a control: whatever the protocol contributes on its own shows
up there.

Because they are the control rather than the subject, they are measured at a
smaller sample -- a quarter of the games and a quarter of the moves. This is not
frugality for its own sake. Max searches fifty times as deep as Casual, so
measuring all four at the same sample would spend nine tenths of the compute on
the two levels nobody has complained about, and the run would take a day instead
of an evening. The reduced sample is recorded next to the numbers so nobody reads
a control interval as though it were a subject interval.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reversi.arena.match import play_match
from reversi.difficulty.calibrate import (
    DifficultyAgent,
    agent_named,
    evaluator_for,
    needs_network,
)
from reversi.difficulty.levels import LEVELS, DifficultyLevel, choose_move, level_by_name
from reversi.game import rules
from reversi.seeding import derive_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reversi.search.evaluator import Evaluator

log = logging.getLogger(__name__)

__all__ = [
    "DROP_SAMPLES",
    "BlockDispersion",
    "DropDistribution",
    "SpreadReport",
    "measure_spread",
    "write_report",
]

# The opponent every level is measured against.
#
# Frozen, deterministic, and already rated on both scales in this project, so a
# reader can place the scores without a new calibration. It also sits *inside*
# the ladder at +358, which keeps the two levels under suspicion off the ceiling:
# a level that wins 99% of its games has almost no variance to measure, and no
# amount of blocking recovers information that the scoreline never contained.
REFERENCE_OPPONENT = "minimax-d4"

# Moves inspected per level for the drop distribution.
#
# Larger than the calibration's 500, because a tail is what is being measured
# here rather than a maximum. At 2,000 moves the 99th percentile rests on about
# twenty observations, which is enough to report and not enough to over-read.
DROP_SAMPLES = 2000

# The same two figures for a level that does not sample. See the module note.
CONTROL_DROP_SAMPLES = 500
CONTROL_BLOCKS = 5

BLOCKS = 15
PAIRS_PER_BLOCK = 10


# ---------------------------------------------------------------------------
# Within a game: the distribution of value drops
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DropDistribution:
    """How far below its best move a level played, over many moves."""

    level: str
    guard: float
    control: bool
    """True for a level that does not sample, measured at a reduced size."""
    samples: int
    """Moves where the level had a real choice and the search had an opinion."""
    forced: int
    """Positions skipped: one legal move, or nothing visited to compare against."""
    best_move_fraction: float
    """Share of moves where it played the best move the search examined."""
    mean: float
    median: float
    p90: float
    p99: float
    worst: float
    near_guard_fraction: float
    """Share of moves within 0.02 of the guardrail -- as bad as it is allowed."""

    def describe(self) -> str:
        tag = " [control]" if self.control else ""
        return (
            f"{self.level}{tag}: {self.samples} moves, best {self.best_move_fraction:.0%}, "
            f"median drop {self.median:.3f}, p99 {self.p99:.3f}, worst {self.worst:.3f} "
            f"against a guard of {self.guard:.2f}"
        )


def measure_drops(
    evaluator: Evaluator,
    level: DifficultyLevel,
    *,
    board_size: int,
    samples: int = DROP_SAMPLES,
    seed: int = 0,
) -> DropDistribution:
    """Play at ``level`` and record the value drop of every move it chose.

    The same measurement the guardrail check makes, kept rather than reduced to a
    maximum. Judging against the best move the search *visited* -- not the best
    over all legal moves -- is deliberate and was the fix for a false alarm: an
    unvisited move carries a placeholder value of 0.0, and treating that as an
    opinion makes every choice in a losing position look like a blunder.
    """
    rng = np.random.default_rng(seed)
    agent = DifficultyAgent(evaluator, level)

    drops: list[float] = []
    forced = 0

    while len(drops) < samples:
        state = rules.initial_state(board_size)
        while not rules.is_terminal(state) and len(drops) < samples:
            legal = rules.legal_actions(state)
            if len(legal) < 2:
                state = rules.apply(state, legal[0])
                continue

            result = agent.mcts.run(state)
            action = choose_move(result, level, rng)

            visited = [i for i in range(len(result.actions)) if result.visits[i] > 0]
            index = result.actions.index(action)
            if not visited or index not in visited:
                forced += 1
                state = rules.apply(state, action)
                continue

            best_q = max(result.q_values[i] for i in visited)
            drops.append(max(0.0, best_q - result.q_values[index]))
            state = rules.apply(state, action)

    array = np.asarray(drops, dtype=np.float64)
    return DropDistribution(
        level=level.name,
        guard=level.guard,
        control=level.temperature <= 0.0,
        samples=int(array.size),
        forced=forced,
        # A drop of exactly zero is the best move; float noise makes an equality
        # test on the value itself unreliable, so the threshold is a hair above.
        best_move_fraction=float((array <= 1e-9).mean()),
        mean=float(array.mean()),
        median=float(np.median(array)),
        p90=float(np.quantile(array, 0.90)),
        p99=float(np.quantile(array, 0.99)),
        worst=float(array.max()),
        near_guard_fraction=float((array >= level.guard - 0.02).mean()) if level.guard > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Between games: dispersion of block scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockDispersion:
    """Whether a level's results vary more than independent games would."""

    level: str
    opponent: str
    samples_moves: bool
    """False for a level that always plays its best move -- see the module note."""
    blocks: int
    pairs_per_block: int
    block_scores: tuple[float, ...]
    mean_score: float
    pair_variance: float
    """Variance of a single opening pair's score. The chance baseline."""
    observed_variance: float
    """Variance of the block scores actually seen."""
    expected_variance: float
    """``pair_variance / pairs_per_block``: what chance alone would give."""
    dispersion: float
    chi_square: float
    degrees_of_freedom: int
    p_value: float

    @property
    def overdispersed(self) -> bool:
        """More variable than chance, at the conventional 5% level."""
        return self.p_value < 0.05 and self.dispersion > 1.0

    def describe(self) -> str:
        verdict = "more variable than chance" if self.overdispersed else "consistent with chance"
        note = "" if self.samples_moves else " [plays deterministically; control]"
        return (
            f"{self.level} vs {self.opponent}: {self.mean_score:.1%} over "
            f"{self.blocks} blocks, dispersion {self.dispersion:.2f} "
            f"(p={self.p_value:.3f}) -- {verdict}{note}"
        )


def dispersion_of(
    pair_scores: list[float],
    *,
    level: str,
    opponent: str,
    samples_moves: bool,
    pairs_per_block: int,
) -> BlockDispersion:
    """Compare the variance between blocks against the variance within them.

    The null hypothesis is that every opening pair is an independent draw from
    one fixed distribution. Under it, a block score is the mean of
    ``pairs_per_block`` such draws, so its variance is the pair variance divided
    by that count, and the sum of squared deviations scaled by it follows a
    chi-square distribution. A level whose form genuinely comes and goes breaks
    the "one fixed distribution" part and shows up as a sum too large to explain.

    Estimating the pair variance from the pooled pairs rather than assuming a
    binomial is what lets draws count as half a result without special-casing.
    """
    from scipy.stats import chi2

    scores = np.asarray(pair_scores, dtype=np.float64)
    blocks = scores.reshape(-1, pairs_per_block)
    block_scores = blocks.mean(axis=1)

    mean = float(scores.mean())
    pair_variance = float(scores.var(ddof=1))
    expected = pair_variance / pairs_per_block
    observed = float(block_scores.var(ddof=1))

    degrees = int(block_scores.size - 1)
    if expected <= 0.0:
        # Every pair scored the same, so there is nothing to disperse. This is
        # the honest answer for a deterministic level that sweeps its opponent:
        # no variance to explain rather than a ratio of two zeros.
        return BlockDispersion(
            level=level,
            opponent=opponent,
            samples_moves=samples_moves,
            blocks=int(block_scores.size),
            pairs_per_block=pairs_per_block,
            block_scores=tuple(round(float(s), 4) for s in block_scores),
            mean_score=mean,
            pair_variance=0.0,
            observed_variance=observed,
            expected_variance=0.0,
            dispersion=float("nan"),
            chi_square=0.0,
            degrees_of_freedom=degrees,
            p_value=float("nan"),
        )

    statistic = float(((block_scores - mean) ** 2).sum() / expected)
    return BlockDispersion(
        level=level,
        opponent=opponent,
        samples_moves=samples_moves,
        blocks=int(block_scores.size),
        pairs_per_block=pairs_per_block,
        block_scores=tuple(round(float(s), 4) for s in block_scores),
        mean_score=mean,
        pair_variance=pair_variance,
        observed_variance=observed,
        expected_variance=expected,
        dispersion=observed / expected,
        chi_square=statistic,
        degrees_of_freedom=degrees,
        # Upper tail only. The question is whether it varies *more* than chance;
        # a level that varies less is not a complaint anybody has made.
        p_value=float(chi2.sf(statistic, degrees)),
    )


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DropJob:
    """One level's drop distribution, described so it survives being sent away."""

    level: str
    model_path: str
    board_size: int
    samples: int
    seed: int
    device: str


def _run_drops(job: _DropJob) -> DropDistribution:
    """Measure one level's drops. Runs in its own process; loads its own network."""
    import torch

    torch.set_num_threads(1)
    return measure_drops(
        evaluator_for(Path(job.model_path), job.device),
        level_by_name(job.level),
        board_size=job.board_size,
        samples=job.samples,
        seed=job.seed,
    )


@dataclass(frozen=True, slots=True)
class _PairScoreJob:
    """One opening pair: two games, colours swapped. Sent to a worker process."""

    level: str
    opponent: str
    model_path: str
    board_size: int
    seed: int
    opening_plies: int
    device: str


def _play_pair(job: _PairScoreJob) -> float:
    """Play one opening pair and return the level's score over the two games."""
    import torch

    # One thread per worker: a tree search asks about one position at a time, so
    # intra-op threads have nothing to divide and eight workers each claiming
    # every core spend their time contending instead of searching.
    torch.set_num_threads(1)

    evaluator = None
    if needs_network(job.level) or needs_network(job.opponent):
        evaluator = evaluator_for(Path(job.model_path), job.device)

    result = play_match(
        agent_named(job.level, evaluator),
        agent_named(job.opponent, evaluator),
        games=2,
        board_size=job.board_size,
        seed=job.seed,
        opening_plies=job.opening_plies,
    )
    return result.score


@dataclass(slots=True)
class SpreadReport:
    """Everything the measurement produced."""

    model: str
    drops: list[DropDistribution]
    dispersions: list[BlockDispersion]
    elapsed_s: float

    def summary(self) -> str:
        lines = ["Value drops within a game:"]
        lines += [f"  {d.describe()}" for d in self.drops]
        lines.append("")
        lines.append("Result variation between blocks of games:")
        lines += [f"  {d.describe()}" for d in self.dispersions]
        return "\n".join(lines)


def measure_spread(
    model_path: Path,
    *,
    board_size: int = 8,
    levels: list[str] | None = None,
    opponent: str = REFERENCE_OPPONENT,
    blocks: int = BLOCKS,
    pairs_per_block: int = PAIRS_PER_BLOCK,
    drop_samples: int = DROP_SAMPLES,
    control_blocks: int = CONTROL_BLOCKS,
    control_drop_samples: int = CONTROL_DROP_SAMPLES,
    opening_plies: int = 4,
    seed: int = 20260907,
    device: str = "cpu",
    workers: int = 1,
) -> SpreadReport:
    """Measure both kinds of spread for every level, and return the evidence."""
    names = levels if levels is not None else [level.name for level in LEVELS]
    rungs = [level_by_name(name) for name in names]
    started = time.perf_counter()

    def blocks_for(rung: DifficultyLevel) -> int:
        return blocks if rung.temperature > 0.0 else min(blocks, control_blocks)

    def drops_for(rung: DifficultyLevel) -> int:
        return drop_samples if rung.temperature > 0.0 else min(drop_samples, control_drop_samples)

    pair_jobs = [
        _PairScoreJob(
            level=rung.name,
            opponent=opponent,
            model_path=str(model_path),
            board_size=board_size,
            # Derived per pair, so a pair's games do not depend on how many
            # processes happened to run it or on what order they finished in.
            seed=derive_seed(seed, "spread", rung.name, opponent, str(index)),
            opening_plies=opening_plies,
            device=device,
        )
        for rung in rungs
        for index in range(blocks_for(rung) * pairs_per_block)
    ]
    drop_jobs = [
        _DropJob(
            level=rung.name,
            model_path=str(model_path),
            board_size=board_size,
            samples=drops_for(rung),
            seed=derive_seed(seed, "drops", rung.name),
            device=device,
        )
        for rung in rungs
    ]

    log.info(
        "%d opening pairs and %d drop samples across %d process(es)",
        len(pair_jobs),
        sum(job.samples for job in drop_jobs),
        workers,
    )

    # The two phases go into one pool together, longest job first.
    #
    # Keeping them separate would leave the tail idle: the drop measurements
    # cannot be split any finer than one per level, and Max's is far longer than
    # any single opening pair, so a pool that starts it last spends its final
    # hour running one process on one core. Started first, it overlaps with
    # everything else.
    if workers <= 1:
        drops = [_run_drops(job) for job in drop_jobs]
        scores = [_play_pair(job) for job in pair_jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(_run_drops, job) for job in drop_jobs]
            futures = [pool.submit(_play_pair, job) for job in pair_jobs]
            drops = [future.result() for future in pending]
            scores = [future.result() for future in futures]

    dispersions = []
    cursor = 0
    for rung in rungs:
        count = blocks_for(rung) * pairs_per_block
        dispersions.append(
            dispersion_of(
                scores[cursor : cursor + count],
                level=rung.name,
                opponent=opponent,
                samples_moves=rung.temperature > 0.0,
                pairs_per_block=pairs_per_block,
            )
        )
        cursor += count

    return SpreadReport(
        model=model_path.name,
        drops=drops,
        dispersions=dispersions,
        elapsed_s=time.perf_counter() - started,
    )


def write_report(destination: Path, report: SpreadReport) -> dict[str, Any]:
    """Write the evidence, in the same shape as every other report here."""

    def clean(value: float) -> float | None:
        """JSON has no NaN. A missing number is missing, not zero."""
        return None if not np.isfinite(value) else round(float(value), 6)

    payload = {
        "kind": "difficulty_spread",
        "report_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": report.model,
        "value_drops": [
            {
                "level": d.level,
                "guard": d.guard,
                "control": d.control,
                "samples": d.samples,
                "forced": d.forced,
                "best_move_fraction": round(d.best_move_fraction, 4),
                "mean": round(d.mean, 4),
                "median": round(d.median, 4),
                "p90": round(d.p90, 4),
                "p99": round(d.p99, 4),
                "worst": round(d.worst, 4),
                "near_guard_fraction": round(d.near_guard_fraction, 4),
            }
            for d in report.drops
        ],
        "block_dispersion": [
            {
                "level": d.level,
                "opponent": d.opponent,
                "samples_moves": d.samples_moves,
                "blocks": d.blocks,
                "pairs_per_block": d.pairs_per_block,
                "block_scores": list(d.block_scores),
                "mean_score": round(d.mean_score, 4),
                "pair_variance": round(d.pair_variance, 6),
                "observed_variance": clean(d.observed_variance),
                "expected_variance": clean(d.expected_variance),
                "dispersion": clean(d.dispersion),
                "chi_square": clean(d.chi_square),
                "degrees_of_freedom": d.degrees_of_freedom,
                "p_value": clean(d.p_value),
                "overdispersed": d.overdispersed,
            }
            for d in report.dispersions
        ],
        "protocol": {
            "reference_opponent": report.dispersions[0].opponent if report.dispersions else None,
            "unit": "opening pair, two games with the colours swapped",
            "null_hypothesis": (
                "every opening pair is an independent draw from one fixed "
                "distribution, so a block score's variance is the pair variance "
                "divided by the pairs per block"
            ),
            "dirichlet_eps": 0.0,
        },
        "elapsed_s": round(report.elapsed_s, 1),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload
