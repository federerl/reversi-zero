"""How many random playouts per move does the knowledge-free opponent need?

The held-out roster (``research/held-out-opponents.md``) asks for a
random-playout tree search "at a fixed number of rollouts per move, chosen so
that it scores roughly even against Minimax-d4". This script measures the
candidates: each playout budget plays a colour-balanced match against
Minimax-d4 and, for context, against Greedy, from the same seeded 4-ply
openings the evaluation protocol uses.

Usage::

    python scripts/calibrate_rollout.py --rollouts 250 1000 3000 --games 60 \
        --out research/results/step2/rollout-calibration.json

Every (budget, opponent) match runs in its own process. The budgets are slow
in pure Python (about 2.5 seconds per opening move at 1,000 playouts), so
expect hours rather than minutes for the larger ones.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

from reversi.arena.entrants import build_agent, parse_entrant
from reversi.arena.match import play_match
from reversi.arena.stats import wilson_interval
from reversi.atomicio import atomic_write_json
from reversi.seeding import derive_seed

OPPONENTS = ("minimax-d4", "greedy")


@dataclass(frozen=True)
class Job:
    rollouts: int
    opponent: str
    games: int
    seed: int
    opening_plies: int


@dataclass(frozen=True)
class Outcome:
    rollouts: int
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    score: float
    ci_low: float
    ci_high: float
    score_as_black: float
    score_as_white: float
    mean_plies: float
    seconds: float
    seed: int


def run(job: Job) -> Outcome:
    rollout = build_agent(parse_entrant(f"rollout-r{job.rollouts}", default_simulations=1))
    opponent = build_agent(parse_entrant(job.opponent, default_simulations=1))
    started = time.perf_counter()
    match = play_match(
        rollout,
        opponent,
        games=job.games,
        board_size=8,
        seed=job.seed,
        opening_plies=job.opening_plies,
    )
    interval = wilson_interval(match.wins + 0.5 * match.draws, match.games)
    return Outcome(
        rollouts=job.rollouts,
        opponent=job.opponent,
        games=match.games,
        wins=match.wins,
        losses=match.losses,
        draws=match.draws,
        score=match.score,
        ci_low=interval.low,
        ci_high=interval.high,
        score_as_black=match.score_as_black,
        score_as_white=match.score_as_white,
        mean_plies=match.mean_plies,
        seconds=time.perf_counter() - started,
        seed=job.seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rollouts", type=int, nargs="+", default=[250, 1000, 3000])
    parser.add_argument("--games", type=int, default=60, help="per match, even")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jobs = [
        Job(
            rollouts=n,
            opponent=opponent,
            games=args.games,
            seed=derive_seed(args.seed, "rollout-calibration", n, opponent),
            opening_plies=args.opening_plies,
        )
        for n in args.rollouts
        for opponent in OPPONENTS
    ]
    # Largest budgets first, so the slowest job is not the last one to start.
    jobs.sort(key=lambda j: -j.rollouts)

    outcomes: list[Outcome] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        # Written out as each match finishes, not in submission order, so the
        # cheap budgets are on disk hours before the most expensive one ends.
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                f"rollout-r{outcome.rollouts} vs {outcome.opponent}: "
                f"{outcome.score:.1%} [{outcome.ci_low:.1%}, {outcome.ci_high:.1%}] "
                f"({outcome.wins}W {outcome.losses}L {outcome.draws}D, "
                f"{outcome.seconds / 60:.0f} min)",
                flush=True,
            )
            atomic_write_json(
                args.out,
                {
                    "purpose": "rollout budget for held-out opponent 2, "
                    "research/held-out-opponents.md section 8",
                    "protocol": {
                        "games_per_match": args.games,
                        "opening_plies": args.opening_plies,
                        "colour_balanced": True,
                        "seed": args.seed,
                    },
                    "matches": [asdict(o) for o in sorted(outcomes, key=lambda o: o.rollouts)],
                },
            )

    print()
    print("| playouts per move | vs Minimax-d4 [95%] | vs Greedy [95%] | seconds per game |")
    print("|---:|---|---|---:|")
    for n in sorted(args.rollouts):
        row = {o.opponent: o for o in outcomes if o.rollouts == n}
        cells = [
            f"{row[op].score:.1%} [{row[op].ci_low:.1%}, {row[op].ci_high:.1%}]" for op in OPPONENTS
        ]
        per_game = sum(o.seconds for o in row.values()) / sum(o.games for o in row.values())
        print(f"| {n} | {cells[0]} | {cells[1]} | {per_game:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
