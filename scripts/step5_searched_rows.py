"""Did the search find the wiping reply at the opponent's positions? (Step 5)

The Step 5 runs (``research/step5-design.md``) search the opponent's positions
in external games and store the search's visit distribution as the policy
target there. The design's first prediction is a check that the mechanism
worked: at the opponent-to-move rows where some legal move takes every one of
the network's discs, the stored target should put most of its mass on such a
move, because that move ends the game as a win and the search reads the exact
result on its first visit.

This script reads every shard a run produced from a given generation on,
finds the rows recorded with the opponent to move (the ``opponent_to_move``
column), and for each one asks the rules whether a wiping move exists. For the
rows where one does, it reports how much of the policy target sits on wiping
moves and how many rows put more than half of it there.

Usage::

    python scripts/step5_searched_rows.py ~/reversi-runs/step5-Bboth-sp-s1 \\
        --from-generation 121 --out research/results/step5/searched-rows.json

Runs without the ``opponent_to_move`` column report zero opponent rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reversi.game import rules
from reversi.game.state import State
from reversi.types import Player


def wiping_actions(state: State) -> list[int]:
    """The legal moves from ``state`` after which the other side has no discs."""
    out = []
    for action in rules.legal_actions(state):
        after = rules.apply(state, action)
        theirs = after.white if state.to_move is Player.BLACK else after.black
        if theirs == 0:
            out.append(int(action))
    return out


def scan(run_dir: Path, *, from_generation: int, board_size: int) -> dict:
    rows = opponent_rows = with_wipe = concentrated = 0
    mass_total = 0.0
    by_generation: dict[int, dict[str, int]] = {}
    for path in sorted((run_dir / "replay").glob("gen_*_w*.npz")):
        with np.load(path) as arrays:
            generation = arrays["generation"]
            keep = generation >= from_generation
            rows += int(keep.sum())
            if "opponent_to_move" not in arrays.files:
                continue
            theirs = arrays["opponent_to_move"].astype(bool) & keep
            opponent_rows += int(theirs.sum())
            black, white, to_move, pi = (
                arrays["black"],
                arrays["white"],
                arrays["to_move"],
                arrays["pi"],
            )
            for row in np.flatnonzero(theirs):
                state = State(
                    black=int(black[row]),
                    white=int(white[row]),
                    to_move=Player(int(to_move[row])),
                    size=board_size,
                )
                wipes = wiping_actions(state)
                if not wipes:
                    continue
                with_wipe += 1
                mass = float(pi[row][wipes].sum())
                mass_total += mass
                concentrated += mass > 0.5
                bucket = by_generation.setdefault(
                    int(generation[row]), {"with_wipe": 0, "concentrated": 0}
                )
                bucket["with_wipe"] += 1
                bucket["concentrated"] += mass > 0.5
    return {
        "run": run_dir.name,
        "rows": rows,
        "opponent_to_move_rows": opponent_rows,
        "with_a_wiping_move": with_wipe,
        "target_mostly_on_a_wiping_move": concentrated,
        "mean_target_mass_on_wiping_moves": (mass_total / with_wipe) if with_wipe else None,
        "by_generation": {str(g): v for g, v in sorted(by_generation.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--from-generation", type=int, default=121)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = [
        scan(run, from_generation=args.from_generation, board_size=args.board_size)
        for run in args.runs
    ]
    for r in results:
        share = (
            f"{r['target_mostly_on_a_wiping_move'] / r['with_a_wiping_move']:.0%}"
            if r["with_a_wiping_move"]
            else "n/a"
        )
        print(
            f"{r['run']}: {r['opponent_to_move_rows']} opponent rows, "
            f"{r['with_a_wiping_move']} with a wiping move, target mostly on one in {share}",
            file=sys.stderr,
        )
    payload = {
        "purpose": "research/step5-design.md, prediction 1: the search's target at the "
        "opponent's positions where a move takes every disc of the network's",
        "from_generation": args.from_generation,
        "runs": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
