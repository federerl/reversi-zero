"""How many training rows carried the wipeout lesson, and how many said the opposite.

Step 3b (``research/step3b-design.md``) recorded the positions the opponent
moves from in external games, so that the value head would be trained at the
position the search reads when it judges the move that led there: the one
where the network holds a disc or two and Greedy is about to take them. The
head probe found the value head unchanged at those positions. This script
asks the replay data why, by counting what the trainer actually saw.

A row "looks like" a fatal position when the game is past the opening and the
network holds at most a few discs. Among the rows that look like that, the
ones labelled as a loss for the network carry the lesson; the ones labelled
as a win say the opposite, that holding few discs with the opponent to move
is fine. The ratio between the two is what the value head learns.

Two kinds of row are counted, for the generations given:

* **opponent to move** (the rows Step 3b added, ``pi_valid`` false): the
  network is the side *not* to move. The row's result is from the mover's
  side, so a label of +1 means the opponent won, the fatal case.
* **network to move** (ordinary rows): the network is the mover, and a label
  of -1 means it lost.

Usage::

    python scripts/step3b_lesson_rows.py ~/reversi-runs/step3-Bboth-s1 \
        ~/reversi-runs/step3-Bboth-s2 --from-generation 121 \
        --out research/results/step3b/lesson-rows.json

Runs without the ``pi_valid`` column (every run before Step 3b) report zero
opponent-to-move rows. ``--max-discs`` and ``--from-ply`` set what "few discs"
and "past the opening" mean; the defaults are 3 discs and ply 16. Shards are
read with numpy alone, so the script runs anywhere the replay directory is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reversi.atomicio import atomic_write_json

BLACK = 0
"""``to_move`` value for black in a shard (``reversi.types.Player.BLACK``)."""


def popcount(bits: np.ndarray) -> np.ndarray:
    """Number of discs in each bitboard of an array of 64-bit boards."""
    return np.array([int(b).bit_count() for b in bits.astype(np.uint64)], dtype=np.int64)


def count_run(
    run_dir: Path, *, from_generation: int, max_discs: int, from_ply: int
) -> dict[str, Any]:
    rows = 0
    value_only = 0
    opp = {"rows": 0, "opponent_won": 0, "network_won": 0}
    net = {"rows": 0, "network_lost": 0, "network_won": 0}
    for path in sorted((run_dir / "replay").glob("gen_*_w*.npz")):
        arrays = np.load(path)
        generation = arrays["generation"]
        keep = generation >= from_generation
        if not keep.any():
            continue
        pi_valid = arrays["pi_valid"] if "pi_valid" in arrays.files else np.ones(len(keep), bool)
        to_move = arrays["to_move"]
        z = arrays["z"]
        ply = arrays["move_no"]
        black = popcount(arrays["black"])
        white = popcount(arrays["white"])
        mover = np.where(to_move == BLACK, black, white)
        other = np.where(to_move == BLACK, white, black)
        late = ply >= from_ply

        rows += int(keep.sum())
        value_only += int((keep & ~pi_valid).sum())

        # Opponent to move: the network is the side not to move, and z is the
        # opponent's result, so z > 0 is the fatal case.
        sel = keep & ~pi_valid & late & (other <= max_discs)
        opp["rows"] += int(sel.sum())
        opp["opponent_won"] += int((z[sel] > 0).sum())
        opp["network_won"] += int((z[sel] < 0).sum())

        # Network to move: z is the network's result.
        sel = keep & pi_valid & late & (mover <= max_discs)
        net["rows"] += int(sel.sum())
        net["network_lost"] += int((z[sel] < 0).sum())
        net["network_won"] += int((z[sel] > 0).sum())

    return {
        "run": run_dir.name,
        "rows": rows,
        "value_only_rows": value_only,
        "opponent_to_move_few_discs": opp,
        "network_to_move_few_discs": net,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--from-generation", type=int, default=121)
    parser.add_argument("--max-discs", type=int, default=3)
    parser.add_argument("--from-ply", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = [
        count_run(
            run,
            from_generation=args.from_generation,
            max_discs=args.max_discs,
            from_ply=args.from_ply,
        )
        for run in args.runs
    ]
    if args.out is not None:
        atomic_write_json(
            args.out,
            {
                "purpose": "research/step3b-results.md: rows that carried the wipeout lesson",
                "from_generation": args.from_generation,
                "few_discs": f"at most {args.max_discs} discs, from ply {args.from_ply}",
                "runs": results,
            },
        )

    print(
        "| run | rows | value-only rows | opponent to move, network holds few discs: rows "
        "| of which the opponent won (the lesson) | of which the network won "
        "| network to move, holds few discs: rows | of which it lost | of which it won |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        o = r["opponent_to_move_few_discs"]
        n = r["network_to_move_few_discs"]
        print(
            f"| {r['run']} | {r['rows']:,} | {r['value_only_rows']:,} | {o['rows']:,} "
            f"| {o['opponent_won']:,} | {o['network_won']:,} | {n['rows']:,} "
            f"| {n['network_lost']:,} | {n['network_won']:,} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
