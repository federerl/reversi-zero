"""Assemble the Step 3 scorecard from one arena report per pairing.

The Step 3 evaluation (``research/step3-design.md``, section 4) plays every
group's final checkpoint, and the checkpoint they all started from, against each
opponent on the frozen roster (``research/held-out-opponents.md``, section 6).
Each pairing is its own ``reversi arena --suite custom`` job with two entrants,
so the evaluation directory holds one report and one recorded-games folder per
pairing, named ``<checkpoint>-vs-<opponent>.json`` and
``<checkpoint>-vs-<opponent>-games/``.

This script reads them all and writes one table: for every checkpoint and every
opponent, the checkpoint's score with its 95% interval, and how many of the 200
games it lost by *wipeout* (ending with no discs on the board, the loss Step 2
found self-play never prepares the network for). Wipeouts the checkpoint
inflicted are counted too, since an opponent that cannot be wiped out says as
much about the opponent as about the network.

Usage::

    python scripts/step3_scorecard.py ~/reversi-runs/step3-eval \
        --out research/results/step3/scorecard.json [--markdown]

The scorecard rows follow the roster document: held out, retention, seen, seen
family, sanity. An opponent whose report is missing shows as a blank cell, so a
half-finished evaluation can still be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKPOINTS = ("control120", "A-s1", "A-s2", "B-s1", "B-s2", "D-s1")
"""Column order: the starting checkpoint, then the groups in the design's order."""

ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("held out", ("edax-l7", "edax-l3", "edax-l5", "rollout-r3000")),
    ("retention", ("gen60",)),
    ("seen", ("greedy", "minimax-d4")),
    ("seen family", ("greedy-corner", "greedy-d2", "noisy-greedy", "corner", "mobility")),
    ("sanity", ("random",)),
)


def read_pairing(eval_dir: Path, checkpoint: str, opponent: str) -> dict | None:
    """The checkpoint's side of one pairing, or None if the report is not there yet."""
    report_path = eval_dir / f"{checkpoint}-vs-{opponent}.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    (matchup,) = report["matchups"]
    if matchup["a"] == checkpoint:
        score, low, high = matchup["score_a"], matchup["ci_low"], matchup["ci_high"]
    else:
        score, low, high = 1 - matchup["score_a"], 1 - matchup["ci_high"], 1 - matchup["ci_low"]

    games_dir = eval_dir / f"{checkpoint}-vs-{opponent}-games"
    lost_by_wipeout = inflicted = None
    games_files = list(games_dir.glob("*.json")) if games_dir.is_dir() else []
    if games_files:
        (games_file,) = games_files
        payload = json.loads(games_file.read_text())
        net_is_a = payload["agents"]["a"] == checkpoint
        lost_by_wipeout = inflicted = 0
        for game in payload["games"]:
            net_is_black = game["a_is_black"] == net_is_a
            mine = game["final"]["black" if net_is_black else "white"]
            theirs = game["final"]["white" if net_is_black else "black"]
            lost_by_wipeout += mine == 0
            inflicted += theirs == 0
    return {
        "score": score,
        "ci_low": low,
        "ci_high": high,
        "games": matchup["games"],
        "record": matchup["record"] if matchup["a"] == checkpoint else None,
        "wipeouts": lost_by_wipeout,
        "wipeouts_inflicted": inflicted,
        "report": report_path.name,
        "commit": report.get("provenance", {}).get("git", {}).get("commit_short"),
    }


def cell(entry: dict | None) -> str:
    if entry is None:
        return "pending"
    text = f"{entry['score']:.1%} [{entry['ci_low']:.1%}, {entry['ci_high']:.1%}]"
    if entry["wipeouts"] is not None:
        text += f", {entry['wipeouts']} wipeouts"
    return text


def markdown(table: dict[str, dict[str, dict | None]]) -> str:
    lines = [
        "| row | opponent | " + " | ".join(CHECKPOINTS) + " |",
        "|---|---|" + "---|" * len(CHECKPOINTS),
    ]
    for row, opponents in ROWS:
        for opponent in opponents:
            cells = " | ".join(cell(table[checkpoint][opponent]) for checkpoint in CHECKPOINTS)
            lines.append(f"| {row} | {opponent} | {cells} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write the table as JSON here.")
    parser.add_argument("--markdown", action="store_true", help="Print the table as Markdown.")
    args = parser.parse_args()

    table = {
        checkpoint: {
            opponent: read_pairing(args.eval_dir, checkpoint, opponent)
            for _, opponents in ROWS
            for opponent in opponents
        }
        for checkpoint in CHECKPOINTS
    }
    done = sum(entry is not None for cells in table.values() for entry in cells.values())
    total = len(CHECKPOINTS) * sum(len(opponents) for _, opponents in ROWS)
    print(f"{done} of {total} pairings reported", file=sys.stderr)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "research/eval-protocol-frozen.json: 200 games per pairing, "
            "50 simulations, 4-ply openings played with both colours, seed 20261025",
            "rows": {row: list(opponents) for row, opponents in ROWS},
            "pairings": table,
        }
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    if args.markdown or args.out is None:
        print(markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
