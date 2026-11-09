"""Assemble the search-budget dose curve from one arena report per pairing.

The Step 3c results say that one extra rule in the search, the terminal check, removes
more wipeouts at 50 simulations a move than five times the budget does without it.
That reading rests on two points. The dose curve makes it a curve: the same checkpoint
plays the same 200 games against the same opponent at 50, 100, 256 and 800 simulations a
move, once with the plain search and once with the check, and the wipeouts are counted
at every point (``research/step8-results.md``, section 8, item 2).

Each pairing is its own ``reversi arena --suite custom`` job, named
``<checkpoint>-s<simulations>[-tc]-vs-<opponent>.json`` with a ``-games`` folder beside
it. The entrant name recorded *inside* the report is the frozen one (``control120``,
``Bw2-s1``, ...), so that every budget draws the same openings as the frozen scorecard,
and the budget and the check are read from the file name. The plain search at 50
simulations is the frozen scorecard's own report and is not re-run: ``--plain-50-from``
names the committed scorecard it is copied from.

Usage::

    python scripts/step3c_dose_curve.py ~/reversi-runs/dose-eval \
        --plain-50-from research/results/step8/scorecard.json \
        --out research/results/step3c/dose-curve.json [--markdown]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CHECKPOINTS = ("control120", "Bw2-s1", "B240w2-s1", "B240w2-s2")
OPPONENTS = ("greedy", "rollout-r3000")
BUDGETS = (50, 100, 256, 800)
STEM = re.compile(r"^(?P<name>.+?)-s(?P<sims>\d+)(?P<tc>-tc)?-vs-(?P<opponent>.+)$")


def read_report(path: Path, name: str) -> dict[str, Any]:
    """The named entrant's side of one report: score, interval, and wipeouts lost."""
    report = json.loads(path.read_text())
    (matchup,) = report["matchups"]
    if matchup["a"] == name:
        score, low, high = matchup["score_a"], matchup["ci_low"], matchup["ci_high"]
    elif matchup["b"] == name:
        score, low, high = 1 - matchup["score_a"], 1 - matchup["ci_high"], 1 - matchup["ci_low"]
    else:
        raise ValueError(f"{path.name}: neither entrant is {name!r}")

    games_dir = path.with_name(path.stem + "-games")
    wipeouts = None
    games_files = list(games_dir.glob("*.json")) if games_dir.is_dir() else []
    if games_files:
        (games_file,) = games_files
        payload = json.loads(games_file.read_text())
        net_is_a = payload["agents"]["a"] == name
        wipeouts = 0
        for game in payload["games"]:
            net_is_black = game["a_is_black"] == net_is_a
            wipeouts += game["final"]["black" if net_is_black else "white"] == 0
    return {
        "score": score,
        "ci_low": low,
        "ci_high": high,
        "games": matchup["games"],
        "wipeouts": wipeouts,
        "report": path.name,
        "commit": report.get("provenance", {}).get("git", {}).get("commit_short"),
    }


def empty_table() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    return {
        name: {opponent: {"plain": {}, "check": {}} for opponent in OPPONENTS}
        for name in CHECKPOINTS
    }


def collect(eval_dir: Path, plain_50: Path | None) -> dict[str, Any]:
    table = empty_table()
    for path in sorted(eval_dir.glob("*.json")):
        m = STEM.match(path.stem)
        if m is None:
            continue
        name, opponent = m["name"], m["opponent"]
        if name not in table or opponent not in OPPONENTS:
            continue
        variant = "check" if m["tc"] else "plain"
        table[name][opponent][variant][m["sims"]] = read_report(path, name)
    if plain_50 is not None:
        scorecard = json.loads(plain_50.read_text())["pairings"]
        for name in CHECKPOINTS:
            for opponent in OPPONENTS:
                entry = scorecard[name][opponent]
                table[name][opponent]["plain"]["50"] = {
                    "score": entry["score"],
                    "ci_low": entry["ci_low"],
                    "ci_high": entry["ci_high"],
                    "games": entry["games"],
                    "wipeouts": entry["wipeouts"],
                    "report": entry["report"],
                    "commit": entry["commit"],
                    "from": str(plain_50).replace("\\", "/"),
                }
    return table


def markdown(table: dict[str, Any], key: str) -> str:
    head = [f"plain {b}" for b in BUDGETS] + [f"check {b}" for b in BUDGETS]
    lines = ["| opponent | checkpoint | " + " | ".join(head) + " |", "|---|---|" + "---:|" * 8]
    for opponent in OPPONENTS:
        for name in CHECKPOINTS:
            cells = []
            for variant in ("plain", "check"):
                for b in BUDGETS:
                    entry = table[name][opponent][variant].get(str(b))
                    if entry is None:
                        cells.append("pending")
                    elif key == "score":
                        cells.append(f"{entry['score']:.1%}")
                    else:
                        cells.append(str(entry["wipeouts"]))
            lines.append(f"| {opponent} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write the table as JSON here.")
    parser.add_argument(
        "--plain-50-from",
        type=Path,
        default=None,
        metavar="SCORECARD",
        help="Committed scorecard whose plain 50-simulation pairings fill the first column.",
    )
    parser.add_argument("--markdown", action="store_true", help="Print the tables as Markdown.")
    args = parser.parse_args()

    table = collect(args.eval_dir, args.plain_50_from)
    done = sum(
        len(cells[variant])
        for by_opponent in table.values()
        for cells in by_opponent.values()
        for variant in ("plain", "check")
    )
    total = len(CHECKPOINTS) * len(OPPONENTS) * len(BUDGETS) * 2
    print(f"{done} of {total} pairings reported", file=sys.stderr)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "purpose": "search-budget dose curve with and without the terminal check: the "
            "control and three trained checkpoints at 50, 100, 256 and 800 simulations a move "
            "against Greedy and the random-playout search (research/step3c-terminal-check.md, "
            "section 7)",
            "protocol": "200 games per pairing, 4-ply openings played with both colours, seed "
            "20261025, one core per pairing; the entrant name is the frozen one so every "
            "budget draws the frozen scorecard's openings; 'plain' is the search as trained, "
            "'check' the same search with terminal_check on",
            "budgets": list(BUDGETS),
            "checkpoints": list(CHECKPOINTS),
            "opponents": list(OPPONENTS),
            "pairings": table,
        }
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    if args.markdown or args.out is None:
        print("wipeouts in 200 games\n")
        print(markdown(table, "wipeouts"))
        print("\nscore\n")
        print(markdown(table, "score"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
