"""Is the value head's confidence borne out by results?

The Step 3 design (``research/step3-design.md``, section 3) asks for a value
calibration check after training: the network's own estimate of a position,
set against how the game actually ended, binned by estimate. A network whose
+0.9 positions are won ninety-five times in a hundred is calibrated; one whose
+0.9 positions are won half the time is overconfident, and a group that paid
for its training change with overconfidence shows it here.

Every recorded network move carries ``net_value``, the raw network estimate
before search, from the network's point of view in [-1, +1]. The game's result
from the same point of view is +1, 0 or -1. Both are mapped to a win
probability, (x + 1) / 2, so a bin's "estimate" and "outcome" columns can be
read side by side; "gap" is outcome minus estimate, positive when the network
undersold itself.

Usage::

    python scripts/value_calibration.py ~/reversi-runs/step3-eval \
        --out research/results/step3/value-calibration.json

The games are grouped by the scorecard rows of ``step3_scorecard.py``, so a
network's calibration against the opponents it never saw can be read apart
from its calibration against Greedy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_scorecard import CHECKPOINTS, ROWS

BINS = np.linspace(-1.0, 1.0, 11)
"""Ten equal bins over the estimate, so each column is a fifth of a win probability."""


def moves_and_outcomes(games_file: Path, network: str) -> tuple[np.ndarray, np.ndarray]:
    """Every recorded estimate the network made in this file, with the game's result."""
    payload = json.loads(games_file.read_text())
    net_is_a = payload["agents"]["a"] == network
    estimates: list[float] = []
    outcomes: list[float] = []
    for game in payload["games"]:
        result = game["outcome_for_a"] if net_is_a else -game["outcome_for_a"]
        for ply in game["plies"]:
            if ply["by_a"] == net_is_a and "net_value" in ply:
                estimates.append(ply["net_value"])
                outcomes.append(result)
    return np.array(estimates), np.array(outcomes, dtype=float)


def calibration(estimates: np.ndarray, outcomes: np.ndarray) -> dict:
    """Per-bin estimate, outcome and count, plus the count-weighted absolute gap."""
    p_est = (estimates + 1) / 2
    p_out = (outcomes + 1) / 2
    which = np.clip(np.digitize(estimates, BINS) - 1, 0, len(BINS) - 2)
    bins = []
    weighted_gap = 0.0
    for b in range(len(BINS) - 1):
        mask = which == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"low": float(BINS[b]), "high": float(BINS[b + 1]), "n": 0})
            continue
        est, out = float(p_est[mask].mean()), float(p_out[mask].mean())
        weighted_gap += n * abs(out - est)
        bins.append(
            {
                "low": float(BINS[b]),
                "high": float(BINS[b + 1]),
                "n": n,
                "estimate": est,
                "outcome": out,
                "gap": out - est,
            }
        )
    return {
        "moves": len(estimates),
        "expected_gap": weighted_gap / max(len(estimates), 1),
        "mean_estimate": float(p_est.mean()) if len(estimates) else None,
        "mean_outcome": float(p_out.mean()) if len(outcomes) else None,
        "bins": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results: dict[str, dict[str, dict]] = {}
    for network in CHECKPOINTS:
        results[network] = {}
        for row, opponents in ROWS:
            est_parts, out_parts = [], []
            for opponent in opponents:
                games_dir = args.eval_dir / f"{network}-vs-{opponent}-games"
                for games_file in games_dir.glob("*.json") if games_dir.is_dir() else []:
                    e, o = moves_and_outcomes(games_file, network)
                    est_parts.append(e)
                    out_parts.append(o)
            if est_parts:
                results[network][row] = calibration(
                    np.concatenate(est_parts), np.concatenate(out_parts)
                )

    print("| network | games against | moves | mean estimate | mean outcome | expected gap |")
    print("|---|---|---:|---:|---:|---:|")
    for network, rows in results.items():
        for row, c in rows.items():
            print(
                f"| {network} | {row} | {c['moves']} | {c['mean_estimate']:.3f} "
                f"| {c['mean_outcome']:.3f} | {c['expected_gap']:.3f} |"
            )
    print()
    print("Top bin (estimate in [0.8, 1.0], as a win probability [0.9, 1.0]):")
    print("| network | games against | moves in top bin | outcome there |")
    print("|---|---|---:|---:|")
    for network, rows in results.items():
        for row, c in rows.items():
            top = c["bins"][-1]
            if top["n"]:
                print(f"| {network} | {row} | {top['n']} | {top['outcome']:.3f} |")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "purpose": "research/step3-design.md, section 3: value calibration "
                    "after training",
                    "bins": "ten equal bins of the raw network value in [-1, 1]; estimate and "
                    "outcome are win probabilities (x + 1) / 2; expected gap is the "
                    "count-weighted mean |outcome - estimate| over bins",
                    "networks": results,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
