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
from its calibration against Greedy. ``--self-play NAME=RUN_DIR`` adds a
"self-play" row for that network: its final checkpoint is run over the
positions of the run's last generation of training games, whose ``z`` is the
result those games actually reached. That needs torch and the run directory,
so it is the part that runs where the runs live.
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


def self_play_estimates(run_dir: Path, *, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """The final checkpoint's raw value on its own last generation of positions, with ``z``.

    The shards of the newest generation in the run's replay folder are read
    directly, without the manifest's checksum, since this is inspection rather
    than training. ``z`` is from the mover's view, as the value head's target is.
    """
    from reversi.data.shards import read_shard
    from reversi.errors import CheckpointError
    from reversi.game.state import State
    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.export import load_export
    from reversi.nn.loader import load_model
    from reversi.types import Player

    shards = sorted((run_dir / "replay").glob("gen_*_w*.npz"))
    if not shards:
        msg = f"no replay shards under {run_dir / 'replay'}"
        raise SystemExit(msg)
    generation = shards[-1].name.split("_")[1]
    shards = [path for path in shards if path.name.startswith(f"gen_{generation}_")]
    checkpoint = run_dir / "checkpoints" / f"gen_{generation}.pt"
    try:
        model = load_export(checkpoint).model
    except CheckpointError:
        model = load_model(checkpoint)
    evaluator = TorchEvaluator(model)

    states: list[State] = []
    z: list[float] = []
    for path in shards:
        arrays = read_shard(path, board_size=8)
        for row in range(len(arrays["black"])):
            states.append(
                State(
                    black=int(arrays["black"][row]),
                    white=int(arrays["white"][row]),
                    to_move=Player(int(arrays["to_move"][row])),
                    size=8,
                )
            )
            z.append(float(arrays["z"][row]))
    rng = np.random.default_rng(20261025)
    keep = rng.permutation(len(states))[:limit]
    states = [states[i] for i in keep]
    values: list[float] = []
    for start in range(0, len(states), 512):
        _, batch_values = evaluator.evaluate(states[start : start + 512])
        values.extend(float(v) for v in batch_values)
    return np.array(values), np.array(z, dtype=float)[keep]


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
    parser.add_argument(
        "--self-play",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Also calibrate NAME's final checkpoint on RUN_DIR's last generation of games.",
    )
    parser.add_argument(
        "--self-play-limit",
        type=int,
        default=20000,
        help="Positions sampled from the last generation for the self-play row.",
    )
    args = parser.parse_args()
    self_play = dict(item.split("=", 1) for item in args.self_play)

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
        if network in self_play:
            estimates, outcomes = self_play_estimates(
                Path(self_play[network]).expanduser(), limit=args.self_play_limit
            )
            results[network]["self-play"] = calibration(estimates, outcomes)

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
