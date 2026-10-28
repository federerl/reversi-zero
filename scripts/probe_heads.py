"""Which head lets the wipeout through: the value estimate, or the move prior?

Step 2 (``research/step2-failure-analysis.md``) found that the network walks
into a wipeout with its value estimate near +1 while a safe move sat in the
search's view. Two different parts of the network could be responsible, and
the recorded games cannot tell them apart:

* the **value head**, which rates the position (and the position one move
  later, with the network holding one disc) as a near-certain win, so the
  search has no reason to avoid it;
* the **policy head**, which gives Greedy's wiping reply such a small prior
  that fifty simulations never look at it, so the search never finds out.

This script asks the network directly, in every position from which it was
wiped out at its last move and the move before. Two readings and a
counterfactual:

1. At the network's position: the raw value, and the prior on the move it
   played versus the prior on the moves that would have survived.
2. One ply later, from Greedy's side: the raw value (the exact answer is +1,
   Greedy wins on the spot), and the prior the network puts on the reply that
   wipes it out.
3. The same search run three ways on the same position: as recorded (real
   prior and real value), with the prior flattened to uniform (so only the
   value head is speaking), and with the value zeroed (so only the prior is
   speaking; finished positions still score exactly). Whichever way stops
   choosing the fatal move points at the head that let it through.

Usage::

    python scripts/probe_heads.py runs/research/pilot-2026-10-games \
        --out research/results/step2/head-probe.json

The checkpoints are read from ``research/eval-protocol-frozen.json`` so the
network probed is the one that played the recorded games.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reversi.atomicio import atomic_write_json
from reversi.game.rules import apply, legal_actions
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import Evaluator
from reversi.search.mcts import MCTS
from reversi.search.node import priors_from_logits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wipeouts import (
    NetMove,
    Trace,
    at,
    load,
    net_discs,
    opponent_replies,
    survives,
)

PROTOCOL = Path("research/eval-protocol-frozen.json")
BUDGETS = (50, 256)
PROBED_MOVES = (1,)
"""Which of the network's moves to probe, counted back from the wipeout: 1 is
its last move. The move before is not probed: a move there counts as safe
whenever *some* surviving line still exists, which Step 2 found is always, so
the question has no content until the last move."""


class OnlyValue:
    """The real network's value with every prior flattened: the value head alone."""

    def __init__(self, inner: Evaluator) -> None:
        self.inner = inner

    def evaluate(self, states: Sequence[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        logits, values = self.inner.evaluate(states)
        return np.zeros_like(logits), values


class OnlyPolicy:
    """The real network's priors with every value zeroed: the policy head alone.

    Finished positions are never sent to an evaluator, so a wipeout the search
    reaches still scores exactly -1; this only removes the network's *opinion*.
    """

    def __init__(self, inner: Evaluator) -> None:
        self.inner = inner

    def evaluate(self, states: Sequence[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        logits, values = self.inner.evaluate(states)
        return logits, np.zeros_like(values)


def load_evaluator(path: Path) -> Evaluator:
    import torch

    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.export import load_export

    torch.set_num_threads(1)
    return TorchEvaluator(load_export(path).model)


def prior_over(evaluator: Evaluator, state: State) -> tuple[dict[int, float], float]:
    """The network's prior over the legal moves and its raw value, for the mover."""
    logits, values = evaluator.evaluate([state])
    actions = legal_actions(state)
    priors = priors_from_logits(logits[0], actions)
    return dict(zip(actions, priors, strict=True)), float(values[0])


def probe_position(evaluator: Evaluator, trace: Trace, move: NetMove, k: int) -> dict[str, Any]:
    """Everything the two heads say about one position the network was wiped out from."""
    state = move.state
    played = max(move.visits, key=lambda av: av[1])[0]
    safe = {a for a in legal_actions(state) if survives(apply(state, a), trace.net_black, k - 1)}
    prior, value = prior_over(evaluator, state)

    child = apply(state, played)
    child_prior, child_value = prior_over(evaluator, child)
    wiping = {r for r in legal_actions(child) if net_discs(apply(child, r), trace.net_black) == 0}
    greedy = set(opponent_replies(child))

    row: dict[str, Any] = {
        "k": k,
        "played": played,
        "played_was_safe": played in safe,
        "safe_moves": sorted(safe),
        "value": value,
        "prior_played": prior[played],
        "prior_safe_total": sum(prior[a] for a in safe),
        "prior_safe_max": max((prior[a] for a in safe), default=0.0),
        "played_has_top_prior": prior[played] == max(prior.values()),
        "child_value_for_greedy": child_value,
        "child_wiping_replies": sorted(wiping),
        "child_prior_wiping_total": sum(child_prior[r] for r in wiping),
        "child_prior_greedy_total": sum(child_prior[r] for r in greedy),
        "child_wiping_has_top_prior": bool(wiping)
        and max(child_prior, key=lambda r: child_prior[r]) in wiping,
        "searches": {},
    }
    for sims in BUDGETS:
        config = SearchConfig(n_simulations=sims)
        for label, eval_ in (
            ("real", evaluator),
            ("value_only", OnlyValue(evaluator)),
            ("policy_only", OnlyPolicy(evaluator)),
        ):
            chosen = MCTS(eval_, config).run(state).best_action()
            row["searches"][f"{label}@{sims}"] = {"chosen": chosen, "safe": chosen in safe}
    return row


def median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def share(xs: list[bool]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"positions": len(rows)}
    if not rows:
        return out
    out.update(
        {
            "value_median": median([r["value"] for r in rows]),
            "prior_played_median": median([r["prior_played"] for r in rows]),
            "prior_safe_total_median": median([r["prior_safe_total"] for r in rows]),
            "played_has_top_prior": share([r["played_has_top_prior"] for r in rows]),
            "child_value_for_greedy_median": median([r["child_value_for_greedy"] for r in rows]),
            "child_prior_wiping_median": median([r["child_prior_wiping_total"] for r in rows]),
            "child_prior_wiping_below_5pct": share(
                [r["child_prior_wiping_total"] < 0.05 for r in rows]
            ),
            "child_wiping_has_top_prior": share([r["child_wiping_has_top_prior"] for r in rows]),
            "searches": {
                key: {
                    "safe_share": share([r["searches"][key]["safe"] for r in rows]),
                    "safe_count": int(sum(r["searches"][key]["safe"] for r in rows)),
                }
                for key in rows[0]["searches"]
            },
            "real_search_replayed_the_record": share(
                [r["searches"]["real@50"]["chosen"] == r["played"] for r in rows]
            ),
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("games_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--networks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="wipeouts per network (smoke)")
    args = parser.parse_args()

    checkpoints = json.loads(args.protocol.read_text(encoding="utf-8"))["checkpoints"]
    results: dict[str, Any] = {}
    for path in sorted(args.games_dir.glob("greedy-vs-*.json")):
        net, traces = load(path)
        if net not in checkpoints or (args.networks and net not in args.networks):
            continue  # a fixed opponent's games, or a network not asked for
        evaluator = load_evaluator(Path(checkpoints[net]["file"]))
        wiped = [t for t in traces if t.wipeout and not t.net_won][: args.limit]
        rows: list[dict[str, Any]] = []
        for trace in wiped:
            for k in PROBED_MOVES:
                move = at(trace, k)
                if move is None:
                    continue
                safe_existed = any(
                    survives(apply(move.state, a), trace.net_black, k - 1)
                    for a in legal_actions(move.state)
                )
                if not safe_existed:
                    continue
                rows.append(probe_position(evaluator, trace, move, k))
        results[net] = {
            "checkpoint": checkpoints[net]["file"],
            "wipeouts": len(wiped),
            "summary": {k: summarise([r for r in rows if r["k"] == k]) for k in PROBED_MOVES},
            "positions": rows,
        }
        print(f"{net}: probed {len(rows)} positions from {len(wiped)} wipeouts", flush=True)

    atomic_write_json(
        args.out,
        {
            "purpose": "research/step2-failure-analysis.md: which head lets the wipeout through",
            "games": str(args.games_dir),
            "budgets": list(BUDGETS),
            "note": "k=1 is the network's last move before the wipeout; only positions "
            "where a surviving move existed are probed",
            "networks": results,
        },
    )

    print()
    print("## What the two heads say")
    print()
    print(
        "| network | move | positions | raw value | prior on the move played | prior on all "
        "safe moves | played move had the top prior | one ply later, raw value for Greedy "
        "(exact: +1) | prior on Greedy's wiping replies | that prior below 5% "
        "| a wiping reply had Greedy's top prior |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for net, result in results.items():
        for k in PROBED_MOVES:
            s = result["summary"][k]
            if not s["positions"]:
                continue
            label = "last" if k == 1 else "2nd last"
            print(
                f"| {net} | {label} | {s['positions']} | {s['value_median']:+.2f} "
                f"| {s['prior_played_median']:.0%} | {s['prior_safe_total_median']:.0%} "
                f"| {s['played_has_top_prior']:.0%} | {s['child_value_for_greedy_median']:+.2f} "
                f"| {s['child_prior_wiping_median']:.1%} "
                f"| {s['child_prior_wiping_below_5pct']:.0%} "
                f"| {s['child_wiping_has_top_prior']:.0%} |"
            )
    print()
    print("## The same search with one head switched off")
    print()
    print("Share of positions where the search chooses a move that survives.")
    print()
    labels = ("real", "value_only", "policy_only")
    keys = [f"{label}@{sims}" for sims in BUDGETS for label in labels]
    print("| network | move | positions | " + " | ".join(keys) + " | real@50 replayed the record |")
    print("|---|---|---:|" + "---:|" * (len(keys) + 1))
    for net, result in results.items():
        for k in PROBED_MOVES:
            s = result["summary"][k]
            if not s["positions"]:
                continue
            label = "last" if k == 1 else "2nd last"
            cells = " | ".join(f"{s['searches'][key]['safe_share']:.0%}" for key in keys)
            print(
                f"| {net} | {label} | {s['positions']} | {cells} "
                f"| {s['real_search_replayed_the_record']:.0%} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
