"""Measure where self-play actually spends its time, before optimising anything.

Run this before changing anything for speed, and again afterwards. The plan's
rule is that no optimisation stays in the repo without a before/after number in
``bench/results/``, and this is what produces them.

    uv run python bench/selfplay_bench.py --device cuda --out bench/results/gpu.json
    uv run python bench/selfplay_bench.py --quick        # a fast sanity pass

Four phases, in the order the numbers depend on each other:

1. **engine** -- legal moves and applies per second. Pure Python, no network.
2. **inference** -- positions scored per second at each batch size. This is the
   phase that decides whether batching is worth building: if batch 1 and batch 48
   are similar, it is not.
3. **tree** -- simulations per second against a stub evaluator, which isolates the
   search's own overhead from the network's.
4. **selfplay** -- games per second, one at a time versus batched. The end-to-end
   number the other three explain.

Multi-process scaling is deliberately not a phase here. It needs a weights file
on disk and spawns real processes, so it belongs to a run rather than to a
microbenchmark -- and the training loop already reports ``games_per_s`` and
``mean_batch`` every generation, which is the same measurement under real
conditions. ``bench/results/`` records what a worker sweep found.

Note what phase 2 measures: a *ceiling*, not a rate you will see. Phase 4 is the
honest number, and it is always lower, because the tree search between calls is
not free.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reversi.game import rules
from reversi.nn.evaluator import TorchEvaluator
from reversi.nn.model import PolicyValueNet
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.search.mcts import MCTS
from reversi.selfplay.game_batch import BatchedSelfPlay
from reversi.selfplay.worker import play_games


@dataclass(slots=True)
class Settings:
    device: str
    board_size: int
    n_blocks: int
    channels: int
    simulations: int
    batches: tuple[int, ...]
    games_solo: int
    games_batched: int
    quick: bool


def build_model(settings: Settings) -> PolicyValueNet:
    model = PolicyValueNet(
        settings.board_size,
        n_blocks=settings.n_blocks,
        channels=settings.channels,
        value_hidden=64,
    )
    model.eval()
    return model


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu_memory_gib"] = round(total / 2**30, 2)
        info["gpu_memory_free_gib"] = round(free / 2**30, 2)
    return info


# ---------------------------------------------------------------------------


def phase_engine(settings: Settings) -> dict[str, float]:
    """The rules engine on its own. If this is not fast, nothing else can be."""
    state = rules.initial_state(settings.board_size)
    iterations = 2_000 if settings.quick else 20_000

    start = time.perf_counter()
    for _ in range(iterations):
        rules.legal_actions(state)
    legal_rate = iterations / (time.perf_counter() - start)

    action = rules.legal_actions(state)[0]
    start = time.perf_counter()
    for _ in range(iterations):
        rules.apply(state, action)
    apply_rate = iterations / (time.perf_counter() - start)

    return {"legal_actions_per_s": legal_rate, "apply_per_s": apply_rate}


def phase_inference(settings: Settings) -> dict[str, dict[str, float]]:
    """Positions per second at each batch size -- the ceiling, not the rate."""
    out: dict[str, dict[str, float]] = {}
    iterations = 10 if settings.quick else 40

    for batch in settings.batches:
        model = build_model(settings).to(settings.device)
        planes = torch.zeros(
            batch, 3, settings.board_size, settings.board_size, device=settings.device
        )
        with torch.inference_mode():
            for _ in range(5):
                model(planes)
            if settings.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                model(planes)
            if settings.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        out[str(batch)] = {
            "positions_per_s": batch * iterations / elapsed,
            "latency_ms": 1000 * elapsed / iterations,
        }
    return out


def phase_tree(settings: Settings) -> dict[str, float]:
    """Search overhead with no network at all: the Python ceiling."""
    state = rules.initial_state(settings.board_size)
    for _ in range(min(20, settings.board_size * 2)):
        if rules.is_terminal(state):
            break
        state = rules.apply(state, rules.legal_actions(state)[0])

    reps = 5 if settings.quick else 20
    sims = 200 if settings.quick else 400
    stub = StubEvaluator()
    start = time.perf_counter()
    for _ in range(reps):
        MCTS(stub, SearchConfig(n_simulations=sims)).run(state)
    elapsed = time.perf_counter() - start
    return {"simulations_per_s": sims * reps / elapsed}


def phase_selfplay(settings: Settings) -> dict[str, Any]:
    """Games per second, one at a time versus batched. The honest number."""
    config = SearchConfig(
        n_simulations=settings.simulations,
        temp_moves=15,
        dirichlet_eps=0.25,
        dirichlet_alpha=1.0,
    )
    out: dict[str, Any] = {}

    evaluator = TorchEvaluator(build_model(settings), device=settings.device)
    start = time.perf_counter()
    for _ in play_games(
        evaluator,
        config,
        board_size=settings.board_size,
        n_games=settings.games_solo,
        root_seed=1,
        generation=1,
    ):
        pass
    elapsed = time.perf_counter() - start
    solo_rate = settings.games_solo / elapsed
    out["stage1"] = {
        "games": settings.games_solo,
        "seconds": elapsed,
        "games_per_s": solo_rate,
        "positions_per_s": evaluator.positions / elapsed,
        "mean_batch": 1.0,
    }

    for in_flight in (8, 16) if settings.quick else (16, 48):
        games = min(settings.games_batched, in_flight)
        evaluator = TorchEvaluator(build_model(settings), device=settings.device)
        play = BatchedSelfPlay(
            evaluator,
            config,
            board_size=settings.board_size,
            games_in_flight=in_flight,
            root_seed=1,
            generation=1,
        )
        start = time.perf_counter()
        for _ in play.play(games):
            pass
        elapsed = time.perf_counter() - start
        out[f"stage2_b{in_flight}"] = {
            "games": games,
            "seconds": elapsed,
            "games_per_s": games / elapsed,
            "positions_per_s": evaluator.positions / elapsed,
            "mean_batch": play.mean_batch,
            "speedup_over_stage1": (games / elapsed) / solo_rate,
        }
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--games-solo", type=int, default=3)
    parser.add_argument("--games-batched", type=int, default=48)
    parser.add_argument("--quick", action="store_true", help="a fast, coarse pass")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings(
        device=args.device,
        board_size=args.board_size,
        n_blocks=args.n_blocks,
        channels=args.channels,
        simulations=args.simulations,
        batches=(1, 8, 32, 48, 128, 256),
        games_solo=args.games_solo,
        games_batched=args.games_batched,
        quick=args.quick,
    )

    if settings.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no GPU is visible")

    report: dict[str, Any] = {
        "settings": {
            "device": settings.device,
            "board_size": settings.board_size,
            "n_blocks": settings.n_blocks,
            "channels": settings.channels,
            "simulations": settings.simulations,
            "quick": settings.quick,
        },
        "machine": machine(),
    }

    for name, phase in (
        ("engine", phase_engine),
        ("inference", phase_inference),
        ("tree", phase_tree),
        ("selfplay", phase_selfplay),
    ):
        print(f"-- {name} ...", flush=True)
        report[name] = phase(settings)

    _print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")


def _print_report(report: dict[str, Any]) -> None:
    print()
    print("ENGINE")
    for key, value in report["engine"].items():
        print(f"  {key:<24}{value:>14,.0f}")

    print("\nINFERENCE -- positions/s (a ceiling, not a rate you will see)")
    print(f"  {'batch':>7}{'positions/s':>16}{'latency ms':>14}")
    for batch, row in report["inference"].items():
        print(f"  {batch:>7}{row['positions_per_s']:>16,.0f}{row['latency_ms']:>14.2f}")

    print("\nTREE -- no network")
    print(f"  simulations/s          {report['tree']['simulations_per_s']:>14,.0f}")

    print("\nSELF-PLAY -- end to end")
    print(f"  {'':<14}{'games/s':>10}{'positions/s':>14}{'mean batch':>12}{'vs stage 1':>12}")
    for name, row in report["selfplay"].items():
        speedup = row.get("speedup_over_stage1")
        print(
            f"  {name:<14}{row['games_per_s']:>10.3f}{row['positions_per_s']:>14,.0f}"
            f"{row['mean_batch']:>12.1f}"
            f"{(f'{speedup:.1f}x' if speedup else '-'):>12}"
        )

    best = max(report["selfplay"].values(), key=lambda r: r["games_per_s"])
    print(f"\n  a 2500-game generation at the best rate: {2500 / best['games_per_s'] / 3600:.2f} h")


if __name__ == "__main__":  # spawn re-imports this module in child processes
    main()
