"""Regenerate every figure from the JSONL and the tournament report.

    uv run python scripts/make_plots.py runs/<run_id>

**Nothing here reads a checkpoint or replays a game.** Every figure comes from
files the run already wrote: the metric streams and the tournament JSON. That is
deliberate -- it means a figure can always be rebuilt from an archived run, by
someone who no longer has the GPU, months later. If a plot needed a model to
regenerate, it would eventually become unreproducible.

The figures, in the order they answer questions:

1. **ratings** -- the headline. Every agent on one scale with its interval, and
   the baselines drawn in so the numbers mean something.
2. **strength over generations** -- win rate against each baseline as training
   progressed, with Wilson bands.
3. **losses** -- policy and value against training step. Labelled a *diagnostic*,
   because it is not evidence of strength and the caption says so.
4. **throughput** -- self-play rate and generation time, for the record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a server
import matplotlib.pyplot as plt

from reversi.arena.stats import wilson_interval
from reversi.obs.metrics import read_jsonl

INK = "#1b1b1b"
GRID = "#d8d8d8"
AGENT = "#1f6f4a"
BASELINE = "#a2521c"
MUTED = "#7a7a7a"


def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(visible=True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_ratings(report: dict[str, Any], out: Path) -> None:
    """The headline figure: everyone on one scale, with intervals."""
    ratings = sorted(report["ratings"], key=lambda r: r["elo"])
    names = [r["name"] for r in ratings]
    elos = [r["elo"] for r in ratings]
    lows = [r["elo"] - (r["ci_low"] if r["ci_low"] is not None else r["elo"]) for r in ratings]
    highs = [(r["ci_high"] if r["ci_high"] is not None else r["elo"]) - r["elo"] for r in ratings]
    colours = [AGENT if n.startswith("gen") else BASELINE for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.barh(
        names,
        elos,
        xerr=[lows, highs],
        color=colours,
        height=0.6,
        error_kw={"ecolor": MUTED, "capsize": 3, "elinewidth": 1},
    )
    style(
        ax,
        "Strength on one scale (Bradley-Terry, Random = 0)",
        "Elo, with 95% bootstrap interval",
        "",
    )
    ax.text(
        0.99,
        -0.16,
        f"{report['protocol']['games_per_pair']} games per pairing, "
        f"{report['protocol']['opening_plies']}-ply opening book, colours balanced",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_strength_over_generations(report: dict[str, Any], out: Path) -> None:
    """Win rate against each baseline as training progressed."""
    baselines = ["random", "greedy", "minimax-d2", "minimax-d4"]
    scores: dict[str, dict[int, tuple[float, float, float]]] = {b: {} for b in baselines}

    for row in report["matchups"]:
        for baseline in baselines:
            # The agent may be on either side of the pairing; flip if needed.
            if row["a"] == baseline and row["b"].startswith("gen"):
                gen, score = int(row["b"][3:]), 1.0 - row["score_a"]
            elif row["b"] == baseline and row["a"].startswith("gen"):
                gen, score = int(row["a"][3:]), row["score_a"]
            else:
                continue
            interval = wilson_interval(score * row["games"], row["games"])
            scores[baseline][gen] = (score, interval.low, interval.high)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    for baseline, marker in zip(baselines, ("o", "s", "^", "D"), strict=True):
        points = sorted(scores[baseline].items())
        if not points:
            continue
        gens = [g for g, _ in points]
        mid = [v[0] for _, v in points]
        low = [v[1] for _, v in points]
        high = [v[2] for _, v in points]
        (line,) = ax.plot(gens, mid, marker=marker, markersize=4, linewidth=1.4, label=baseline)
        ax.fill_between(gens, low, high, alpha=0.13, color=line.get_color())

    ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle="--")
    ax.text(ax.get_xlim()[1], 0.5, " even", va="center", fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.02)
    style(
        ax,
        "Strength against each baseline over training",
        "generation",
        "score (win = 1, draw = 0.5)",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_losses(train: list[dict[str, Any]], out: Path) -> None:
    """A diagnostic. The caption says so, because it is not evidence of strength."""
    gens = [r["generation"] for r in train]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.2), dpi=160, sharex=True)
    top.plot(gens, [r["policy_loss"] for r in train], color=AGENT, linewidth=1.6)
    style(
        top, "Training loss -- a diagnostic, NOT evidence of strength", "", "policy (cross-entropy)"
    )

    bottom.plot(gens, [r["value_loss"] for r in train], color=BASELINE, linewidth=1.6)
    style(bottom, "", "generation", "value (mean squared error)")

    top.text(
        0.0,
        -0.22,
        "A falling loss shows the network is learning to predict its own search. "
        "Whether it plays better is a separate question, answered by the arena.",
        transform=top.transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_throughput(selfplay: list[dict[str, Any]], out: Path) -> None:
    gens = [r["generation"] for r in selfplay]

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
    ax.plot(gens, [r["seconds"] / 60 for r in selfplay], color=AGENT, linewidth=1.6)
    style(ax, "Self-play time per generation", "generation", "minutes")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a run directory")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/plots")
    args = parser.parse_args()

    run: Path = args.run
    out: Path = args.out or run / "plots"
    out.mkdir(parents=True, exist_ok=True)

    made = []
    train = read_jsonl(run / "metrics" / "train.jsonl")
    selfplay = read_jsonl(run / "metrics" / "selfplay.jsonl")

    if train:
        plot_losses(train, out / "losses.png")
        made.append("losses.png")
    if selfplay:
        plot_throughput(selfplay, out / "throughput.png")
        made.append("throughput.png")

    tournament = run / "arena" / "crossgen.json"
    if tournament.exists():
        report = json.loads(tournament.read_text(encoding="utf-8"))
        plot_ratings(report, out / "ratings.png")
        plot_strength_over_generations(report, out / "strength.png")
        made.extend(["ratings.png", "strength.png"])
    else:
        print(f"no tournament at {tournament}; skipping the rating figures")

    for name in made:
        print(f"  {out / name}")


if __name__ == "__main__":
    main()
