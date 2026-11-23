"""Reliability curves: what the value head says against what happened, by row group.

The value head's output is a number in [-1, 1] that the search treats as the
probability of winning from the position, scaled. The calibration check bins every
position the network reached in the evaluation games by that estimate and asks how
often the network went on to win from positions in the bin. A calibrated head sits
on the diagonal; a head that overrates its positions sits below it.

This figure draws the curve for the checkpoints of Steps 3, 3b and 4, the three
routes into the value head, one panel per row group, so that two things can be
read: the value head is calibrated on self-play and overrates its chances against
the held-out opponents by the same margin on every checkpoint, so none of the
three routes changed what it believes; and the one checkpoint trained on a
material term in the value target (group D) is no longer a probability at all, its
curve an S that reads certain loss below 0.3 and certain win above 0.7.

Everything is read from the committed JSON under research/results/:

    python scripts/figures/reliability.py --out research/figures/reliability.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chain import GRID, INK, INK_2, INK_3, ROUTE_COLOUR, SURFACE, load, style
from matplotlib.lines import Line2D

# Every checkpoint: the calibration file it is read from, its name, its recipe.
NETWORKS: list[tuple[str, str, str]] = [
    ("step3", "control120", "control"),
    ("step3", "A-s1", "control"),  # the control recipe continued for twenty generations
    ("step3", "A-s2", "control"),
    ("step3", "B-s1", "mix"),  # the opponent mix, the network's own positions recorded
    ("step3", "B-s2", "mix"),
    ("step3b", "Bboth-s1", "both-sides"),  # the mix, the opponent's positions recorded too
    ("step3b", "Bboth-s2", "both-sides"),
    ("step4", "Btc-s1", "check"),  # the mix, the check inside the training search
    ("step4", "Btc-s2", "check"),
    ("step3", "D-s1", "material"),  # a material term in the value target
]
RECIPE_COLOUR = {
    "control": INK_3,
    "mix": ROUTE_COLOUR["games"],
    "both-sides": ROUTE_COLOUR["both"],
    "check": ROUTE_COLOUR["weight"],
    "material": INK,
}
RECIPE_LABEL = {
    "control": "the control recipe (generation 120, and continued to 140, two seeds)",
    "mix": "the opponent mix (Step 3 B)",
    "both-sides": "the mix with the opponent's positions recorded (Step 3b)",
    "check": "the mix with the check inside the training search (Step 4)",
    "material": "a material term in the value target (Step 3 D)",
}
PANELS: list[tuple[str, str]] = [
    ("self-play", "self-play"),
    ("retention", "generation 60"),
    ("seen", "Greedy and Minimax"),
    ("seen family", "the Greedy family"),
    ("held out", "the held-out opponents"),
]


def collect() -> list[dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    curves: list[dict[str, Any]] = []
    for step, name, recipe in NETWORKS:
        if step not in files:
            files[step] = load(f"{step}/value-calibration.json")
        groups = files[step]["networks"][name]
        curves.append(
            {
                "name": name,
                "recipe": recipe,
                "groups": {
                    g: {
                        "x": [float(b["estimate"]) for b in groups[g]["bins"] if b["n"]],
                        "y": [float(b["outcome"]) for b in groups[g]["bins"] if b["n"]],
                        "gap": float(groups[g]["expected_gap"]),
                    }
                    for g, _ in PANELS
                },
            }
        )
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("research/figures/reliability.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    curves = collect()

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.70, bottom=0.24, wspace=0.12)

    for ax, (group, title) in zip(axes, PANELS, strict=True):
        style(ax, title, "")
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
        # The material-target checkpoint on top, the control recipe underneath.
        for c in sorted(curves, key=lambda c: c["recipe"] == "material"):
            g = c["groups"][group]
            ax.plot(
                g["x"],
                g["y"],
                color=RECIPE_COLOUR[c["recipe"]],
                linewidth=2.0 if c["recipe"] == "material" else 1.6,
                linestyle=(0, (4, 2)) if c["recipe"] == "material" else "solid",
                marker="o",
                markersize=3.2,
                markeredgecolor=SURFACE,
                markeredgewidth=0.5,
                alpha=0.95 if c["recipe"] in ("material", "control") else 0.8,
                zorder=3 if c["recipe"] == "material" else 2,
            )
        control = next(c for c in curves if c["name"] == "control120")["groups"][group]["gap"]
        material = next(c for c in curves if c["name"] == "D-s1")["groups"][group]["gap"]
        others = [
            c["groups"][group]["gap"] for c in curves if c["name"] not in ("control120", "D-s1")
        ]
        ax.text(
            0.97,
            0.05,
            f"gap: control {control:.2f}, D {material:.2f}\n"
            f"the other seven {min(others):.2f}\u2013{max(others):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color=INK_2,
            linespacing=1.4,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    axes[0].set_ylabel(
        "share of those positions the network went on to win", color=INK_2, fontsize=8
    )
    fig.text(
        0.52,
        0.165,
        "the value head's estimate, as a win probability",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )

    handles = [
        Line2D(
            [],
            [],
            color=RECIPE_COLOUR[r],
            linewidth=2.0,
            linestyle=(0, (4, 2)) if r == "material" else "solid",
            label=RECIPE_LABEL[r],
        )
        for r in ("control", "mix", "both-sides", "check", "material")
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.0),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        labelcolor=INK_2,
        columnspacing=1.6,
    )

    fig.suptitle(
        "What the value head believes, by opponent group: three routes into it left the curve "
        "where the control had it, and the material target broke it",
        x=0.05,
        ha="left",
        fontsize=12,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.05,
        0.905,
        "Each line is one checkpoint; positions from its evaluation games are binned by the "
        "value head's estimate (ten bins) and the share it went on to win is plotted against "
        "the bin's mean estimate.\n"
        'The diagonal is a calibrated head. "Gap" is the count-weighted mean distance from '
        "it."
        + chr(10)
        + "The held-out opponents are Edax at three levels and the random-playout search: "
        "every checkpoint but D's overrates its chances there by 8 to 12 points.",
        fontsize=8,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for c in curves:
        gaps = "  ".join(f"{g}={c['groups'][g]['gap']:.3f}" for g, _ in PANELS)
        print(f"  {c['name']:>10}  {gaps}")


if __name__ == "__main__":
    main()
