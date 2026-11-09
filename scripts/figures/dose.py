"""The dose figure: wipeouts against the search budget, with and without the terminal check.

Step 3c found that one extra rule in the search, looking one move ahead for a reply
that ends the game before trusting the network's estimate, removes more wipeouts at
50 simulations a move than five times the budget does without it. That was two
points. This figure is the curve: the untrained control and three checkpoints
trained on the opponent's positions (Steps 7 and 8), each playing the same 200 games
against Greedy and against the random-playout search at 50, 100, 256 and 800
simulations a move, once with the plain search (solid) and once with the check
(dashed). Two things are read off it: how far the search budget alone has to go to
reach what the check reaches at 50, and how the trained prior changes that, since
a higher prior on the wiping reply lets a modest budget find it.

Everything is read from the committed JSON under research/results/:

    python scripts/figures/dose.py --out research/figures/dose-curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = Path("research/results/step3c/dose-curve.json")

# Surface and ink, as in chain.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e6e5e1"

# The control is the reference and wears ink; the two trained recipes wear the route
# colours of the chain figure (orange: a weight on the searched rows; aqua: more
# games and a weight at once). The two seeds of the combined recipe share a colour.
COLOUR = {
    "control120": INK_2,
    "Bw2-s1": "#eb6834",
    "B240w2-s1": "#1baf7a",
    "B240w2-s2": "#1baf7a",
}
LABEL = {
    "control120": "control",
    "Bw2-s1": "weight \u00d72",
    "B240w2-s1": "240 games \u00d72",
}
LEGEND = {
    "control120": "untrained control",
    "Bw2-s1": "searched rows counted twice in the policy loss (Step 7)",
    "B240w2-s1": "240 external games and the rows counted twice, two seeds (Step 8)",
}
PANELS = (
    ("greedy", "Against Greedy: the positions the rows came from"),
    ("rollout-r3000", "Against the random-playout search: the held-out boards"),
)


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def series(table: dict[str, Any], name: str, opponent: str, variant: str) -> list[tuple[int, int]]:
    """(simulations, wipeouts) for the pairings that have reported, in budget order."""
    cells = table[name][opponent][variant]
    points = [(int(b), int(e["wipeouts"])) for b, e in cells.items() if e["wipeouts"] is not None]
    return sorted(points)


def style(ax: Any, title: str, budgets: list[int]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("simulations a move", color=INK_2, fontsize=9)
    ax.grid(visible=True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_3, labelsize=8, length=3)


def draw_panel(
    ax: Any,
    table: dict[str, Any],
    opponent: str,
    checkpoints: list[str],
    budgets: list[int],
    *,
    title: str,
    label_offsets: dict[str, tuple[float, float]],
) -> None:
    style(ax, title, budgets)
    for name in checkpoints:
        c = COLOUR[name]
        for variant, dash, face in (("plain", "solid", c), ("check", (0, (4, 3)), SURFACE)):
            pts = series(table, name, opponent, variant)
            if not pts:
                continue
            ax.plot(
                [x for x, _ in pts],
                [y for _, y in pts],
                color=c,
                linewidth=1.6,
                linestyle=dash,
                alpha=0.9 if variant == "plain" else 0.75,
                zorder=2,
            )
            ax.plot(
                [x for x, _ in pts],
                [y for _, y in pts],
                marker="o",
                markersize=6.5,
                markerfacecolor=face,
                markeredgecolor=c,
                markeredgewidth=1.4,
                linestyle="none",
                zorder=3,
            )
    # One label per recipe, beside its plain line's first point, where the lines are
    # furthest apart; the two seeds of the combined recipe share one label.
    for name, text in LABEL.items():
        pts = series(table, name, opponent, "plain")
        if not pts:
            continue
        x, y = pts[0]
        if name == "B240w2-s1":
            other = series(table, "B240w2-s2", opponent, "plain")
            if other:
                y = (y + other[0][1]) / 2
        dx, dy = label_offsets.get(name, (-9, 0))
        far = abs(dx) > 14 or abs(dy) > 14
        ax.annotate(
            text,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "color": INK_3, "linewidth": 0.7, "shrinkB": 5}
            if far
            else None,
            zorder=4,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=Path("research/figures/dose-curve.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    data = load(args.data)
    table = data["pairings"]
    checkpoints = list(data["checkpoints"])
    budgets = [int(b) for b in data["budgets"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.25, wspace=0.2)

    draw_panel(
        axes[0],
        table,
        PANELS[0][0],
        checkpoints,
        budgets,
        title=PANELS[0][1],
        label_offsets={},
    )
    draw_panel(
        axes[1],
        table,
        PANELS[1][0],
        checkpoints,
        budgets,
        title=PANELS[1][1],
        label_offsets={},
    )
    axes[0].set_ylabel("wipeouts in 200 games", color=INK_2, fontsize=9)
    axes[0].set_ylim(0, 70)
    axes[1].set_ylim(0, 40)
    for ax in axes:
        ax.set_xlim(21, 1000)

    fig.suptitle(
        "Sixteen times the search budget buys less than one ply of exact reading",
        x=0.07,
        ha="left",
        fontsize=12.5,
        color=INK,
        y=0.975,
    )
    fig.text(
        0.07,
        0.895,
        "Each line is one checkpoint playing the same 200 games (the frozen openings, both "
        "colours) at four search budgets.\n"
        "Solid, filled: the search as trained. Dashed, open: the same search with the "
        "terminal check, which looks one move ahead for a game-ending reply.",
        fontsize=8,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    style_handles = [
        Line2D([], [], color=INK_2, linewidth=1.6, label="plain search"),
        Line2D([], [], color=INK_2, linewidth=1.6, linestyle=(0, (4, 3)), label="with the check"),
    ]
    colour_handles = [
        Line2D([], [], color=COLOUR[n], marker="o", markersize=6, linewidth=1.6, label=text)
        for n, text in LEGEND.items()
    ]
    common = {
        "loc": "lower left",
        "frameon": False,
        "fontsize": 8,
        "labelcolor": INK_2,
        "handletextpad": 0.6,
        "columnspacing": 1.6,
    }
    fig.legend(handles=style_handles, bbox_to_anchor=(0.07, 0.055), ncol=2, **common)
    fig.legend(handles=colour_handles, bbox_to_anchor=(0.07, 0.0), ncol=3, **common)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for opponent, _ in PANELS:
        for name in checkpoints:
            plain = series(table, name, opponent, "plain")
            check = series(table, name, opponent, "check")
            print(f"  {opponent:<14}{name:<11} plain {plain}  check {check}")


if __name__ == "__main__":
    main()
