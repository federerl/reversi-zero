"""The chain figure: what the searched opponent rows bought, against how much of them there was.

Steps 5 to 8 train the policy head on the opponent's positions by two routes, more
external games (120, 240, 480 a generation) and a weight on the searched rows in
the policy loss (2, 4, 8), and once by both at once (240 games, weight 2). The
quantity the two routes share is the number of fatal searched rows a run delivered
times the weight on them. This figure puts every checkpoint on that axis and reads
two rows of the scorecard off it: the wipeouts summed over the four disc-hungry
opponents the network trained against (the Greedy family), and the wipeouts
against the random-playout search, the held-out opponent that finds the same hole
with no knowledge of Greedy and on different boards.

The checkpoints trained without any searched rows sit at zero on the axis and are
drawn as a grey band, since on the held-out row they spread from 21 to 39 and no
single checkpoint of 200 games separates a trained group from them; the exact
one-move check on the untrained control, which removes the wipeouts without
training, is the dashed reference line.

Everything is read from the committed JSON under research/results/:

    python scripts/figures/chain.py --out research/figures/chain-weighted-count.png
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

RESULTS = Path("research/results")
FAMILY = ("greedy", "greedy-corner", "greedy-d2", "noisy-greedy")
HELD_OUT = "rollout-r3000"

# Surface and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e6e5e1"
BAND = "#e3e2de"

# One hue per route, in a fixed order; the three validate as a set on a scatter.
ROUTE_COLOUR = {
    "games": "#2a78d6",  # more external games
    "weight": "#eb6834",  # a weight on the searched rows
    "both": "#1baf7a",  # both at once
}
ROUTE_LABEL = {
    "games": "more external games (Step 6)",
    "weight": "a weight on the searched rows (Step 7, 8)",
    "both": "both at once (Step 8)",
}

# Each trained group: the scorecard it is read from, its two evaluation names,
# the searched-row file and the run names in it, the weight on the rows, the route
# and the label printed beside its points.
GROUPS: list[dict[str, Any]] = [
    {
        "label": "120 games",
        "scorecard": "step6",
        "names": ["Bsp-s1", "Bsp-s2"],
        "rows": ("step5/searched-rows.json", ["step5-Bboth-sp-s1", "step5-Bboth-sp-s2"]),
        "weight": 1,
        "route": "games",
    },
    {
        "label": "120, gens 141\u2013160",
        "scorecard": "step6",
        "names": ["Bsp160-s1", "Bsp160-s2"],
        "rows": (
            ("step5/searched-rows.json", ["step5-Bboth-sp-s1", "step5-Bboth-sp-s2"]),
            ("step6/searched-rows-continued.json", ["step5-Bboth-sp-s1", "step5-Bboth-sp-s2"]),
        ),
        "weight": 1,
        "route": "games",
    },
    {
        "label": "240",
        "scorecard": "step6",
        "names": ["Bsp240-s1", "Bsp240-s2"],
        "rows": ("step6/searched-rows-dose.json", ["step6-Bsp240-s1", "step6-Bsp240-s2"]),
        "weight": 1,
        "route": "games",
    },
    {
        "label": "480",
        "scorecard": "step6",
        "names": ["Bsp480-s1", "Bsp480-s2"],
        "rows": ("step6/searched-rows-dose.json", ["step6-Bsp480-s1", "step6-Bsp480-s2"]),
        "weight": 1,
        "route": "games",
    },
    {
        "label": "\u00d72",
        "scorecard": "step8",
        "names": ["Bw2-s1", "Bw2-s2"],
        "rows": ("step7/searched-rows.json", ["step7-Bsp-w2-s1", "step7-Bsp-w2-s2"]),
        "weight": 2,
        "route": "weight",
    },
    {
        "label": "\u00d74",
        "scorecard": "step8",
        "names": ["Bw4-s1", "Bw4-s2"],
        "rows": ("step7/searched-rows.json", ["step7-Bsp-w4-s1", "step7-Bsp-w4-s2"]),
        "weight": 4,
        "route": "weight",
    },
    {
        "label": "\u00d78",
        "scorecard": "step8",
        "names": ["Bw8-s1", "Bw8-s2"],
        "rows": ("step8/searched-rows.json", ["step8-Bsp-w8-s1", "step8-Bsp-w8-s2"]),
        "weight": 8,
        "route": "weight",
    },
    {
        "label": "240 games \u00d72",
        "scorecard": "step8",
        "names": ["B240w2-s1", "B240w2-s2"],
        "rows": ("step8/searched-rows.json", ["step8-Bsp240-w2-s1", "step8-Bsp240-w2-s2"]),
        "weight": 2,
        "route": "both",
    },
]

# The checkpoints trained without searched rows: the scorecard each is read from.
NO_SEARCHED: list[tuple[str, str]] = [
    ("step3", "control120"),
    ("step3", "A-s1"),  # the control recipe continued for twenty generations
    ("step3", "A-s2"),
    ("step3", "B-s1"),  # the opponent mix
    ("step3", "B-s2"),
    ("step3b", "Bboth-s1"),  # the opponent mix with the opponent's positions recorded
    ("step3b", "Bboth-s2"),
    ("step4", "Btc-s1"),  # the same, with the check inside the training search
    ("step4", "Btc-s2"),
]


def load(path: str) -> dict[str, Any]:
    with (RESULTS / path).open(encoding="utf-8") as f:
        return json.load(f)


def wipeouts(scorecard: dict[str, Any], name: str, opponent: str) -> int:
    return int(scorecard["pairings"][name][opponent]["wipeouts"])


def family_sum(scorecard: dict[str, Any], name: str) -> int:
    return sum(wipeouts(scorecard, name, o) for o in FAMILY)


def fatal_rows(spec: Any) -> list[int]:
    """Fatal searched rows per seed; a tuple of files is summed (a continued run)."""
    if isinstance(spec[0], tuple):
        parts = [fatal_rows(s) for s in spec]
        return [sum(p[i] for p in parts) for i in range(len(parts[0]))]
    file, runs = spec
    by_run = {r["run"]: int(r["with_a_wiping_move"]) for r in load(file)["runs"]}
    return [by_run[r] for r in runs]


def collect() -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, int | None]]:
    scorecards: dict[str, dict[str, Any]] = {}

    def sc(step: str) -> dict[str, Any]:
        if step not in scorecards:
            scorecards[step] = load(f"{step}/scorecard.json")
        return scorecards[step]

    points: list[dict[str, Any]] = []
    for g in GROUPS:
        rows = fatal_rows(g["rows"])
        for name, n in zip(g["names"], rows, strict=True):
            s = sc(g["scorecard"])
            points.append(
                {
                    "group": g["label"],
                    "route": g["route"],
                    "name": name,
                    "x": n * g["weight"],
                    "family": family_sum(s, name),
                    "held_out": wipeouts(s, name, HELD_OUT),
                }
            )

    reference = {"family": [], "held_out": []}
    for step, name in NO_SEARCHED:
        s = sc(step)
        reference["family"].append(family_sum(s, name))
        reference["held_out"].append(wipeouts(s, name, HELD_OUT))

    checked = load("step3c/scorecard.json")["pairings"]["control120-tc"]
    check = {
        "held_out": int(checked[HELD_OUT]["wipeouts"]) if checked.get(HELD_OUT) else None,
        "family": sum(int(checked[o]["wipeouts"]) for o in FAMILY)
        if all(checked.get(o) for o in FAMILY)
        else None,
    }
    return points, reference, check


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(visible=True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_3, labelsize=8, length=3)


def draw_panel(
    ax: Any,
    points: list[dict[str, Any]],
    key: str,
    reference: list[int],
    check: int | None,
    *,
    title: str,
    ylabel: str,
    label_offsets: dict[str, tuple[float, float]],
) -> None:
    style(ax, title, ylabel)
    lo, hi = min(reference), max(reference)
    ax.axhspan(lo, hi, color=BAND, zorder=0, linewidth=0)
    ax.text(
        0.995,
        hi,
        f"trained without searched rows, {len(reference)} checkpoints: {lo}\u2013{hi}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )
    if check is not None:
        ax.axhline(check, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(
            0.995,
            check,
            f"the exact check on the untrained control: {check}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)
    for ps in groups.values():
        ax.plot(
            [p["x"] for p in ps],
            [p[key] for p in ps],
            color=ROUTE_COLOUR[ps[0]["route"]],
            linewidth=1.2,
            alpha=0.55,
            zorder=2,
        )
    for p in points:
        c = ROUTE_COLOUR[p["route"]]
        ax.plot(
            p["x"],
            p[key],
            marker="o",
            markersize=7.5,
            markerfacecolor=c,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=3,
        )

    # One label per group, beside the mean of its two seeds; a label placed far
    # from its points, where the points crowd, gets a thin leader line.
    for label, ps in groups.items():
        x = sum(p["x"] for p in ps) / len(ps)
        y = sum(p[key] for p in ps) / len(ps)
        dx, dy = label_offsets.get(label, (8, 0))
        far = abs(dx) > 14 or abs(dy) > 14
        ax.annotate(
            label,
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
    parser.add_argument(
        "--out", type=Path, default=Path("research/figures/chain-weighted-count.png")
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    points, reference, check = collect()

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, (ax_family, ax_held) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.17, wspace=0.22)

    draw_panel(
        ax_family,
        points,
        "family",
        reference["family"],
        check["family"],
        title="Against the Greedy family: the positions the rows came from",
        ylabel="wipeouts in 800 games (four opponents \u00d7 200)",
        label_offsets={
            "120 games": (9, 6),
            "120, gens 141\u2013160": (-14, -30),
            "240": (6, 12),
            "480": (-9, -4),
            "\u00d72": (14, -10),
            "\u00d74": (-4, 18),
            "\u00d78": (9, 0),
            "240 games \u00d72": (26, -10),
        },
    )
    draw_panel(
        ax_held,
        points,
        "held_out",
        reference["held_out"],
        check["held_out"],
        title="Against the random-playout search: the held-out boards",
        ylabel="wipeouts in 200 games",
        label_offsets={
            "120 games": (9, 0),
            "120, gens 141\u2013160": (-25, -28),
            "240": (-8, 28),
            "480": (-9, -12),
            "\u00d72": (8, 10),
            "\u00d74": (9, 4),
            "\u00d78": (9, 0),
            "240 games \u00d72": (12, 0),
        },
    )
    for ax in (ax_family, ax_held):
        ax.set_xlabel(
            "fatal searched rows delivered \u00d7 weight on them", color=INK_2, fontsize=9
        )
        ax.set_xlim(0, 780)
        ax.set_ylim(bottom=0)
    ax_family.set_ylim(0, 300)
    ax_held.set_ylim(0, 45)

    fig.suptitle(
        "What the searched opponent rows bought, Steps 5 to 8: "
        "the family keeps falling, the held-out row stops",
        x=0.07,
        ha="left",
        fontsize=12.5,
        color=INK,
        y=0.975,
    )
    fig.text(
        0.07,
        0.895,
        "Each point is one checkpoint, two seeds per group joined by a line, read under the "
        "frozen protocol (200 games a pairing, 50 simulations a move).\n"
        "The x-axis is the number of opponent positions with a game-ending reply that the "
        "training search was run at, times the weight those rows carry in the policy loss.",
        fontsize=8,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=ROUTE_COLOUR[r],
            markersize=7,
            linestyle="none",
            label=ROUTE_LABEL[r],
        )
        for r in ("games", "weight", "both")
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for p in points:
        print(
            f"  {p['name']:>11}  x={p['x']:>4}  family={p['family']:>3}  held-out={p['held_out']}"
        )
    ref_f, ref_h = reference["family"], reference["held_out"]
    print(f"  reference family {min(ref_f)}-{max(ref_f)}, held-out {min(ref_h)}-{max(ref_h)}")
    print(f"  the check on the control: {check}")


if __name__ == "__main__":
    main()
