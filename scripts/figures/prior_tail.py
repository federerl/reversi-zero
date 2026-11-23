"""The prior's tail: how far the policy head's blind spot closed, against the signal.

The fixed-position head probe reads every network on the same 63 positions, the
untrained control's fatal positions against Greedy: the network to move, one reply
for Greedy that takes every disc. At each one the probe records the prior the
policy head gives that wiping reply at the position one ply later, with Greedy to
move. The search spends simulations in proportion to the prior, so a reply with a
prior under 5% is one a 50-simulation search rarely looks at: that share is the
tail of the blind spot, and the median prior is its centre.

This figure puts both against the axis the chain figure uses, the number of fatal
searched opponent rows a run delivered times the weight on them in the policy loss.
The checkpoints trained without searched rows sit at zero and are drawn as a grey
band. The tail closes from 65% to about 10% by a weighted count of 320 and stops
there; the median keeps rising, a little, past it.

Everything is read from the committed JSON under research/results/:

    python scripts/figures/prior_tail.py --out research/figures/prior-tail.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chain import (
    BAND,
    GROUPS,
    INK,
    INK_2,
    INK_3,
    ROUTE_COLOUR,
    ROUTE_LABEL,
    SURFACE,
    fatal_rows,
    load,
    style,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

# The fixed-position probe file each evaluation name is read from.
PROBE_FILE = {
    "Bsp160-s1": "step6/head-probe-fixed.json",
    "Bsp160-s2": "step6/head-probe-fixed.json",
}
DEFAULT_PROBE = "step8/head-probe-fixed.json"

# The checkpoints trained without searched rows that the fixed probe was run on.
NO_SEARCHED_PROBED: list[tuple[str, str]] = [
    ("step8/head-probe-fixed.json", "control120"),
    ("step5/head-probe-fixed.json", "Bboth-s1"),  # the opponent's positions recorded
    ("step5/head-probe-fixed.json", "Bboth-s2"),
    ("step5/head-probe-fixed.json", "Btc-s1"),  # the check inside the training search
    ("step5/head-probe-fixed.json", "Btc-s2"),
]

TAIL = 0.05


def tail_and_median(probe: dict[str, Any], name: str) -> tuple[float, float]:
    priors = [float(p["child_prior_wiping_total"]) for p in probe["networks"][name]["positions"]]
    return sum(x < TAIL for x in priors) / len(priors), statistics.median(priors)


def collect() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    probes: dict[str, dict[str, Any]] = {}

    def probe(path: str) -> dict[str, Any]:
        if path not in probes:
            probes[path] = load(path)
        return probes[path]

    points: list[dict[str, Any]] = []
    for g in GROUPS:
        for name, n in zip(g["names"], fatal_rows(g["rows"]), strict=True):
            tail, median = tail_and_median(probe(PROBE_FILE.get(name, DEFAULT_PROBE)), name)
            points.append(
                {
                    "group": g["label"],
                    "route": g["route"],
                    "name": name,
                    "x": n * g["weight"],
                    "tail": tail,
                    "median": median,
                }
            )
    reference: dict[str, list[dict[str, Any]]] = {"tail": [], "median": []}
    for path, name in NO_SEARCHED_PROBED:
        tail, median = tail_and_median(probe(path), name)
        reference["tail"].append({"name": name, "y": tail})
        reference["median"].append({"name": name, "y": median})
    return points, reference


def draw_panel(
    ax: Any,
    points: list[dict[str, Any]],
    key: str,
    reference: list[dict[str, Any]],
    *,
    title: str,
    ylabel: str,
    label_offsets: dict[str, tuple[float, float]],
) -> None:
    style(ax, title, ylabel)
    ys = [r["y"] for r in reference]
    lo, hi = min(ys), max(ys)
    ax.axhspan(lo, hi, color=BAND, zorder=0, linewidth=0)
    ax.text(
        0.995,
        hi,
        f"trained without searched rows, {len(reference)} checkpoints: {lo:.1%}\u2013{hi:.1%}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )
    control = next(r for r in reference if r["name"] == "control120")
    ax.plot(
        0,
        control["y"],
        marker="o",
        markersize=7.5,
        markerfacecolor=INK_2,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        linestyle="none",
        zorder=3,
    )
    ax.annotate(
        "the control",
        (0, control["y"]),
        xytext=(9, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK_2,
        va="center",
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)
    for ps in groups.values():
        cx = sum(p["x"] for p in ps) / len(ps)
        cy = sum(p[key] for p in ps) / len(ps)
        for p in ps:
            ax.plot(
                [cx, p["x"]],
                [cy, p[key]],
                color=ROUTE_COLOUR[p["route"]],
                linewidth=1.2,
                alpha=0.55,
                zorder=2,
            )
    for p in points:
        ax.plot(
            p["x"],
            p[key],
            marker="o",
            markersize=7.5,
            markerfacecolor=ROUTE_COLOUR[p["route"]],
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=3,
        )
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
    parser.add_argument("--out", type=Path, default=Path("research/figures/prior-tail.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    points, reference = collect()

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, (ax_tail, ax_median) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.2, wspace=0.22)

    draw_panel(
        ax_tail,
        points,
        "tail",
        reference["tail"],
        title="The tail: positions where the prior on the wiping reply is under 5%",
        ylabel="share of the 63 fatal positions",
        label_offsets={
            "120 games": (9, 0),
            "120, gens 141\u2013160": (50, 20),
            "240": (-9, 10),
            "480": (-10, -10),
            "\u00d72": (10, 10),
            "\u00d74": (12, -12),
            "\u00d78": (9, 0),
            "240 games \u00d72": (26, -2),
        },
    )
    draw_panel(
        ax_median,
        points,
        "median",
        reference["median"],
        title="The centre: the median prior on the wiping reply",
        ylabel="prior, median over the 63 positions",
        label_offsets={
            "120 games": (9, 0),
            "120, gens 141\u2013160": (6, 34),
            "240": (-40, -12),
            "480": (-6, 12),
            "\u00d72": (24, -34),
            "\u00d74": (30, -22),
            "\u00d78": (9, 0),
            "240 games \u00d72": (34, 12),
        },
    )
    for ax in (ax_tail, ax_median):
        ax.set_xlabel(
            "fatal searched rows delivered \u00d7 weight on them", color=INK_2, fontsize=9
        )
        ax.set_xlim(-25, 780)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_tail.set_ylim(0, 0.75)
    ax_median.set_ylim(0, 0.15)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=7,
            markerfacecolor=ROUTE_COLOUR[r],
            markeredgecolor=SURFACE,
            linestyle="none",
            label=ROUTE_LABEL[r],
        )
        for r in ("games", "weight", "both")
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        columnspacing=1.8,
    )

    fig.suptitle(
        "The policy head's blind spot, Steps 5 to 8: the tail closes by a weighted count "
        "of about 300 and stops",
        x=0.07,
        ha="left",
        fontsize=12.5,
        color=INK,
        y=0.975,
    )
    fig.text(
        0.07,
        0.895,
        "Each point is one checkpoint read on the control's 63 fatal positions, the seeds "
        "of a group joined by a line; the prior is the policy head's probability\n"
        "on Greedy's wiping reply at the position one ply below, where the search reads it. "
        "A prior under 5% is one a 50-simulation search rarely visits.",
        fontsize=8,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for p in points:
        print(f"  {p['name']:>11}  x={p['x']:>4}  tail={p['tail']:.0%}  median={p['median']:.1%}")
    for r in reference["tail"]:
        m = next(q for q in reference["median"] if q["name"] == r["name"])
        print(f"  {r['name']:>11}  x=   0  tail={r['y']:.0%}  median={m['y']:.1%}")


if __name__ == "__main__":
    main()
