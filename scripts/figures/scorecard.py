"""The scorecard as a dot plot: every checkpoint on every row, against the control.

Steps 3 to 8 evaluated 27 checkpoints under one frozen protocol on the same
thirteen rows. The results pages read those rows one step at a time; this figure
puts every checkpoint's score on every row in one picture, so that the two things
the series claims can be seen at once: the rows that moved (the Greedy family under
the policy route) and the rows that did not (Edax, the earlier generation, the
held-out random-playout search, at 50 simulations). The untrained control's 95%
interval is the grey band on each row; a dot outside it is a checkpoint that reads
differently from the control by more than 200 games' noise.

The checkpoints are sorted into four lanes by how they were trained: without any
searched opponent rows (the control recipe continued, the opponent mix, the mix
with the opponent's positions recorded, the mix with the check inside the training
search), and the three routes into the policy head of Steps 5 to 8 (more external
games, a weight on the searched rows, both). The one checkpoint trained on a
material term in the value target (Step 3, group D) is the open diamond: it is the
one that moved the held-out row at 50 simulations, and the one that paid for it on
the Edax and retention rows.

Everything is read from the committed JSON under research/results/:

    python scripts/figures/scorecard.py --out research/figures/scorecard-dots.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chain import BAND, GRID, INK, INK_2, INK_3, ROUTE_COLOUR, SURFACE, load, style
from matplotlib.lines import Line2D

# The rows in the order they are drawn, top to bottom, grouped as the roster groups them.
ROW_GROUPS: list[tuple[str, list[str]]] = [
    ("held out", ["edax-l7", "edax-l5", "edax-l3", "rollout-r3000"]),
    ("retention", ["gen60"]),
    ("seen", ["greedy", "minimax-d4"]),
    ("seen family", ["greedy-corner", "greedy-d2", "noisy-greedy", "corner", "mobility"]),
    ("sanity", ["random"]),
]
ROW_LABEL = {
    "edax-l7": "Edax level 7",
    "edax-l5": "Edax level 5",
    "edax-l3": "Edax level 3",
    "rollout-r3000": "random-playout search",
    "gen60": "generation 60",
    "greedy": "Greedy",
    "minimax-d4": "Minimax depth 4",
    "greedy-corner": "greedy-corner",
    "greedy-d2": "greedy-d2",
    "noisy-greedy": "noisy-greedy",
    "corner": "corner",
    "mobility": "mobility",
    "random": "random mover",
}

# Every checkpoint: the scorecard it is read from, its evaluation name, its lane.
CHECKPOINTS: list[tuple[str, str, str]] = [
    ("step3", "A-s1", "none"),
    ("step3", "A-s2", "none"),
    ("step3", "B-s1", "none"),
    ("step3", "B-s2", "none"),
    ("step3", "D-s1", "material"),
    ("step3b", "Bboth-s1", "none"),
    ("step3b", "Bboth-s2", "none"),
    ("step4", "Btc-s1", "none"),
    ("step4", "Btc-s2", "none"),
    ("step8", "Bsp-s1", "games"),
    ("step8", "Bsp-s2", "games"),
    ("step6", "Bsp160-s1", "games"),
    ("step6", "Bsp160-s2", "games"),
    ("step8", "Bsp240-s1", "games"),
    ("step8", "Bsp240-s2", "games"),
    ("step8", "Bsp480-s1", "games"),
    ("step8", "Bsp480-s2", "games"),
    ("step8", "Bw2-s1", "weight"),
    ("step8", "Bw2-s2", "weight"),
    ("step8", "Bw4-s1", "weight"),
    ("step8", "Bw4-s2", "weight"),
    ("step8", "Bw8-s1", "weight"),
    ("step8", "Bw8-s2", "weight"),
    ("step8", "B240w2-s1", "both"),
    ("step8", "B240w2-s2", "both"),
    ("step8", "B240w2-s3", "both"),
]
CONTROL = ("step3", "control120")

# Vertical offset of each lane inside a row, and its colour. The material-target
# checkpoint shares the "none" lane, drawn as an open diamond.
LANE_OFFSET = {"none": -0.27, "material": -0.27, "games": -0.09, "weight": 0.09, "both": 0.27}
LANE_COLOUR = {
    "none": INK_3,
    "material": INK,
    "games": ROUTE_COLOUR["games"],
    "weight": ROUTE_COLOUR["weight"],
    "both": ROUTE_COLOUR["both"],
}
LANE_LABEL = {
    "none": "trained without searched opponent rows (Steps 3, 3b, 4)",
    "material": "material term in the value target (Step 3, group D)",
    "games": "more external games (Steps 5, 6)",
    "weight": "a weight on the searched rows (Steps 7, 8)",
    "both": "both at once (Step 8)",
}


def collect() -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """The control's score and interval per row, and every checkpoint's score per row."""
    scorecards: dict[str, dict[str, Any]] = {}

    def sc(step: str) -> dict[str, Any]:
        if step not in scorecards:
            scorecards[step] = load(f"{step}/scorecard.json")
        return scorecards[step]

    control_pairings = sc(CONTROL[0])["pairings"][CONTROL[1]]
    control = {
        row: {
            "score": float(control_pairings[row]["score"]),
            "lo": float(control_pairings[row]["ci_low"]),
            "hi": float(control_pairings[row]["ci_high"]),
        }
        for _, rows in ROW_GROUPS
        for row in rows
    }
    points: list[dict[str, Any]] = []
    for step, name, lane in CHECKPOINTS:
        pairings = sc(step)["pairings"][name]
        points.append(
            {
                "name": name,
                "lane": lane,
                "scores": {row: float(pairings[row]["score"]) for row in control},
            }
        )
    return control, points


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("research/figures/scorecard-dots.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    control, points = collect()
    rows = [row for _, rows in ROW_GROUPS for row in rows]
    y_of = {row: i for i, row in enumerate(rows)}

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(10, 8.2))
    fig.subplots_adjust(left=0.2, right=0.975, top=0.85, bottom=0.14)
    style(ax, "", "")
    ax.grid(visible=False)
    ax.grid(visible=True, axis="x", color=GRID, linewidth=0.6)

    # The control's interval as a band, its score as a tick.
    for row, c in control.items():
        y = y_of[row]
        ax.fill_betweenx([y - 0.42, y + 0.42], c["lo"] * 100, c["hi"] * 100, color=BAND, zorder=0)
        ax.plot(
            [c["score"] * 100, c["score"] * 100],
            [y - 0.42, y + 0.42],
            color=INK,
            linewidth=1.4,
            zorder=2,
        )

    for p in points:
        lane = p["lane"]
        for row, s in p["scores"].items():
            y = y_of[row] + LANE_OFFSET[lane]
            if lane == "material":
                ax.plot(
                    s * 100,
                    y,
                    marker="D",
                    markersize=6,
                    markerfacecolor=SURFACE,
                    markeredgecolor=INK,
                    markeredgewidth=1.3,
                    linestyle="none",
                    zorder=4,
                )
            else:
                ax.plot(
                    s * 100,
                    y,
                    marker="o",
                    markersize=6,
                    markerfacecolor=LANE_COLOUR[lane],
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.9,
                    linestyle="none",
                    alpha=0.9,
                    zorder=3,
                )

    # Group separators and group names in the left margin.
    boundary = 0
    for group, group_rows in ROW_GROUPS:
        top, bottom = boundary - 0.5, boundary + len(group_rows) - 0.5
        if boundary:
            ax.axhline(top, color=GRID, linewidth=0.8, zorder=1)
        ax.text(
            -0.215,
            (top + bottom) / 2,
            group,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color=INK_2,
            fontweight="bold",
        )
        boundary += len(group_rows)

    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([ROW_LABEL[r] for r in rows], fontsize=8.5, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 101.5)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_xlabel(
        "the network's score in 200 games (a win counts 1, a draw \u00bd)",
        color=INK_2,
        fontsize=9,
    )

    handles = [
        Line2D([], [], color=INK, linewidth=1.4, label="the untrained control (generation 120)"),
        Line2D(
            [],
            [],
            marker="s",
            markersize=9,
            markerfacecolor=BAND,
            markeredgecolor=BAND,
            linestyle="none",
            label="the control's 95% interval",
        ),
    ]
    for lane in ("none", "material", "games", "weight", "both"):
        handles.append(
            Line2D(
                [],
                [],
                marker="D" if lane == "material" else "o",
                markersize=6,
                markerfacecolor=SURFACE if lane == "material" else LANE_COLOUR[lane],
                markeredgecolor=INK if lane == "material" else SURFACE,
                markeredgewidth=1.3 if lane == "material" else 0.9,
                linestyle="none",
                label=LANE_LABEL[lane],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.2, 0.0),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        labelcolor=INK_2,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    fig.suptitle(
        "Every checkpoint on every row of the scorecard: what moved, and what did not",
        x=0.2,
        ha="left",
        fontsize=12.5,
        color=INK,
        y=0.975,
    )
    fig.text(
        0.2,
        0.93,
        f"{len(points)} checkpoints from Steps 3 to 8, each read under the frozen protocol: "
        "200 games a row, 50 simulations a move,"
        + chr(10)
        + "the same openings for every entrant. A dot outside the band reads differently "
        "from the control by more than the noise of 200 games."
        + chr(10)
        + "Rows are grouped as the held-out roster groups them.",
        fontsize=8,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for row in rows:
        c = control[row]
        by_lane: dict[str, list[float]] = {}
        for p in points:
            by_lane.setdefault(p["lane"], []).append(p["scores"][row] * 100)
        spread = "  ".join(
            f"{lane}={min(v):.0f}\u2013{max(v):.0f}" for lane, v in sorted(by_lane.items())
        )
        print(
            f"  {row:>14}  control {c['score'] * 100:.1f} "
            f"[{c['lo'] * 100:.1f}, {c['hi'] * 100:.1f}]  {spread}"
        )


if __name__ == "__main__":
    main()
