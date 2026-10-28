"""Read a Step 1 pilot off its report and its recorded games.

The frozen protocol (``research/eval-protocol-frozen.json``) names four things
to read from the pilot, all decided before it was played: the score and
interval of every pairing, every pairing split by colour, the number of
wipeouts, and the two one-variable comparisons between the checkpoints. This
script prints exactly those, as Markdown tables.

Usage::

    python scripts/analyze_pilot.py runs/research/pilot-2026-10.json \
        runs/research/pilot-2026-10-games

Terms. A *wipeout* is a game that ends with one side holding no discs at all.
A game's *length* here is the number of moves after the seeded opening, passes
not counted; a game that fills the board is 56 moves long after a 4-ply
opening. The *colour difference* of a pairing is the first agent's score with
black minus its score with white. The two games of an opening pair share a
position, so they are resampled together when putting an interval on that
difference: drawing pairs with replacement many times and reading off the
middle 95% of the differences is a *bootstrap* interval.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOOTSTRAP_DRAWS = 10_000
LONG_GAME_MOVES = 50
"""A loss of this many moves or more was decided late, not by an early wipeout."""


@dataclass(frozen=True)
class Game:
    a_is_black: bool
    score_a: float
    moves: int
    black: int
    white: int

    @property
    def wipeout(self) -> bool:
        return self.black == 0 or self.white == 0

    @property
    def a_wiped_out(self) -> bool:
        return (self.black == 0) if self.a_is_black else (self.white == 0)


def load_games(path: Path) -> list[Game]:
    payload = json.loads(path.read_text())
    pass_action = payload["board_size"] ** 2
    games = []
    for g in payload["games"]:
        # ``plies`` holds the moves after the opening, passes included; the
        # opening itself is in ``opening``.
        moves = sum(p["action"] != pass_action for p in g["plies"])
        games.append(
            Game(
                a_is_black=g["a_is_black"],
                score_a={1: 1.0, 0: 0.5, -1: 0.0}[g["outcome_for_a"]],
                moves=moves,
                black=g["final"]["black"],
                white=g["final"]["white"],
            )
        )
    return games


def colour_difference(games: list[Game], rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of (score as black minus score as white) over opening pairs, with a bootstrap interval.

    The two games of a pair share an opening, so pairs are what gets resampled.
    """
    diffs = []
    for i in range(0, len(games) - 1, 2):
        x, y = games[i], games[i + 1]
        black, white = (x, y) if x.a_is_black else (y, x)
        diffs.append(black.score_a - white.score_a)
    sample = np.array(diffs)
    draws = rng.choice(sample, size=(BOOTSTRAP_DRAWS, len(sample)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(sample.mean()), float(low), float(high)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def main(report_path: Path, games_dir: Path) -> int:
    report = json.loads(report_path.read_text())
    protocol = report["protocol"]
    rng = np.random.default_rng(protocol["seed"])
    networks = [n for n, s in report["agent_specs"].items() if s.get("kind") == "alphazero"]
    fixed = [
        n for n, s in report["agent_specs"].items() if s.get("kind") not in {"alphazero", "random"}
    ]
    matchups = {(m["a"], m["b"]): m for m in report["matchups"]}
    games = {(a, b): load_games(games_dir / f"{a}-vs-{b}.json") for a, b in matchups}

    def pairing(x: str, y: str) -> tuple[tuple[str, str], bool]:
        """The pairing between ``x`` and ``y``, and whether ``x`` was its first agent."""
        return ((x, y), True) if (x, y) in matchups else ((y, x), False)

    per_pair, seed = protocol["games_per_pair"], protocol["seed"]
    print(f"report: {report_path}  ({per_pair} games per pairing, seed {seed})")
    print()

    print("## Every pairing")
    print()
    print(
        "| pairing | score of first | 95% Wilson | as black | as white "
        "| black minus white, per opening pair [95% bootstrap] | wipeouts (first / second) |"
    )
    print("|---|---:|---|---:|---:|---|---|")
    for (a, b), row in matchups.items():
        played = games[(a, b)]
        point, low, high = colour_difference(played, rng)
        wiped_a = sum(g.a_wiped_out for g in played)
        wiped_b = sum(g.wipeout and not g.a_wiped_out for g in played)
        print(
            f"| {a} vs {b} | {pct(row['score_a'])} | [{pct(row['ci_low'])}, {pct(row['ci_high'])}] "
            f"| {pct(row['score_as_black'])} | {pct(row['score_as_white'])} "
            f"| {pts(point)} pts [{pts(low)}, {pts(high)}] | {wiped_a} / {wiped_b} |"
        )
    print()

    print("## How the networks lose to each fixed opponent")
    print()
    print(
        "| network | opponent | losses | of which wipeouts "
        f"| moves after the opening at each wipeout | losses of {LONG_GAME_MOVES}+ moves |"
    )
    print("|---|---|---:|---:|---|---:|")
    for opponent in fixed:
        for net in networks:
            key, net_first = pairing(net, opponent)
            lost = 0.0 if net_first else 1.0
            losses = [g for g in games[key] if g.score_a == lost]
            wiped = sorted(g.moves for g in losses if g.wipeout)
            long = sum(g.moves >= LONG_GAME_MOVES for g in losses)
            when = ", ".join(map(str, wiped)) or "none"
            print(f"| {net} | {opponent} | {len(losses)} | {len(wiped)} | {when} | {long} |")
    print()

    print("## Wipeouts by the colour of the side wiped out")
    print()
    print(
        "| pairing | first agent wiped out as black | as white "
        "| second agent wiped out as black | as white |"
    )
    print("|---|---:|---:|---:|---:|")
    for (a, b), played in games.items():
        if not any(g.wipeout for g in played):
            continue
        a_black = sum(g.a_wiped_out and g.a_is_black for g in played)
        a_white = sum(g.a_wiped_out and not g.a_is_black for g in played)
        b_black = sum(g.wipeout and not g.a_wiped_out and not g.a_is_black for g in played)
        b_white = sum(g.wipeout and not g.a_wiped_out and g.a_is_black for g in played)
        print(f"| {a} vs {b} | {a_black} | {a_white} | {b_black} | {b_white} |")
    print()

    print("## The checkpoints side by side")
    print()
    ratings = {r["name"]: r for r in report["ratings"]}
    columns = " | ".join(f"vs {o} (wiped out)" for o in fixed)
    print(f"| checkpoint | rating [95%] | {columns} |")
    print("|---|---|" + "---:|" * len(fixed))
    for net in networks:
        r = ratings[net]
        cells = []
        for opponent in fixed:
            key, net_first = pairing(net, opponent)
            score = matchups[key]["score_a"]
            score = score if net_first else 1 - score
            wiped = sum(g.a_wiped_out == net_first and g.wipeout for g in games[key])
            cells.append(f"{pct(score)} ({wiped})")
        print(
            f"| {net} | {r['elo']:+.0f} [{r['ci_low']:+.0f}, {r['ci_high']:+.0f}] | "
            + " | ".join(cells)
            + " |"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
