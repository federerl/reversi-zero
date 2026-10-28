"""What did the network believe while Greedy was wiping it out?

Step 2, question 1 of ``research/step1-pilot-results.md``. Every recorded
network move carries two numbers: ``net_value``, the network's own estimate
of the position before any search, and ``search_value``, the estimate after
the tree search, both from the network's point of view in [-1, +1] where +1
means "I win". This script replays each game the network lost by wipeout
against Greedy, lines the network's moves up by distance from the end, and
asks when, if ever, the estimates turned negative, and how many discs the
network still held at each of those moves.

Usage::

    python scripts/analyze_wipeouts.py runs/research/pilot-2026-10-games \
        [--opponent greedy] [--figure research/figures/wipeout-values.png]

``--opponent`` names the fixed opponent whose games to read (the
``<opponent>-vs-*.json`` files). It must be one that reports its tie set
through ``choices``, as Greedy and its variants do, because the "was it
avoidable" analysis has to consider every reply the opponent might have
chosen. The games are replayed with the rules module to recover disc counts,
which the records do not store per move.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reversi.arena.entrants import build_agent, parse_entrant
from reversi.game.rules import apply, is_terminal, legal_actions, must_pass
from reversi.game.scoring import disc_counts
from reversi.game.state import State, initial_state

LOOKBACK = 10
"""How many of the network's own moves before the end to look at."""

OPPONENT = "greedy"
"""The fixed opponent being analysed; ``set_opponent`` changes it."""
_replies = None


def set_opponent(name: str) -> None:
    """Choose the fixed opponent. Its ``choices`` method supplies the reply sets."""
    global OPPONENT, _replies
    agent = build_agent(parse_entrant(name, default_simulations=1))
    choices = getattr(agent, "choices", None)
    if choices is None:
        msg = f"{name} does not report its tie set (no choices method); cannot enumerate replies"
        raise SystemExit(msg)
    OPPONENT, _replies = name, choices


@dataclass(frozen=True)
class NetMove:
    moves_before_end: int
    """1 for the network's last move, 2 for the one before, and so on."""
    net_value: float
    search_value: float
    my_discs: int
    """The network's discs when it made the move."""
    state: State
    """The position the network faced, network to move."""
    visits: tuple[tuple[int, int], ...]
    """(action, visit count) at the root of the network's search for this move."""


@dataclass(frozen=True)
class Trace:
    net_won: bool
    net_black: bool
    wipeout: bool
    moves: tuple[NetMove, ...]


def net_discs(state: State, net_black: bool) -> int:
    black, white = disc_counts(state)
    return black if net_black else white


def opponent_replies(state: State) -> list[int]:
    """Every move the opponent might make here: for Greedy, the ones flipping the most discs."""
    if _replies is None:
        set_opponent(OPPONENT)
    assert _replies is not None
    return list(_replies(state))


def survives(state: State, net_black: bool, moves_left: int) -> bool:
    """Can the network keep at least one disc through its next ``moves_left`` moves?

    The network may choose any legal move; the opponent may play any move in
    its tie set (Greedy breaks ties at random, so every tie must be safe).
    Passes are forced and cost nothing. ``True`` means a line exists in which
    the network is never wiped out, whatever the opponent's tie-breaks.
    """
    pass_action = state.size**2
    if is_terminal(state):
        return net_discs(state, net_black) > 0
    if must_pass(state):
        return survives(apply(state, pass_action), net_black, moves_left)
    black_to_move = state.to_move == initial_state(state.size).to_move
    if black_to_move == net_black:
        if moves_left == 0:
            return True
        return any(
            survives(apply(state, a), net_black, moves_left - 1) for a in legal_actions(state)
        )
    for reply in opponent_replies(state):
        after = apply(state, reply)
        if net_discs(after, net_black) == 0 or not survives(after, net_black, moves_left):
            return False
    return True


def replay(game: dict, net_is_a: bool, board_size: int) -> Trace:
    pass_action = board_size**2
    state = initial_state(board_size)
    for action in game["opening"]:
        state = apply(state, action)
    net_is_black = game["a_is_black"] == net_is_a
    seen: list[tuple[float, float, int, State, tuple[tuple[int, int], ...]]] = []
    for ply in game["plies"]:
        if ply["by_a"] == net_is_a and "search_value" in ply:
            visits = tuple((int(a), int(n)) for a, n in ply["visits"])
            seen.append(
                (
                    ply["net_value"],
                    ply["search_value"],
                    net_discs(state, net_is_black),
                    state,
                    visits,
                )
            )
        state = (
            apply(state, ply["action"])
            if ply["action"] != pass_action
            else apply(state, pass_action)
        )
    black, white = disc_counts(state)
    outcome = game["outcome_for_a"] if net_is_a else -game["outcome_for_a"]
    moves = tuple(
        NetMove(len(seen) - i, nv, sv, discs, s, v) for i, (nv, sv, discs, s, v) in enumerate(seen)
    )
    return Trace(
        net_won=outcome > 0, net_black=net_is_black, wipeout=black == 0 or white == 0, moves=moves
    )


def load(path: Path) -> tuple[str, list[Trace]]:
    payload = json.loads(path.read_text())
    a, b = payload["agents"]["a"], payload["agents"]["b"]
    net = b if a == OPPONENT else a
    net_is_a = net == a
    traces = [replay(g, net_is_a, payload["board_size"]) for g in payload["games"]]
    return net, traces


def at(trace: Trace, k: int) -> NetMove | None:
    for m in trace.moves:
        if m.moves_before_end == k:
            return m
    return None


def first_persistent_negative(trace: Trace) -> int | None:
    """Moves before the end at which the searched value went negative and stayed there.

    ``None`` if the searched value was still positive at the network's last move.
    """
    ordered = sorted(trace.moves, key=lambda m: m.moves_before_end)  # last move first
    # A game can end before the network moves at all: when the opening leaves
    # it with one disc and Greedy moves first. There is no estimate to inspect.
    if not ordered or ordered[0].search_value >= 0:
        return None
    k = 0
    for m in ordered:
        if m.search_value < 0:
            k = m.moves_before_end
        else:
            break
    return k


def median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("games_dir", type=Path)
    parser.add_argument("--opponent", default="greedy")
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()
    set_opponent(args.opponent)

    results: dict[str, list[Trace]] = {}
    for path in sorted(args.games_dir.glob(f"{OPPONENT}-vs-*.json")):
        net, traces = load(path)
        if not any(trace.moves for trace in traces):
            continue  # the other side is a fixed opponent too; nothing was estimated
        results[net] = traces
    if not results:
        raise SystemExit(f"no {OPPONENT}-vs-*.json under {args.games_dir}")

    print("## The network's estimates in the moves before a wipeout")
    print()
    print(
        "Values are the searched value from the network's view, +1 = sure win. "
        "'positive' is the share of wipeouts where it was still above zero at that move."
    )
    print()
    print(
        "| network | wipeouts | move | median searched value | positive "
        "| median raw network value | median discs held |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for net, traces in results.items():
        wiped = [t for t in traces if t.wipeout and not t.net_won]
        for k in (1, 2, 3, 5, 8):
            rows = [m for t in wiped if (m := at(t, k)) is not None]
            if not rows:
                continue
            label = "last move" if k == 1 else f"{k - 1} before last"
            print(
                f"| {net} | {len(wiped)} | {label} | {median([m.search_value for m in rows]):+.2f} "
                f"| {np.mean([m.search_value > 0 for m in rows]):.0%} "
                f"| {median([m.net_value for m in rows]):+.2f} "
                f"| {median([m.my_discs for m in rows]):.0f} |"
            )
    print()

    print("## When the searched value first turned negative for good")
    print()
    print(
        "| network | wipeouts in which the network moved | never (positive at the last move) "
        "| only at the last move | 2 to 3 moves before | 4 or more before |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for net, traces in results.items():
        wiped = [t for t in traces if t.wipeout and not t.net_won and t.moves]
        firsts = [first_persistent_negative(t) for t in wiped]
        never = sum(f is None for f in firsts)
        last = sum(f == 1 for f in firsts)
        soon = sum(f is not None and 2 <= f <= 3 for f in firsts)
        early = sum(f is not None and f >= 4 for f in firsts)
        print(f"| {net} | {len(wiped)} | {never} | {last} | {soon} | {early} |")
    print()

    print("## Was the wipeout still avoidable?")
    print()
    print(
        "From the position at the network's k-th last move: does any line of play keep at least "
        f"one network disc through those k moves, against every reply {OPPONENT} "
        "might choose?"
    )
    print()
    print("| network | wipeouts | avoidable at the last move | at the 2nd last | at the 3rd last |")
    print("|---|---:|---:|---:|---:|")
    for net, traces in results.items():
        wiped = [t for t in traces if t.wipeout and not t.net_won]
        counts = []
        for k in (1, 2, 3):
            rows = [(t, m) for t in wiped if (m := at(t, k)) is not None]
            avoidable = sum(survives(m.state, t.net_black, k) for t, m in rows)
            counts.append(f"{avoidable} of {len(rows)}")
        print(f"| {net} | {len(wiped)} | " + " | ".join(counts) + " |")
    print()

    print("## Did the search look at the safe move?")
    print()
    print(
        "At the network's last move, where a safe move existed: how many of the search's visits "
        "went to safe moves, and in how many games every safe move went unvisited."
    )
    print()
    print(
        "| network | wipeouts with a safe last move | median share of visits on safe moves "
        "| games where no safe move was visited | games where a safe move was the runner-up |"
    )
    print("|---|---:|---:|---:|---:|")
    for net, traces in results.items():
        wiped = [t for t in traces if t.wipeout and not t.net_won]
        shares, unvisited, runner_up, n = [], 0, 0, 0
        for t in wiped:
            m = at(t, 1)
            if m is None:
                continue
            safe = {
                a for a in legal_actions(m.state) if survives(apply(m.state, a), t.net_black, 0)
            }
            if not safe:
                continue
            n += 1
            total = sum(v for _, v in m.visits) or 1
            on_safe = sum(v for a, v in m.visits if a in safe)
            shares.append(on_safe / total)
            unvisited += on_safe == 0
            ranked = sorted(m.visits, key=lambda av: -av[1])
            runner_up += len(ranked) > 1 and ranked[1][0] in safe
        print(f"| {net} | {n} | {median(shares):.0%} | {unvisited} | {runner_up} |")
    print()

    print(f"## For comparison: the same moves in games the network won against {OPPONENT}")
    print()
    print(
        "Median searched value at the network's n-th move after the opening, "
        "in wins versus in wipeouts."
    )
    print()
    print("| network | n-th move | in wins | in wipeouts (those that lasted that long) |")
    print("|---|---:|---:|---:|")
    for net, traces in results.items():
        wins = [t for t in traces if t.net_won]
        wiped = [t for t in traces if t.wipeout and not t.net_won]
        for n in (1, 3, 5, 7):

            def nth(t: Trace, n: int = n) -> NetMove | None:
                ordered = sorted(t.moves, key=lambda m: -m.moves_before_end)
                return ordered[n - 1] if len(ordered) >= n else None

            w = [m.search_value for t in wins if (m := nth(t)) is not None]
            x = [m.search_value for t in wiped if (m := nth(t)) is not None]
            print(f"| {net} | {n} | {median(w):+.2f} | {median(x):+.2f} (of {len(x)}) |")
    print()

    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, len(results), figsize=(4.2 * len(results), 6.4), sharex=True, sharey="row"
        )
        for col, (net, traces) in enumerate(results.items()):
            wiped = [t for t in traces if t.wipeout and not t.net_won]
            top, bottom = axes[0, col], axes[1, col]
            xs = list(range(-LOOKBACK, 0))
            values = np.full((len(wiped), LOOKBACK), np.nan)
            discs = np.full((len(wiped), LOOKBACK), np.nan)
            for i, t in enumerate(wiped):
                for m in t.moves:
                    if m.moves_before_end <= LOOKBACK:
                        values[i, LOOKBACK - m.moves_before_end] = m.search_value
                        discs[i, LOOKBACK - m.moves_before_end] = m.my_discs
            for row in values:
                top.plot(xs, row, color="0.75", linewidth=0.6)
            for row in discs:
                bottom.plot(xs, row, color="0.75", linewidth=0.6)
            top.plot(xs, np.nanmedian(values, axis=0), color="black", linewidth=2, label="median")
            bottom.plot(xs, np.nanmedian(discs, axis=0), color="black", linewidth=2)
            top.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
            top.set_title(f"{net}: {len(wiped)} wipeouts by {OPPONENT}")
            bottom.set_xlabel("network's moves before the wipeout")
            if col == 0:
                top.set_ylabel("searched value (+1 = network wins)")
                bottom.set_ylabel("network's discs at that move")
                top.legend(loc="lower left")
            top.set_ylim(-1.05, 1.05)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130)
        print(f"figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
