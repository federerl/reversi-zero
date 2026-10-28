"""Are two opponents different enough for one to count as unseen?

The admission rule in ``research/held-out-opponents.md`` section 5. The
training opponents for the opponent-diversity experiment are Greedy and
Minimax-d4. A test opponent is only *held out* if it does not play like
either of them, and "plays like" is measured rather than argued: put every
opponent in front of the same 500 mid-game positions, write down the move
each one chooses, and count how often the candidate's move is one the
training opponent would have played.

**Where the threshold comes from.** The first version of this check used
the agreement between the two training opponents themselves as the ceiling.
Measured, that number (17%) is *below* the chance level (23% against Greedy,
14% against Minimax-d4): two different strategies agree less often than a
random player agrees with either, so it cannot be a ceiling. What the data
does show is a gap. Opponents built from the same ideas as a training
opponent (Greedy with a corner rule, Greedy two plies deep, Minimax at other
depths) agree with it far more often than strong opponents built from other
knowledge (Edax, which learned from a game database; the self-play networks).
The threshold is therefore placed halfway between the two groups whose source
is known by construction: the lowest same-source agreement and the highest
different-source agreement. A candidate is held out, by this check, when it
lands on the different-source side for both training opponents.

Agreement with a competent player partly measures competence (every good
player takes the corner), so this check cannot prove two opponents unrelated.
It can show that a candidate is *not* a near-copy of a training opponent, and
that is what it is used for; the source-of-knowledge rule in the roster
document decides the rest.

Agreement is directional and allows for ties. An agent that breaks ties at
random (Greedy, and the one-idea controls) has a *set* of moves it might play
in a position; the agreement of candidate ``A`` with reference ``B`` at one
position is the share of ``A``'s moves that are in ``B``'s set, which is the
probability that a random tie-break lands on a move ``B`` could have chosen.
Averaged over the positions. ``random``'s row is therefore the chance level:
the share of legal moves the reference would play.

Usage::

    python scripts/similarity_check.py runs/research/pilot-2026-10-games \
        --positions research/results/step2/similarity-positions.json \
        --out research/results/step2/similarity-check.json

The positions are sampled once from the Step 1 game logs, seeded, and saved;
a later run with more candidates reads the saved file so every opponent
answers the same questions.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from reversi.arena.entrants import build_agent, describe_entrant, parse_entrant
from reversi.atomicio import atomic_write_json
from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State, initial_state
from reversi.seeding import derive_seed
from reversi.seeding import rng as make_rng
from reversi.types import Player

SEEN = ("greedy", "minimax-d4")
SAME_SOURCE: dict[str, tuple[str, ...]] = {
    "greedy": ("greedy-corner", "greedy-d2", "noisy-greedy"),
    "minimax-d4": ("minimax-d2", "minimax-d6"),
}
"""Opponents that are, by construction, a training opponent's own ideas."""
DIFFERENT_SOURCE = ("edax-l3", "edax-l5", "edax-l7", "gen60", "control120")
"""Strong opponents whose knowledge comes from elsewhere: a game database, self-play."""
DEFAULT_AGENTS = (
    "random",
    "greedy",
    "minimax-d4",
    "edax-l3",
    "edax-l5",
    "edax-l7",
    "rollout-r250",
    "rollout-r1000",
    "gen60=models/reversi-8x8-gen60.pt@50",
    "control120=models/reversi-8x8-control-gen120.pt@50",
    "greedy-corner",
    "greedy-d2",
    "noisy-greedy",
    "corner",
    "mobility",
    "minimax-d2",
    "minimax-d6",
)
MIN_EMPTIES, MAX_EMPTIES = 20, 44
RULE = (
    "per training opponent, threshold = midpoint between the lowest agreement of a "
    "same-source reference and the highest agreement of a different-source reference; "
    "a candidate passes when its agreement with every training opponent is at most "
    "that opponent's threshold"
)

Packed = tuple[int, int, int]
"""A position as (black, white, to_move) so it can be sent to another process."""


def pack(state: State) -> Packed:
    return (state.black, state.white, int(state.to_move))


def unpack(packed: Packed, size: int = 8) -> State:
    black, white, to_move = packed
    return State(black=black, white=white, to_move=Player(to_move), size=size)


# ---------------------------------------------------------------------------
# Sampling the positions
# ---------------------------------------------------------------------------


def sample_positions(games_dir: Path, *, count: int, seed: int) -> list[dict[str, Any]]:
    """``count`` distinct mid-game positions with a real choice to make, from the pilot's games."""
    candidates: dict[Packed, str] = {}
    for path in sorted(games_dir.glob("*-vs-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        size = payload["board_size"]
        for game in payload["games"]:
            state = initial_state(size)
            for action in game["opening"]:
                state = rules.apply(state, action)
            for ply, record in enumerate(game["plies"]):
                empties = size * size - state.disc_count()
                if MIN_EMPTIES <= empties <= MAX_EMPTIES and len(rules.legal_actions(state)) >= 2:
                    candidates.setdefault(pack(state), f"{path.stem}:{game['index']}:{ply}")
                state = rules.apply(state, record["action"])
    if len(candidates) < count:
        msg = f"only {len(candidates)} usable positions under {games_dir}, asked for {count}"
        raise SystemExit(msg)
    keys = sorted(candidates)
    chosen = make_rng(seed).choice(len(keys), size=count, replace=False)
    out = []
    for index in sorted(chosen.tolist()):
        black, white, to_move = keys[index]
        out.append(
            {
                "black": f"0x{black:016x}",
                "white": f"0x{white:016x}",
                "to_move": Player(to_move).label,
                "empties": 64 - popcount(black | white),
                "source": candidates[keys[index]],
            }
        )
    return out


def read_positions(payload: dict[str, Any]) -> list[Packed]:
    return [
        (int(p["black"], 16), int(p["white"], 16), int(Player.from_label(p["to_move"])))
        for p in payload["positions"]
    ]


# ---------------------------------------------------------------------------
# One agent, every position
# ---------------------------------------------------------------------------


def moves_of(text: str, positions: list[Packed], seed: int) -> tuple[str, list[list[int]], float]:
    """The set of moves ``text``'s agent might play in each position.

    Agents that expose their tie set (``choices``) report it exactly. Agents
    whose randomness is not a tie-break (random playouts, noisy Greedy) are
    sampled once, with a seed derived from the position, so a re-run gives the
    same answer. Random reports every legal move, which makes its row the
    chance level.
    """
    spec = parse_entrant(text, default_simulations=50)
    if spec.needs_network:
        import torch

        torch.set_num_threads(1)
    agent = build_agent(spec)
    started = time.perf_counter()
    out: list[list[int]] = []
    for index, packed in enumerate(positions):
        state = unpack(packed)
        choices = getattr(agent, "choices", None)
        if spec.kind == "random":
            out.append(rules.legal_actions(state))
        elif choices is not None:
            out.append(sorted(choices(state)))
        else:
            rng = make_rng(derive_seed(seed, "similarity", spec.name, index))
            out.append([int(agent.select(state, rng))])
    close = getattr(agent, "close", None)
    if close is not None:
        close()
    return spec.name, out, time.perf_counter() - started


def agreement(candidate: list[list[int]], reference: list[list[int]]) -> float:
    """Mean over positions of the share of the candidate's moves the reference could play."""
    shares = [len(set(a) & set(b)) / len(a) for a, b in zip(candidate, reference, strict=True)]
    return float(np.mean(shares))


def thresholds(matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per training opponent: the two reference groups' edges and the midpoint between them."""
    out: dict[str, dict[str, float]] = {}
    for seen in SEEN:
        same = [matrix[c][seen] for c in SAME_SOURCE[seen] if c in matrix]
        other = [matrix[c][seen] for c in DIFFERENT_SOURCE if c in matrix]
        if not same or not other:
            msg = f"the reference opponents for {seen} are missing from the run"
            raise SystemExit(msg)
        low, high = min(same), max(other)
        out[seen] = {
            "lowest_same_source": low,
            "highest_different_source": high,
            "threshold": (low + high) / 2,
            "chance": matrix["random"][seen],
        }
    return out


def verdicts(matrix: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limits = thresholds(matrix)
    rows = []
    for name in matrix:
        if name in SEEN:
            continue
        with_seen = {ref: matrix[name][ref] for ref in SEEN}
        below = {ref: with_seen[ref] <= limits[ref]["threshold"] for ref in SEEN}
        rows.append(
            {
                "candidate": name,
                "agreement_with": with_seen,
                "chance_level": {ref: limits[ref]["chance"] for ref in SEEN},
                "passes_check": all(below.values()),
                "fails_against": [ref for ref in SEEN if not below[ref]],
            }
        )
    return rows, limits


def report(names: list[str], matrix: dict[str, dict[str, float]]) -> None:
    rows, limits = verdicts(matrix)
    print()
    print(
        "| training opponent | chance | lowest same-source | highest different-source | threshold |"
    )
    print("|---|---:|---:|---:|---:|")
    for seen, limit in limits.items():
        print(
            f"| {seen} | {limit['chance']:.1%} | {limit['lowest_same_source']:.1%} "
            f"| {limit['highest_different_source']:.1%} | **{limit['threshold']:.1%}** |"
        )
    print()
    print("| candidate | agrees with greedy (chance) | agrees with minimax-d4 (chance) | check |")
    print("|---|---:|---:|---|")
    for row in rows:
        cells = [
            f"{row['agreement_with'][ref]:.1%} ({row['chance_level'][ref]:.1%})" for ref in SEEN
        ]
        verdict = "passes" if row["passes_check"] else "fails: " + ", ".join(row["fails_against"])
        print(f"| {row['candidate']} | {cells[0]} | {cells[1]} | {verdict} |")
    print()
    print("agreement matrix, row = candidate, column = reference:")
    print()
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for a in names:
        print(f"| {a} | " + " | ".join(f"{matrix[a][b]:.0%}" for b in names) + " |")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("games_dir", type=Path)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--reanalyse",
        action="store_true",
        help="Recompute the verdicts from the moves already saved in --out; plays nothing.",
    )
    args = parser.parse_args()

    if args.reanalyse:
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        moves: dict[str, list[list[int]]] = saved["moves"]
        names = list(moves)
        matrix = {a: {b: agreement(moves[a], moves[b]) for b in names} for a in names}
        rows, limits = verdicts(matrix)
        saved.update({"agreement": matrix, "thresholds": limits, "verdicts": rows, "rule": RULE})
        saved.pop("baseline", None)
        atomic_write_json(args.out, saved)
        report(names, matrix)
        return 0

    if args.positions.is_file():
        payload = json.loads(args.positions.read_text(encoding="utf-8"))
        print(f"using the {len(payload['positions'])} saved positions in {args.positions}")
    else:
        payload = {
            "purpose": "research/held-out-opponents.md section 5, step 1",
            "sampled_from": str(args.games_dir),
            "rule": f"{MIN_EMPTIES} to {MAX_EMPTIES} empty squares, at least two legal moves, "
            "distinct, chosen uniformly at random with the seed",
            "seed": args.seed,
            "positions": sample_positions(args.games_dir, count=args.count, seed=args.seed),
        }
        atomic_write_json(args.positions, payload)
        print(f"sampled {len(payload['positions'])} positions into {args.positions}")
    positions = read_positions(payload)

    for name in SEEN:
        if name not in args.agents:
            args.agents.append(name)

    moves: dict[str, list[list[int]]] = {}
    specs: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(moves_of, text, positions, args.seed) for text in args.agents]
        for text, future in zip(args.agents, futures, strict=True):
            name, chosen, seconds = future.result()
            moves[name] = chosen
            specs[name] = describe_entrant(parse_entrant(text, default_simulations=50))
            print(f"{name:14s} answered {len(chosen)} positions in {seconds:6.0f}s", flush=True)

    names = list(moves)
    matrix = {a: {b: agreement(moves[a], moves[b]) for b in names} for a in names}
    rows, limits = verdicts(matrix)

    atomic_write_json(
        args.out,
        {
            "purpose": "research/held-out-opponents.md section 5: the admission check",
            "positions_file": str(args.positions),
            "positions": len(positions),
            "seed": args.seed,
            "definition": "agreement(candidate, reference) = mean over positions of "
            "|candidate moves & reference moves| / |candidate moves|",
            "seen": list(SEEN),
            "rule": RULE,
            "thresholds": limits,
            "agent_specs": specs,
            "agreement": matrix,
            "verdicts": rows,
            "moves": moves,
        },
    )
    report(names, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
