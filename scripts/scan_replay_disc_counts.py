"""How often did self-play ever put a side down to a handful of discs?

Step 2, question 2 of ``research/step1-pilot-results.md``. The replay window
is the set of self-play positions the trainer learned from; each stored
position has both sides' discs as 64-bit boards. This script counts how many
of those positions have a side holding very few discs, overall and by stage
of the game, and what outcome those positions were labelled with. If such
positions are nearly absent, the network had little chance to learn that a
tiny disc count is dangerous.

Usage::

    python scripts/scan_replay_disc_counts.py runs/<run>/replay

Only the window that is still on disk is scanned (the trainer deletes older
shards), so the counts describe what the last generations trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THRESHOLDS = (1, 2, 3, 5, 8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits in each uint64."""
    bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1)
    return bits.sum(axis=1)


def main(replay_dir: Path) -> int:
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz shards under {replay_dir}")
    black, white, to_move, move_no, z, generation = [], [], [], [], [], []
    for f in files:
        with np.load(f) as shard:
            black.append(popcount64(shard["black"]))
            white.append(popcount64(shard["white"]))
            to_move.append(shard["to_move"])
            move_no.append(shard["move_no"])
            z.append(shard["z"])
            generation.append(shard["generation"])
    b = np.concatenate(black)
    w = np.concatenate(white)
    tm = np.concatenate(to_move)
    mv = np.concatenate(move_no)
    zz = np.concatenate(z)
    gen = np.concatenate(generation)
    n = len(b)
    fewest = np.minimum(b, w)
    mover = np.where(tm == 0, b, w)  # to_move: 0 is black, 1 is white
    games = int(np.sum(mv == 0))
    print(
        f"{n:,} positions in {len(files)} shards, generations {gen.min()} to {gen.max()}; "
        f"about {games:,} games"
    )
    print()

    # The first few plies of every game pass through tiny disc counts by rule
    # (after black's first move, white holds exactly one disc), so they say
    # nothing about danger. The pilot's wipeouts happened from ply 9 onwards
    # (4 opening plies, then 5 or more moves); count from ply 8.
    print("## Positions from ply 8 onwards where a side holds few discs")
    print()
    later = mv >= 8
    print(f"{later.sum():,} positions from ply 8 on, {later.mean():.1%} of the window")
    print()
    print(
        "| a side holds at most | positions | share of those positions | per game "
        "| of which the side to move |"
    )
    print("|---:|---:|---:|---:|---:|")
    for t in THRESHOLDS:
        mask = later & (fewest <= t)
        print(
            f"| {t} | {mask.sum():,} | {mask.sum() / later.sum():.4%} | {mask.sum() / games:.3f} "
            f"| {np.sum(later & (mover <= t)):,} |"
        )
    print()

    print("## Where in the game those positions sit (a side holds at most 3, ply 8 onwards)")
    print()
    mask = later & (fewest <= 3)
    print("| plies | positions |")
    print("|---|---:|")
    for lo, hi in ((8, 15), (16, 23), (24, 31), (32, 39), (40, 61)):
        print(f"| {lo} to {hi} | {np.sum(mask & (mv >= lo) & (mv <= hi)):,} |")
    print()

    print("## What the value target said when the side to move held few discs (ply 8 onwards)")
    print()
    print(
        "`z` is the game's final outcome from the mover's view, +1 / 0 / -1. "
        "If holding few discs were dangerous in self-play, these rows would sit well below zero."
    )
    print()
    print("| mover holds at most | positions | mean z | share lost | share won |")
    print("|---:|---:|---:|---:|---:|")
    for t in THRESHOLDS:
        mask = later & (mover <= t)
        if not mask.any():
            print(f"| {t} | 0 | | | |")
            continue
        lost = np.mean(zz[mask] <= -0.999)
        won = np.mean(zz[mask] >= 0.999)
        print(f"| {t} | {mask.sum():,} | {zz[mask].mean():+.2f} | {lost:.0%} | {won:.0%} |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main(Path(sys.argv[1])))
