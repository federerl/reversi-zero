"""Seed derivation (backlog T03).

One ``seed`` per run; everything else is derived. The critical property is
cross-process stability: a worker spawned by the parent must derive the same
seed the parent expects, which rules out the builtin ``hash()`` (randomised per
process unless PYTHONHASHSEED is pinned).
"""

from __future__ import annotations

import os
import subprocess
import sys

from reversi.seeding import derive_seed, game_seed, matchup_seed, rng, worker_seed


def test_derivation_is_deterministic() -> None:
    assert derive_seed(1337, "worker", 3, 0) == derive_seed(1337, "worker", 3, 0)


def test_derivation_is_sensitive_to_every_component() -> None:
    base = derive_seed(1337, "worker", 3, 0)
    assert derive_seed(1338, "worker", 3, 0) != base
    assert derive_seed(1337, "game", 3, 0) != base
    assert derive_seed(1337, "worker", 4, 0) != base
    assert derive_seed(1337, "worker", 3, 1) != base


def test_seeds_fit_in_32_bits() -> None:
    for i in range(200):
        value = derive_seed(1337, "worker", 0, i)
        assert 0 <= value < 2**32


def test_worker_seeds_are_distinct_within_a_generation() -> None:
    seeds = {worker_seed(1337, 5, w) for w in range(64)}
    assert len(seeds) == 64


def test_game_seeds_are_distinct_across_workers_and_indices() -> None:
    seeds = {game_seed(1337, 2, w, g) for w in range(8) for g in range(64)}
    assert len(seeds) == 8 * 64


def test_matchup_seeds_are_distinct() -> None:
    assert matchup_seed(1, "gen10-vs-greedy") != matchup_seed(1, "gen10-vs-random")


def test_derivation_is_stable_across_processes() -> None:
    """Rules out hash() randomisation: a spawned worker must agree with its parent."""
    code = (
        "from reversi.seeding import derive_seed;"
        "print(derive_seed(1337, 'worker', 3, 0))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert int(result.stdout.strip()) == derive_seed(1337, "worker", 3, 0)


def test_rng_is_reproducible() -> None:
    a = rng(derive_seed(1337, "game", 0, 0)).integers(0, 1000, size=20)
    b = rng(derive_seed(1337, "game", 0, 0)).integers(0, 1000, size=20)
    assert list(a) == list(b)


def test_different_games_get_different_streams() -> None:
    a = rng(game_seed(1337, 0, 0, 0)).integers(0, 10_000, size=20)
    b = rng(game_seed(1337, 0, 0, 1)).integers(0, 10_000, size=20)
    assert list(a) != list(b)
