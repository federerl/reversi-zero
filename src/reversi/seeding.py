"""Deterministic seed derivation.

One ``seed`` per run; everything else is *derived* from it, so a run is fully
described by that single integer plus its config.

Derivation uses BLAKE2b rather than the builtin ``hash()``: ``hash()`` of a str
or tuple containing a str is randomised per interpreter process unless
``PYTHONHASHSEED`` is pinned, which would make worker seeds differ between the
parent and a spawned child on the same run. BLAKE2b is stable across processes,
platforms, and Python versions.

Scope of determinism (documented non-goal, see docs/architecture.md):

* **Training** is NOT bitwise reproducible across machines -- cuDNN autotuning,
  atomic accumulation order, and process scheduling all vary. Seeds make it
  *statistically* reproducible, not identical.
* **Evaluation** IS reproducible: the arena runs single-process with
  ``dirichlet_eps=0``, ``tau=0``, fixed opening seeds, and deterministic algos.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.random import Generator

_MAX_SEED = 2**32


def derive_seed(root_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from a root seed and arbitrary labels.

    >>> derive_seed(1337, "selfplay", 3, 0) == derive_seed(1337, "selfplay", 3, 0)
    True
    """
    payload = "|".join([str(root_seed), *(str(p) for p in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _MAX_SEED


def worker_seed(root_seed: int, generation: int, worker_id: int) -> int:
    """Seed for one self-play worker process in one generation."""
    return derive_seed(root_seed, "worker", generation, worker_id)


def game_seed(root_seed: int, generation: int, worker_id: int, game_index: int) -> int:
    """Seed for a single self-play game."""
    return derive_seed(root_seed, "game", generation, worker_id, game_index)


def matchup_seed(root_seed: int, matchup_id: str) -> int:
    """Seed for one arena matchup's opening book and tie-breaking."""
    return derive_seed(root_seed, "matchup", matchup_id)


def rng(seed: int) -> Generator:
    """A fresh NumPy Generator. Prefer this over the global numpy random state."""
    return np.random.default_rng(seed)


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and (if importable) torch.

    ``deterministic_torch`` is for the evaluation path only: it trades throughput
    for exact reproducibility and must not be enabled during training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Legacy global state: seeded only so third-party code that reaches for it
    # is reproducible. Our own code uses explicit Generators via `rng()`.
    np.random.seed(seed % _MAX_SEED)

    try:
        import torch
    except ImportError:  # pragma: no cover - pure-CPU game/search layers
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
