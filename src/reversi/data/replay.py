"""The sliding window of recent games that the trainer learns from.

**Why a window and not everything ever played.** The agent is chasing a moving
target: games from generation 3 were played by a much weaker player than the one
we are training now, and their policy targets say things that are no longer true.
Keeping them forever drags the network back toward old habits. Throwing them away
immediately is worse -- training on only the newest generation makes the network
oscillate, chasing whatever quirk the last few thousand games happened to
contain. A window of a few hundred thousand positions is the compromise, and its
size is one of the highest-leverage numbers in the whole configuration.

**The per-generation cap.** If one generation ran fast and contributed twice as
many positions as the others, uniform sampling would let it dominate the window
in proportion. The cap limits how much of a batch any single generation may
supply, so the window stays a mixture of recent play rather than a snapshot of
one lucky afternoon.

**Augmentation happens here, at sampling time, not on the way in.** A Reversi
position has eight equivalent orientations. We could store all eight, but that
would be eight times the disk for no new information. Instead each sample gets a
random orientation as it is drawn -- so the same stored position is shown in a
different orientation each time it is sampled, which is strictly better variety
than storing all eight once and sampling those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reversi.data.schema import FIELDS, Arrays
from reversi.data.shards import Manifest, read_shard
from reversi.errors import ReplayError
from reversi.game import symmetry
from reversi.game.state import State
from reversi.nn import features
from reversi.types import Player, policy_size

__all__ = ["Batch", "ReplayBuffer"]


@dataclass(frozen=True, slots=True)
class Batch:
    """One training batch, ready for the network."""

    planes: NDArray[np.float32]
    """``(batch, 3, size, size)`` -- what the network reads."""
    pi: NDArray[np.float32]
    """``(batch, size**2 + 1)`` -- what the policy head should say."""
    z: NDArray[np.float32]
    """``(batch,)`` -- what the value head should say."""

    def __len__(self) -> int:
        return len(self.z)


class ReplayBuffer:
    """Recent positions, in memory, with sampling that mixes generations fairly."""

    def __init__(
        self,
        *,
        board_size: int,
        window: int,
        per_gen_cap_factor: float = 2.0,
        symmetry_aug: bool = True,
    ) -> None:
        self.board_size = board_size
        self.window = window
        self.per_gen_cap_factor = per_gen_cap_factor
        self.symmetry_aug = symmetry_aug

        self._columns: Arrays = {}
        self._by_generation: dict[int, NDArray[np.int64]] = {}

        # For each of the eight orientations, the index array that reorders a
        # policy vector to match. Precomputed because it is used once per sample.
        n_squares = board_size * board_size
        self._policy_gather = tuple(
            np.array([*list(sym.inverse), n_squares], dtype=np.int64)
            for sym in symmetry.symmetries(board_size)
        )

    # -----------------------------------------------------------------
    # Filling it
    # -----------------------------------------------------------------

    def load_from(self, manifest: Manifest) -> int:
        """Rebuild the window from the shards on disk, newest first.

        Reads backwards through the manifest and stops once the window is full,
        so a run with thirty shards on disk and a window of two generations only
        touches two files.
        """
        self._columns = {}
        blocks: list[Arrays] = []
        total = 0

        for shard in reversed(manifest.shards):
            if total >= self.window:
                break
            arrays = read_shard(
                manifest.file(shard),
                board_size=self.board_size,
                expect_sha256=shard.sha256,
            )
            blocks.append(arrays)
            total += len(arrays["black"])

        for arrays in reversed(blocks):  # back into chronological order
            self._append(arrays)

        self._trim()
        self._reindex()
        return len(self)

    def add(self, arrays: Arrays) -> None:
        """Add a freshly produced generation without re-reading it from disk."""
        self._append(arrays)
        self._trim()
        self._reindex()

    def _append(self, arrays: Arrays) -> None:
        if not self._columns:
            self._columns = {field: np.asarray(arrays[field]).copy() for field in FIELDS}
            return
        for field in FIELDS:
            self._columns[field] = np.concatenate((self._columns[field], np.asarray(arrays[field])))

    def _trim(self) -> None:
        """Drop the oldest positions once the window is full."""
        size = len(self)
        if size <= self.window:
            return
        keep = slice(size - self.window, size)
        self._columns = {field: self._columns[field][keep] for field in FIELDS}

    def _reindex(self) -> None:
        if not self._columns:
            self._by_generation = {}
            return
        generations = np.asarray(self._columns["generation"], dtype=np.int64)
        self._by_generation = {
            int(generation): np.flatnonzero(generations == generation)
            for generation in np.unique(generations)
        }

    # -----------------------------------------------------------------
    # Reading it
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return 0 if not self._columns else len(self._columns["black"])

    @property
    def generations(self) -> list[int]:
        return sorted(self._by_generation)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Draw a batch: pick a generation, then a position inside it, then turn it."""
        size = len(self)
        if size == 0:
            msg = "the replay buffer is empty; nothing to sample"
            raise ReplayError(msg)
        if batch_size < 1:
            msg = f"batch_size must be at least 1, got {batch_size}"
            raise ReplayError(msg)

        chosen = self._choose_indices(batch_size, rng)
        return self._build_batch(chosen, rng)

    def _choose_indices(self, batch_size: int, rng: np.random.Generator) -> NDArray[np.int64]:
        """Sample row indices, respecting the per-generation cap."""
        generations = self.generations
        if len(generations) == 1:
            only = self._by_generation[generations[0]]
            return only[rng.integers(0, len(only), size=batch_size)]

        weights = self._generation_weights()
        picked_generations = rng.choice(len(generations), size=batch_size, p=weights)
        out = np.empty(batch_size, dtype=np.int64)
        for row, which in enumerate(picked_generations):
            rows = self._by_generation[generations[int(which)]]
            out[row] = rows[rng.integers(0, len(rows))]
        return out

    def _generation_weights(self) -> NDArray[np.float64]:
        """How likely each generation is to supply a sample, respecting the cap.

        Capping is not as simple as clipping the shares and renormalising: after
        a clip, renormalising scales everything back up, which pushes the clipped
        generation straight over the line again. A generation with ten times the
        positions of its neighbours still ended up supplying 85% of every batch
        that way.

        So instead the over-cap generations are *fixed* at the cap, and the
        probability left over is shared among the rest in proportion to their
        sizes -- repeatedly, since giving the small generations more mass can push
        one of them over the cap in turn.
        """
        generations = self.generations
        count = len(generations)
        sizes = np.array([len(self._by_generation[g]) for g in generations], dtype=np.float64)

        # A cap below the fair share is a contradiction rather than a cap: every
        # generation would be limited to less than an equal portion, and the
        # shares could not add up to one. So it never goes below 1/n.
        cap = max(self.per_gen_cap_factor / count, 1.0 / count)
        if cap >= 1.0:
            return sizes / sizes.sum()

        weights = np.zeros(count, dtype=np.float64)
        free = np.ones(count, dtype=bool)
        remaining = 1.0

        while free.any() and remaining > 1e-12:
            pool = sizes[free]
            allocation = remaining * pool / pool.sum()
            over = allocation > cap
            if not over.any():
                weights[free] = allocation
                break
            capped = np.flatnonzero(free)[over]
            weights[capped] = cap
            remaining -= cap * len(capped)
            free[capped] = False

        total = weights.sum()
        return weights / total if total > 0 else sizes / sizes.sum()

    def _build_batch(self, indices: NDArray[np.int64], rng: np.random.Generator) -> Batch:
        size = self.board_size
        width = policy_size(size)
        turns = symmetry.symmetries(size)

        planes = np.empty((len(indices), features.IN_PLANES, size, size), dtype=np.float32)
        pi = np.empty((len(indices), width), dtype=np.float32)
        z = np.empty(len(indices), dtype=np.float32)

        which_turn = (
            rng.integers(0, len(turns), size=len(indices))
            if self.symmetry_aug
            else np.zeros(len(indices), dtype=np.int64)
        )

        for row, index in enumerate(indices):
            state = State(
                black=int(self._columns["black"][index]),
                white=int(self._columns["white"][index]),
                to_move=Player(int(self._columns["to_move"][index])),
                size=size,
            )
            target = np.asarray(self._columns["pi"][index], dtype=np.float32)

            turn = int(which_turn[row])
            if turn != 0:
                state = symmetry.transform_state(state, turns[turn])
                target = target[self._policy_gather[turn]]

            planes[row] = features.encode(state)
            pi[row] = target
            z[row] = self._columns["z"][index]

        return Batch(planes=planes, pi=pi, z=z)

    # -----------------------------------------------------------------

    def stats(self, current_generation: int) -> dict[str, Any]:
        """Numbers worth watching every generation.

        ``age`` is how stale the window is: if the mean age creeps up, the trainer
        is spending its time on games played by a noticeably weaker version of
        itself. ``unique_fraction`` catches the opposite failure -- self-play
        collapsing onto the same handful of positions, which is what happens when
        the exploration noise or the opening temperature is switched off by
        accident.
        """
        size = len(self)
        if size == 0:
            return {"buffer_size": 0}

        generations = np.asarray(self._columns["generation"], dtype=np.int64)
        ages = current_generation - generations

        positions = np.stack(
            (
                np.asarray(self._columns["black"], dtype=np.uint64),
                np.asarray(self._columns["white"], dtype=np.uint64),
                np.asarray(self._columns["to_move"], dtype=np.uint64),
            ),
            axis=1,
        )
        unique = len(np.unique(positions, axis=0))

        return {
            "buffer_size": size,
            "buffer_generations": len(self._by_generation),
            "buffer_age_mean": float(ages.mean()),
            "buffer_age_p95": float(np.percentile(ages, 95)),
            "unique_positions_fraction": unique / size,
        }
