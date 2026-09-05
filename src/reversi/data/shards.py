"""One file per batch of games, and a manifest that knows what is on disk.

**Why many small files instead of one big one.** A training run is a sequence of
interruptions: an eight-hour job limit, a preemption, a crash, a Ctrl-C. If all
the collected games lived in one file that gets rewritten every generation, an
interruption mid-write destroys the whole history. With one immutable file per
generation, the worst case is losing the file currently being written -- one
generation out of thirty, which the sliding window absorbs without noticing.

Every write goes to a temporary file first and is then moved into place, so a
half-written shard never exists under its real name. Every shard's SHA-256 goes
in the manifest, and resume drops any shard whose contents no longer match. That
combination is what makes "we were killed at an arbitrary moment" a survivable
event rather than a debugging session.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reversi.atomicio import atomic_write_json, atomic_write_with, sha256_file
from reversi.data.schema import FIELDS, OPTIONAL_FIELDS, Arrays, validate_arrays
from reversi.errors import ReplayError

__all__ = ["Manifest", "ShardInfo", "read_shard", "shard_filename", "write_shard"]

MANIFEST_NAME = "manifest.json"


def shard_filename(generation: int, worker_id: int = 0) -> str:
    """``gen_00007_w03.npz`` -- zero-padded so the files sort chronologically."""
    return f"gen_{generation:05d}_w{worker_id:02d}.npz"


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """What the manifest records about one shard file."""

    filename: str
    n_positions: int
    generation: int
    sha256: str

    @property
    def worker_id(self) -> int:
        return int(self.filename.split("_w")[-1].removesuffix(".npz"))


def write_shard(
    path: Path,
    arrays: Arrays,
    *,
    board_size: int,
) -> ShardInfo:
    """Validate, compress, and write one shard atomically. Returns its manifest entry."""
    validate_arrays(arrays, board_size=board_size)

    def save(tmp: Path) -> None:
        # Write through an open handle rather than passing the path: numpy's
        # savez helpfully appends ".npz" to any filename that lacks it, which
        # would leave the real data next to the temporary name we are about to
        # rename, and rename an empty file into place.
        # Typed loosely on purpose: savez_compressed takes the array names as
        # keyword arguments, which a checker cannot tell apart from its own
        # keyword parameters.
        columns: dict[str, Any] = dict(arrays)
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **columns)

    atomic_write_with(path, save)

    return ShardInfo(
        filename=path.name,
        n_positions=len(arrays["black"]),
        generation=int(arrays["generation"][0]),
        sha256=sha256_file(path),
    )


def read_shard(
    path: Path,
    *,
    board_size: int,
    expect_sha256: str | None = None,
) -> Arrays:
    """Load one shard, checking it is the file the manifest remembers.

    The checksum is the part that matters. A truncated or corrupted shard usually
    still *loads* -- numpy will happily hand back whatever arrays it can parse --
    so without this the damage shows up as slightly wrong training data rather
    than as an error.
    """
    if not path.exists():
        msg = f"shard {path} is missing"
        raise ReplayError(msg)

    if expect_sha256 is not None:
        actual = sha256_file(path)
        if actual != expect_sha256:
            msg = (
                f"shard {path.name} does not match its manifest checksum "
                f"(expected {expect_sha256[:12]}..., found {actual[:12]}...). "
                "It was truncated, overwritten, or written by a different run."
            )
            raise ReplayError(msg)

    try:
        with np.load(path) as loaded:
            arrays = {
                field: loaded[field] for field in (*FIELDS, *OPTIONAL_FIELDS) if field in loaded
            }
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        msg = f"shard {path.name} could not be read: {error}"
        raise ReplayError(msg) from error

    validate_arrays(arrays, board_size=board_size)
    return arrays


@dataclass(slots=True)
class Manifest:
    """The list of shards on disk, in the order they were produced.

    Written atomically after every change, so it is never inconsistent with the
    files it describes -- at worst it is one shard behind, which resume handles by
    ignoring files the manifest has not heard of.
    """

    directory: Path
    shards: list[ShardInfo]

    @classmethod
    def load(cls, directory: Path) -> Manifest:
        """Read the manifest, or start an empty one if this is a fresh run."""
        path = directory / MANIFEST_NAME
        if not path.exists():
            return cls(directory=directory, shards=[])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            shards = [ShardInfo(**entry) for entry in payload["shards"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            msg = f"replay manifest at {path} is unreadable: {error}"
            raise ReplayError(msg) from error
        return cls(directory=directory, shards=shards)

    @property
    def path(self) -> Path:
        return self.directory / MANIFEST_NAME

    @property
    def total_positions(self) -> int:
        return sum(shard.n_positions for shard in self.shards)

    @property
    def generations(self) -> list[int]:
        return sorted({shard.generation for shard in self.shards})

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "version": 1,
                "total_positions": self.total_positions,
                "shards": [asdict(shard) for shard in self.shards],
            },
        )

    def add(self, shard: ShardInfo) -> None:
        self.shards = [s for s in self.shards if s.filename != shard.filename]
        self.shards.append(shard)
        self.shards.sort(key=lambda s: (s.generation, s.filename))
        self.save()

    def file(self, shard: ShardInfo) -> Path:
        return self.directory / shard.filename

    def verify(self) -> list[ShardInfo]:
        """Drop shards that are missing or no longer match their checksum.

        Called on resume. Returns what was dropped, so the caller can say so out
        loud rather than quietly training on less data than it thinks it has.
        """
        good: list[ShardInfo] = []
        bad: list[ShardInfo] = []
        for shard in self.shards:
            path = self.file(shard)
            if path.exists() and sha256_file(path) == shard.sha256:
                good.append(shard)
            else:
                bad.append(shard)
        if bad:
            self.shards = good
            self.save()
        return bad

    def prune(self, keep: int) -> list[ShardInfo]:
        """Delete all but the newest ``keep`` shards. Returns what was removed.

        Replay shards are the bulk of a run's disk usage -- tens to hundreds of
        megabytes per generation -- and anything older than the sampling window is
        dead weight.
        """
        if keep >= len(self.shards):
            return []

        removed = self.shards[: len(self.shards) - keep]
        self.shards = self.shards[len(self.shards) - keep :]
        for shard in removed:
            self.file(shard).unlink(missing_ok=True)
        self.save()
        return removed
