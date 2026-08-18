"""Atomic file writes.

Every artifact that a resume depends on -- checkpoints, replay shards, the shard
manifest, run metadata -- is written to a temporary file in the *same directory*
and then moved into place with ``os.replace``, which is atomic on both POSIX and
Windows for same-filesystem renames.

The failure this prevents: a job killed mid-write leaves a truncated file that
loads without error but contains garbage, silently poisoning a resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically."""
    _atomic(path, lambda tmp: tmp.write_bytes(data))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (always UTF-8, always LF-agnostic)."""
    _atomic(path, lambda tmp: tmp.write_text(text, encoding=encoding))


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialise ``payload`` as JSON and write it atomically."""
    text = json.dumps(payload, indent=indent, sort_keys=True, default=str) + "\n"
    atomic_write_text(path, text)


def atomic_write_with(path: Path, writer: Callable[[Path], object]) -> None:
    """Run ``writer(tmp_path)`` then move the result onto ``path`` atomically.

    ``writer`` may return anything (``Path.write_bytes`` returns an ``int``); the
    return value is ignored.

    Use this for binary formats with their own serialiser (``torch.save``,
    ``numpy.savez_compressed``) that need a path rather than a buffer.
    """
    _atomic(path, writer)


def _atomic(path: Path, writer: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        # Same-directory rename: atomic on POSIX and on Windows.
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, used for checkpoint and shard integrity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
