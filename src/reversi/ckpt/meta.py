"""What a checkpoint is, described in plain JSON next to the weights.

Every ``gen_00007.pt`` has a ``gen_00007.json`` beside it holding the same
metadata. That duplication is deliberate and it buys three things:

**The checksum has somewhere to live.** A checkpoint cannot contain its own
hash -- writing the hash in would change the hash. The sidecar can, which is what
lets a resume verify that the file it is about to trust has not been truncated
by the interruption that made a resume necessary in the first place.

**Lineage is readable without torch.** ``grep generation runs/*/checkpoints/*.json``
answers "what happened in this run?" on a machine with no GPU, no virtualenv, and
no patience. Answering the same question from the ``.pt`` files means loading
each one.

**Architecture mismatches fail with a sentence.** The ``arch`` block says what
shape the weights are. Loading checks it first, so a stale checkpoint produces a
message naming what differs rather than a shape error deep inside torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reversi.errors import CheckpointError

__all__ = ["FORMAT_VERSION", "CheckpointMeta", "checkpoint_name", "sidecar_for"]

# 0 was day 5's weights-only stub. 1 is the first format that can be resumed
# from: it carries optimiser state, RNG state, and lineage.
FORMAT_VERSION = 1


def checkpoint_name(generation: int) -> str:
    return f"gen_{generation:05d}.pt"


def sidecar_for(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """Everything about a checkpoint except the weights themselves."""

    format_version: int
    run_id: str
    generation: int
    global_step: int
    arch: dict[str, Any]
    config_sha256: str
    created_utc: str
    sha256: str
    """Of the ``.pt`` file. Checked before the file is trusted on resume."""

    parent: str | None = None
    """Filename of the checkpoint this one was trained from. Following the chain
    backwards reconstructs the whole history of a run, including across restarts
    where the generation numbers alone would not tell you what came from what."""

    games_played: int = 0
    positions_seen: int = 0
    replay_manifest_sha256: str | None = None
    elo_estimate: float | None = None
    git: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CheckpointMeta:
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            # A newer version of this project wrote it. Refusing is safer than
            # loading half of it and behaving as though the rest was not there.
            msg = (
                f"checkpoint metadata has unexpected fields {sorted(unknown)}; "
                "it was probably written by a newer version of reversi-zero"
            )
            raise CheckpointError(msg)
        missing = (known - set(payload)) - {
            "parent",
            "games_played",
            "positions_seen",
            "replay_manifest_sha256",
            "elo_estimate",
            "git",
            "env",
        }
        if missing:
            msg = f"checkpoint metadata is missing required fields {sorted(missing)}"
            raise CheckpointError(msg)
        return cls(**payload)

    @classmethod
    def read(cls, path: Path) -> CheckpointMeta:
        try:
            return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
        except OSError as error:
            msg = f"checkpoint metadata {path} could not be read: {error}"
            raise CheckpointError(msg) from error
        except json.JSONDecodeError as error:
            msg = f"checkpoint metadata {path} is not valid JSON: {error}"
            raise CheckpointError(msg) from error

    def describe(self) -> str:
        return (
            f"generation {self.generation} (step {self.global_step}, "
            f"{self.games_played} games, {self.positions_seen} positions)"
        )
