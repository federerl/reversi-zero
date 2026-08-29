"""Turning a training checkpoint into something the web app can serve.

A training checkpoint carries the optimiser state, the RNG state, the replay
manifest hash, the git commit, the environment -- everything needed to *resume*.
An export carries only what is needed to *play*: the weights and the architecture
that shapes them.

**Why that separation is worth a module.** A training checkpoint here is about
9 MB, most of it optimiser state that is meaningless outside the run that made
it. An export is under 2 MB and is a plain artifact: no run directory, no
lineage, nothing that only makes sense in context. That is what gets attached to
a release and downloaded by anyone who wants to play against the agent.

It also cuts a dependency. The web API loads an export and never touches
``reversi.train`` or ``reversi.data`` -- so shipping the app does not ship the
training pipeline, and an accident cannot wire the two together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from reversi.atomicio import atomic_write_with, sha256_file
from reversi.ckpt.manager import CheckpointManager
from reversi.errors import CheckpointError
from reversi.nn.model import PolicyValueNet

__all__ = ["EXPORT_VERSION", "ExportedModel", "export_checkpoint", "load_export"]

EXPORT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExportedModel:
    """A network ready to play, plus what it came from."""

    model: PolicyValueNet
    meta: dict[str, Any]

    @property
    def board_size(self) -> int:
        return self.model.board_size

    @property
    def label(self) -> str:
        """A short human-readable identity, shown in the app's health endpoint."""
        return f"{self.meta.get('run_id', 'unknown')}-gen{self.meta.get('generation', '?')}"


def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    verify: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    """Write a play-only copy of ``checkpoint`` to ``destination``.

    ``verify`` re-checks the source against its sidecar checksum first. Exporting
    a corrupt checkpoint would produce a corrupt artifact that nothing downstream
    could distinguish from a good one -- the app would happily serve nonsense.
    """
    manager = CheckpointManager(checkpoint.parent, run_id="export", config_sha256="")
    if verify:
        manager.verify(checkpoint)
    source_meta = manager.read_meta(checkpoint)
    payload = manager.load(checkpoint)

    arch = payload.get("arch")
    if not isinstance(arch, dict):
        msg = f"{checkpoint.name} has no architecture block; it cannot be exported"
        raise CheckpointError(msg)

    meta = {
        "export_version": EXPORT_VERSION,
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": source_meta.run_id,
        "generation": source_meta.generation,
        "global_step": source_meta.global_step,
        "arch": arch,
        "source_checkpoint": checkpoint.name,
        "source_sha256": source_meta.sha256,
        "config_sha256": source_meta.config_sha256,
        "git": source_meta.git,
        "notes": notes,
    }

    atomic_write_with(
        destination,
        lambda tmp: torch.save(
            {"meta": meta, "model_state_dict": payload["model_state_dict"]}, tmp
        ),
    )

    # The checksum of the export itself goes in a sidecar, so a download can be
    # verified without trusting the transport that delivered it.
    meta["sha256"] = sha256_file(destination)
    destination.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return meta


def load_export(path: Path, *, device: str = "cpu", verify: bool = True) -> ExportedModel:
    """Load an exported model, ready to play with.

    Returned in ``eval()`` mode: every use of a loaded model is inference, and a
    model left in training mode answers differently depending on what it was
    batched with.
    """
    if not path.exists():
        msg = f"exported model {path} does not exist"
        raise CheckpointError(msg)

    sidecar = path.with_suffix(".json")
    if verify and sidecar.exists():
        expected = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
        actual = sha256_file(path)
        if expected and expected != actual:
            msg = (
                f"exported model {path.name} does not match its recorded checksum "
                f"(expected {expected[:12]}..., found {actual[:12]}...). "
                "It was damaged in transit or replaced."
            )
            raise CheckpointError(msg)

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        msg = f"exported model {path.name} could not be read: {error}"
        raise CheckpointError(msg) from error

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        msg = f"{path.name} is not an exported reversi model"
        raise CheckpointError(msg)

    meta = payload.get("meta", {})
    arch = meta.get("arch")
    if not isinstance(arch, dict):
        msg = f"{path.name} carries no architecture block, so its weights cannot be placed"
        raise CheckpointError(msg)

    version = meta.get("export_version", 0)
    if version > EXPORT_VERSION:
        msg = (
            f"{path.name} is export version {version} but this code understands up "
            f"to {EXPORT_VERSION}"
        )
        raise CheckpointError(msg)

    model = PolicyValueNet(
        arch["board_size"],
        n_blocks=arch["n_blocks"],
        channels=arch["channels"],
        value_hidden=arch["value_hidden"],
        in_planes=arch["in_planes"],
    )
    try:
        model.load_state_dict(payload["model_state_dict"])
    except (RuntimeError, KeyError, TypeError) as error:
        msg = (
            f"{path.name} describes {arch} but its weights do not fit that shape: "
            f"{error}. The file and the code that reads it disagree."
        )
        raise CheckpointError(msg) from error

    model.to(device)
    model.eval()
    return ExportedModel(model=model, meta=meta)
