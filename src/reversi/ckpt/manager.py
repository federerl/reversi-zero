"""Writing checkpoints that survive being interrupted, and finding one to resume.

**The ordering is the whole trick.** Each save writes three things: the weights,
then the metadata sidecar carrying the weights' checksum, then a ``latest`` copy.
They go in that order on purpose. If the process dies partway through:

* died before the weights landed -- nothing changed, the previous checkpoint is
  still the newest;
* died between weights and sidecar -- the ``.pt`` exists with no metadata, so
  resume ignores it and falls back to the previous generation;
* died between sidecar and ``latest`` -- the numbered pair is complete and valid,
  and resume searches by generation number rather than trusting ``latest``.

There is no window in which a resume can pick up a half-written checkpoint and
believe it. That matters because the interruption which makes a resume necessary
is exactly the kind of event that leaves half-written files.

**Every individual write is atomic too** -- to a temporary name, then renamed --
so a file never exists under its real name with partial contents.

**What gets saved beyond the weights.** The optimiser state is the piece people
forget: AdamW keeps a running estimate per parameter, and resuming without it
means the first few hundred steps after every restart are taken with the wrong
step sizes. The step counter matters for the same reason -- it is what the
learning-rate schedule reads, so losing it restarts the warmup and the run never
reaches its configured rate.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reversi.atomicio import atomic_write_json, atomic_write_with, sha256_file
from reversi.ckpt.meta import FORMAT_VERSION, CheckpointMeta, checkpoint_name, sidecar_for
from reversi.errors import CheckpointError
from reversi.nn.model import PolicyValueNet
from reversi.obs.runmeta import environment_info, git_info

__all__ = ["CheckpointManager", "RestoredRun"]

log = logging.getLogger(__name__)

LATEST = "latest"
BEST = "best"


class RestoredRun:
    """What a resume recovered, so the caller knows where to carry on from."""

    __slots__ = ("meta", "payload")

    def __init__(self, meta: CheckpointMeta, payload: dict[str, Any]) -> None:
        self.meta = meta
        self.payload = payload

    @property
    def next_generation(self) -> int:
        return self.meta.generation + 1

    @property
    def global_step(self) -> int:
        return self.meta.global_step


class CheckpointManager:
    """Owns one run's checkpoint directory."""

    def __init__(self, directory: Path, *, run_id: str, config_sha256: str) -> None:
        self.directory = directory
        self.run_id = run_id
        self.config_sha256 = config_sha256

    # -----------------------------------------------------------------
    # Writing
    # -----------------------------------------------------------------

    def save(
        self,
        *,
        model: PolicyValueNet,
        generation: int,
        global_step: int,
        optimizer_state: dict[str, Any] | None = None,
        games_played: int = 0,
        positions_seen: int = 0,
        replay_manifest_sha256: str | None = None,
        parent: str | None = None,
        capture_rng: bool = True,
    ) -> CheckpointMeta:
        """Write one checkpoint plus its sidecar, and refresh ``latest``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / checkpoint_name(generation)

        payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "run_id": self.run_id,
            "generation": generation,
            "global_step": global_step,
            "arch": model.arch(),
            "config_sha256": self.config_sha256,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "rng": _capture_rng() if capture_rng else None,
        }

        # 1. the weights
        atomic_write_with(path, lambda tmp: torch.save(payload, tmp))

        # 2. the metadata, including the checksum of what was just written
        meta = CheckpointMeta(
            format_version=FORMAT_VERSION,
            run_id=self.run_id,
            generation=generation,
            global_step=global_step,
            arch=model.arch(),
            config_sha256=self.config_sha256,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            sha256=sha256_file(path),
            parent=parent,
            games_played=games_played,
            positions_seen=positions_seen,
            replay_manifest_sha256=replay_manifest_sha256,
            git=git_info(),
            env=environment_info(),
        )
        atomic_write_json(sidecar_for(path), meta.to_json())

        # 3. the convenience copies. Copies rather than symlinks: symlinks need
        #    elevated privileges on Windows, and this has to run on a laptop as
        #    well as a server.
        self._copy_as(path, meta, LATEST)
        return meta

    def mark_best(self, generation: int) -> None:
        """Promote a generation to ``best`` -- used by the arena from day 9."""
        path = self.directory / checkpoint_name(generation)
        meta = self.read_meta(path)
        self._copy_as(path, meta, BEST)

    def _copy_as(self, source: Path, meta: CheckpointMeta, label: str) -> None:
        target = self.directory / f"{label}.pt"
        atomic_write_with(target, lambda tmp: tmp.write_bytes(source.read_bytes()))
        atomic_write_json(self.directory / f"{label}.json", meta.to_json())

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    def read_meta(self, checkpoint: Path) -> CheckpointMeta:
        sidecar = sidecar_for(checkpoint)
        if not sidecar.exists():
            msg = (
                f"{checkpoint.name} has no metadata sidecar. It was probably being "
                "written when the process stopped, so its contents cannot be trusted."
            )
            raise CheckpointError(msg)
        return CheckpointMeta.read(sidecar)

    def verify(self, checkpoint: Path) -> CheckpointMeta:
        """Load the metadata and confirm the weights still match its checksum."""
        meta = self.read_meta(checkpoint)
        if not checkpoint.exists():
            msg = f"{checkpoint.name} is described by a sidecar but the file is gone"
            raise CheckpointError(msg)

        actual = sha256_file(checkpoint)
        if actual != meta.sha256:
            msg = (
                f"{checkpoint.name} does not match its recorded checksum "
                f"(expected {meta.sha256[:12]}..., found {actual[:12]}...). "
                "It was truncated or overwritten."
            )
            raise CheckpointError(msg)
        return meta

    def generations(self) -> list[int]:
        """Numbered checkpoints present on disk, oldest first."""
        found = []
        for path in self.directory.glob("gen_*.pt"):
            try:
                found.append(int(path.stem.removeprefix("gen_")))
            except ValueError:
                continue
        return sorted(found)

    def newest_valid(self) -> RestoredRun | None:
        """The most recent checkpoint that passes its checks, or None.

        Walks backwards rather than trusting the newest file. A run killed during
        a save leaves exactly one bad checkpoint at the end; falling back one
        generation costs a few minutes of self-play, while trusting it would
        resume from garbage.
        """
        for generation in reversed(self.generations()):
            path = self.directory / checkpoint_name(generation)
            try:
                meta = self.verify(path)
                payload = self.load(path)
            except CheckpointError as error:
                log.warning("skipping %s: %s", path.name, error)
                continue
            return RestoredRun(meta, payload)
        return None

    def load(self, path: Path) -> dict[str, Any]:
        """Read a checkpoint's contents, checking it is one of ours."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            msg = f"checkpoint {path.name} could not be loaded: {error}"
            raise CheckpointError(msg) from error

        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            msg = f"{path.name} is not a reversi checkpoint"
            raise CheckpointError(msg)

        version = payload.get("format_version", 0)
        if version > FORMAT_VERSION:
            msg = (
                f"{path.name} is format version {version} but this code understands "
                f"up to {FORMAT_VERSION}; it was written by a newer version"
            )
            raise CheckpointError(msg)
        if version < FORMAT_VERSION:
            log.warning(
                "%s is format version %d (pre-resume). Its weights can be loaded, "
                "but it carries no optimiser or RNG state, so a resume from it will "
                "restart the optimiser.",
                path.name,
                version,
            )
        return payload

    # -----------------------------------------------------------------
    # Tidying
    # -----------------------------------------------------------------

    def prune(self, *, keep_last: int = 10, keep_every: int = 5) -> list[int]:
        """Delete old checkpoints, keeping a thinned history. Returns what went.

        Everything from the last ``keep_last`` generations stays, because that is
        what a resume or a rollback would reach for. Beyond that, one in every
        ``keep_every`` is kept so the cross-generation tournament on day 11 still
        has a spread of opponents to measure improvement against -- deleting them
        all would make that comparison impossible to reconstruct.
        """
        generations = self.generations()
        if len(generations) <= keep_last:
            return []

        newest = set(generations[-keep_last:])
        removed = []
        for generation in generations[:-keep_last]:
            if generation % keep_every == 0 or generation in newest:
                continue
            path = self.directory / checkpoint_name(generation)
            path.unlink(missing_ok=True)
            sidecar_for(path).unlink(missing_ok=True)
            removed.append(generation)
        return removed


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def _capture_rng() -> dict[str, Any]:
    """Snapshot every random number generator that could affect a run.

    Belt and braces, honestly. This project derives every seed it uses from
    ``(run seed, generation, worker, game index)``, so generation 8 plays the
    same games whether it was reached in one sitting or after three restarts --
    no saved state required. These are captured anyway because the cost is a few
    kilobytes and the failure they guard against, a resumed run silently
    replaying identical games, is invisible.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    """Put the generators back where they were. Silently does nothing if absent."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
