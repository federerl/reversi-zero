"""Run identity and provenance.

Contract: **a run that cannot write its metadata does not start.** ``init_run``
is called before any self-play game is played or any weight is initialised, and
it raises rather than proceeding if the run directory cannot be created.

Every run directory contains, at minimum::

    config.yaml    fully resolved configuration (+ its sha256 in meta.json)
    env.json       python / torch / CUDA / driver / OS / CPU / GPU / hostname
    git.json       commit, branch, dirty flag, diff sha256 (+ diff.patch if dirty)
    meta.json      run_id, seed, config sha256, created_utc
    cmdline.txt    the exact argv that launched the run

This is what makes "reproduce experiment 3" a mechanical operation rather than
an archaeology project.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reversi.atomicio import atomic_write_json, atomic_write_text, sha256_text
from reversi.config import Config

_GIT_TIMEOUT_S = 10


# ===========================================================================
# Run identity
# ===========================================================================


def make_run_id(config_name: str, seed: int, *, git_short: str | None = None) -> str:
    """``20260817-221030-full8x8-a1b2c3d-s1337`` -- sortable and self-describing."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    short = git_short or git_info().get("commit_short") or "nogit"
    return f"{stamp}-{config_name}-{short}-s{seed}"


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Canonical layout of a run directory."""

    run_id: str
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def replay(self) -> Path:
        return self.root / "replay"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def arena(self) -> Path:
        return self.root / "arena"

    @property
    def tb(self) -> Path:
        return self.root / "tb"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    @property
    def meta_file(self) -> Path:
        return self.root / "meta.json"

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.checkpoints,
            self.replay,
            self.logs,
            self.metrics,
            self.arena,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_root(config: Config) -> Path:
    """Resolve the run root, honouring ``RZ_RUN_ROOT`` (set by the SLURM wrapper).

    This is the only place outside the sbatch scripts that knows about cluster
    storage conventions; ``src/`` otherwise contains no scheduler-specific logic.
    """
    override = os.environ.get("RZ_RUN_ROOT")
    return Path(override) if override else config.run_root


def init_run(
    config: Config,
    *,
    run_id: str | None = None,
    argv: list[str] | None = None,
) -> RunPaths:
    """Create the run directory and write all provenance files.

    Returns the ``RunPaths`` for the created run. Safe to call again for the same
    ``run_id`` (resume): existing files are overwritten with current values, and
    the previous config sha256 is preserved in ``meta.json['config_sha256_history']``.
    """
    resolved_id = run_id or make_run_id(config.name, config.seed)
    paths = RunPaths(run_id=resolved_id, root=run_root(config) / resolved_id)
    paths.ensure()

    atomic_write_text(paths.config_file, config.to_yaml())
    atomic_write_json(paths.root / "env.json", environment_info())

    git = git_info()
    atomic_write_json(paths.root / "git.json", git)
    if git.get("dirty") and git.get("diff"):
        atomic_write_text(paths.root / "diff.patch", str(git["diff"]))

    atomic_write_text(
        paths.root / "cmdline.txt",
        " ".join(argv if argv is not None else sys.argv) + "\n",
    )

    history = _previous_config_hashes(paths)
    if config.sha256 not in history:
        history.append(config.sha256)
    atomic_write_json(
        paths.meta_file,
        {
            "run_id": resolved_id,
            "config_name": config.name,
            "seed": config.seed,
            "config_sha256": config.sha256,
            "config_sha256_history": history,
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    return paths


def _previous_config_hashes(paths: RunPaths) -> list[str]:
    if not paths.meta_file.is_file():
        return []
    import json

    try:
        previous = json.loads(paths.meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt meta is non-fatal
        return []
    history = previous.get("config_sha256_history", [])
    return [str(h) for h in history] if isinstance(history, list) else []


# ===========================================================================
# Environment capture
# ===========================================================================


def environment_info() -> dict[str, Any]:
    """Everything needed to explain why two runs behaved differently."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "ram_bytes": _total_ram_bytes(),
        "slurm": _slurm_info(),
        "torch": _torch_info(),
    }
    return info


def _cpu_affinity() -> int | None:
    """Cores this process may actually use -- often < cpu_count under SLURM."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:  # pragma: no cover - Windows
        return None
    return len(getter(0))


def _total_ram_bytes() -> int | None:
    # os.sysconf is POSIX-only; it does not exist on the Windows dev box.
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:  # pragma: no cover - Windows
        return None
    try:
        return int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):  # pragma: no cover - unsupported names
        return None


def _slurm_info() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_CPUS_ON_NODE",
            "SLURM_JOB_NODELIST",
            "SLURM_RESTART_COUNT",
        }
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        # torch.version is a real submodule, but not every torch release's
        # type stubs declare it. Asking for it defensively keeps this working
        # across the cpu and cu124 builds, which resolve to different versions.
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        count = torch.cuda.device_count()
        info["gpu_count"] = count
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_bytes"] = props.total_memory
    return info


# ===========================================================================
# Git provenance
# ===========================================================================


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Commit, branch, dirty flag, and a hash of the uncommitted diff.

    A dirty working tree is recorded, never rejected -- but the diff is captured
    so the exact code that produced a result can always be reconstructed.
    """
    cwd = repo or _repo_root()

    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"available": False}

    status = _git(["status", "--porcelain"], cwd) or ""
    dirty = bool(status.strip())
    diff = _git(["diff", "HEAD"], cwd) if dirty else ""

    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:7],
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": dirty,
        "diff_sha256": sha256_text(diff or ""),
        "diff": diff or None,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        # Fixed argv, no shell: `args` is never attacker-controlled.
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
