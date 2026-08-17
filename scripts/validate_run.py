#!/usr/bin/env python
"""Validate that a run directory is reproducible (success criterion S19).

A run that cannot be reproduced is not an experiment, it is an anecdote. This
script is the mechanical check: it verifies that every provenance file exists,
parses, and carries the fields needed to reconstruct the run.

Usage::

    python scripts/validate_run.py runs/20260817-221030-full8x8-a1b2c3d-s1337
    python scripts/validate_run.py runs/*            # validate every run

Exit code 0 = valid, 1 = one or more runs are missing required provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FILES = ("config.yaml", "env.json", "git.json", "meta.json", "cmdline.txt")
REQUIRED_DIRS = ("checkpoints", "replay", "logs", "metrics", "arena")
REQUIRED_META = ("run_id", "config_name", "seed", "config_sha256", "created_utc")
REQUIRED_ENV = ("python", "platform", "hostname", "cpu_count_logical", "torch")


def validate(run_dir: Path) -> list[str]:
    """Return a list of problems; empty means the run is valid."""
    problems: list[str] = []

    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            problems.append(f"missing file: {name}")
    for name in REQUIRED_DIRS:
        if not (run_dir / name).is_dir():
            problems.append(f"missing directory: {name}/")

    config = _load(run_dir / "config.yaml", yaml.safe_load, problems, "config.yaml")
    meta = _load(run_dir / "meta.json", json.loads, problems, "meta.json")
    env = _load(run_dir / "env.json", json.loads, problems, "env.json")
    git = _load(run_dir / "git.json", json.loads, problems, "git.json")

    if isinstance(meta, dict):
        problems += [f"meta.json missing field: {k}" for k in REQUIRED_META if k not in meta]
        if meta.get("run_id") and meta["run_id"] != run_dir.name:
            problems.append(f"meta.json run_id {meta['run_id']!r} != directory {run_dir.name!r}")

    if isinstance(env, dict):
        problems += [f"env.json missing field: {k}" for k in REQUIRED_ENV if k not in env]

    if isinstance(git, dict) and git.get("available") and not git.get("commit"):
        problems.append("git.json claims availability but records no commit")

    if isinstance(config, dict) and not config.get("game", {}).get("board_size"):
        problems.append("config.yaml has no game.board_size")

    return problems


def _load(path: Path, parser: Any, problems: list[str], label: str) -> Any:
    if not path.is_file():
        return None
    try:
        return parser(path.read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError) as exc:
        problems.append(f"{label} does not parse: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = 0
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            print(f"[SKIP] {run_dir} is not a directory")
            continue
        problems = validate(run_dir)
        if problems:
            failed += 1
            print(f"[FAIL] {run_dir}")
            for problem in problems:
                print(f"         - {problem}")
        else:
            print(f"[ OK ] {run_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
