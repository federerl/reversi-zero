"""The standard line-ups: who plays whom in each kind of evaluation.

A *suite* is a named set of entrants. Naming them means the same tournament can be
re-run on a new run directory and produce a report in the same shape, which is
what lets `docs/crossgen.json`, the README figures, and the web app's opponent
list all be regenerated from one command.

* ``baselines`` -- one checkpoint against the fixed opponents. The quick question:
  how good is this network?
* ``crossgen`` -- several checkpoints from one run, plus the baselines, all rated
  on one scale. The question the project exists to answer: did it keep getting
  better, and how sure are we?
* ``final`` -- ``crossgen`` plus the external Edax engine, when it is installed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from reversi.arena.entrants import EntrantSpec, parse_entrant
from reversi.errors import ArenaError

__all__ = [
    "CROSSGEN_BASELINES",
    "baseline_entrants",
    "checkpoint_entrant",
    "checkpoint_label",
    "crossgen_entrants",
    "list_generations",
    "select_generations",
]

log = logging.getLogger(__name__)

CROSSGEN_BASELINES: tuple[str, ...] = ("random", "greedy", "minimax-d2", "minimax-d4")
"""The four opponents that have anchored every cross-generation table so far."""

_CHECKPOINT = re.compile(r"^gen_(\d+)\.pt$")


def baseline_entrants(names: tuple[str, ...] | list[str]) -> list[EntrantSpec]:
    """Build the fixed opponents from their names; ``minimax4`` and ``minimax-d4`` both work."""
    return [parse_entrant(name, default_simulations=1) for name in names]


def list_generations(checkpoints: Path) -> list[int]:
    """Numbered checkpoints present in a run's ``checkpoints/`` directory, oldest first."""
    found = []
    for path in checkpoints.glob("gen_*.pt"):
        match = _CHECKPOINT.match(path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def select_generations(available: list[int], *, max_checkpoints: int) -> list[int]:
    """Pick up to ``max_checkpoints`` generations spread evenly across the run.

    The first and last available are always included, because the two questions a
    reader asks are "where did it start" and "where did it end". The pruner keeps
    every fifth checkpoint plus the last ten, so the spread is over what is on
    disk, not over an idealised range.
    """
    if max_checkpoints < 2:
        msg = f"a cross-generation table needs at least 2 checkpoints, got {max_checkpoints}"
        raise ArenaError(msg)
    if len(available) <= max_checkpoints:
        return list(available)
    positions = np.linspace(0, len(available) - 1, num=max_checkpoints)
    picked = sorted({available[int(np.rint(p))] for p in positions})
    return picked


def checkpoint_label(path: Path) -> str:
    """The name a checkpoint file gets in a report: ``gen60`` for ``gen_00060.pt``.

    ``latest.pt`` and exported models carry their generation in the JSON sidecar,
    so they get the same kind of name. Anything else is named after its file.
    """
    match = _CHECKPOINT.match(path.name)
    if match:
        return f"gen{int(match.group(1)):02d}"
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("generation"), int):
            return f"gen{payload['generation']:02d}"
    return path.stem


def checkpoint_entrant(checkpoints: Path, generation: int, *, simulations: int) -> EntrantSpec:
    """The entrant for one saved generation, named the way the reports have always named them."""
    from reversi.ckpt.meta import checkpoint_name

    path = checkpoints / checkpoint_name(generation)
    if not path.is_file():
        msg = f"no checkpoint for generation {generation} at {path}"
        raise ArenaError(msg)
    return EntrantSpec(
        name=f"gen{generation:02d}",
        kind="checkpoint",
        path=str(path),
        simulations=simulations,
    )


def crossgen_entrants(
    checkpoints: Path,
    *,
    simulations: int,
    max_checkpoints: int = 6,
    baselines: tuple[str, ...] = CROSSGEN_BASELINES,
) -> list[EntrantSpec]:
    """The ``crossgen`` line-up for one run directory."""
    available = list_generations(checkpoints)
    if len(available) < 2:
        msg = (
            f"{checkpoints} holds {len(available)} numbered checkpoint(s); a "
            "cross-generation table needs at least two"
        )
        raise ArenaError(msg)
    chosen = select_generations(available, max_checkpoints=max_checkpoints)
    log.info("rating generations %s of %d available", chosen, len(available))
    entrants = baseline_entrants(baselines)
    entrants.extend(checkpoint_entrant(checkpoints, g, simulations=simulations) for g in chosen)
    return entrants
