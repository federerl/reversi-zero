"""A competitor described as data, so it can be sent to another process and built there.

A tournament plays its pairings in separate processes. A process cannot receive a
live agent -- a loaded network is not something Python can pickle and send -- so
what travels is a small description: *what kind of agent, and which file*. Each
process builds its own agent from that, loading its own copy of the weights.

The same description is what gets written into the report under ``agent_specs``,
so a reader of ``crossgen.json`` can see exactly what "gen60" meant: which
checkpoint file, how many simulations, which minimax weights.

Kinds:

* ``random``, ``greedy`` -- the two baselines with no settings.
* ``minimax`` -- alpha-beta search to ``depth`` plies with the frozen weights.
* ``edax`` -- the external Edax engine at level ``depth`` (its own scale, 1..60).
* ``checkpoint`` -- a saved network from a run, searching ``simulations`` per move
  with no exploration noise. Either a training checkpoint or an exported model.
* ``level`` -- one rung of the difficulty ladder on top of a saved network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reversi.agents.base import Agent
from reversi.errors import ConfigError
from reversi.search.evaluator import Evaluator

__all__ = ["EntrantSpec", "build_agent", "describe_entrant", "parse_entrant"]

Kind = Literal["random", "greedy", "minimax", "edax", "checkpoint", "level"]

_BASELINE_PATTERNS: tuple[tuple[re.Pattern[str], Kind], ...] = (
    (re.compile(r"^minimax-?d?(\d+)$"), "minimax"),
    (re.compile(r"^edax-?l?(\d+)$"), "edax"),
)


@dataclass(frozen=True, slots=True)
class EntrantSpec:
    """Everything needed to rebuild one competitor, and nothing that cannot be pickled."""

    name: str
    kind: Kind
    path: str | None = None
    """The network file, for ``checkpoint`` and ``level``."""
    simulations: int | None = None
    """Search budget per move, for ``checkpoint``."""
    depth: int | None = None
    """Search depth for ``minimax``; the engine level for ``edax``."""
    level: str | None = None
    """The rung name, for ``level``."""
    c_puct: float = 1.5
    fpu_reduction: float = 0.25

    @property
    def needs_network(self) -> bool:
        return self.kind in {"checkpoint", "level"}


def parse_entrant(text: str, *, default_simulations: int) -> EntrantSpec:
    """Turn the command-line form of an entrant into a spec.

    Accepted forms::

        random                      greedy
        minimax-d4   (or minimax4)  edax-l5  (or edax5)
        NAME=PATH                   a network, searching ``default_simulations``
        NAME=PATH@200               the same, at 200 simulations
        PATH                        name taken from the file's stem
        level:casual=PATH           a difficulty rung on top of a network

    The equals sign separates the name from the file because file names on
    Windows may contain colons and drive letters, and a name never does.
    """
    text = text.strip()
    if not text:
        msg = "an entrant cannot be empty"
        raise ConfigError(msg)

    if text in {"random", "greedy"}:
        return EntrantSpec(name=text, kind=text)  # type: ignore[arg-type]

    for pattern, kind in _BASELINE_PATTERNS:
        found = pattern.match(text)
        if found:
            number = int(found.group(1))
            letter = "d" if kind == "minimax" else "l"
            return EntrantSpec(name=f"{kind}-{letter}{number}", kind=kind, depth=number)

    if text.startswith("level:"):
        body = text.removeprefix("level:")
        if "=" not in body:
            msg = f"a level entrant looks like level:casual=PATH, got {text!r}"
            raise ConfigError(msg)
        level, path = body.split("=", 1)
        return EntrantSpec(name=level, kind="level", path=path, level=level)

    name, _, rest = text.rpartition("=") if "=" in text else ("", "", text)
    path, _, sims = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if not name:
        name = Path(path).stem
    try:
        simulations = int(sims) if sims else default_simulations
    except ValueError as error:
        msg = f"simulations must be a whole number, got {sims!r} in {text!r}"
        raise ConfigError(msg) from error
    if simulations < 1:
        msg = f"simulations must be at least 1, got {simulations} in {text!r}"
        raise ConfigError(msg)
    if not Path(path).is_file():
        msg = (
            f"no network file at {path!r} (from entrant {text!r}). Baselines are "
            "random, greedy, minimax-dN and edax-lN; anything else is read as a file."
        )
        raise ConfigError(msg)
    return EntrantSpec(name=name, kind="checkpoint", path=path, simulations=simulations)


def _load_evaluator(path: Path, device: str) -> Evaluator:
    """Load either kind of network file this project produces.

    A *training checkpoint* keeps its architecture at the top level; an *exported*
    model keeps it under ``meta``. Both are reasonable things to rate -- the export
    is what ships, and the checkpoint is what you have mid-run -- so this accepts
    either rather than making the caller remember which is which.
    """
    from reversi.errors import CheckpointError
    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.export import load_export
    from reversi.nn.loader import load_model

    try:
        model = load_export(path, device=device).model
    except CheckpointError:
        model = load_model(path, device=device)
    return TorchEvaluator(model, device=device)


def build_agent(spec: EntrantSpec, *, device: str = "cpu") -> Agent:
    """Build the agent a spec describes. Imports torch only when a network is needed."""
    if spec.kind == "random":
        from reversi.agents.random_agent import RandomAgent

        return RandomAgent()
    if spec.kind == "greedy":
        from reversi.agents.greedy import GreedyAgent

        return GreedyAgent()
    if spec.kind == "minimax":
        from reversi.agents.minimax import MinimaxAgent

        return MinimaxAgent(_required(spec.depth, spec, "depth"), name=spec.name)
    if spec.kind == "edax":
        from reversi.agents.edax import EdaxAgent

        return EdaxAgent(_required(spec.depth, spec, "depth"), name=spec.name)

    path = Path(_required(spec.path, spec, "path"))
    evaluator = _load_evaluator(path, device)

    if spec.kind == "level":
        from reversi.difficulty.calibrate import DifficultyAgent
        from reversi.difficulty.levels import level_by_name

        return DifficultyAgent(evaluator, level_by_name(_required(spec.level, spec, "level")))

    from reversi.agents.az_agent import AZAgent
    from reversi.search.config import SearchConfig

    return AZAgent(
        evaluator,
        SearchConfig(
            n_simulations=_required(spec.simulations, spec, "simulations"),
            c_puct=spec.c_puct,
            fpu_reduction=spec.fpu_reduction,
            dirichlet_eps=0.0,
            temp_moves=0,
        ),
        name=spec.name,
    )


def describe_entrant(spec: EntrantSpec, *, board_size: int = 8) -> dict[str, Any]:
    """What goes into the report under ``agent_specs``: enough to reproduce the entrant."""
    if spec.kind in {"random", "greedy"}:
        return {"kind": spec.kind}
    if spec.kind == "minimax":
        from reversi.agents.minimax import weights_fingerprint

        return {"kind": "minimax", "depth": spec.depth, "weights": weights_fingerprint(board_size)}
    if spec.kind == "edax":
        return {"kind": "edax", "level": spec.depth}

    path = Path(spec.path or "")
    out: dict[str, Any] = {"checkpoint": path.name}
    out.update(_sidecar_facts(path))
    if spec.kind == "level":
        out.update({"kind": "difficulty", "level": spec.level})
    else:
        out.update({"kind": "alphazero", "simulations": spec.simulations})
        if spec.c_puct != 1.5 or spec.fpu_reduction != 0.25:
            out.update({"c_puct": spec.c_puct, "fpu_reduction": spec.fpu_reduction})
    return out


def _sidecar_facts(path: Path) -> dict[str, Any]:
    """Generation and run id from the JSON that sits beside every checkpoint and export."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    facts: dict[str, Any] = {}
    for key in ("generation", "run_id"):
        if isinstance(payload, dict) and key in payload:
            facts[key] = payload[key]
    return facts


def _required(value: Any, spec: EntrantSpec, field_name: str) -> Any:
    if value is None:
        msg = f"entrant {spec.name!r} of kind {spec.kind!r} needs {field_name}"
        raise ConfigError(msg)
    return value
