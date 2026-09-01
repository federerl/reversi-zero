"""The list of opponents the web app offers, built from measured results.

The site lets a visitor pick which generation of the training run to play
against. That is the most direct way to show what the training actually did:
instead of reading that the agent improved, you play generation 5, then
generation 60, and feel the difference.

**Every label here is generated from the tournament report, never typed by
hand.** The repository's definition of done says difficulty labels state
measured strength rather than marketing adjectives, and the way to keep a rule
like that is to make the alternative impossible. If a rating changes, the
manifest changes with it; if a generation was never rated, it does not appear.

The manifest carries no weights -- only names, file paths, checksums and Elo. It
is small and committed. The ``.onnx`` files it points at are attached to a
GitHub Release and fetched at build time, exactly like the ``.pt`` files they
came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reversi.errors import ConfigError

__all__ = ["MANIFEST_VERSION", "build_manifest", "write_manifest"]

MANIFEST_VERSION = 1

# What each generation is worth saying about it, beyond its rating. Kept short:
# the number is the claim, this is only context for a reader who does not know
# what an Elo is.
NOTES: dict[int, str] = {
    5: "About six hours into training. Already beats a classical search.",
    10: "Still learning what corners are for.",
    20: "A third of the way through training.",
    30: "Halfway.",
    40: "Two thirds through -- already close to its final strength.",
    50: "Near the end of the run.",
    60: "The final agent. Beat the depth-4 search 30 games to nil.",
}


def build_manifest(
    tournament: Path,
    *,
    generations: list[int] | None = None,
    board_size: int = 8,
    url_prefix: str = "/models",
    calibration: Path | None = None,
) -> dict[str, Any]:
    """Turn a cross-generation tournament report into the app's opponent list.

    Only entrants named ``genNN`` become playable opponents. The baselines in the
    same report -- random, greedy, the minimax searches -- are carried along
    separately, because they are what make the agent's numbers mean anything and
    the interface should be able to show them on the same scale.
    """
    report = json.loads(tournament.read_text(encoding="utf-8"))
    ratings = report.get("ratings")
    if not ratings:
        msg = f"{tournament} has no ratings block; it is not a tournament report"
        raise ConfigError(msg)

    by_name = {entry["name"]: entry for entry in ratings}

    models: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []

    for entry in ratings:
        name = entry["name"]
        if not name.startswith("gen"):
            baselines.append(
                {
                    "name": name,
                    "elo": round(entry["elo"], 1),
                    "interval": [round(entry["ci_low"], 1), round(entry["ci_high"], 1)],
                }
            )
            continue

        generation = int(name.removeprefix("gen"))
        if generations is not None and generation not in generations:
            continue

        models.append(
            {
                "id": f"gen{generation:02d}",
                "generation": generation,
                "label": f"Generation {generation}",
                "url": f"{url_prefix}/reversi-{board_size}x{board_size}-gen{generation}.onnx",
                "boardSize": board_size,
                "elo": round(entry["elo"], 1),
                "eloInterval": [round(entry["ci_low"], 1), round(entry["ci_high"], 1)],
                "games": entry.get("games"),
                "note": NOTES.get(generation, ""),
            }
        )

    if not models:
        available = sorted(n for n in by_name if n.startswith("gen"))
        msg = (
            f"no playable generations found in {tournament.name}. "
            f"It rates: {', '.join(available) or 'nothing'}"
        )
        raise ConfigError(msg)

    models.sort(key=lambda entry: entry["generation"], reverse=True)

    levels: list[dict[str, Any]] = []
    if calibration is not None:
        levels = _levels_from(calibration)

    return {
        "manifest_version": MANIFEST_VERSION,
        "levels": levels,
        "board_size": board_size,
        "run_id": report.get("provenance", {}).get("run_id"),
        "rating_anchor": "random = 0",
        "rating_method": "Bradley-Terry, 95% bootstrap interval",
        "games_per_entrant": models[0].get("games"),
        "source": tournament.name,
        "models": models,
        "baselines": baselines,
    }


def _levels_from(calibration: Path) -> list[dict[str, Any]]:
    """The difficulty levels' ratings, from the report that measured them.

    Same rule as the opponents: a label states a measured number or it does not
    appear. Before this was run, "Casual" and "Club" were an assertion that
    moving up gets you a harder game -- reasonable, and not evidence. Reading the
    ratings from the report means the interface cannot claim a separation that
    was not measured, and cannot go stale without the file it reads going stale
    too.
    """
    report = json.loads(calibration.read_text(encoding="utf-8"))
    if not report.get("passed"):
        msg = (
            f"{calibration.name} records a calibration that did not meet S15, so its "
            "ratings should not be shown as if the ladder were separated"
        )
        raise ConfigError(msg)

    rated = {entry["name"]: entry for entry in report["ratings"]}
    out = []
    for name in report["levels"]:
        entry = rated.get(name)
        if entry is None:
            msg = f"{calibration.name} rates no level called {name!r}"
            raise ConfigError(msg)
        out.append(
            {
                "id": name,
                "elo": round(entry["elo"], 1),
                "eloInterval": [round(entry["ci_low"], 1), round(entry["ci_high"], 1)],
            }
        )
    return out


def write_manifest(destination: Path, manifest: dict[str, Any]) -> int:
    """Write the manifest, returning its size in bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return destination.stat().st_size
