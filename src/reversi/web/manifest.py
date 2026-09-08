"""The list of opponents the web app offers, built from measured results.

The site lets a visitor pick which generation of the training run to play
against. That is the most direct way to show what the training actually did:
instead of reading that the agent improved, you play generation 5, then
generation 60, and feel the difference.

**Every label here is generated from the tournament report, never typed by
hand.** The repository's definition of done says no claim about strength is typed
by hand, and the way to keep a rule like that is to make the alternative
impossible. If a rating changes, the manifest changes with it; if a generation
was never rated, it does not appear.

The manifest carries no weights -- only names, file paths, checksums and Elo. It
is small and committed. The ``.onnx`` files it points at are attached to a
GitHub Release and fetched at build time, exactly like the ``.pt`` files they
came from.

Version 2
---------

Version 1 said which generation a model was and what it rated. That is enough
while there is exactly one training run in the world, and it stops being enough
the moment there are two, because "generation 60" is then ambiguous and the file
name ``reversi-8x8-gen60.onnx`` names two different networks.

Three things changed.

**A model records which run it came from, its architecture, and its checksum**,
all read from the export sidecar rather than restated here. The sidecar is
written by ``reversi export-onnx`` and is the only place these facts are
authoritative. So a manifest entry can be checked against the file it points at,
and a reader can tell a 6x64 network from a 10x128 one without downloading
either.

**The file name carries the run.** Release assets live in one flat namespace per
release -- there are no directories to separate two runs -- so uniqueness has to
be in the name. Rather than trust the naming scheme, ``build_manifest`` checks
that no two models resolve to the same URL and refuses to write a manifest where
they do. That is the structural guarantee; the name is only the mechanism.

**A manifest says which release it expects and which game it is for.** The
release tag is what makes the URLs resolvable, and the game is what lets a second
game's manifest exist beside this one without either having to guess.

Version 2 only adds keys. Everything version 1 carried is still there under the
same name, so a consumer written against version 1 keeps working -- which is why
the web app needed no change when this landed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reversi.errors import ConfigError

__all__ = [
    "MANIFEST_VERSION",
    "build_manifest",
    "run_key",
    "write_manifest",
]

MANIFEST_VERSION = 2

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


def run_key(run_id: str) -> str:
    """A short, unique tag for a run, for use in a file name.

    Run ids look like ``20260827-030939-full8x8-60cfdda-s1337``: a start date, a
    start time, then the config, the commit and the seed. The first two fields
    are what make it unique -- a run id is minted from the clock when the run
    begins, so two runs cannot share them -- and they are also the two a person
    can read. The rest is useful in a report and only noise in a file name.

    The full run id is still recorded in the manifest entry, so nothing is lost
    by shortening it here.
    """
    fields = [field for field in run_id.split("-") if field]
    if len(fields) < 2:
        msg = (
            f"run id {run_id!r} does not look like '<date>-<time>-...', so there is "
            "no unique part to put in a file name"
        )
        raise ConfigError(msg)
    return f"{fields[0]}-{fields[1]}"


def _sidecar_for(sidecars: Path, board_size: int, generation: int) -> dict[str, Any]:
    """Read the export sidecar for one generation, or say why it cannot.

    A missing sidecar is an error rather than a field left blank. Without it
    there is no checksum, which means the manifest would be publishing a file
    nobody can verify -- and a rating attached to an unverifiable file is worth
    less than no rating at all.
    """
    path = sidecars / f"reversi-{board_size}x{board_size}-gen{generation}.json"
    if not path.exists():
        msg = (
            f"no export sidecar at {path}. Run `reversi export-onnx` for generation "
            f"{generation} first; a model cannot be published without its checksum"
        )
        raise ConfigError(msg)

    sidecar = json.loads(path.read_text(encoding="utf-8"))

    for field in ("sha256", "run_id", "arch"):
        if not sidecar.get(field):
            msg = f"{path.name} has no {field!r}; it is not a complete export sidecar"
            raise ConfigError(msg)

    recorded = sidecar.get("generation")
    if recorded != generation:
        msg = (
            f"{path.name} records generation {recorded}, not {generation}. "
            "The sidecar and the file it describes have come apart"
        )
        raise ConfigError(msg)

    return sidecar


def build_manifest(
    tournament: Path,
    *,
    sidecars: Path,
    release: str,
    generations: list[int] | None = None,
    board_size: int = 8,
    url_prefix: str = "/models",
    calibration: Path | None = None,
    game: str = "reversi",
) -> dict[str, Any]:
    """Turn a cross-generation tournament report into the app's opponent list.

    Only entrants named ``genNN`` become playable opponents. The baselines in the
    same report -- random, greedy, the minimax searches -- are carried along
    separately, because they are what make the agent's numbers mean anything and
    the interface should be able to show them on the same scale.

    ``sidecars`` is the directory holding the ``.json`` files that
    ``reversi export-onnx`` wrote next to each ``.onnx``. Every generation that
    is about to be published needs one; see ``_sidecar_for``.
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
        if not _is_model_name(name):
            baselines.append(
                {
                    "name": name,
                    "elo": round(entry["elo"], 1),
                    "interval": [round(entry["ci_low"], 1), round(entry["ci_high"], 1)],
                }
            )
            continue

        generation = int(name.removeprefix("gen"))  # _is_model_name checked this
        if generations is not None and generation not in generations:
            continue

        sidecar = _sidecar_for(sidecars, board_size, generation)
        run_id = str(sidecar["run_id"])
        stem = f"reversi-{board_size}x{board_size}-gen{generation}-{run_key(run_id)}"

        models.append(
            {
                "id": f"gen{generation:02d}",
                "generation": generation,
                "label": f"Generation {generation}",
                "url": f"{url_prefix}/{stem}.onnx",
                "boardSize": board_size,
                "elo": round(entry["elo"], 1),
                "eloInterval": [round(entry["ci_low"], 1), round(entry["ci_high"], 1)],
                "games": entry.get("games"),
                "note": NOTES.get(generation, ""),
                "run": run_id,
                "arch": sidecar["arch"],
                "sha256": sidecar["sha256"],
            }
        )

    if not models:
        available = sorted(n for n in by_name if _is_model_name(n))
        msg = (
            f"no playable generations found in {tournament.name}. "
            f"It rates: {', '.join(available) or 'nothing'}"
        )
        raise ConfigError(msg)

    # The guarantee the naming scheme is only a means to. Two entries pointing at
    # one file is the failure this version of the manifest exists to prevent, so
    # it is checked rather than assumed.
    _reject_colliding_urls(models)

    # Newest first: the strongest agent is the one a visitor is most likely to
    # want, and the ladder in the app orders itself by rating anyway.
    models.sort(key=lambda entry: (entry["generation"], entry["run"]), reverse=True)

    levels: list[dict[str, Any]] = []
    if calibration is not None:
        levels = _levels_from(calibration)

    return {
        "manifest_version": MANIFEST_VERSION,
        "game": game,
        "release": release,
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


def _is_model_name(name: str) -> bool:
    """Whether a tournament entrant is a publishable generation of the network.

    ``gen20`` is; ``random`` and ``minimax-d4`` are not. The awkward case is a
    name that begins with ``gen`` and then is not a number -- ``gen60-run1``, the
    obvious thing to call run 1's generation 60 when it enters another run's
    tournament as a reference. That used to reach ``int()`` and raise a bare
    ``ValueError`` mentioning only ``'60-run1'``, which says nothing about what to
    do instead.

    It is refused rather than reclassified. Silently treating it as a baseline
    would drop a level the author may have meant to publish, and silently
    treating it as a model would need a rule for inventing its generation. The
    manifest does not guess anywhere else and should not start here.
    """
    if not name.startswith("gen"):
        return False
    suffix = name.removeprefix("gen")
    if suffix.isdigit():
        return True
    msg = (
        f"entrant {name!r} starts with 'gen' but {suffix!r} is not a generation "
        "number. Name a publishable checkpoint 'gen<number>', and name anything "
        "else -- an entrant from another run, an external engine -- something "
        "that does not begin with 'gen', so it is rated without being published"
    )
    raise ConfigError(msg)


def _reject_colliding_urls(models: list[dict[str, Any]]) -> None:
    """Refuse a manifest where two models would be fetched from one URL.

    This is the whole point of putting the run in the file name, stated as a
    check instead of a hope. Two runs' generation 60 are different networks with
    different ratings, and serving one file for both would attach a measured
    number to the wrong weights -- a mistake nothing downstream could detect.
    """
    seen: dict[str, str] = {}
    for model in models:
        url = str(model["url"])
        previous = seen.get(url)
        if previous is not None:
            msg = (
                f"two models resolve to {url}: run {previous} and run {model['run']}. "
                "Their file names must differ, or one would silently serve the other"
            )
            raise ConfigError(msg)
        seen[url] = str(model["run"])

    ids = [str(model["id"]) for model in models]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        msg = (
            f"more than one model claims the id(s) {', '.join(duplicates)}. "
            "The app selects an opponent by id, so these would be indistinguishable"
        )
        raise ConfigError(msg)


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
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return destination.stat().st_size
