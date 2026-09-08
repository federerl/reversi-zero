"""The manifest that tells the web app what it may offer.

This file is the join between what was measured and what a visitor is told, and
it is generated precisely so that nobody can put a number on the screen by hand.
The tests are therefore about the *rules* rather than the shape: a generation
that was never rated must not appear, a model with no checksum must not be
publishable, and two runs' generation 60 must not be able to resolve to one file.

The last of those is the reason version 2 exists, and it is the one failure
nothing downstream could detect: a manifest that served run 1's weights under
run 2's rating would look completely healthy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reversi.errors import ConfigError
from reversi.web.manifest import MANIFEST_VERSION, build_manifest, run_key, write_manifest

RUN_ONE = "20260827-030939-full8x8-60cfdda-s1337"
RUN_TWO = "20260903-114500-full8x8-e12own-s99"


def _rating(name: str, elo: float, games: int = 210) -> dict[str, Any]:
    return {
        "name": name,
        "elo": elo,
        "ci_low": elo - 100,
        "ci_high": elo + 100,
        "games": games,
    }


def _tournament(tmp_path: Path, ratings: list[dict[str, Any]]) -> Path:
    path = tmp_path / "crossgen.json"
    path.write_text(
        json.dumps({"ratings": ratings, "provenance": {"run_id": RUN_ONE}}),
        encoding="utf-8",
    )
    return path


def _sidecar(
    directory: Path,
    generation: int,
    *,
    run_id: str = RUN_ONE,
    board_size: int = 8,
    channels: int = 64,
    omit: str | None = None,
    generation_field: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generation": generation if generation_field is None else generation_field,
        "run_id": run_id,
        "sha256": f"{generation:064x}",
        "arch": {
            "board_size": board_size,
            "channels": channels,
            "n_blocks": 6,
            "in_planes": 3,
            "policy_size": 65,
            "value_hidden": 64,
        },
    }
    if omit is not None:
        payload.pop(omit)
    path = directory / f"reversi-{board_size}x{board_size}-gen{generation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestRunKey:
    def test_the_key_is_the_date_and_time_the_run_started(self) -> None:
        assert run_key(RUN_ONE) == "20260827-030939"
        assert run_key(RUN_TWO) == "20260903-114500"

    def test_two_runs_of_the_same_config_get_different_keys(self) -> None:
        """The point of the key. Same config, same commit, different clock."""
        a = "20260827-030939-full8x8-60cfdda-s1337"
        b = "20260827-194412-full8x8-60cfdda-s1337"
        assert run_key(a) != run_key(b)

    def test_a_run_id_with_no_unique_part_is_refused(self) -> None:
        """Better to fail here than to name two files the same thing."""
        with pytest.raises(ConfigError, match="no unique part"):
            run_key("legacy")


class TestWhatMayAppear:
    def test_only_rated_generations_become_opponents(self, tmp_path: Path) -> None:
        """The rule the manifest exists to enforce, stated as a test.

        `gen40` has an export and a checksum and is entirely playable -- and it
        is not in the tournament, so it must not reach the app. An unrated
        opponent on the ladder would be a difficulty the interface implies and
        nothing measured.
        """
        models = tmp_path / "models"
        for generation in (5, 40, 60):
            _sidecar(models, generation)
        report = _tournament(
            tmp_path,
            [_rating("gen60", 877.3), _rating("gen05", 547.3), _rating("random", 0.0)],
        )

        manifest = build_manifest(report, sidecars=models, release="models-v2")

        assert [model["id"] for model in manifest["models"]] == ["gen60", "gen05"]
        assert [b["name"] for b in manifest["baselines"]] == ["random"]

    def test_a_generation_with_no_sidecar_cannot_be_published(self, tmp_path: Path) -> None:
        """No checksum, no publication.

        A rating attached to a file nobody can verify is worth less than no
        rating: it looks like evidence and is not.
        """
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3), _rating("gen05", 547.3)])

        with pytest.raises(ConfigError, match="no export sidecar"):
            build_manifest(report, sidecars=models, release="models-v2")

    @pytest.mark.parametrize("field", ["sha256", "run_id", "arch"])
    def test_an_incomplete_sidecar_is_refused(self, tmp_path: Path, field: str) -> None:
        models = tmp_path / "models"
        _sidecar(models, 60, omit=field)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])

        with pytest.raises(ConfigError, match=f"no {field!r}"):
            build_manifest(report, sidecars=models, release="models-v2")

    def test_a_sidecar_describing_another_generation_is_refused(self, tmp_path: Path) -> None:
        """The sidecar and its file coming apart is silent otherwise."""
        models = tmp_path / "models"
        _sidecar(models, 60, generation_field=40)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])

        with pytest.raises(ConfigError, match="records generation 40"):
            build_manifest(report, sidecars=models, release="models-v2")

    def test_a_gen_prefixed_name_that_is_not_a_generation_is_refused(self, tmp_path: Path) -> None:
        """The obvious name for a reference entrant from another run.

        Putting two runs on one scale means one run's checkpoint enters the
        other's tournament, and `gen60-run1` is the natural thing to call it.
        It used to reach `int()` and raise a bare ValueError mentioning only
        '60-run1'. Now it says what to do instead, and says it before anything
        is written.
        """
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60-run1", 877.3), _rating("random", 0.0)])

        with pytest.raises(ConfigError, match="is not a generation number"):
            build_manifest(report, sidecars=models, release="models-v2")

    def test_a_reference_entrant_named_clearly_is_rated_but_not_published(
        self, tmp_path: Path
    ) -> None:
        """The other half of that rule, and the reason it is a rule.

        `run1-gen60` is rated on the same scale as everything else and appears in
        the manifest, but as a rating row rather than a level. A ladder's levels
        are one run's progression; a network from a different run belongs beside
        them as a reference, not inside them as a rung.
        """
        models = tmp_path / "models"
        _sidecar(models, 120, run_id=RUN_TWO)
        report = _tournament(
            tmp_path,
            [
                _rating("gen120", 967.9),
                _rating("run1-gen60", 877.3),
                _rating("random", 0.0),
            ],
        )

        manifest = build_manifest(report, sidecars=models, release="models-v2")

        assert [m["id"] for m in manifest["models"]] == ["gen120"]
        rated = {b["name"]: b["elo"] for b in manifest["baselines"]}
        assert rated["run1-gen60"] == 877.3
        assert rated["random"] == 0.0

    def test_a_tournament_rating_nothing_playable_is_refused(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        report = _tournament(tmp_path, [_rating("random", 0.0), _rating("greedy", 313.1)])

        with pytest.raises(ConfigError, match="no playable generations"):
            build_manifest(report, sidecars=models, release="models-v2")


class TestTwoRunsCannotCollide:
    def test_the_run_reaches_the_url_and_the_entry(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])

        manifest = build_manifest(report, sidecars=models, release="models-v2")
        model = manifest["models"][0]

        assert model["url"] == "/models/reversi-8x8-gen60-20260827-030939.onnx"
        assert model["run"] == RUN_ONE
        assert model["sha256"] == f"{60:064x}"
        assert model["arch"]["channels"] == 64

    def test_the_same_generation_from_two_runs_gets_two_urls(self, tmp_path: Path) -> None:
        """The case version 1 could not express.

        Two networks, both called generation 60, with different weights and
        different ratings. Under version 1 both entries pointed at
        `reversi-8x8-gen60.onnx` and whichever file was uploaded last won.
        """
        models = tmp_path / "models"
        _sidecar(models, 60, run_id=RUN_ONE)
        _sidecar(models, 120, run_id=RUN_TWO, channels=128)
        report = _tournament(tmp_path, [_rating("gen60", 877.3), _rating("gen120", 940.0)])

        manifest = build_manifest(report, sidecars=models, release="models-v2")
        urls = {model["id"]: model["url"] for model in manifest["models"]}

        assert urls["gen60"].endswith("gen60-20260827-030939.onnx")
        assert urls["gen120"].endswith("gen120-20260903-114500.onnx")
        # And the architectures are visible without downloading either.
        by_id = {model["id"]: model for model in manifest["models"]}
        assert by_id["gen60"]["arch"]["channels"] == 64
        assert by_id["gen120"]["arch"]["channels"] == 128

    def test_a_collision_is_refused_rather_than_written(self, tmp_path: Path) -> None:
        """The structural guarantee, not the naming convention.

        Two sidecars whose run ids share a date *and* a time would defeat the
        naming scheme. Rather than trust that this cannot happen, the builder
        checks the URLs it produced and refuses.
        """
        models = tmp_path / "models"
        clock = "20260827-030939"
        _sidecar(models, 60, run_id=f"{clock}-runA-aaaaaaa-s1")
        _sidecar(models, 61, run_id=f"{clock}-runB-bbbbbbb-s2")
        report = _tournament(tmp_path, [_rating("gen60", 877.3), _rating("gen61", 880.0)])

        # Same clock but different generations still differ, so this pair is fine.
        manifest = build_manifest(report, sidecars=models, release="models-v2")
        assert len({model["url"] for model in manifest["models"]}) == 2

        # Force the real collision: one generation, one clock, two runs.
        colliding = [
            dict(manifest["models"][0]),
            dict(manifest["models"][0]) | {"run": f"{clock}-runB-bbbbbbb-s2", "id": "gen60b"},
        ]
        from reversi.web.manifest import _reject_colliding_urls

        with pytest.raises(ConfigError, match="two models resolve to"):
            _reject_colliding_urls(colliding)


class TestVersionTwoIsAdditive:
    def test_everything_version_one_carried_is_still_there(self, tmp_path: Path) -> None:
        """A consumer written against version 1 must keep working.

        This is why the web app needed no change: version 2 adds keys and
        renames none. If a future version has to remove one, this test is where
        that decision becomes visible.
        """
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3), _rating("random", 0.0)])

        manifest = build_manifest(report, sidecars=models, release="models-v2")

        for key in (
            "manifest_version",
            "levels",
            "board_size",
            "run_id",
            "rating_anchor",
            "rating_method",
            "games_per_entrant",
            "source",
            "models",
            "baselines",
        ):
            assert key in manifest, key
        for key in ("id", "generation", "label", "url", "boardSize", "elo", "eloInterval", "note"):
            assert key in manifest["models"][0], key

        assert manifest["manifest_version"] == MANIFEST_VERSION == 2
        assert manifest["game"] == "reversi"
        assert manifest["release"] == "models-v2"

    def test_it_round_trips_through_the_file(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])
        manifest = build_manifest(report, sidecars=models, release="models-v2")

        destination = tmp_path / "out" / "models.json"
        size = write_manifest(destination, manifest)

        assert size > 0
        assert json.loads(destination.read_text(encoding="utf-8")) == manifest


class TestLevels:
    def test_level_ratings_come_from_a_calibration_that_passed(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])

        calibration = tmp_path / "difficulty.json"
        calibration.write_text(
            json.dumps(
                {
                    "passed": True,
                    "levels": ["casual", "max"],
                    "ratings": [_rating("casual", 431.0), _rating("max", 1053.3)],
                }
            ),
            encoding="utf-8",
        )

        manifest = build_manifest(
            report, sidecars=models, release="models-v2", calibration=calibration
        )

        assert [level["id"] for level in manifest["levels"]] == ["casual", "max"]
        assert manifest["levels"][0]["elo"] == 431.0

    def test_a_failed_calibration_is_not_shown_as_a_ladder(self, tmp_path: Path) -> None:
        """A ladder that did not separate must not be presented as one."""
        models = tmp_path / "models"
        _sidecar(models, 60)
        report = _tournament(tmp_path, [_rating("gen60", 877.3)])

        calibration = tmp_path / "difficulty.json"
        calibration.write_text(
            json.dumps({"passed": False, "levels": [], "ratings": []}), encoding="utf-8"
        )

        with pytest.raises(ConfigError, match="did not meet S15"):
            build_manifest(report, sidecars=models, release="models-v2", calibration=calibration)
