"""Run provenance (backlog T04, success criterion S19).

S19: every run directory records config, seed, git commit, environment, and
hardware. A run missing any of these is not reproducible, so the schema is
asserted here rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reversi.config import Config, load_config
from reversi.obs.runmeta import RunPaths, environment_info, git_info, init_run, make_run_id

REQUIRED_FILES = ("config.yaml", "env.json", "git.json", "meta.json", "cmdline.txt")
REQUIRED_DIRS = ("checkpoints", "replay", "logs", "metrics", "arena")


def test_run_id_is_sortable_and_descriptive() -> None:
    run_id = make_run_id("full8x8", 1337, git_short="a1b2c3d")
    assert run_id.endswith("-full8x8-a1b2c3d-s1337")
    stamp = run_id.split("-")[0]
    assert len(stamp) == 8 and stamp.isdigit()


def test_run_ids_sort_chronologically() -> None:
    early = "20260101-000000-dev8x8-aaaaaaa-s1"
    late = "20260817-235959-dev8x8-aaaaaaa-s1"
    assert sorted([late, early]) == [early, late]


def test_init_run_writes_every_required_artifact(run_root: Path, smoke_config: Config) -> None:
    paths = init_run(smoke_config, argv=["reversi", "train", "-c", "configs/smoke4x4.yaml"])

    assert paths.root.is_relative_to(run_root), "RZ_RUN_ROOT must be honoured"
    for name in REQUIRED_FILES:
        assert (paths.root / name).is_file(), f"missing {name}"
    for name in REQUIRED_DIRS:
        assert (paths.root / name).is_dir(), f"missing {name}/"


def test_written_config_round_trips(run_root: Path, smoke_config: Config) -> None:
    paths = init_run(smoke_config)
    restored = Config.model_validate(yaml.safe_load(paths.config_file.read_text(encoding="utf-8")))
    assert restored.sha256 == smoke_config.sha256


def test_meta_records_identity(run_root: Path, smoke_config: Config) -> None:
    paths = init_run(smoke_config)
    meta = json.loads(paths.meta_file.read_text(encoding="utf-8"))
    assert meta["run_id"] == paths.run_id
    assert meta["seed"] == smoke_config.seed
    assert meta["config_sha256"] == smoke_config.sha256
    assert meta["config_name"] == smoke_config.name
    assert meta["created_utc"]


def test_cmdline_is_recorded(run_root: Path, smoke_config: Config) -> None:
    argv = ["reversi", "train", "--set", "mcts.n_simulations=99"]
    paths = init_run(smoke_config, argv=argv)
    assert (paths.root / "cmdline.txt").read_text(encoding="utf-8").strip() == " ".join(argv)


def test_resume_preserves_config_hash_history(run_root: Path, smoke_config: Config) -> None:
    """A resumed run that changed config must leave both hashes on the record."""
    paths = init_run(smoke_config)
    changed = smoke_config.model_copy(update={"seed": smoke_config.seed + 1})
    init_run(changed, run_id=paths.run_id)

    meta = json.loads(paths.meta_file.read_text(encoding="utf-8"))
    assert meta["config_sha256"] == changed.sha256
    assert smoke_config.sha256 in meta["config_sha256_history"]
    assert changed.sha256 in meta["config_sha256_history"]


def test_env_json_has_result_explaining_fields(run_root: Path, smoke_config: Config) -> None:
    paths = init_run(smoke_config)
    env = json.loads((paths.root / "env.json").read_text(encoding="utf-8"))
    for key in ("python", "platform", "hostname", "cpu_count_logical", "torch", "slurm"):
        assert key in env, f"env.json is missing {key}"
    assert "available" in env["torch"]


def test_git_json_identifies_the_code(run_root: Path, smoke_config: Config) -> None:
    paths = init_run(smoke_config)
    git = json.loads((paths.root / "git.json").read_text(encoding="utf-8"))
    if not git.get("available"):
        pytest.skip("not a git checkout")
    assert len(git["commit"]) == 40
    assert git["commit_short"] == git["commit"][:7]
    assert isinstance(git["dirty"], bool)
    assert git["diff_sha256"]


def test_environment_info_is_json_serialisable() -> None:
    json.dumps(environment_info(), default=str)


def test_git_info_never_raises_outside_a_repo(tmp_path: Path) -> None:
    info = git_info(tmp_path)
    assert isinstance(info, dict)
    assert "available" in info


def test_runpaths_layout_is_stable(tmp_path: Path) -> None:
    paths = RunPaths(run_id="rid", root=tmp_path / "rid")
    assert paths.checkpoints.name == "checkpoints"
    assert paths.metrics.name == "metrics"
    assert paths.config_file.name == "config.yaml"


def test_default_run_root_is_used_without_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RZ_RUN_ROOT", raising=False)
    config = load_config().model_copy(update={"run_root": tmp_path / "custom"})
    paths = init_run(config)
    assert paths.root.is_relative_to(tmp_path / "custom")
