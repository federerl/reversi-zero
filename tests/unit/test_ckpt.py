"""Checkpoints that survive being interrupted (test matrix T22, T23).

The event that makes a resume necessary -- a kill, a reboot, a dropped session --
is exactly the event that leaves half-written files. So these tests are mostly
about damage: what happens when a checkpoint is truncated, when its metadata is
missing, when the architecture no longer matches. In every case the answer has to
be a clear refusal, never a silent load of something wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from reversi.ckpt import CheckpointManager, CheckpointMeta
from reversi.ckpt.meta import FORMAT_VERSION, checkpoint_name, sidecar_for
from reversi.config import TrainConfig
from reversi.data.replay import Batch
from reversi.errors import CheckpointError
from reversi.nn.model import PolicyValueNet
from reversi.train.trainer import Trainer


def model(board_size: int = 4, channels: int = 8) -> PolicyValueNet:
    return PolicyValueNet(board_size, n_blocks=1, channels=channels, value_hidden=16)


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "checkpoints", run_id="test-run", config_sha256="abc123")


def save_one(manager: CheckpointManager, generation: int, net: PolicyValueNet) -> CheckpointMeta:
    return manager.save(
        model=net,
        generation=generation,
        global_step=generation * 10,
        optimizer_state={"state": {}, "param_groups": []},
        games_played=100,
        positions_seen=900,
    )


# ===========================================================================
# Round trip (T22)
# ===========================================================================


def test_a_checkpoint_round_trips(manager: CheckpointManager) -> None:
    net = model()
    meta = save_one(manager, 3, net)

    restored = manager.newest_valid()
    assert restored is not None
    assert restored.meta.generation == 3
    assert restored.meta.global_step == 30
    assert restored.next_generation == 4

    rebuilt = model()
    rebuilt.load_state_dict(restored.payload["model_state_dict"])
    for before, after in zip(net.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(before, after)

    assert meta.arch == net.arch()
    assert meta.format_version == FORMAT_VERSION


def test_the_optimiser_state_survives(manager: CheckpointManager) -> None:
    """The piece people forget.

    AdamW keeps a running estimate per parameter. Resume without it and the first
    few hundred steps after every restart are taken with the wrong step sizes --
    which does not fail, it just quietly wastes the beginning of every session.
    """
    net = model()
    trainer = Trainer(net, TrainConfig(lr=1e-2, warmup_steps=0), total_steps=100)

    # One real step, so the optimiser has accumulated something worth saving.
    # Saving an untouched optimiser would pass a shallower version of this test
    # while proving nothing.
    rng = np.random.default_rng(0)
    trainer.train_on(
        Batch(
            planes=rng.integers(0, 2, size=(4, 3, 4, 4)).astype(np.float32),
            pi=np.full((4, 17), 1 / 17, dtype=np.float32),
            z=np.ones(4, dtype=np.float32),
        )
    )

    manager.save(
        model=net,
        generation=1,
        global_step=trainer.global_step,
        optimizer_state=trainer.optimizer.state_dict(),
    )

    restored = manager.newest_valid()
    assert restored is not None
    saved = restored.payload["optimizer_state_dict"]
    assert saved is not None
    assert saved["state"], "the running per-parameter estimates should be there"

    # And it loads back into a fresh optimiser without complaint.
    fresh = Trainer(model(), TrainConfig(lr=1e-2, warmup_steps=0), total_steps=100)
    fresh.optimizer.load_state_dict(saved)


def test_the_random_generators_are_captured(manager: CheckpointManager) -> None:
    save_one(manager, 1, model())
    restored = manager.newest_valid()

    assert restored is not None
    rng = restored.payload["rng"]
    assert rng is not None
    assert {"python", "numpy", "torch_cpu"} <= set(rng)


def test_the_sidecar_is_readable_without_torch(manager: CheckpointManager) -> None:
    """Lineage has to be greppable on a machine with no virtualenv."""
    save_one(manager, 5, model())
    sidecar = manager.directory / "gen_00005.json"

    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["generation"] == 5
    assert payload["run_id"] == "test-run"
    assert payload["arch"]["board_size"] == 4
    assert len(payload["sha256"]) == 64


def test_lineage_links_each_checkpoint_to_its_parent(manager: CheckpointManager) -> None:
    """Following the chain backwards reconstructs a run across restarts, where
    generation numbers alone would not say what came from what."""
    net = model()
    manager.save(model=net, generation=1, global_step=10)
    manager.save(model=net, generation=2, global_step=20, parent=checkpoint_name(1))

    second = manager.read_meta(manager.directory / checkpoint_name(2))
    assert second.parent == "gen_00001.pt"


def test_latest_is_a_copy_not_a_link(manager: CheckpointManager) -> None:
    """Symlinks need elevated privileges on Windows; this runs on a laptop too."""
    save_one(manager, 2, model())
    latest = manager.directory / "latest.pt"

    assert latest.is_file()
    assert not latest.is_symlink()
    assert latest.read_bytes() == (manager.directory / checkpoint_name(2)).read_bytes()


# ===========================================================================
# Damage (T23)
# ===========================================================================


def test_a_truncated_checkpoint_is_refused(manager: CheckpointManager) -> None:
    save_one(manager, 1, model())
    path = manager.directory / checkpoint_name(1)
    path.write_bytes(path.read_bytes()[:100])

    with pytest.raises(CheckpointError, match="does not match its recorded checksum"):
        manager.verify(path)


def test_a_checkpoint_with_no_sidecar_is_refused(manager: CheckpointManager) -> None:
    """The window between writing the weights and writing the metadata.

    A process killed there leaves a .pt whose contents nothing has vouched for.
    Resume must treat it as absent, not as newest.
    """
    save_one(manager, 1, model())
    sidecar_for(manager.directory / checkpoint_name(1)).unlink()

    with pytest.raises(CheckpointError, match="no metadata sidecar"):
        manager.verify(manager.directory / checkpoint_name(1))


def test_resume_falls_back_to_the_previous_generation(manager: CheckpointManager) -> None:
    """The realistic failure: killed while saving. One bad checkpoint at the end.

    Falling back one generation costs a few minutes of self-play. Trusting the
    damaged file would resume from garbage and never say so.
    """
    net = model()
    save_one(manager, 1, net)
    save_one(manager, 2, net)
    save_one(manager, 3, net)

    broken = manager.directory / checkpoint_name(3)
    broken.write_bytes(b"killed halfway through")

    restored = manager.newest_valid()

    assert restored is not None
    assert restored.meta.generation == 2, "should skip the damaged newest one"


def test_an_empty_directory_resumes_to_nothing(manager: CheckpointManager) -> None:
    assert manager.newest_valid() is None


def test_a_newer_format_version_is_refused(manager: CheckpointManager) -> None:
    """Loading half of a format we do not understand is worse than stopping."""
    save_one(manager, 1, model())
    path = manager.directory / checkpoint_name(1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = FORMAT_VERSION + 5
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="written by a newer version"):
        manager.load(path)


def test_metadata_with_unknown_fields_is_refused(tmp_path: Path) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "run_id": "r",
        "generation": 1,
        "global_step": 1,
        "arch": {},
        "config_sha256": "x",
        "created_utc": "now",
        "sha256": "y",
        "something_from_the_future": 42,
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="unexpected fields"):
        CheckpointMeta.read(path)


def test_metadata_missing_required_fields_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"generation": 1}), encoding="utf-8")

    with pytest.raises(CheckpointError, match="missing required fields"):
        CheckpointMeta.read(path)


def test_corrupt_metadata_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(CheckpointError, match="not valid JSON"):
        CheckpointMeta.read(path)


# ===========================================================================
# Pruning
# ===========================================================================


def test_pruning_keeps_the_recent_ones_and_a_thinned_history(
    manager: CheckpointManager,
) -> None:
    """Old checkpoints are the opponents for the cross-generation tournament.

    Deleting all of them would save disk and make "is generation 40 stronger than
    generation 10?" impossible to answer afterwards -- so a spread is kept.
    """
    net = model()
    for generation in range(1, 26):
        manager.save(model=net, generation=generation, global_step=generation)

    removed = manager.prune(keep_last=10, keep_every=5)
    left = manager.generations()

    assert set(range(16, 26)) <= set(left), "the ten newest must all survive"
    assert 5 in left and 10 in left and 15 in left, "every fifth is kept as an anchor"
    assert 7 in removed and 13 in removed
    for generation in removed:
        assert not sidecar_for(manager.directory / checkpoint_name(generation)).exists()


def test_pruning_a_short_run_removes_nothing(manager: CheckpointManager) -> None:
    net = model()
    for generation in (1, 2, 3):
        manager.save(model=net, generation=generation, global_step=generation)

    assert manager.prune(keep_last=10) == []
    assert manager.generations() == [1, 2, 3]
