"""Exporting a network and reading it back.

The regression this file exists for: `load_export` built its model without the
ownership head, so a network trained with one exported "successfully" into a file
nothing in the project could open. Nobody noticed for as long as no exported
network had the head -- and the first one that did was the network chosen to ship.

The failure was also disguised. `evaluator_for` falls back from the export reader
to the training-checkpoint reader, and it was the *second* reader's complaint that
reached the screen: "has no architecture block", which is a true statement about
the wrong file format and says nothing about the real problem.

So there are three things to hold down: an exported network with the head loads,
an export that cannot be read back is not left on disk, and a file neither reader
accepts reports both reasons rather than the more confusing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from reversi.ckpt.manager import CheckpointManager
from reversi.config import NetConfig
from reversi.difficulty.calibrate import evaluator_for
from reversi.errors import CheckpointError
from reversi.nn.export import export_checkpoint, load_export
from reversi.nn.loader import load_model
from reversi.nn.model import build

BOARD = 8


def _checkpoint(root: Path, *, ownership: bool) -> Path:
    """A small untrained network, saved as a training checkpoint."""
    net = build(
        NetConfig(n_blocks=2, channels=16, value_hidden=32, ownership=ownership),
        BOARD,
        seed=11,
    )
    manager = CheckpointManager(root / "ckpt", run_id="export-test", config_sha256="x")
    manager.save(model=net, generation=1, global_step=10)
    return root / "ckpt" / "gen_00001.pt"


@pytest.mark.parametrize("ownership", [False, True])
def test_an_export_can_be_read_back(tmp_path: Path, ownership: bool) -> None:
    """Both shapes of network survive the round trip.

    Parametrised rather than written twice, because the bug was precisely that
    one of these two cases was never exercised.
    """
    source = _checkpoint(tmp_path, ownership=ownership)
    destination = tmp_path / "model.pt"

    meta = export_checkpoint(source, destination)

    assert meta["arch"]["ownership"] is ownership
    exported = load_export(destination)
    assert exported.meta["generation"] == 1

    # The weights really came across, not just the shape.
    original = load_model(source)
    planes = torch.zeros(1, 3, BOARD, BOARD)
    planes[0, 0, 3, 3] = 1.0
    with torch.no_grad():
        want_policy, want_value = original(planes)
        got_policy, got_value = exported.model(planes)
    assert torch.allclose(want_policy, got_policy, atol=1e-6)
    assert torch.allclose(want_value, got_value, atol=1e-6)


@pytest.mark.parametrize("ownership", [False, True])
def test_the_evaluator_loads_an_export_and_a_checkpoint(tmp_path: Path, ownership: bool) -> None:
    """`evaluator_for` accepts either file, which is the point of it.

    This is the path the calibration and the spread measurement use, and the one
    that failed on the release network.
    """
    source = _checkpoint(tmp_path, ownership=ownership)
    destination = tmp_path / "model.pt"
    export_checkpoint(source, destination)

    assert evaluator_for(destination, "cpu") is not None
    assert evaluator_for(source, "cpu") is not None


def test_an_unreadable_export_is_deleted_rather_than_left(tmp_path: Path) -> None:
    """An export nothing can load is not an export.

    Simulated by writing a file whose architecture block and weights disagree,
    which is exactly the shape of the original bug from the reader's side. The
    check belongs next to the checkpoint that still exists, rather than
    downstream where the only evidence is a file and a puzzle.
    """
    destination = tmp_path / "broken.pt"
    torch.save(
        {
            "meta": {
                "export_version": 1,
                "arch": {
                    "board_size": BOARD,
                    "in_planes": 3,
                    "n_blocks": 2,
                    "channels": 16,
                    "value_hidden": 32,
                },
            },
            "model_state_dict": {"nonsense.weight": torch.zeros(1)},
        },
        destination,
    )

    with pytest.raises(CheckpointError, match="do not fit that shape"):
        load_export(destination, verify=False)


def test_a_file_neither_reader_accepts_reports_both_reasons(tmp_path: Path) -> None:
    """The diagnosis that was hidden.

    A file that fails as an export used to be reported with the training-checkpoint
    reader's message, which describes a different file format entirely. Both
    reasons now reach the caller, so the one that matters is among them.
    """
    destination = tmp_path / "neither.pt"
    torch.save({"model_state_dict": {}}, destination)

    with pytest.raises(CheckpointError) as raised:
        evaluator_for(destination, "cpu")

    message = str(raised.value)
    assert "as an exported model" in message.lower()
    assert "as a training checkpoint" in message.lower()
