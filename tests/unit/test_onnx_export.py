"""The browser's copy of the network must be the same network.

An ONNX export that loads fine but computes something slightly different is the
failure worth testing for. Nothing would raise. The web app would play an agent
that no measurement in this repository was taken against, and it would look like
a weak model rather than a broken file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx", reason="the `web` extra is not installed")
pytest.importorskip("onnxruntime", reason="the `web` extra is not installed")

from reversi.ckpt import CheckpointManager
from reversi.config import NetConfig
from reversi.errors import CheckpointError
from reversi.nn.export import export_checkpoint
from reversi.nn.model import build
from reversi.nn.onnx import ONNX_OPSET, check_agreement, export_onnx, run_onnx

BOARD = 8


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small untrained network. These tests are about the export, not strength."""
    root = tmp_path_factory.mktemp("onnx")
    net = build(NetConfig(n_blocks=2, channels=16, value_hidden=32), BOARD, seed=7)
    manager = CheckpointManager(root / "ckpt", run_id="onnx-test", config_sha256="x")
    manager.save(model=net, generation=1, global_step=10)

    exported = root / "model.pt"
    export_checkpoint(root / "ckpt" / "gen_00001.pt", exported)
    return exported


def test_the_export_answers_what_pytorch_answers(model_path: Path, tmp_path: Path) -> None:
    """The whole point. Float32 rounding is fine; anything larger is not."""
    destination = tmp_path / "model.onnx"
    meta = export_onnx(model_path, destination)

    agreement = meta["agreement"]
    assert agreement is not None
    assert agreement["max_policy_diff"] < 1e-4
    assert agreement["max_value_diff"] < 1e-4
    assert meta["onnx_opset"] == ONNX_OPSET


def test_a_drifting_export_is_deleted_rather_than_shipped(model_path: Path, tmp_path: Path) -> None:
    """A bad file left on disk is worse than no file.

    Downstream, an .onnx is just bytes to copy to a CDN. If a broken one survives
    the export it gets published, and the first sign of trouble is an agent that
    plays badly for no visible reason.
    """
    destination = tmp_path / "drifted.onnx"

    from reversi.nn import onnx as onnx_module

    real = onnx_module.check_agreement

    def drifted(*args: object, **kwargs: object) -> onnx_module.AgreementReport:
        report = real(*args, **kwargs)  # type: ignore[arg-type]
        return onnx_module.AgreementReport(
            positions=report.positions,
            max_policy_diff=0.5,
            max_value_diff=report.max_value_diff,
            tolerance=report.tolerance,
        )

    onnx_module.check_agreement = drifted  # type: ignore[assignment]
    try:
        with pytest.raises(CheckpointError, match="does not agree"):
            export_onnx(model_path, destination)
    finally:
        onnx_module.check_agreement = real  # type: ignore[assignment]

    assert not destination.exists(), "a disagreeing export was left on disk"


def test_the_export_accepts_any_batch_size(model_path: Path, tmp_path: Path) -> None:
    """The browser may or may not batch its search leaves.

    Fixing the batch axis at 1 would make that an export-time decision rather
    than a runtime one, and finding out would mean re-exporting every model.
    """
    import onnxruntime as ort

    destination = tmp_path / "model.onnx"
    export_onnx(model_path, destination)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    for batch in (1, 3, 16):
        board = rng.random((batch, 3, BOARD, BOARD), dtype=np.float32)
        policy, value = run_onnx(session, board)
        assert policy.shape == (batch, BOARD * BOARD + 1)
        assert value.shape == (batch, 1)


def test_one_position_is_scored_the_same_alone_as_in_a_batch(
    model_path: Path, tmp_path: Path
) -> None:
    """Batching must not change an answer.

    It would be an easy thing to break -- a batch norm left in training mode
    normalises across the batch, so a position's score would depend on which
    other positions happened to be searched with it.
    """
    import onnxruntime as ort

    destination = tmp_path / "model.onnx"
    export_onnx(model_path, destination)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(1)
    batch = rng.random((8, 3, BOARD, BOARD), dtype=np.float32)

    grouped_policy, grouped_value = run_onnx(session, batch)
    for row in range(8):
        alone_policy, alone_value = run_onnx(session, batch[row : row + 1])
        assert np.abs(alone_policy[0] - grouped_policy[row]).max() < 1e-5
        assert abs(float(alone_value[0][0]) - float(grouped_value[row][0])) < 1e-5


def test_agreement_is_measured_on_the_whole_graph(model_path: Path, tmp_path: Path) -> None:
    """Random inputs rather than real positions, deliberately.

    A real board is mostly zeros, so a layer that got dropped in the export might
    never be reached by the check that is supposed to notice.
    """
    destination = tmp_path / "model.onnx"
    export_onnx(model_path, destination, verify=False)

    from reversi.nn.export import load_export

    model = load_export(model_path, device="cpu").model
    report = check_agreement(model, destination, positions=16)

    assert report.ok
    assert report.positions == 16
    assert "16 positions" in report.describe()


def test_the_export_is_one_self_contained_file(model_path: Path, tmp_path: Path) -> None:
    """The browser fetches one URL, so the weights have to be inside the file.

    torch's newer exporter writes tensors to a sibling `.onnx.data` when it
    judges them large enough, and the `.onnx` then refers to it by name. The
    browser cannot follow that reference -- it fails with "Module.MountedFiles is
    not available" -- and because the behaviour depends on the torch version, it
    embedded on the machine this was written on and split on CI.
    """
    from reversi.nn.onnx import _has_external_data

    destination = tmp_path / "model.onnx"
    meta = export_onnx(model_path, destination)

    assert not _has_external_data(destination)
    assert meta["self_contained"] is True
    strays = list(tmp_path.glob("*.data"))
    assert strays == [], f"weights were left outside the model: {strays}"


def test_a_model_split_across_files_is_repaired_rather_than_shipped(
    model_path: Path, tmp_path: Path
) -> None:
    """Deliberately split a model, then export over it and check it comes back whole.

    This is the failure mode reproduced rather than described: without the
    embedding step the .onnx would still load in Python -- which is why it went
    unnoticed -- and fail only in a browser.
    """
    import onnx

    from reversi.nn.onnx import _embed_weights, _has_external_data

    destination = tmp_path / "split.onnx"
    export_onnx(model_path, destination)

    # Push the weights out to a sidecar, the way the newer exporter does.
    model = onnx.load(str(destination))
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        location="split.onnx.data",
        size_threshold=0,
    )
    assert _has_external_data(destination), "the split did not take effect"

    _embed_weights(destination)

    assert not _has_external_data(destination)
    assert not (tmp_path / "split.onnx.data").exists()
