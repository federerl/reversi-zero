"""Turning an exported model into something a browser can run.

The web app runs the network on the visitor's own machine rather than on a
server. That needs the weights in a format the browser understands, and ONNX is
that format: a plain description of the arithmetic -- these convolutions, this
addition, that tanh -- with no PyTorch anywhere in the picture.

**Why this is worth doing at all.** A search at the strongest level asks the
network about 800 positions before it plays a move. Done on a server, that is
seconds of one CPU core per move, per player, and a machine that has to stay
awake. Done in the browser it costs us nothing and cannot run out. The whole
network is 458k numbers -- about the size of a photograph -- so handing it to
each visitor is cheaper than renting a machine to hold it for them.

**The check that matters here is agreement, not file size.** An export that
loads happily but computes something slightly different would not fail. The
browser would play a subtly different agent than the one every measurement in
this repository was taken against, and nothing would say so. ``export_onnx``
therefore runs both versions on random inputs and refuses to write a file whose
answers do not match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from reversi.atomicio import sha256_file
from reversi.errors import CheckpointError
from reversi.nn.export import load_export

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from reversi.nn.model import PolicyValueNet

__all__ = [
    "ONNX_OPSET",
    "AgreementReport",
    "check_agreement",
    "export_onnx",
    "run_onnx",
]

# Opset 17 covers every operation this network uses (convolution, batch norm,
# ReLU, matmul, tanh) and is what ONNX Runtime Web supports without a shim.
# Nothing here needs a newer one.
ONNX_OPSET = 17

# Float32 arithmetic reordered by a different runtime drifts in the last few
# digits. Anything beyond this is a real behaviour change, not rounding.
AGREEMENT_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How far the exported network's answers drift from PyTorch's."""

    positions: int
    max_policy_diff: float
    max_value_diff: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return max(self.max_policy_diff, self.max_value_diff) <= self.tolerance

    def describe(self) -> str:
        return (
            f"{self.positions} positions, worst policy difference "
            f"{self.max_policy_diff:.2e}, worst value difference "
            f"{self.max_value_diff:.2e} (tolerance {self.tolerance:.0e})"
        )


def run_onnx(
    session: Any, board: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Run a batch through a session and return ``(policy, value)`` as arrays.

    ``InferenceSession.run`` is typed as returning any of several output kinds,
    because in general an ONNX graph may emit sparse tensors, lists or maps.
    This one emits two dense tensors and nothing else, so the narrowing happens
    here, once, with a message that names the real problem if it ever does not.
    """
    outputs = session.run(None, {"board": board})
    if len(outputs) != 2:
        msg = f"expected the network to return a policy and a value, got {len(outputs)} outputs"
        raise CheckpointError(msg)

    policy, value = outputs
    if not isinstance(policy, np.ndarray) or not isinstance(value, np.ndarray):
        msg = (
            "the network returned something other than plain arrays "
            f"({type(policy).__name__}, {type(value).__name__}). The .onnx file is "
            "not the one this code expects."
        )
        raise CheckpointError(msg)
    return policy, value


def check_agreement(
    model: PolicyValueNet,
    onnx_path: Path,
    *,
    positions: int = 64,
    seed: int = 0,
    tolerance: float = AGREEMENT_TOLERANCE,
) -> AgreementReport:
    """Run both versions of the network on the same inputs and compare.

    Random inputs rather than real positions, deliberately: a real position
    lights up a fraction of the network, and a layer that got dropped or
    reordered in the export might simply not be reached. Random values in every
    plane exercise the whole graph.
    """
    import onnxruntime as ort  # imported here so the package is optional at runtime

    size = model.board_size
    rng = np.random.default_rng(seed)
    batch = rng.random((positions, model.in_planes, size, size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        torch_policy, torch_value = model(torch.from_numpy(batch))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = run_onnx(session, batch)

    return AgreementReport(
        positions=positions,
        max_policy_diff=float(np.abs(torch_policy.numpy() - onnx_policy).max()),
        max_value_diff=float(np.abs(torch_value.numpy() - onnx_value).max()),
        tolerance=tolerance,
    )


def export_onnx(
    exported_model: Path,
    destination: Path,
    *,
    verify: bool = True,
    check_positions: int = 64,
) -> dict[str, Any]:
    """Convert an exported ``.pt`` model to ONNX, refusing a version that drifts.

    ``verify`` runs :func:`check_agreement` and raises if the two disagree by
    more than float32 noise. Turning it off is only sensible when deliberately
    inspecting a bad export.
    """
    loaded = load_export(exported_model, device="cpu")
    model = loaded.model
    size = model.board_size

    destination.parent.mkdir(parents=True, exist_ok=True)

    # A dynamic batch axis, because the browser may or may not batch its search
    # leaves. Fixing it at 1 would make that a re-export rather than a decision.
    dummy = torch.zeros(1, model.in_planes, size, size)
    torch.onnx.export(
        model,
        (dummy,),
        str(destination),
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=ONNX_OPSET,
    )

    report: AgreementReport | None = None
    if verify:
        report = check_agreement(model, destination, positions=check_positions)
        if not report.ok:
            destination.unlink(missing_ok=True)
            msg = (
                f"the ONNX export of {exported_model.name} does not agree with the "
                f"PyTorch model it came from: {report.describe()}. The exported file "
                "has been removed rather than left in place, because a browser "
                "running it would silently play a different agent."
            )
            raise CheckpointError(msg)

    meta = {
        "onnx_opset": ONNX_OPSET,
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_export": exported_model.name,
        "source_sha256": loaded.meta.get("sha256"),
        "run_id": loaded.meta.get("run_id"),
        "generation": loaded.meta.get("generation"),
        "label": loaded.label,
        "arch": loaded.meta.get("arch"),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "agreement": None
        if report is None
        else {
            "positions": report.positions,
            "max_policy_diff": report.max_policy_diff,
            "max_value_diff": report.max_value_diff,
            "tolerance": report.tolerance,
        },
    }
    destination.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return meta
