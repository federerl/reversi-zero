"""Rebuilding a network from a saved file.

A checkpoint stores the weights plus an ``arch`` block describing the shape they
belong to. Loading checks the two agree before copying anything, so a mismatch
comes out as a sentence naming what differs rather than as a shape error forty
lines deep inside torch.

That matters more than it sounds. The failure it prevents is loading last week's
6-block checkpoint into today's 8-block code, which torch would reject -- but a
*channel* change of the same total size might not be, and would then produce an
agent whose weights mean nothing while playing perfectly legal moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from reversi.errors import CheckpointError
from reversi.nn.model import PolicyValueNet

__all__ = ["load_model", "read_payload"]

_REQUIRED_ARCH = ("board_size", "in_planes", "n_blocks", "channels", "value_hidden")


def read_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint file without building anything from it."""
    if not path.exists():
        msg = f"checkpoint {path} does not exist"
        raise CheckpointError(msg)
    try:
        # weights_only=False because the payload carries plain metadata (ints,
        # strings, a small dict) alongside the tensors. These files are produced
        # by this project, not downloaded.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        msg = f"checkpoint {path} could not be read: {error}"
        raise CheckpointError(msg) from error

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        msg = f"checkpoint {path} is not a reversi checkpoint (no model_state_dict)"
        raise CheckpointError(msg)
    return payload


def load_model(path: Path, *, device: str | torch.device = "cpu") -> PolicyValueNet:
    """Rebuild the network a checkpoint describes, ready to play with.

    Returned in ``eval()`` mode, because every use of a loaded model is inference.
    """
    payload = read_payload(path)

    arch = payload.get("arch")
    if not isinstance(arch, dict):
        msg = f"checkpoint {path} has no architecture block, so its weights cannot be placed"
        raise CheckpointError(msg)

    missing = [key for key in _REQUIRED_ARCH if key not in arch]
    if missing:
        msg = f"checkpoint {path} is missing architecture fields {missing}"
        raise CheckpointError(msg)

    model = PolicyValueNet(
        arch["board_size"],
        n_blocks=arch["n_blocks"],
        channels=arch["channels"],
        value_hidden=arch["value_hidden"],
        in_planes=arch["in_planes"],
        ownership=bool(arch.get("ownership", False)),
    )

    try:
        model.load_state_dict(payload["model_state_dict"])
    except (RuntimeError, KeyError, TypeError) as error:
        msg = (
            f"checkpoint {path} describes {arch} but its weights do not fit that "
            f"shape: {error}. The file and the code that wrote it disagree."
        )
        raise CheckpointError(msg) from error

    model.to(device)
    model.eval()
    return model
