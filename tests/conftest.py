"""Shared test fixtures.

The important one is ``_block_cuda``: unless a test is explicitly marked
``@pytest.mark.gpu``, CUDA is made to look unavailable. This enforces the rule
that the whole test suite -- and therefore CI -- runs on CPU, and catches any
module that silently assumes a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from reversi.config import Config


@pytest.fixture(autouse=True)
def _block_cuda(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CUDA invisible to every test that has not opted in."""
    if request.node.get_closest_marker("gpu") is not None:
        return
    try:
        import torch
    except ImportError:
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect run artifacts into tmp_path so tests never write to ./runs."""
    root = tmp_path / "runs"
    monkeypatch.setenv("RZ_RUN_ROOT", str(root))
    yield root


@pytest.fixture
def smoke_config() -> Config:
    """A tiny 4x4 config that is fast enough to use inside unit tests."""
    return Config.model_validate(
        {
            "name": "test4x4",
            "seed": 7,
            "game": {"board_size": 4},
            "net": {"n_blocks": 1, "channels": 8, "value_hidden": 16},
            "mcts": {"n_simulations": 8, "temp_moves": 2, "dirichlet_alpha": 2.0},
            "selfplay": {
                "games_per_generation": 4,
                "n_workers": 1,
                "games_in_flight": 2,
                "max_generations": 2,
            },
            "replay": {"window": 512, "retain_shards": 4},
            "train": {"steps_per_generation": 2, "batch_size": 8},
            "arena": {"every_n_generations": 1, "games": 4, "opening_plies": 1},
        }
    )
