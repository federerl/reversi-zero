"""CLI surface (backlog T03).

The CLI is the single entry point for laptop, GPU server, and SLURM. These tests
pin the surface so the sbatch scripts cannot silently drift from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from reversi.cli import app

runner = CliRunner()

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
EXPECTED_COMMANDS = ("train", "bench", "arena", "calibrate", "serve", "play", "config", "init-run")


def test_help_lists_every_workflow() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "reversi-zero" in result.output


def test_config_prints_resolved_yaml_and_hash() -> None:
    result = runner.invoke(app, ["config", "-c", str(CONFIG_DIR / "smoke4x4.yaml")])
    assert result.exit_code == 0
    assert "board_size: 4" in result.output
    assert "sha256:" in result.output


def test_config_sha_only_is_a_bare_hash() -> None:
    result = runner.invoke(app, ["config", "--sha-only"])
    assert result.exit_code == 0
    digest = result.output.strip()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_applies_overrides() -> None:
    result = runner.invoke(app, ["config", "-s", "mcts.n_simulations=222"])
    assert result.exit_code == 0
    assert "n_simulations: 222" in result.output


def test_bad_override_exits_2_with_a_readable_message() -> None:
    result = runner.invoke(app, ["config", "-s", "mcts.n_simulations=-5"])
    assert result.exit_code == 2
    assert "n_simulations" in result.output


def test_unknown_field_exits_2() -> None:
    result = runner.invoke(app, ["config", "-s", "mcts.simulations=100"])
    assert result.exit_code == 2


def test_init_run_creates_a_run_directory(run_root: Path) -> None:
    result = runner.invoke(app, ["init-run", "-c", str(CONFIG_DIR / "smoke4x4.yaml")])
    assert result.exit_code == 0
    created = Path(result.output.strip())
    assert (created / "config.yaml").is_file()
    assert (created / "meta.json").is_file()


@pytest.mark.parametrize("command", ["bench", "arena", "calibrate", "serve", "play"])
def test_unimplemented_commands_fail_loudly_and_name_their_task(command: str) -> None:
    """A stub must never look like a successful no-op."""
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented" in result.output


def test_train_runs_a_generation(run_root: Path) -> None:
    """``train`` is real as of day 5, so it must not be in the stub list above.

    Overridden down to a few games and a couple of steps: this checks the command
    wires up -- config, run directory, metrics, loop, summary -- not that anything
    learns. Learning is measured by the integration tests and by day 6's gate.
    """
    result = runner.invoke(
        app,
        [
            "train",
            "-c",
            "configs/smoke4x4.yaml",
            "--generations",
            "1",
            "--resume",
            "off",
            "-s",
            "selfplay.games_per_generation=2",
            "-s",
            "train.steps_per_generation=2",
            "-s",
            "train.batch_size=8",
            "-s",
            "mcts.n_simulations=4",
            "-s",
            "net.n_blocks=1",
            "-s",
            "net.channels=8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "finished 1 generation" in result.output

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run = run_dirs[0]
    assert (run / "config.yaml").is_file()
    assert (run / "checkpoints" / "latest.pt").is_file()
    assert list((run / "replay").glob("*.npz"))
    assert (run / "metrics" / "train.jsonl").is_file()
