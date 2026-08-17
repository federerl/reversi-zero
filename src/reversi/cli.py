"""Command-line entry point.

Every workflow -- local, GPU server, and SLURM -- goes through this one CLI. The
sbatch scripts in ``slurm/`` add scheduler directives, module loads, and
``RZ_RUN_ROOT``, then invoke exactly the same command as a laptop would. Nothing
under ``src/`` reads a ``SLURM_*`` variable except ``obs/runmeta.py``, which only
records them.

Subcommands that are not yet implemented exit with code 2 and name the backlog
task that will deliver them, rather than failing obscurely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from reversi import __version__
from reversi.config import Config, load_config
from reversi.errors import ReversiError
from reversi.obs import init_run, setup_logging

app = typer.Typer(
    name="reversi",
    help="AlphaZero-style Reversi: self-play training, evaluation, and web play.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Profile YAML, layered on configs/base.yaml."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override, e.g. --set mcts.n_simulations=200. Repeatable."),
]
RunIdOpt = Annotated[
    str | None,
    typer.Option("--run-id", help="Reuse an existing run id (required to resume)."),
]
ResumeOpt = Annotated[
    str,
    typer.Option("--resume", help="'auto', 'off', or an explicit checkpoint path."),
]
SuiteOpt = Annotated[
    str,
    typer.Option("--suite", help="'baselines', 'crossgen', or 'final'."),
]
ShaOnlyOpt = Annotated[
    bool,
    typer.Option("--sha-only", help="Print only the config hash."),
]
ValidateOpt = Annotated[
    bool,
    typer.Option("--validate", help="Check the shipped ladder instead of searching."),
]


def _load(config: Path | None, overrides: list[str] | None) -> Config:
    try:
        return load_config(config, overrides=overrides)
    except ReversiError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _not_yet(task: str, what: str) -> None:
    message = f"{what} is not implemented yet (backlog {task})."
    typer.secho(message, fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


# `invoke_without_command=True` is required for `--version`: a Click group runs
# its callback only when a subcommand is present, so without it `reversi
# --version` fails with "Missing command" before the callback is ever reached.
@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        typer.echo(f"reversi-zero {__version__}")
        raise typer.Exit


# ===========================================================================
# Config inspection -- implemented
# ===========================================================================


@app.command("config")
def show_config(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    sha_only: ShaOnlyOpt = False,
) -> None:
    """Resolve base + profile + overrides and print the result."""
    resolved = _load(config, set_)
    if sha_only:
        typer.echo(resolved.sha256)
        return
    typer.echo(resolved.to_yaml().rstrip())
    typer.echo(f"# sha256: {resolved.sha256}")


@app.command("init-run")
def init_run_cmd(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    run_id: RunIdOpt = None,
) -> None:
    """Create a run directory and write its provenance files.

    Useful on its own to verify that a cluster filesystem is writable *before*
    queueing an eight-hour job.
    """
    resolved = _load(config, set_)
    paths = init_run(resolved, run_id=run_id)
    setup_logging(log_dir=paths.logs, run_id=paths.run_id, component="cli")
    typer.echo(paths.root)


# ===========================================================================
# Pipeline -- stubs until their backlog tasks land
# ===========================================================================


@app.command()
def train(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    run_id: RunIdOpt = None,
    resume: ResumeOpt = "auto",
) -> None:
    """Run the generational self-play + training loop."""
    _load(config, set_)
    _ = (run_id, resume)
    _not_yet("T18", "training loop")


@app.command()
def bench(
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Measure engine, MCTS, inference, and self-play throughput."""
    _load(config, set_)
    _not_yet("T24", "benchmark suite")


@app.command()
def arena(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    suite: SuiteOpt = "baselines",
) -> None:
    """Play evaluation matches and report Wilson intervals and Bradley-Terry Elo."""
    _load(config, set_)
    _ = suite
    _not_yet("T27/T28", "arena")


@app.command()
def calibrate(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    validate: ValidateOpt = False,
) -> None:
    """Search for and validate the four difficulty levels."""
    _load(config, set_)
    _ = validate
    _not_yet("T33", "difficulty calibration")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Serve the web app and inference API."""
    _ = (host, port)
    _not_yet("T35", "web backend")


@app.command()
def play(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    difficulty: Annotated[str, typer.Option("--difficulty")] = "strong",
) -> None:
    """Play a game against the agent in the terminal."""
    _load(config, set_)
    _ = difficulty
    _not_yet("T29", "terminal play")


if __name__ == "__main__":  # pragma: no cover
    app()
