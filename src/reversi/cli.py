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

import os
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


def _pick_device(requested: str) -> str:
    """Resolve --device, preferring the GPU when one is there.

    Defaulting to the CPU would be the safe-looking choice and the wrong one: it
    fails silently. An 8x8 run on the CPU is roughly ten times slower, so the
    first sign of the mistake would be a night that produced three generations
    instead of nineteen.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            typer.secho(
                "--device cuda was asked for but no GPU is visible.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        return "cuda"
    if requested != "auto":
        typer.secho(
            f"--device must be 'auto', 'cpu', or 'cuda'; got {requested!r}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    generations: Annotated[
        int | None,
        typer.Option("--generations", help="Override selfplay.max_generations for this run."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="'auto', 'cpu', or 'cuda'. Auto prefers the GPU."),
    ] = "auto",
) -> None:
    """Run the generational self-play + training loop.

    Writes everything under ``runs/<run_id>/``: the resolved config, the
    provenance files, one replay shard and one checkpoint per generation, and the
    JSONL metric streams that every figure in the final report is built from.
    """
    from reversi.obs.metrics import MetricsHub
    from reversi.obs.signals import cooperative_stop
    from reversi.train.loop import run_training

    if resume not in {"auto", "off"}:
        typer.secho(
            f"--resume must be 'auto' or 'off', got {resume!r}. Resuming from an "
            "explicit checkpoint path is not supported: a run resumes from its own "
            "directory, which is what keeps the replay shards and the weights in step.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    resolved = _load(config, set_)
    chosen = _pick_device(device)
    paths = init_run(resolved, run_id=run_id)
    setup_logging(log_dir=paths.logs, run_id=paths.run_id, component="train")

    typer.echo(f"run: {paths.root}")
    typer.echo(f"device: {chosen}")
    if chosen == "cpu" and device == "auto":
        typer.secho(
            "no GPU visible, so this will train on the CPU. Measured on 8x8 that is "
            'roughly ten times slower -- check `python -c "import torch; '
            'print(torch.cuda.is_available())"` if you expected otherwise.',
            fg=typer.colors.YELLOW,
            err=True,
        )
    if resume == "off":
        typer.secho(
            "--resume off: starting from freshly initialised weights even if this "
            "directory already holds checkpoints.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # A signal now means "finish this generation, then stop" rather than dying
    # partway through writing a shard. Send it twice to stop immediately.
    with (
        cooperative_stop() as stop,
        MetricsHub(
            paths.metrics,
            run_id=paths.run_id,
            tb_dir=paths.tb if resolved.obs.tensorboard else None,
        ) as metrics,
    ):
        reports = run_training(
            resolved,
            paths,
            metrics=metrics,
            generations=generations,
            should_stop=stop,
            resume=resume == "auto",
            device=chosen,
        )

    if not reports:
        typer.echo("nothing to do: this run has already reached its generation target")
        return

    if stop.requested:
        typer.secho(
            f"stopped after generation {reports[-1].generation} on {stop.reason}. "
            "Re-run the same command to carry on.",
            fg=typer.colors.YELLOW,
        )

    last = reports[-1]
    typer.echo(
        f"finished {len(reports)} generation(s): "
        f"{sum(r.positions for r in reports)} positions, "
        f"final loss {last.training['total_loss']:.4f} "
        f"(policy {last.training['policy_loss']:.4f}, value {last.training['value_loss']:.4f})"
    )
    typer.echo(f"checkpoint: {last.checkpoint}")


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
    model: Annotated[Path, typer.Argument(help="An exported model, from `reversi export`")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the report.")] = Path(
        "runs/calibration/difficulty_report.json"
    ),
    games: Annotated[
        int, typer.Option("--games", help="Games per pairing. 300 is the plan's figure.")
    ] = 300,
    board_size: Annotated[int, typer.Option("--board-size")] = 8,
    guard_samples: Annotated[
        int, typer.Option("--guard-samples", help="Moves to inspect for the guardrail check.")
    ] = 500,
    seed: Annotated[int, typer.Option("--seed")] = 20260830,
    device: Annotated[
        str,
        typer.Option("--device", help="'cpu' or 'cuda'. CPU is usually faster here."),
    ] = "cpu",
    workers: Annotated[
        int,
        typer.Option("--workers", help="Pairings to play at once. 1 runs in this process."),
    ] = 1,
    no_baselines: Annotated[
        bool,
        typer.Option("--no-baselines", help="Rate the levels against each other only."),
    ] = False,
    write_config: Annotated[
        Path | None,
        typer.Option("--write-config", help="Also write the settings, with their ratings."),
    ] = None,
) -> None:
    """Measure the difficulty ladder and check it against criterion S15.

    The four levels are *designed* to differ -- each searches more than the one
    below it and tolerates less of a drop in value before refusing a move -- but
    designed is not measured. A ladder whose rungs are not actually separated is
    four names for one opponent, and a player who moves up a level and notices
    nothing has been told something untrue.

    This plays every level against every other and against the frozen baselines,
    fits the whole result matrix at once, and reports whether the ratings rise,
    whether adjacent rungs are at least 80 Elo apart with intervals that do not
    overlap, whether the easiest still beats random play, and whether its
    guardrail held over 500 of its own moves.

    Exits non-zero when the ladder does not hold up, because a difficulty setting
    that is not separated is a claim the interface should not be making.
    """
    from reversi.difficulty.calibrate import calibrate as run_calibration
    from reversi.difficulty.calibrate import write_difficulty_config, write_report
    from reversi.difficulty.levels import LEVELS

    setup_logging()
    try:
        report = run_calibration(
            model,
            board_size=board_size,
            games_per_pair=games,
            seed=seed,
            device=device,
            guard_samples=guard_samples,
            include_baselines=not no_baselines,
            workers=workers,
        )
    except ReversiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    typer.echo("")
    typer.echo(report.tournament.ratings.describe())
    typer.echo("")
    typer.echo(report.summary())

    write_report(out, report, model)
    typer.echo("")
    typer.echo(f"report: {out}")

    if write_config is not None:
        write_difficulty_config(write_config, report, list(LEVELS), model)
        typer.echo(f"config: {write_config}")

    if not report.passed:
        typer.secho(
            "S15 is not met. The plan's documented options, in order: widen the "
            "simulation ratio, or ship fewer levels and say so -- never four tiers "
            "that are not actually different.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("export")
def export_cmd(
    checkpoint: Annotated[Path, typer.Argument(help="A training checkpoint, e.g. .../latest.pt")],
    destination: Annotated[Path, typer.Argument(help="Where to write the play-only model")],
    notes: Annotated[str, typer.Option("--notes", help="Recorded in the sidecar.")] = "",
) -> None:
    """Write a play-only copy of a checkpoint, for serving or releasing.

    A training checkpoint carries optimiser and RNG state that only mean anything
    inside the run that made them. An export carries the weights and the
    architecture, and nothing else -- about half the size, and no run directory
    required to use it.
    """
    from reversi.nn.export import export_checkpoint

    try:
        meta = export_checkpoint(checkpoint, destination, notes=notes)
    except ReversiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    size = destination.stat().st_size / 1e6
    typer.echo(f"{destination}  ({size:.1f} MB)")
    typer.echo(f"  run {meta['run_id']}, generation {meta['generation']}")
    typer.echo(f"  sha256 {meta['sha256']}")


@app.command("export-onnx")
def export_onnx_cmd(
    model: Annotated[Path, typer.Argument(help="An exported model, from `reversi export`")],
    destination: Annotated[Path, typer.Argument(help="Where to write the .onnx file")],
    positions: Annotated[
        int, typer.Option("--check-positions", help="How many random inputs to compare on.")
    ] = 64,
) -> None:
    """Convert an exported model to ONNX, for running it in a browser.

    The web app runs the network on the visitor's own machine, which needs the
    weights in a form that does not involve PyTorch. This writes that form, and
    then checks it: both versions are run on random inputs and the file is
    deleted rather than kept if their answers differ by more than float32
    rounding. An export that computes something slightly different would not
    fail anywhere -- the browser would just play a different agent than the one
    every measurement in this repository was taken against.
    """
    from reversi.nn.onnx import export_onnx

    try:
        meta = export_onnx(model, destination, check_positions=positions)
    except ReversiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    agreement = meta["agreement"]
    typer.echo(f"{destination}  ({meta['bytes'] / 1e6:.2f} MB)")
    typer.echo(f"  {meta['label']}, opset {meta['onnx_opset']}")
    if agreement:
        typer.echo(
            f"  agrees with PyTorch over {agreement['positions']} positions: "
            f"policy within {agreement['max_policy_diff']:.1e}, "
            f"value within {agreement['max_value_diff']:.1e}"
        )
    typer.echo(f"  sha256 {meta['sha256']}")


@app.command("export-fixtures")
def export_fixtures_cmd(
    destination: Annotated[Path, typer.Argument(help="Directory to write the fixture JSON into")],
    onnx: Annotated[
        Path | None,
        typer.Option("--onnx", help="An .onnx file, to also generate the network fixture."),
    ] = None,
    board_size: Annotated[int, typer.Option("--board-size")] = 8,
    rules_count: Annotated[
        int, typer.Option("--rules", help="Positions in the rules fixture.")
    ] = 1000,
    seed: Annotated[int, typer.Option("--seed")] = 20260830,
) -> None:
    """Write the data a TypeScript port of this engine has to reproduce.

    The rules of Reversi are implemented twice in this repository and checked
    against each other over 50,000 games. Running the agent in a browser means
    writing them a third time, and a wrong port would not crash -- it would just
    play a slightly different game and make the agent look weak.

    So the expectations are generated from the frozen engine rather than written
    by hand: legal moves and flip masks, the input encoding, the network's own
    answers, search visit counts, and whole games. Regenerating them and finding
    a difference means something changed that should not have.
    """
    from reversi.web.fixtures import write_fixtures

    try:
        sizes = write_fixtures(
            destination,
            board_size=board_size,
            seed=seed,
            rules_count=rules_count,
            onnx_path=onnx,
        )
    except ReversiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    for name in sorted(sizes):
        typer.echo(f"  {name + '.json':16s} {sizes[name] / 1024:7.1f} KiB")
    typer.echo(f"  {'total':16s} {sum(sizes.values()) / 1024:7.1f} KiB")
    if onnx is None:
        typer.echo("  (no --onnx given, so the network fixture was skipped)")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    model: Annotated[
        Path | None,
        typer.Option("--model", help="Exported model. Defaults to $RZ_MODEL_PATH."),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="'auto', 'cpu' or 'cuda'.")] = "cpu",
) -> None:
    """Serve the web app and inference API.

    Defaults to the CPU rather than the GPU: a served game is one position at a
    time, which a GPU barely notices, and leaving the card free means the server
    can run alongside a training job on the same machine.
    """
    import uvicorn

    from reversi.api.app import build_app

    chosen = _pick_device(device)
    path = model or Path(os.environ.get("RZ_MODEL_PATH", "models/reversi-8x8-gen60.pt"))
    if not path.exists():
        typer.secho(
            f"no model at {path}. Export one from a training run first:",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "  uv run reversi export runs/<run_id>/checkpoints/latest.pt models/model.pt",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(f"serving {path} on {chosen} at http://{host}:{port}")
    uvicorn.run(build_app(path, device=chosen), host=host, port=port, log_level="info")


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
