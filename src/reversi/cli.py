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
    typer.Option(
        "--suite",
        help="'baselines' (one checkpoint vs the fixed opponents), 'crossgen' (several "
        "generations of one run, on one scale), 'final' (crossgen plus Edax), or "
        "'custom' (only what --entrant lists).",
    ),
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
    run_id: RunIdOpt = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="The network to rate in the baselines suite. Default: the run's latest.pt.",
        ),
    ] = None,
    entrant: Annotated[
        list[str] | None,
        typer.Option(
            "--entrant",
            "-e",
            help="An extra entrant: random, greedy, minimax-d4, edax-l5, or NAME=PATH[@SIMS] "
            "for a network file. Repeatable. With --suite custom these are the whole field.",
        ),
    ] = None,
    games: Annotated[
        int | None,
        typer.Option("--games", help="Games per pairing, even. Default: arena.games."),
    ] = None,
    simulations: Annotated[
        int | None,
        typer.Option(
            "--simulations",
            help="Search budget per move for network entrants. Default: mcts.n_simulations.",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Pairings to play at once. 1 runs in this process."),
    ] = 1,
    opening_plies: Annotated[
        int | None,
        typer.Option(
            "--opening-plies",
            help="Random moves before each game, so pairings do not replay one line. "
            "Default: arena.opening_plies.",
        ),
    ] = None,
    max_checkpoints: Annotated[
        int,
        typer.Option("--max-checkpoints", help="Generations to rate in crossgen and final."),
    ] = 6,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Tournament seed. Default: derived from the config seed."),
    ] = None,
    bootstrap: Annotated[
        int | None,
        typer.Option(
            "--bootstrap",
            help="Resamples for the rating intervals. Default: arena.bootstrap_resamples.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write the report. Default: <run>/arena/<suite>.json."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="'cpu' or 'cuda'. CPU is usually faster here."),
    ] = "cpu",
    notes: Annotated[str, typer.Option("--notes", help="Recorded in the report.")] = "",
) -> None:
    """Play a tournament and report ratings with confidence intervals.

    Every entrant plays every other, the same number of colour-balanced games
    from the same seeded openings, and the whole result matrix is fitted at once
    (Bradley-Terry, anchored at random play = 0). That is what turns "it won 60%
    of its games" into a rating with an error bar, on a scale that Random, Greedy
    and Minimax also sit on.

    The report has the shape every consumer already reads: ``crossgen.json`` under
    the run's ``arena/`` directory feeds the README figures and the web app's
    opponent list, so a new run gets a rated table with one command.
    """
    from reversi.arena.entrants import describe_entrant, parse_entrant
    from reversi.arena.suites import baseline_entrants, checkpoint_label, crossgen_entrants
    from reversi.arena.tournament import round_robin_parallel, write_tournament
    from reversi.errors import ConfigError
    from reversi.obs.runmeta import run_root
    from reversi.seeding import derive_seed

    resolved = _load(config, set_)
    setup_logging()

    suites = {"baselines", "crossgen", "final", "custom"}
    if suite not in suites:
        typer.secho(
            f"--suite must be one of {sorted(suites)}, got {suite!r}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    run_dir = run_root(resolved) / run_id if run_id else None
    checkpoints_dir = run_dir / "checkpoints" if run_dir else None
    sims = simulations if simulations is not None else resolved.mcts.n_simulations
    n_games = games if games is not None else resolved.arena.games
    plies = opening_plies if opening_plies is not None else resolved.arena.opening_plies
    resamples = bootstrap if bootstrap is not None else resolved.arena.bootstrap_resamples
    tournament_seed = seed if seed is not None else derive_seed(resolved.seed, "arena", suite)

    try:
        entrants = []
        label = suite
        if suite == "baselines":
            path = checkpoint
            if path is None and checkpoints_dir is not None:
                path = checkpoints_dir / "latest.pt"
            if path is None:
                msg = (
                    "the baselines suite needs --checkpoint, or --run-id to use the run's latest.pt"
                )
                raise ConfigError(msg)
            rated = parse_entrant(f"{checkpoint_label(path)}={path}", default_simulations=sims)
            entrants = [*baseline_entrants(resolved.arena.baselines), rated]
            label = f"baselines_{rated.name}"
        elif suite in {"crossgen", "final"}:
            if checkpoints_dir is None:
                msg = f"the {suite} suite needs --run-id, so it knows which checkpoints to rate"
                raise ConfigError(msg)
            entrants = crossgen_entrants(
                checkpoints_dir, simulations=sims, max_checkpoints=max_checkpoints
            )
            if suite == "final":
                from reversi.agents.edax import find_edax

                try:
                    find_edax()
                except ReversiError as missing:
                    typer.secho(
                        f"Edax is not installed, so the final suite is crossgen only. {missing}",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
                else:
                    entrants.append(parse_entrant("edax-l5", default_simulations=sims))
        for text in entrant or []:
            entrants.append(parse_entrant(text, default_simulations=sims))
        if suite == "custom" and len(entrants) < 2:
            msg = "the custom suite needs at least two --entrant"
            raise ConfigError(msg)

        names = [e.name for e in entrants]
        anchor = "random" if "random" in names else names[0]
        if anchor != "random":
            typer.secho(
                f"random is not in this field, so ratings are anchored at {anchor} = 0 and are "
                "not comparable to tables anchored at random.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        typer.echo(f"{len(entrants)} entrants, {n_games} games per pairing, {sims} simulations")
        result = round_robin_parallel(
            entrants,
            games_per_pair=n_games,
            board_size=resolved.game.board_size,
            seed=tournament_seed,
            workers=workers,
            opening_plies=plies,
            anchor=anchor,
            bootstrap=resamples,
            device=device,
        )
    except ReversiError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    typer.echo("")
    typer.echo(result.describe())

    if out is not None:
        destination = out
    elif run_dir is not None:
        destination = run_dir / "arena" / f"{label}.json"
    else:
        destination = Path(resolved.run_root) / "arena" / f"{label}.json"
    write_tournament(
        destination,
        result,
        specs={e.name: describe_entrant(e, board_size=resolved.game.board_size) for e in entrants},
        notes=notes,
    )
    typer.echo("")
    typer.echo(f"report: {destination}  ({result.seconds / 60:.1f} min)")


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
    # Check the model before importing the server.
    #
    # Both are things that can be missing on a first run, and this order reports
    # the more likely and more actionable one. Importing first means somebody
    # without the `api` extra installed is told about uvicorn when their real
    # problem is that they have not exported a model yet -- and looking at a file
    # path needs no dependencies at all.
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

    try:
        import uvicorn

        from reversi.api.app import build_app
    except ImportError as error:
        typer.secho(
            f"the web server needs the `api` extra, which is not installed ({error}).",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "  uv sync --extra cpu --extra api",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2) from error

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
