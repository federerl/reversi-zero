"""The generational cycle: play games, keep them, train on them, repeat.

One generation, in order:

1. Freeze the current network and play a batch of games against itself.
2. Write those positions to a shard, and add them to the sliding window.
3. Take a few hundred optimisation steps, sampling from that window.
4. Save the result, and go again with the improved network.

**Why the network is frozen during self-play.** Every game in a generation is
played by the *same* weights. That is what makes a generation a meaningful unit:
the games it produced can be attributed to one specific network, so "which data
trained which checkpoint" has an answer. Updating the weights continuously while
games are in flight would use the hardware slightly better and make that question
unanswerable -- a bad trade when the deliverable is evidence rather than a score.

**Why the loop is synchronous.** Self-play and training take turns rather than
running at once. Same reason: a resume has an unambiguous point to restart from,
and every checkpoint has an exact data lineage.

**What is deliberately missing until day 7.** Resume, signal handling, and proper
checkpoint metadata. The weights are saved each generation so a run leaves
something behind, but restarting from one is not supported yet, and this file will
hand that job to the checkpoint manager rather than growing one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from reversi.atomicio import atomic_write_with
from reversi.config import Config
from reversi.data.replay import ReplayBuffer
from reversi.data.schema import samples_to_arrays
from reversi.data.shards import Manifest, shard_filename, write_shard
from reversi.errors import WorkerError
from reversi.nn.evaluator import TorchEvaluator
from reversi.nn.model import PolicyValueNet, build
from reversi.obs.metrics import MetricsHub
from reversi.obs.runmeta import RunPaths
from reversi.search.config import SearchConfig
from reversi.seeding import derive_seed
from reversi.seeding import rng as make_rng
from reversi.selfplay.worker import SelfPlaySummary, play_games
from reversi.train.trainer import Trainer

__all__ = ["GenerationReport", "run_training"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationReport:
    """What one generation did. Returned so tests and the CLI can assert on it."""

    generation: int
    games: int
    positions: int
    buffer_size: int
    seconds: float
    selfplay: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    checkpoint: Path | None = None


def run_training(
    config: Config,
    paths: RunPaths,
    *,
    metrics: MetricsHub | None = None,
    generations: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    device: str | torch.device = "cpu",
) -> list[GenerationReport]:
    """Run the loop. Returns one report per completed generation.

    ``generations`` overrides the configured count, which is what the pipeline
    test uses to run three generations instead of fifteen. ``should_stop`` is
    checked between generations, so day 7 can wire a signal handler to it without
    touching this function.
    """
    board_size = config.game.board_size
    paths.ensure()

    model = build(config.net, board_size, seed=config.seed)
    search_config = SearchConfig.for_selfplay(config.mcts)

    manifest = Manifest.load(paths.replay)
    dropped = manifest.verify()
    if dropped:
        log.warning(
            "dropped %d replay shard(s) that no longer match their checksum: %s",
            len(dropped),
            ", ".join(shard.filename for shard in dropped),
        )

    buffer = ReplayBuffer(
        board_size=board_size,
        window=config.replay.window,
        per_gen_cap_factor=config.replay.per_gen_cap_factor,
        symmetry_aug=config.train.symmetry_aug,
    )
    recovered = buffer.load_from(manifest)
    if recovered:
        log.info("recovered %d positions from %d shard(s)", recovered, len(manifest.shards))

    total_generations = generations if generations is not None else config.selfplay.max_generations
    trainer = Trainer(
        model,
        config.train,
        total_steps=total_generations * config.train.steps_per_generation,
        device=device,
    )

    reports: list[GenerationReport] = []
    for generation in range(1, total_generations + 1):
        if should_stop is not None and should_stop():
            log.info("stopping before generation %d as asked", generation)
            break

        started = time.perf_counter()

        # ---- 1. play -------------------------------------------------
        # eval() matters: batch norm must use its accumulated averages, not the
        # statistics of whatever positions happen to share a batch. The evaluator
        # refuses to run a model in training mode, so this is belt and braces.
        model.eval()
        evaluator = TorchEvaluator(model, device=device)
        summary = SelfPlaySummary()
        samples = []

        for record, branching in play_games(
            evaluator,
            search_config,
            board_size=board_size,
            n_games=config.selfplay.games_per_generation,
            root_seed=config.seed,
            generation=generation,
            max_plies=4 * board_size * board_size,
        ):
            summary.observe(record, branching)
            samples.extend(record.samples)

        if not samples:
            msg = (
                f"generation {generation} produced no training positions from "
                f"{summary.games} games. Every position had a single legal move, "
                "which should be impossible -- suspect the rules engine or the config."
            )
            raise WorkerError(msg)

        # ---- 2. store ------------------------------------------------
        arrays = samples_to_arrays(samples, generation=generation, board_size=board_size)
        shard = write_shard(
            paths.replay / shard_filename(generation),
            arrays,
            board_size=board_size,
        )
        manifest.add(shard)
        buffer.add(arrays)
        played_seconds = time.perf_counter() - started

        # ---- 3. train ------------------------------------------------
        train_rng = make_rng(derive_seed(config.seed, "train", generation))
        totals: dict[str, float] = {}
        for _ in range(config.train.steps_per_generation):
            batch = buffer.sample(config.train.batch_size, train_rng)
            for key, value in trainer.train_on(batch).items():
                totals[key] = totals.get(key, 0.0) + float(value)
        steps = config.train.steps_per_generation
        training: dict[str, Any] = {key: value / steps for key, value in totals.items()}

        # ---- 4. save and tidy ----------------------------------------
        checkpoint = _save_weights(paths, model, config, generation, trainer.global_step)
        removed = manifest.prune(config.replay.retain_shards)
        if removed:
            log.debug("pruned %d shard(s) older than the retention window", len(removed))

        elapsed = time.perf_counter() - started
        buffer_stats = buffer.stats(generation)
        selfplay_metrics = summary.as_metrics()

        report = GenerationReport(
            generation=generation,
            games=summary.games,
            positions=len(samples),
            buffer_size=len(buffer),
            seconds=elapsed,
            selfplay=selfplay_metrics,
            training=training,
            checkpoint=checkpoint,
        )
        reports.append(report)

        log.info(
            "generation %d/%d: %d games, %d positions, loss %.4f (policy %.4f, value %.4f), %.1fs",
            generation,
            total_generations,
            summary.games,
            len(samples),
            training.get("total_loss", float("nan")),
            training.get("policy_loss", float("nan")),
            training.get("value_loss", float("nan")),
            elapsed,
        )

        if metrics is not None:
            metrics.log(
                "selfplay",
                generation=generation,
                global_step=trainer.global_step,
                seconds=played_seconds,
                games_per_s=summary.games / max(played_seconds, 1e-9),
                **selfplay_metrics,
            )
            metrics.log(
                "train",
                generation=generation,
                global_step=trainer.global_step,
                steps=steps,
                **training,
            )
            metrics.log(
                "replay",
                generation=generation,
                global_step=trainer.global_step,
                shards=len(manifest.shards),
                **buffer_stats,
            )

    return reports


def _save_weights(
    paths: RunPaths,
    model: PolicyValueNet,
    config: Config,
    generation: int,
    global_step: int,
) -> Path:
    """Write the weights for this generation, plus a ``latest`` copy.

    Deliberately minimal. The real checkpoint -- optimiser state, RNG state,
    lineage, environment, a checksummed sidecar, and the resume logic that reads
    all of it -- is day 7's job. This exists so that a run today leaves a usable
    network behind rather than only a log file.

    ``latest.pt`` is a copy, not a symlink: symlinks need elevated privileges on
    Windows, and this project has to run on a laptop as well as a cluster.
    """
    payload = {
        "format_version": 0,  # 0 = pre-checkpoint-manager; day 7 bumps this
        "generation": generation,
        "global_step": global_step,
        "arch": model.arch(),
        "model_state_dict": model.state_dict(),
        "config_sha256": config.sha256,
    }

    path = paths.checkpoints / f"gen_{generation:05d}.pt"
    atomic_write_with(path, lambda tmp: torch.save(payload, tmp))
    atomic_write_with(paths.checkpoints / "latest.pt", lambda tmp: torch.save(payload, tmp))
    return path
