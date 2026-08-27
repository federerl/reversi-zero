"""Running several self-play workers at once, with nothing shared between them.

After batching, the network is no longer the bottleneck -- measured on the target
hardware, batched self-play reaches 5,464 positions per second against a GPU
ceiling of 19,296. The remaining time goes on the tree search, which is Python,
which means the way to go faster is more processes.

**Nothing is shared.** Each worker is handed a weights file and a range of game
indices, plays its games, writes its own shard, and exits. No queues, no shared
memory, no long-lived connection to the parent. That is a deliberate rejection of
the textbook design (one central inference server that every worker submits
positions to), for three reasons:

* It removes the deadlock and backpressure risk entirely. A server has queues,
  queues have failure modes, and those failure modes present as a training run
  that simply stops making progress at 3am.
* The gain it would offer is already collected. Each worker's own batching
  submits 48 positions at a time; a central batcher would combine those into
  larger ones, but the GPU is not the limit any more.
* A crashed worker costs its own shard and nothing else. The parent notices a
  non-zero exit code, re-runs that worker's games once, and gives up loudly on a
  second failure rather than quietly producing a short generation.

**Workers report back through files, not pipes.** Each writes a small JSON
summary next to its shard. That keeps the no-IPC property honest and means the
evidence survives the process that produced it.

**One thread each.** ``torch.set_num_threads(1)`` in every worker: they are
tree-bound, so torch's own intra-op threads only contend with the other workers
for the same cores.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from reversi.config import Config
from reversi.data.schema import samples_to_arrays
from reversi.data.shards import ShardInfo, shard_filename, write_shard
from reversi.errors import WorkerError
from reversi.selfplay.worker import SelfPlaySummary

__all__ = ["WorkerResult", "merge_summaries", "run_workers", "split_games"]

log = logging.getLogger(__name__)


def split_games(total: int, workers: int) -> list[tuple[int, int]]:
    """Divide game indices into contiguous ranges, one per worker.

    Contiguous and index-based rather than round-robin, because a game's seed is
    derived from its index: whichever worker plays game 57, it is the same game.
    That makes a re-run of a failed worker reproduce exactly the games that were
    lost, instead of a different set of the same size.
    """
    if workers < 1:
        msg = f"need at least one worker, got {workers}"
        raise WorkerError(msg)
    if total < workers:
        msg = f"cannot split {total} games across {workers} workers"
        raise WorkerError(msg)

    base, extra = divmod(total, workers)
    ranges = []
    start = 0
    for index in range(workers):
        count = base + (1 if index < extra else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


@dataclass(slots=True)
class WorkerResult:
    worker_id: int
    shard: ShardInfo
    summary: dict[str, Any]
    seconds: float


def _summary_path(replay_dir: Path, generation: int, worker_id: int) -> Path:
    return replay_dir / f"gen_{generation:05d}_w{worker_id:02d}.summary.json"


def _worker_main(
    weights_path: str,
    config_yaml: str,
    replay_dir: str,
    generation: int,
    worker_id: int,
    first_game: int,
    last_game: int,
    device: str,
) -> None:
    """One worker process, start to finish. Runs in a fresh interpreter."""
    import dataclasses
    import time

    import torch

    from reversi.config import Config as _Config
    from reversi.nn.evaluator import TorchEvaluator
    from reversi.nn.loader import load_model
    from reversi.search.config import SearchConfig
    from reversi.selfplay.game_batch import BatchedSelfPlay

    # Tree-bound work: torch's own threads would only fight the other workers.
    torch.set_num_threads(1)

    config = _Config.model_validate(json.loads(config_yaml))
    board_size = config.game.board_size

    model = load_model(Path(weights_path), device=device)
    evaluator = TorchEvaluator(model, device=device)

    started = time.perf_counter()
    play = BatchedSelfPlay(
        evaluator,
        SearchConfig.for_selfplay(config.mcts),
        board_size=board_size,
        games_in_flight=min(config.selfplay.games_in_flight, last_game - first_game),
        root_seed=config.seed,
        generation=generation,
        worker_id=worker_id,
        max_plies=4 * board_size * board_size,
    )

    summary = SelfPlaySummary()
    samples = []
    # Game indices are global, so this worker's first game is `first_game` of the
    # generation -- the seed derivation uses the worker id as well, so no two
    # workers can produce the same game even if the ranges were wrong.
    for record, branching in play.play(last_game - first_game):
        summary.observe(record, branching)
        samples.extend(record.samples)

    if not samples:
        msg = f"worker {worker_id} produced no training positions"
        raise WorkerError(msg)

    arrays = samples_to_arrays(samples, generation=generation, board_size=board_size)
    shard = write_shard(
        Path(replay_dir) / shard_filename(generation, worker_id),
        arrays,
        board_size=board_size,
    )

    payload = {
        "worker_id": worker_id,
        "seconds": time.perf_counter() - started,
        "mean_batch": play.mean_batch,
        "shard": {
            "filename": shard.filename,
            "n_positions": shard.n_positions,
            "generation": shard.generation,
            "sha256": shard.sha256,
        },
        "summary": dataclasses.asdict(summary),
    }
    _summary_path(Path(replay_dir), generation, worker_id).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def run_workers(
    *,
    weights_path: Path,
    config: Config,
    replay_dir: Path,
    generation: int,
    n_workers: int,
    device: str = "cpu",
    retry_failed: bool = True,
) -> list[WorkerResult]:
    """Play one generation across ``n_workers`` processes. Returns their shards.

    A worker that exits non-zero is re-run once -- transient failures on a shared
    machine are common enough to be worth absorbing. A second failure raises,
    because a generation that is quietly short is worse than one that stops.
    """
    replay_dir.mkdir(parents=True, exist_ok=True)
    ranges = split_games(config.selfplay.games_per_generation, n_workers)
    config_json = config.model_dump_json()

    # `spawn` rather than `fork`: fork copies the parent's CUDA context, which is
    # unsupported and fails in confusing ways, and it is the only start method
    # Windows has.
    context = mp.get_context("spawn")

    def launch(worker_id: int) -> BaseProcess:
        first, last = ranges[worker_id]
        process = context.Process(
            target=_worker_main,
            args=(
                str(weights_path),
                config_json,
                str(replay_dir),
                generation,
                worker_id,
                first,
                last,
                device,
            ),
            name=f"selfplay-w{worker_id:02d}",
            daemon=False,
        )
        process.start()
        return process

    running: dict[int, BaseProcess] = {
        worker_id: launch(worker_id) for worker_id in range(n_workers)
    }
    failed: list[int] = []
    for worker_id, process in running.items():
        process.join()
        if process.exitcode != 0:
            log.warning("worker %d exited with code %s", worker_id, process.exitcode)
            failed.append(worker_id)

    if failed and retry_failed:
        log.warning("re-running %d failed worker(s): %s", len(failed), failed)
        retried = {worker_id: launch(worker_id) for worker_id in failed}
        still_failed = []
        for worker_id, process in retried.items():
            process.join()
            if process.exitcode != 0:
                still_failed.append(worker_id)
        failed = still_failed

    if failed:
        msg = (
            f"worker(s) {failed} failed twice in generation {generation}. "
            "Continuing would produce a generation with a hole in it, which is "
            "worse than stopping: the missing games are not random."
        )
        raise WorkerError(msg)

    return [_collect(replay_dir, generation, worker_id) for worker_id in range(n_workers)]


def merge_summaries(results: list[WorkerResult]) -> SelfPlaySummary:
    """Add the workers' counters together.

    Summing raw counts rather than averaging each worker's averages: with
    unequal game counts the average of averages is simply the wrong number, and
    it would be wrong quietly.
    """
    merged = SelfPlaySummary()
    for result in results:
        counters = result.summary
        merged.games += counters["games"]
        merged.positions += counters["positions"]
        merged.plies.extend(counters["plies"])
        merged.passes += counters["passes"]
        merged.forced_moves += counters["forced_moves"]
        merged.black_wins += counters["black_wins"]
        merged.white_wins += counters["white_wins"]
        merged.draws += counters["draws"]
        merged.branching_total += counters["branching_total"]
        merged.branching_count += counters["branching_count"]
    return merged


def _collect(replay_dir: Path, generation: int, worker_id: int) -> WorkerResult:
    path = _summary_path(replay_dir, generation, worker_id)
    if not path.exists():
        msg = (
            f"worker {worker_id} exited cleanly but left no summary at {path.name}; "
            "its shard cannot be trusted"
        )
        raise WorkerError(msg)

    payload = json.loads(path.read_text(encoding="utf-8"))
    path.unlink(missing_ok=True)
    return WorkerResult(
        worker_id=worker_id,
        shard=ShardInfo(**payload["shard"]),
        summary=payload["summary"],
        seconds=payload["seconds"],
    )


def sensible_worker_count(requested: int) -> int:
    """Clamp to what the machine actually has, leaving room for the parent."""
    cores = os.cpu_count() or 2
    return max(1, min(requested, cores - 1))
