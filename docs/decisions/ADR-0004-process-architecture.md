# ADR-0004: Independent workers that write their own shards

**Status:** accepted, day 8 — and extended to the arena on day 15
**Applies to:** `selfplay/runner.py`, `selfplay/game_batch.py`, `train/loop.py`,
`difficulty/calibrate.py`
**Supersedes:** nothing. The single-process version from day 5 still runs and is
still the fallback.

## The problem

One generation of `full8x8` is 600 games × ~58 plies × 200 simulations ≈
**7×10⁷ node expansions**, each needing legal-move generation, a state copy, a
PUCT argmax and a backup — all in Python. Training on that data is 800 steps over
a 458k-parameter network: seconds of GPU time.

The ratio is roughly 100:1. **Every hour spent optimising the trainer is wasted.**
The only thing that matters is how fast games can be produced.

And a measurement made that concrete: at batch size 1 the GPU does 499 positions
per second against the CPU's 329 — only **1.5×**. A tree search asks about one
position at a time, so run naively, a GPU buys almost nothing. The GPU is not
slow; it is starved.

## The decision

Three stages, each justified by a measurement.

**Stage 1 — one process, one game, batch 1.** Correct and simple. Enough for the
4×4 gate, and it still exists as the fallback path.

**Stage 2 — many games in flight per worker.** One worker holds `B = 24` games
and advances all of them in lockstep, collecting exactly *one* leaf per tree per
round, then issuing **one** forward pass of batch `B`. Finished games are replaced
immediately so `B` stays saturated.

> Measured: **13.6×** over stage 1.

That is the whole answer to the starved-GPU problem, and it is about forty lines.

**Stage 3 — six independent worker processes.** `spawn` start method, each loading
its own model replica from the checkpoint file, playing its share of the games,
writing **its own** `.npz` shard, and exiting.

> Measured: **2.25×** on top of stage 2.

**No queues. No shared memory. No long-lived IPC. No central inference server.**

## Why not a central inference server

It is the textbook answer, and it was rejected deliberately.

A batching server that collects leaves from all workers would extract more GPU
utilisation. But the gain it targets is *already captured* by stage 2 — each
worker is submitting batches of 24 — and it introduces this project's largest
deadlock and backpressure surface in exchange.

Independent processes are embarrassingly parallel, deadlock-free by construction,
crash-tolerant (a dead worker loses only its own shard, and its games are re-run
once), and work identically on Windows. For a project whose deliverable is
*rigour*, a component that can hang under load is a bad trade for utilisation.

The trigger for revisiting was written down in advance: **GPU utilisation below
40% while all six worker CPUs are above 90%.** That has not happened.

## The trainer stays synchronous

Generate → train → checkpoint → repeat, with every generation a resumable
boundary.

Asynchronous training uses hardware better. It also makes "which data trained
which checkpoint" ambiguous, and makes resume and reproducibility harder to reason
about. When the deliverable is a defensible measurement, that ambiguity costs more
than the utilisation is worth.

## What it cost, and the mistakes made

**A generation ends with the slowest worker.** Skew is logged as
`worker_skew_pct` rather than assumed away.

**T23 was not met.** The plan asked for ≥3× from six workers; the measurement was
**2.25×**, and eight workers were *slower* than six — contention on one 4 GB
laptop GPU. That is recorded as not met rather than reworded, and the practical
target it protected (a generation inside 20 minutes) was met by sizing generations
instead.

**The day-8 benchmark was itself confounded.** It gave each worker too few games,
so `games_in_flight` silently collapsed from 24 to ~6 — a worker cannot hold more
games in flight than it was given. Every rate in that sweep was ~2.4× low. It was
caught because the real run disagreed with the prediction, and the correction is
recorded in `bench/results/worker-sweep.json` rather than quietly overwritten.

**One thread per worker.** `torch.set_num_threads(1)`. Workers are tree-bound, so
intra-op threads only contend. This was learned twice: once here, and again in
the arena, where omitting it turned a one-hour calibration into one that managed
three of twenty-one pairings in two hours.

## The same shape, applied to the arena

Day 15 extended this to `difficulty/calibrate.py`. A round robin's pairings are
independent, so they parallelise the same way — one process per pairing, each
loading its own network, no shared state.

With one twist that follows from the same measurement as before: **the arena runs
on the CPU, not the GPU.** At batch size 1 the GPU is only 1.5× a core, and an
arena plays one position at a time with nothing to batch. Eight CPU processes beat
one GPU process by roughly four times and leave the GPU free.

That took the difficulty calibration from ~32 hours to under six, which is the
difference between a criterion that could be measured and one that stayed
theoretical for the life of the project.

## Revisit if

* GPU utilisation is under 40% while worker CPUs are above 90% — then the P2
  central batcher earns its complexity.
* A machine with real GPU memory becomes available, at which point the six-worker
  contention that capped this at 2.25× may simply disappear.
* Self-play stops being the bottleneck. It has not been close.
