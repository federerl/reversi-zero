# Running a training job

## The short version

```bash
uv run reversi train -c configs/full8x8.yaml
```

That prints a run id. Everything the run produces goes under `runs/<run_id>/`.

To carry on later — the next night, or after anything interrupted it — run the
**same command with that run id**:

```bash
uv run reversi train -c configs/full8x8.yaml --run-id 20260826-033307-full8x8-529c8fc-s1337
```

It picks up from the newest checkpoint that passes its checks and continues.

## `--generations` is a total, not "how many more"

A run resumed at generation 9 with `--generations 12` does three more, not twelve.
This is deliberate: it means the command you type on night three is identical to
the one you typed on night one, and it is what makes the learning-rate schedule
land where it was aimed — the schedule reads the *cumulative* step count, so a run
that restarted its counter every night would repeat the warmup every night and
never reach its configured rate.

Re-running a finished run does nothing and says so, rather than quietly training
it a second time.

## Stopping a run

**Press Ctrl-C once.** The run finishes the generation it is in, writes its
checkpoint and shard, and exits cleanly. That wait is bounded by design —
generations are sized to about twenty minutes on the full profile — and it is what
keeps the files on disk complete.

**Press Ctrl-C twice** and the process dies immediately. That is supported, and
it is safe in the sense that matters: the resume logic assumes it might happen
and checks every file before trusting it. You may lose the generation that was in
flight.

The same applies to `SIGTERM` (what most kill commands and shutdown scripts send)
and `SIGUSR1` (what a job scheduler sends before a wall-clock limit, when
configured to warn first).

## Running longer than your ssh session

An ssh connection that drops takes your training run with it. Use `tmux`:

```bash
tmux new -s reversi                  # start a named session
uv run reversi train -c configs/full8x8.yaml
# detach with Ctrl-B then D -- the run keeps going
```

Later, from anywhere:

```bash
tmux attach -t reversi               # back to the same terminal
tmux ls                              # what sessions exist
```

This is the single most useful habit for long jobs on a shared machine. Without
it, closing your laptop lid can end an eight-hour run.

## What a run leaves behind

```
runs/<run_id>/
  config.yaml            the fully resolved configuration, hashed into every checkpoint
  env.json  git.json     what machine and what commit produced this
  meta.json  cmdline.txt the run id, seed, and the exact command
  checkpoints/
    gen_00007.pt         weights, optimiser state, RNG state
    gen_00007.json       the same metadata as plain JSON, plus the .pt's checksum
    latest.pt/.json      a copy of the newest (a copy, not a symlink -- Windows)
  replay/
    gen_00007_w00.npz    one shard per generation
    manifest.json        every shard with its checksum
  metrics/
    train.jsonl  selfplay.jsonl  replay.jsonl
  logs/
```

The `.json` sidecars are there so you can inspect a run without loading torch:

```bash
grep -h '"generation"' runs/*/checkpoints/gen_*.json | tail
```

## How resume decides what to trust

Not "the newest file". Every checkpoint is written in a fixed order — weights,
then the sidecar carrying the weights' checksum, then the `latest` copy — and
resume walks backwards from the newest generation until it finds one whose
checksum still matches.

That ordering closes the window the interruption opens. A process killed between
writing the weights and writing the sidecar leaves a `.pt` that nothing has
vouched for, and resume treats it as absent rather than as newest. Falling back
one generation costs a few minutes of self-play; trusting a torn file would
resume from garbage and never say so.

Replay shards work the same way: the manifest records a checksum for each, and
any shard that no longer matches is dropped from the window with a warning.

## Starting over in an existing run directory

```bash
uv run reversi train -c configs/smoke4x4.yaml --run-id <id> --resume off
```

Weights start fresh. Replay shards already in the directory are still picked up —
they are games, and games do not go stale just because you restarted the network.

## Changing the config between sessions

Changing hyperparameters — learning rate, simulations, games per generation —
between nights is fine. You get a warning that the config hash differs, and the
run continues.

Changing the **network shape** is not fine, and is refused with a message naming
what differs. Different `n_blocks` or `channels` means the saved numbers do not
describe the network you just built. torch would reject an outright shape
mismatch anyway, but the message is unhelpful, and a change that happened to
preserve shapes would not be caught at all.

## On a job scheduler

There is no SLURM support **yet**, and that is a timing accident rather than a
decision.

The GPU servers this project targets are offline for maintenance and are due back
*as a Slurm cluster* before the start of term. Until they return there is nothing
to write a batch script against and nothing to test it on, so the plan's
criterion S9 — verify that a real job hits its wall clock, requeues, and resumes —
is recorded as **deferred**, not as met and not as inapplicable.

What a batch script will need already exists:

* `--signal=B:USR1@300` to get a warning five minutes before the wall clock.
  `obs/signals.py` already treats `SIGUSR1` as "finish this generation and stop".
* `--requeue` so the job comes back.
* the same `reversi train -c ... --run-id ...` command, unchanged. It resumes on
  its own, which is the part that actually took the work.

In other words the hard half is done and the sbatch file is a dozen lines of
directives. Writing it before there is a cluster to run it on would produce
something untested, which for this particular file is worth nothing.

## Troubleshooting

**"nothing to do: this run has already reached its generation target."**
It finished. Raise `--generations` to continue it, or start a new run.

**"skipping gen_000NN.pt: ... does not match its recorded checksum."**
Expected after a hard kill. The run fell back one generation and carried on.

**"dropped N replay shard(s) that no longer match their checksum."**
Same cause, same handling. The sliding window absorbs a lost generation.

**"cannot resume: the checkpoint was built as {...} but this config builds {...}."**
You changed the network shape. Either put the config back, or start a new run.

**The learning rate looks far too small.**
Check for `train.warmup_steps is N but this run is only M steps`. Warmup is capped
at half the run so a short run still reaches its intended rate, but if you see
this on a *long* run, `warmup_steps` is genuinely misconfigured.
