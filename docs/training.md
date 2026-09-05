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

## Rating what a run produced

Training loss says nothing about strength. The number that does is a rating from
games, and `reversi arena` produces it:

```bash
uv run reversi arena -c configs/full8x8.yaml --suite crossgen --run-id <id> --workers 8
```

`crossgen` takes up to six of the run's saved generations, spread from first to
last, adds Random, Greedy and two Minimax depths, plays every pairing the same
number of colour-balanced games from seeded openings, and fits one rating table
to the whole matrix (Bradley–Terry, anchored at Random = 0, with bootstrap
intervals). The report lands at `runs/<id>/arena/crossgen.json`, in the shape the
README figures and the web app's opponent list read.

Two other suites: `baselines` rates one checkpoint (the run's `latest.pt`, or
`--checkpoint`) against the fixed opponents; `final` is `crossgen` plus the Edax
engine when it is installed. `--entrant` adds anyone to any suite, which is how
a checkpoint from a *different* run joins the table:

```bash
uv run reversi arena -c configs/full8x8.yaml --suite custom     -e random -e greedy -e minimax-d4     -e control-g60=runs/control/checkpoints/gen_00060.pt@50     -e run1-g60=runs/<run 1>/checkpoints/gen_00060.pt@50 --out runs/compare.json
```

Pairings run in separate CPU processes, one thread each. A tournament asks the
network about one position at a time, so a GPU barely helps here and eight CPU
processes beat one GPU process comfortably. On the cluster this is a
`slurm/cpu.sbatch` job.

## On a job scheduler

Training runs on the CSSE Slurm cluster. The scripts live in `slurm/`, and
`slurm/README.md` is the operator's page: which partition, which flags, how to
submit, how to read a log. This section is about *why* the arrangement is what it
is, and what has been verified.

**The constraint.** Every partition has a 24-hour wall-clock limit, and a full
8×8 run takes longer than that. So a run is a chain of identical jobs, each one
resuming where the last stopped. `slurm/submit_chain.sh` submits them with
`--dependency=afterany`, so job *k*+1 starts when job *k* ends for any reason.

**Why no new code was needed.** Three things the training loop already did make
the chain safe:

* `--generations N` means "stop once the run has *N* generations in total". A
  job that starts after the run is finished exits in seconds, so asking for one
  job too many costs nothing.
* `--run-id` reuses the run directory, and resume loads the newest checkpoint
  that verifies (see above).
* `SIGUSR1` means "finish this generation, then stop". The sbatch header asks
  Slurm for that signal fifteen minutes before the limit (`--signal=B:USR1@900`),
  and `rz_run` in `slurm/env.sh` forwards it from the batch shell to the training
  process. Without the forwarding, the signal stops at the shell and the process
  is killed mid-checkpoint when the limit arrives.

**What was verified, on 2026-09-04, on gus.** Criterion S9 from the original plan
was: a real job hits its wall clock, stops cleanly, and a later job resumes.

1. `sbatch slurm/smoke4x4.sbatch` (job 6843): twelve generations of the 4×4
   profile on one L40S in 2 minutes 15 seconds, checkpoint written, exit status 0.
2. The same profile with a two-minute limit and the warning at sixty seconds
   (job 6844): the warning arrived after generation 3, the loop finished
   generation 4, wrote its checkpoint, and exited with status 0 forty seconds
   before the limit would have killed it.
3. The same run id resubmitted without a limit (job 6845): resumed at generation
   5 and finished at 12.

The resumed run reproduced the uninterrupted run's losses digit for digit
(generation 5: 1.7663; generation 12: 1.3038). Stopping and resuming changed
nothing about the result, which is the strongest evidence the resume path can
give. The `env.json` the resuming job wrote is kept as
`docs/slurm_smoke_evidence.json`: node, partition, job id, torch and CUDA
versions, and the GPU.

**Which GPU node, and why.** The cluster has two: gebru (eight Quadro RTX 6000,
48 CPU threads) and gus (six L40S, 255 CPU threads). Training goes to gus.
Self-play is CPU-bound tree search, with the GPU only scoring positions in
batches; six CPU threads per GPU on gebru would leave the card waiting on the
workers. The request `--gres=gpu:nvidia_l40s:1` selects gus without naming it.

**Where evaluation goes.** Round robins between agents are many independent
games, one per core. They run on the `cpu` partition (984 cores across four
nodes) through `slurm/cpu.sbatch`, not on a GPU.

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


---

## Calibrating the difficulty ladder (S15)

The four difficulty levels are *designed* to differ. Until this is run, that is
all they are: a ladder whose rungs are not measurably separated is four names for
one opponent.

```bash
uv run reversi calibrate models/reversi-8x8-gen60.pt   --games 300 --workers 8 --device cpu --guard-samples 500   --out runs/calibration/difficulty_report.json   --write-config configs/difficulty.yaml
```

On Windows, PowerShell continues a line with a backtick rather than a backslash.
A backslash there fails with `Missing expression after unary operator '--'`,
which is PowerShell reading the next line as a fresh command:

```powershell
uv run reversi calibrate models/reversi-8x8-gen60.pt `
  --games 300 --workers 8 --device cpu --guard-samples 500 `
  --out runs/calibration/difficulty_report.json `
  --write-config configs/difficulty.yaml
```

**Budget about four hours.** Run it in a terminal you can leave alone -- `tmux`,
or just a window you do not close. It is an overnight job in the same sense the
training runs are.

### Why it costs that much

`max` searches 800 simulations per move. On one CPU thread that is roughly 1.6
seconds a move, so a game where `max` plays costs about 50 seconds, and the
plan's 300 games across 21 pairings is some 32 hours of work. `--workers` spreads
the pairings across processes, which brings the wall-clock down to about four.

| games/pair | wall clock, 8 workers |
|---|---|
| 60 | ~1 hour |
| 300 (the criterion) | ~4 hours |

### Why the CPU, and not the GPU

Counterintuitive, and it follows from the day-7 measurement: at batch size one
the GPU is only 1.5x a CPU core. An arena plays one position at a time, so there
is nothing to batch and nothing for the GPU to be good at. Eight CPU processes
beat one GPU process by roughly four times, and leave the GPU free.

Each worker sets `torch.set_num_threads(1)`. Without it, eight processes each
try to use every core and spend their time contending -- measured, that turned a
one-hour run into one that managed three pairings in two hours.

### Reading the result

The report is written whether or not the ladder holds up, and the command exits
non-zero when it does not, because an unmeasured difficulty label is a claim the
interface should not be making. Five checks:

| check | what it means |
|---|---|
| `monotonic` | ratings rise from Casual to Max |
| `adjacent_gap` | neighbouring rungs at least 80 Elo apart |
| `intervals_disjoint` | neighbouring 95% intervals do not overlap |
| `easiest_beats_random` | Casual beats Random, Wilson lower bound above 0.60 |
| `guardrail` | Casual never played a move far below the best available |

`intervals_disjoint` is the one that fails first, and usually for a reason that
has nothing to do with the ladder: intervals narrow with the square root of the
number of games, so a short run can show a genuinely ordered ladder as
unseparated. If the ratings rise cleanly and only that check fails, the answer is
more games rather than different levels.

**If it genuinely does not separate**, the plan's documented options in order are
to widen the simulation ratio (8 / 64 / 400 / 1500), and failing that to ship
three levels and say so in the README. Never four tiers that are not different.

### The obvious next optimisation, not yet done

The arena plays one game at a time. Self-play got 13.6x by advancing many games
in lockstep so their network calls batch together, and the same trick applies
here -- a pairing is 300 independent games. That would take the four hours down
to something like twenty minutes. It is a real piece of work, and it has not been
done.
