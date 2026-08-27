# reversi-zero

An AlphaZero-style Reversi system with self-play training, PUCT MCTS, calibrated difficulty levels,
and an interactive web app.

The agent starts from randomly initialised weights and learns **only** from self-play — no human
games, no opening books, no hand-written evaluation. The point of the project is not just that it
learns, but that the claim is *measured*: playing strength is established by tournaments with
confidence intervals and Bradley–Terry ratings, never by pointing at a training-loss curve.

> **Status: day 6 of 14. The pipeline gate passes.** A 4×4 agent trained from randomly initialised
> weights reaches **97.2%** against a random opponent and **93.2%** against a greedy one, over 200
> colour-balanced games, after 7.2 minutes of self-play training on a laptop CPU.
>
> **That is a pipeline check, not a result.** 4×4 exists only to prove the machinery works before an
> overnight GPU run is spent on it; no 4×4 number belongs in a strength claim. The headline is 8×8,
> which has not been run yet.
>
> **Day 7 done:** a run now survives being killed. Checkpoints carry optimiser and RNG state, every
> file is checksummed, and `reversi train --run-id <id>` picks up from the newest checkpoint that
> still verifies — losing at most one generation. Proved by killing a real training process
> mid-write and resuming it, not by simulating one.
>
> **Day 8 done:** self-play now advances many games in lockstep so their network calls batch
> together (**13.6×**) and runs across six worker processes (**2.25×** on top). Measured, not
> assumed — `bench/results/` holds the numbers, and `configs/full8x8.yaml` was rewritten around
> them.
>
> Still to come: the real baselines and rating machinery (days 9–10), the web app (days 13–14).
> Commands marked *(planned)* below exit with code 2 and name the task that will deliver them.

## Quickstart

```bash
uv sync --extra cpu --extra dev --extra api --extra obs   # GPU cluster: --extra cu124
make test        # fast suite, CPU only, target < 90s
make quality     # ruff + pyright
```

On Windows, `.\make.ps1 test` is an equivalent shim for the same targets.

```bash
reversi config -c configs/full8x8.yaml     # resolved config + its sha256
reversi init-run -c configs/smoke4x4.yaml  # create a run dir and write provenance
python scripts/validate_run.py runs/*      # assert every run is reproducible

reversi train -c configs/smoke4x4.yaml     # 4x4 pipeline validation, ~8 min CPU
reversi train -c configs/full8x8.yaml      # 8x8 run (needs a GPU to be worth starting)
reversi arena                              # (planned) evaluate vs Random / Greedy / Minimax-d4
reversi calibrate --validate               # (planned) verify the four difficulty levels
reversi serve                              # (planned) play in the browser
```

## Configuration

Three profiles, layered on `configs/base.yaml`:

| Profile | Purpose | Board | Sims | Notes |
|---|---|---|---|---|
| `smoke4x4` | Pipeline validation only — **never a result** | 4×4 | 25 | Must learn in ≤ 10 min on CPU |
| `dev8x8` | Shake-out run before committing a night | 8×8 | 100 | ~60–90 min on one GPU |
| `full8x8` | The headline result | 8×8 | 300 | 6 workers; generations sized to ≤ 20 min so an 8-hour job ceiling costs little on preemption |

Override anything from the command line: `reversi train -c configs/full8x8.yaml --set mcts.n_simulations=200`.
Unknown keys are errors, not silent no-ops.

## Reproducibility

Every run writes, before doing any work:

```
runs/<run_id>/
  config.yaml   fully resolved configuration (hashed into every checkpoint)
  env.json      python / torch / CUDA / driver / OS / CPU / GPU / hostname / SLURM job
  git.json      commit, branch, dirty flag, diff sha256 (+ diff.patch when dirty)
  meta.json     run_id, seed, config sha256, config-hash history across resumes
  cmdline.txt   the exact argv that launched the run
```

`run_id` is `{timestamp}-{profile}-{git_short}-s{seed}`. A run that cannot write these does not start.

Each generation also writes `checkpoints/gen_NNNNN.pt` (weights, optimiser state, RNG state) beside
`gen_NNNNN.json` — the same metadata as plain text, plus the checkpoint's SHA-256. Resume walks
backwards from the newest generation until it finds one whose checksum still matches, so a process
killed partway through a write falls back one generation rather than loading a torn file. See
`docs/training.md`.

## Does it learn?

Yes, on the validation board — measured by playing, never by pointing at a loss curve:

| | score over 200 colour-balanced games |
|---|---|
| 4×4 agent vs Random | **97.2%** |
| 4×4 agent vs Greedy | **93.2%** |

The agent scores near 100% as white and lower as black. That is not a lopsided agent: **white wins
4×4 Reversi with perfect play**, which `tests/unit/test_solved_4x4.py` proves by solving the game
exactly (3,306 positions). As black it is defending a theoretically lost position. That test doubles
as an independent check on the rules engine — it never inspects a flip or a legal-move list, it just
plays every possible game to the end and asks who wins.

## The training loop

One generation: freeze the network, play a batch of games against itself, write those positions to a
shard, then take a few hundred optimisation steps sampling from a sliding window of recent games.

Two things the network learns from each position: what the search concluded (its visit distribution
over the legal moves), and how the game actually ended, seen from the point of view of whoever was to
move there. The search's answer is better than the network's own, which is what makes the loop climb.

Positions are stored as **bitboards, not encoded planes** — so changing the network's input format
does not invalidate games already collected. Every shard is checksummed, every write is atomic, and
a shard that no longer matches its checksum is dropped rather than trained on.

Measured on the dev laptop, `smoke4x4`: ~40s per generation, so the 12-generation profile lands
around 8 minutes on CPU.

## Throughput

Self-play is where essentially all the compute goes — one 8×8 generation is tens of millions of
network calls against a few seconds of backpropagation. Measured on an RTX A1000 laptop GPU:

| | games/s | speedup |
|---|---|---|
| one game at a time | 0.072 | — |
| 48 games batched | 0.977 | **13.6×** |
| 6 worker processes | — | **2.25×** on top |

The reason batching matters so much is that a single position barely occupies a GPU at all — 499
positions/s at batch 1 versus 26,328 at batch 48, on the same card. It was never a hardware problem;
the GPU was being starved.

`bench/selfplay_bench.py` produces these numbers, and `bench/results/` holds them. Nothing in this
repo is optimised without a before/after measurement to point at.

## The rules engine

Implemented twice on purpose:

* `game/reference.py` — a grid of squares, walking outward in eight directions. Slow, and correct by
  inspection. This is the specification.
* `game/rules.py` + `game/bitboard.py` — two 64-bit integers per board, one bit per square. About ten
  times faster, and *not* correct by inspection.

Neither was written from the other; both were written from the rules of Othello. A test plays 20,000
random games through both and compares them at **every move** — legal moves, which discs each move
would flip, the resulting position bit for bit, and the final score. That agreement is the entire
correctness argument, and it exists because a rules bug does not crash: the AI would simply learn a
slightly different game and every measurement afterwards would be meaningless.

See `docs/how-the-engine-works.md`.

## Documentation

* `docs/configuration.md` — every setting, what it does, and what breaks if you change it
* `docs/how-the-engine-works.md` — bitboards, passing, perspective, and the eight board rotations
* `docs/decisions/` — one file per decision that would be expensive to revisit
* `docs/architecture.md` — layering, dependency rules, and the seven correctness contracts *(planned)*
* `docs/training.md` — running a job, stopping one, resuming, and what a run leaves behind
* `docs/experiments.md` — one entry per run: hypothesis, config delta, outcome, decision *(planned)*
* `docs/model_card.md` — training compute, data provenance, measured strength, limitations *(planned)*

## License

MIT
