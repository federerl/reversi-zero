# reversi-zero

An AlphaZero-style Reversi system with self-play training, PUCT MCTS, calibrated difficulty levels,
and an interactive web app.

The agent starts from randomly initialised weights and learns **only** from self-play — no human
games, no opening books, no hand-written evaluation. The point of the project is not just that it
learns, but that the claim is *measured*: playing strength is established by tournaments with
confidence intervals and Bradley–Terry ratings, never by pointing at a training-loss curve.

> **Status: day 3 of 14.** The rules engine is finished and frozen — implemented twice and
> cross-checked over 20,000 games. Configuration, run provenance, metrics, and the CLI skeleton are
> in place. The network, search, training loop, evaluation, and web app are not yet implemented.
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

reversi train -c configs/smoke4x4.yaml     # (planned) 4x4 pipeline validation, ~10 min CPU
reversi train -c configs/full8x8.yaml      # (planned) 8x8 overnight run
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
* `docs/architecture.md` — layering, dependency rules, and the seven correctness contracts *(planned)*
* `docs/training.md` — local and SLURM runbook, resume, troubleshooting *(planned)*
* `docs/experiments.md` — one entry per run: hypothesis, config delta, outcome, decision *(planned)*
* `docs/model_card.md` — training compute, data provenance, measured strength, limitations *(planned)*

## License

MIT
