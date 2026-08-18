# reversi-zero

An AlphaZero-style Reversi system with self-play training, PUCT MCTS, calibrated difficulty levels,
and an interactive web app.

The agent starts from randomly initialised weights and learns **only** from self-play — no human
games, no opening books, no hand-written evaluation. The point of the project is not just that it
learns, but that the claim is *measured*: playing strength is established by tournaments with
confidence intervals and Bradley–Terry ratings, never by pointing at a training-loss curve.

> **Status: day 1 of 14.** Configuration, run provenance, metrics, and the CLI skeleton are in place
> and tested. The engine, network, MCTS, training loop, evaluation, and web app are not yet
> implemented — see the roadmap. Commands marked *(planned)* below exit with code 2 and name the
> task that will deliver them.

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

## Documentation

* `docs/architecture.md` — layering, dependency rules, and the seven correctness contracts
* `docs/training.md` — local and SLURM runbook, resume, troubleshooting
* `docs/experiments.md` — one entry per run: hypothesis, config delta, outcome, decision
* `docs/model_card.md` — training compute, data provenance, measured strength, limitations

## License

MIT
