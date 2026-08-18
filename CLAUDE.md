# Working agreements for this repo

## Git: ask before anything leaves my machine

**Never `git push` unless I explicitly ask for it.** Same for opening a PR,
updating a PR, merging, or pushing to an existing branch.

Committing locally is fine — but say so, don't assume it implies pushing.

When work is ready to go out, stop and tell me:
- what changed
- what you verified (and what you couldn't)
- the exact command you'd run

Then wait. "The change is finished and tested" is not permission to push.

Also never: `push --force`, rewrite pushed history, or commit directly to `main`.
Branch first.

## Writing: I have taken one quarter of RL

Write for that level, in code comments, docs, commit messages, and PR
descriptions alike.

- Explain the term the first time it appears. Don't assume AlphaZero vocabulary.
- Say what a thing *does* and *what breaks without it*, not what category it belongs to.
- Prefer "the GPU sits idle waiting for one position at a time" over "inference
  latency bounds throughput."
- Keep the real reasoning. Simpler language, not less substance — I need to be
  able to defend these decisions in an interview.

If I have to re-read a sentence to parse it, it's too dense.

## Things not to re-derive

- **Quality gates:** `ruff check`, `ruff format --check`, `pyright`, `pytest -m "not slow and not gpu"`. All must pass before you tell me something is done. `uv` is not installed yet; a throwaway venv in the scratchpad is the workaround, and it must not touch my conda base.
- **CPU-only rule:** every test and the whole CI pipeline run without a GPU. `reversi.game` and `reversi.config` must not import torch.
- **Never commit:** checkpoints, replay data, run folders (`runs/`, `models/`, `*.pt`, `*.npz`). `.gitignore` and `.gitattributes` already cover this — don't loosen them.
- **The plan:** `.claude/plans/assume-one-rtx-6000-l40a-crispy-dusk.md` holds the 14-day roadmap, the task backlog, and the correctness contracts. Follow it; flag it if something in it turns out to be wrong.

## Reporting

Say what actually happened. If a check didn't run, say it didn't run — don't
imply it passed. If something is half-finished, say which half.
