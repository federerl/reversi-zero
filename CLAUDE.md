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
  able to defend every decision in this repo from first principles.

If I have to re-read a sentence to parse it, it's too dense.

## Things not to re-derive

- **Quality gates:** `ruff check`, `ruff format --check`, `pyright`, `pytest -m "not slow and not gpu"`. All must pass before you tell me something is done. `uv` is not installed yet; a throwaway venv in the scratchpad is the workaround, and it must not touch my conda base.
- **CPU-only rule:** every test and the whole CI pipeline run without a GPU. `reversi.game` and `reversi.config` must not import torch.
- **Never commit:** checkpoints, replay data, run folders (`runs/`, `models/`, `*.pt`, `*.npz`). `.gitignore` and `.gitattributes` already cover this — don't loosen them.
- **The plan:** `.claude/plans/assume-one-rtx-6000-l40a-crispy-dusk.md` holds the 14-day roadmap, the task backlog, and the correctness contracts. Follow it; flag it if something in it turns out to be wrong.

## The repo is public and read by strangers

Recruiters read this repository. Nothing in it should read like a transcript of
our conversation.

Commit messages, PR titles and bodies, code comments, and docs are written **for
someone who has never seen this chat**. That means:

- No first-person narration of your own reasoning or mistakes ("four things I had
  wrong", "as I mentioned earlier", "you asked me to").
- No addressing me. No "your run", "as requested", "per your question".
- Describe the change and why it is right, not the process that produced it or
  the order things were discovered in.
- Corrections are stated as facts about the code or the docs, not as an account
  of who believed what.

Anything that only makes sense with our conversation as context belongs in
`notes/`, which is gitignored.

## Reporting

Say what actually happened. If a check didn't run, say it didn't run — don't
imply it passed. If something is half-finished, say which half.
