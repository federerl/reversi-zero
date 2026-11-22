# Evaluation recheck across code versions

The evaluation reports in `step3-eval` were produced over several weeks, and each one records the git commit of the code that ran it: the Step 3 reports at a commit from early November, the Step 8 reports at `e75fa39`. Between those commits the search code gained options that the evaluation protocol does not use (the terminal check as a self-play setting in Step 4, the policy-loss weight in Step 7). Every comparison across steps assumes the plain search at 50 simulations behaved the same throughout.

The protocol is seeded, so the assumption can be checked by re-running old pairings at the new code: the same checkpoint, opponent, openings and seed must give the same games. Five pairings were re-run at `e75fa39` on November 8, 2026 and compared with the originals on every field of `matchups`:

| pairing | original report | re-run at `e75fa39` | identical |
|---|---|---|---|
| control120 vs greedy | Step 3 | `control120-vs-greedy.json` | yes |
| control120 vs greedy-d2 | Step 3 | `control120-vs-greedy-d2.json` | yes |
| control120 vs edax-l5 | Step 3 | `control120-vs-edax-l5.json` | yes |
| Bsp-s1 vs greedy | Step 5 | `Bsp-s1-vs-greedy.json` | yes |
| Bsp-s1 vs edax-l5 | Step 5 | `Bsp-s1-vs-edax-l5.json` | yes |

The files here are the re-runs; the originals are the ones the Step 3 and Step 5 scorecards were built from. The command is the scorecard command of the Step 3 design, section 7, with `--out` pointed at a separate directory:

```bash
reversi arena --suite custom -e "control120=runs/control/checkpoints/gen_00120.pt" -e greedy \
    --games 200 --simulations 50 --workers 1 --seed 20261025 --out results/eval-recheck/control120-vs-greedy.json
```
