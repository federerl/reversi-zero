# Step 4 — Results: the Terminal Check in Self-Play

**Applies to:** the [Step 4 design](step4-design.md); every number below is read against the predictions in its section 6
**Runs:** `step4-Bboth-tc-s1`, `step4-Bboth-tc-s2`, continued from `control120` for twenty generations under `configs/full8x8_step4_check.yaml`; compared with the Step 3b B-both seeds, not re-run
**Date:** November 4, 2026
**Data:** `research/results/step4/` (scorecard, the checked pairings, head probe, row counts, per-generation training metrics, calibration)

**Verdict: the null the design predicted, with the mechanism visible in the data.** With the terminal check on during training, the network stopped walking into wipeouts: it was wiped out in 0.5% and 0.6% of its external training games against the Step 3b runs' 6.0% and 6.9%. Evaluated with the plain search, it was wiped out by Greedy 59 and 58 times in 200 games, against 61 and 69 for the Step 3b seeds and 65 for the checkpoint all runs started from. The value head did not move at the fatal positions; neither did the policy head's prior on the fatal move. The check at evaluation still removes the wipeouts on these weights (5 against Greedy, 9 against the random-playout search), and nothing else on the scorecard moved. Exact resolution inside the search teaches the network nothing it can use without the search, and the row counts say why: the check removed the training rows that carried the lesson rather than adding any.

## 1. What was run

| run | seed | games | of which external | training rows | of which value-only | wall time | network's score in the external games | wiped out in the external games |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| step3-Bboth-s1 (Step 3b) | 1337 | 12,000 | 2,400 | 684,605 | 58,461 | 1.32 h | 93.8% | 6.0% |
| step3-Bboth-s2 (Step 3b) | 1338 | 12,000 | 2,400 | 684,173 | 58,201 | 1.29 h | 92.6% | 6.9% |
| **step4-Bboth-tc-s1** | 1337 | 12,000 | 2,400 | 689,380 | 60,203 | 3.9 h | **99.2%** | **0.5%** |
| **step4-Bboth-tc-s2** | 1338 | 12,000 | 2,400 | 688,987 | 60,054 | 3.9 h | **98.8%** | **0.6%** |

The Step 4 runs trained on the cluster's RTX 6000 node with 14 self-play workers; the Step 3b runs trained on an L40S with 22. The wall times are therefore not comparable across the two rows. The like-for-like cost of the check comes from Step 5, whose runs used the same node and worker count without it: 285 to 289 seconds of self-play per generation against 460 to 463 here. The check cost about 60% more self-play time, well above the tenth to a third the design expected; every opened node tries each legal move in Python, and a generation opens several million nodes.

**Prediction 1 held.** The network's wipeout rate in the external games fell from 6 to 7% to under 1% from the first generation (0.8% and 0.0% at generation 121) and stayed there: 0.0 to 1.7% on seed 1, 0.0 to 2.5% on seed 2 across the twenty generations, 0.54% and 0.62% overall. Its score in those games rose from 93 to 99%. The games were a little longer on average, 61.1 plies against 60.5, as fewer of them ended early. The switch reached the workers, and the rest of the design is readable.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. Every column is the plain search at 50 simulations, the check off. The control and the Step 3b columns are the Step 3 and 3b numbers, not re-run.

| row | opponent | control120 | B-both-s1 | B-both-s2 | B-both-tc-s1 | B-both-tc-s2 |
|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 10.0% [6.6, 14.9], 9 wo | 14.8% [10.5, 20.3], 2 wo | 11.8% [8.0, 16.9], 6 wo | 11.5% [7.8, 16.7], 4 wo | 8.8% [5.6, 13.5], 7 wo |
| **held out** | Edax L5 | 29.8% [23.8, 36.4], 0 wo | 26.2% [20.6, 32.8], 2 wo | 37.0% [30.6, 43.9], 1 wo | 29.0% [23.2, 35.6], 2 wo | 34.0% [27.8, 40.8], 3 wo |
| **held out** | Edax L3 | 72.0% [65.4, 77.8], 0 wo | 68.0% [61.2, 74.1], 0 wo | 64.8% [57.9, 71.0], 1 wo | 66.8% [60.0, 72.9], 1 wo | 70.5% [63.8, 76.4], 0 wo |
| **held out** | rollout-r3000 | 81.0% [75.0, 85.8], 34 wo | 80.8% [74.7, 85.6], 36 wo | 80.8% [74.7, 85.6], 37 wo | 86.5% [81.1, 90.6], 26 wo | 85.0% [79.4, 89.3], 29 wo |
| retention | gen60 | 62.5% [55.6, 68.9], 0 wo | 61.8% [54.9, 68.2], 0 wo | 62.3% [55.4, 68.7], 0 wo | 62.0% [55.1, 68.4], 0 wo | 62.3% [55.4, 68.7], 0 wo |
| seen | Greedy | 67.5% [60.7, 73.6], **65 wo** | 69.0% [62.3, 75.0], **61 wo** | 65.5% [58.7, 71.7], **69 wo** | 70.0% [63.3, 75.9], **59 wo** | 71.0% [64.4, 76.8], **58 wo** |
| seen | Minimax-d4 | 99.0% [96.4, 99.7], 0 wo | 96.2% [92.6, 98.1], 0 wo | 97.5% [94.3, 98.9], 0 wo | 96.0% [92.3, 98.0], 0 wo | 97.8% [94.6, 99.1], 0 wo |
| seen family | greedy-corner | 63.5% [56.6, 69.9], 72 wo | 60.5% [53.6, 67.0], 77 wo | 65.0% [58.2, 71.3], 70 wo | 70.5% [63.8, 76.4], 58 wo | 72.5% [65.9, 78.2], 55 wo |
| seen family | greedy-d2 | 59.0% [52.1, 65.6], 82 wo | 58.0% [51.1, 64.6], 84 wo | 55.5% [48.6, 62.2], 89 wo | 58.5% [51.6, 65.1], 82 wo | 62.5% [55.6, 68.9], 75 wo |
| seen family | noisy-greedy | 73.5% [67.0, 79.1], 52 wo | 78.0% [71.8, 83.2], 44 wo | 79.5% [73.4, 84.5], 41 wo | 73.5% [67.0, 79.1], 52 wo | 79.5% [73.4, 84.5], 41 wo |
| seen family | corner | 98.0% [95.0, 99.2], 4 wo | 96.5% [93.0, 98.3], 7 wo | 97.5% [94.3, 98.9], 5 wo | 97.0% [93.6, 98.6], 6 wo | 97.5% [94.3, 98.9], 5 wo |
| seen family | mobility | 91.5% [86.8, 94.6], 17 wo | 91.0% [86.2, 94.2], 18 wo | 95.0% [91.0, 97.3], 10 wo | 94.2% [90.1, 96.7], 11 wo | 88.5% [83.3, 92.2], 23 wo |
| sanity | Random | 100% [98.1, 100], 0 wo | 99.0% [96.4, 99.7], 2 wo | 98.0% [95.0, 99.2], 4 wo | 99.5% [97.2, 99.9], 1 wo | 97.5% [94.3, 98.9], 5 wo |

**The plain search loses the same games.** Greedy wipeouts 59 and 58, inside the 45-to-66 range the design named and within seed noise of every earlier run. The random-playout row, the one held-out opponent that attacks the same weakness, reads 26 and 29 against the Step 3b seeds' 36 and 37 and the Step 3 A seeds' 26 and 38: inside the control range, at its low edge. The Edax rows and the generation-60 row are where the Step 3b seeds left them; so are Minimax, the corner and mobility players and Random.

**The greedy-corner row leans.** 58 and 55 wipeouts against 77 and 70, fifteen to nineteen fewer on both seeds. The scorecard resolves about ±6 to 7 points per row and the seeds differ by up to 10 ([Step 3 results](step3-results.md), section 9), so this is at the edge of what two seeds at 200 games can see, and it is reported as a lean, not a result; greedy-d2 (82 and 75 against 84 and 89) and noisy-greedy (52 and 41 against 44 and 41) do not lean with it.

**With the check on at evaluation**, the trained weights read as the control did: B-both-tc-s1 scores 97.5% against Greedy with 5 wipeouts and 95.5% against the random-playout search with 9, against `control120-tc`'s 94.0% and 12, and 93.2% and 12 ([Step 3c](step3c-terminal-check.md)). These two pairings used the same entrant name as the plain run, so they drew the same hundred openings; the Step 3c comparison did not, and is unpaired.

## 3. The predictions, one by one

1. **The training games change.** Held: 6 to 7% to 0.5 to 0.6%, from the first generation.
2. **The value head does not move.** Held: at the fatal positions the value with Greedy to move reads −1.00 on both seeds, and the value at the network's own position +0.99 (section 4).
3. **The policy head moves at the network's own position.** Failed: the prior on the fatal move reads 27% and 30%, against the Step 3b seeds' 29% and 28%; the design asked for a fall below 20%. Thousands of policy targets at low-disc positions that avoid the fatal move did not move the prior at the fatal positions the evaluation produced.
4. **The plain search still loses the same games.** Held: Greedy 59 and 58, the random-playout row 26 and 29, the held-out and retention rows within seed noise.
5. **With the check at evaluation, the run reads like `control120-tc`.** Held: 5 and 9 wipeouts, 97.5% and 95.5%.
6. **No cost.** Held: the generation-60 row 62.0% and 62.3% against 61.8% and 62.3%; the Edax rows within seed noise; calibration in section 6.

Five of six. The one that failed is the one with the most training signal behind it, and its failure is read in section 4.

## 4. The head probe

As in Steps 3b and 3c: for every evaluation game the network lost by wipeout against Greedy where a surviving move existed at its last decision, the two heads are read at that position (P) and at the position after the move it played, with Greedy to move (C). Medians over the probed positions. `scripts/probe_heads.py`, output `research/results/step4/head-probe.json`.

| network | wipeouts probed | raw value at P | raw value at C, for Greedy (exact: +1) | prior on Greedy's wiping reply | prior on the move played | prior on all safe moves | plain search escapes | checked search escapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control120 | 63 | +0.99 | −0.99 | 3.6% | 36% | 44% | 0% | 95% |
| B-both-s1 | 60 | +0.93 | −0.98 | 4.6% | 29% | 46% | 0% | 97% |
| B-both-s2 | 66 | +0.96 | −0.99 | 4.4% | 28% | 35% | 0% | 98% |
| **B-both-tc-s1** | 57 | **+0.99** | **−1.00** | 3.6% | 27% | 38% | 0% | 100% |
| **B-both-tc-s2** | 51 | **+0.99** | **−1.00** | 4.1% | 30% | 41% | 0% | 100% |

Two things to read off.

**The value at P went back up.** The Step 3b seeds read +0.93 and +0.96 at the network's own fatal position, down from the control's +0.99; those were the only value-head movements any training produced, and they came from the 307 and 311 rows at low-disc positions that Step 3b's runs labelled as losses. The Step 4 runs read +0.99 again. The check steered the network away from losing at those positions, so the rows that said "this can be lost" went from about 300 to 17 and 27 (section 5), and the value head returned to the control's answer. The mechanism the design's section 3 described is visible: the check removes the lesson from the data.

**The policy head learned nothing it could carry to these positions.** Every row at a P-like position in training now carried a target that avoided the fatal move, several thousand of them per run. At the fatal positions the evaluation games produced, the prior on the fatal move is where it was. What the policy head would have needed to learn is a two-ply fact, that *this* move allows a reply that takes everything, and the rows showed it only which move was avoided, not why.

The probe reads each network at its own wipeouts, which selects the positions it has not learned; the fixed-position probe in the [Step 5 results](step5-results.md), section 4, reads every network on the control's 63 positions and gives the same answer for these two: priors of 3.9% and 3.8% on the wiping reply, 26% and 28% on the fatal move.

## 5. The training rows

The Step 3b row count (`scripts/step3b_lesson_rows.py`), on the Step 4 shards: rows past the opening in which the network holds at most three discs, split by who is to move and how the game ended. `research/results/step4/lesson-rows.json`.

| run | rows | value-only rows | opponent to move, network ≤3 discs | of which the opponent won (the lesson) | of which the network won | network to move, ≤3 discs | of which it lost | of which it won |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step3-Bboth-s1 | 684,605 | 58,461 | 230 | **77** | 153 | 4,573 | 307 | 4,266 |
| step3-Bboth-s2 | 684,173 | 58,201 | 235 | **83** | 152 | 4,171 | 311 | 3,858 |
| **step4-Bboth-tc-s1** | 689,380 | 60,203 | 135 | **2** | 133 | 4,616 | **17** | 4,599 |
| **step4-Bboth-tc-s2** | 688,987 | 60,054 | 160 | **10** | 150 | 4,377 | **27** | 4,350 |

The states are still there: four and a half thousand network-to-move rows at three discs or fewer, as in Step 3b. The lesson is gone from them: the network lost from such a position 17 and 27 times instead of 307 and 311, and the opponent-to-move rows labelled as the opponent's win, the rows that carried the lesson in Step 3b, number 2 and 10 instead of 77 and 83. The check made the network a player that does not lose these positions, and a player that does not lose them produces no examples of losing them. The value head learned exactly what the data said, which is now even more firmly that a low-disc position is fine.

## 6. Value calibration

As in Step 3b: the network's win-probability estimate against the outcome, by row group, on its own self-play positions and on the evaluation games. `scripts/value_calibration.py`, `research/results/step4/value-calibration.json`.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against the Greedy family | against Random |
|---|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.89, won 0.95, gap 0.09 | est. 0.95, won 0.93, gap 0.06 | est. 0.96, won 1.00, gap 0.04 |
| B-both-s1 | est. 0.52, won 0.52, gap 0.02 | est. 0.58, won 0.49, gap 0.09 | est. 0.58, won 0.62, gap 0.06 | est. 0.89, won 0.94, gap 0.07 | est. 0.94, won 0.93, gap 0.06 | est. 0.95, won 1.00, gap 0.05 |
| B-both-s2 | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.50, gap 0.08 | est. 0.59, won 0.62, gap 0.06 | est. 0.90, won 0.94, gap 0.08 | est. 0.94, won 0.94, gap 0.06 | est. 0.95, won 1.00, gap 0.04 |
| **B-both-tc-s1** | est. 0.50, won 0.52, gap 0.02 | est. 0.58, won 0.50, gap 0.08 | est. 0.55, won 0.62, gap 0.08 | est. 0.89, won 0.94, gap 0.09 | est. 0.95, won 0.94, gap 0.07 | est. 0.96, won 1.00, gap 0.04 |
| **B-both-tc-s2** | est. 0.51, won 0.52, gap 0.01 | est. 0.59, won 0.51, gap 0.08 | est. 0.57, won 0.62, gap 0.07 | est. 0.91, won 0.95, gap 0.08 | est. 0.95, won 0.94, gap 0.06 | est. 0.96, won 0.99, gap 0.04 |

Unchanged: calibrated to within two points on its own positions, the same 8 to 9 points of overrating against the held-out opponents that every self-play network shows. The value target did not change and neither did the value head's calibration.

## 7. What this establishes

- **Exact resolution inside the training search does not teach the heads the tactic.** The strongest data-side route, every training game played by a search that already knows the one-move answer, left the value head where it was, the prior on the wiping reply where it was, the prior on the fatal move where it was, and the plain search losing the same games. The data-side routes have now been tried in three forms, opponent exposure (Step 3), the value target at the position the search reads (Step 3b), and the exact answer inside the search that writes the targets (Step 4), and none transferred.
- **The check in self-play removes the lesson rather than adding it.** It steers the network away from the positions where the lesson is written, so the loss labels at low-disc positions fall by a factor of ten and the value head unlearns the little Step 3b had taught it. A training intervention that makes the agent play better during training can make the data worse for the thing one wants learned.
- **The policy head did not generalise a two-ply avoidance.** Thousands of targets that avoided the fatal move at seen positions did not lower its prior at unseen ones. That is the policy-side twin of the Step 3b value result, and the reason the [Step 5 design](step5-design.md) gives the policy head a one-ply target instead.
- **The check's effect is a property of the search, not of the weights.** On the trained weights it removes the wipeouts as it did on the control (5 and 9 against 12 and 12), with paired openings this time.

**What it does not establish.** It does not test the check with a larger search budget in self-play, or for longer than twenty generations; the row counts say more of the same training would produce more of the same data. It does not show whether the policy head learned the training positions themselves; the probe reads evaluation positions, and generalisation is the question the report asks. The greedy-corner lean is within resolution and is not a finding. The scorecard resolves about ±6 to 7 points per row and the seeds differ by up to 10, and "did not move" means "did not move by more than that" throughout.

## 8. What follows

The [Step 5 design](step5-design.md): the network's search run at the opponent's positions, so that the policy head gets a target at the position the search reads, one ply below the fatal move, where the quantity the search is missing is a single reply rather than a two-ply consequence. Its results are in the [Step 5 results](step5-results.md).

## 9. Reproduction

On the cluster, with the Step 3 and 3b checkpoints, recorded games and the `control120` run under `~/reversi-runs/`:

```bash
# training: the Step 3b recipe plus the check, continued from control120
cp -r ~/reversi-runs/control ~/reversi-runs/step4-Bboth-tc-s1
CONFIG=configs/full8x8_step4_check.yaml RUN_ID=step4-Bboth-tc-s1 GENERATIONS=140 WORKERS=14 \
    sbatch --job-name=rz-step4-Bboth-tc-s1 --nodelist=gebru --gres=gpu:quadro_rtx_6000:1 --cpus-per-task=16 slurm/train.sbatch
# the second seed adds EXTRA_ARGS="--set seed=1338"

# evaluation: one one-core CPU job per pairing, the frozen protocol, into the Step 3 directory,
# plus the two checked pairings under the same entrant name into a directory of their own
ckpt=~/reversi-runs/step4-Bboth-tc-s1/checkpoints/gen_00140.pt
for opp in edax-l7 edax-l3 edax-l5 rollout-r3000 "gen60=$HOME/reversi-models/reversi-8x8-gen60.pt" \
           greedy minimax-d4 greedy-corner greedy-d2 noisy-greedy corner mobility random; do
  sbatch --job-name="rz-eval-Btc-s1-${opp%%=*}" --cpus-per-task=1 --mem=6G slurm/cpu.sbatch \
    uv run reversi arena --suite custom -e "Btc-s1=$ckpt" -e "$opp" \
      --games 200 --simulations 50 --workers 1 --seed 20261025 --record-games \
      --out ~/reversi-runs/step3-eval/Btc-s1-vs-${opp%%=*}.json
done
for opp in greedy rollout-r3000; do
  sbatch --job-name="rz-eval-Btc-s1-tc-$opp" --cpus-per-task=1 --mem=6G slurm/cpu.sbatch \
    uv run reversi arena --suite custom -e "Btc-s1=$ckpt@50;terminal_check" -e "$opp" \
      --games 200 --simulations 50 --workers 1 --seed 20261025 --record-games \
      --out ~/reversi-runs/step4-eval-tc/Btc-s1-vs-$opp.json
done

# the tables
python scripts/step3_scorecard.py ~/reversi-runs/step3-eval --checkpoints control120 Bboth-s1 Bboth-s2 Btc-s1 Btc-s2 \
    --out research/results/step4/scorecard.json --markdown
python scripts/step3_scorecard.py ~/reversi-runs/step4-eval-tc --checkpoints Btc-s1 --out research/results/step4/scorecard-checked.json
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol ~/reversi-runs/step3-eval/step4-checkpoints.json \
    --networks control120 Bboth-s1 Bboth-s2 Btc-s1 Btc-s2 --out research/results/step4/head-probe.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step4-Bboth-tc-s1 ~/reversi-runs/step4-Bboth-tc-s2 \
    --from-generation 121 --out research/results/step4/lesson-rows.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval --checkpoints control120 Bboth-s1 Bboth-s2 Btc-s1 Btc-s2 \
    --self-play Btc-s1=~/reversi-runs/step4-Bboth-tc-s1 --self-play Btc-s2=~/reversi-runs/step4-Bboth-tc-s2 \
    --out research/results/step4/value-calibration.json
```

`step4-checkpoints.json` is the Step 3b checkpoint map with the two Step 4 checkpoints added, by evaluation name. The per-generation training metrics in `research/results/step4/training.json` are the runs' `metrics/selfplay.jsonl` rows from generation 121 on.
