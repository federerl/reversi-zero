# Step 5 — Results: Searching the Opponent's Positions

**Applies to:** the [Step 5 design](step5-design.md); every number below is read against the predictions in its section 5
**Runs:** `step5-Bboth-sp-s1`, `step5-Bboth-sp-s2`, continued from `control120` for twenty generations under `configs/full8x8_step5_search.yaml`; compared with the Step 3b B-both seeds, not re-run
**Date:** November 4, 2026
**Data:** `research/results/step5/` (scorecard, the checked pairings, head probe at each network's own wipeouts and at one fixed set of positions, the searched-row count, row counts, per-generation training metrics, calibration)

**Verdict: the first training route that moved the wipeouts, by a third, on both seeds, at no cost, and without reaching the held-out opponent.** Giving the policy head a target at the opponent's positions, the position the search reads one ply below the fatal move, cut the wipeouts against Greedy from 61 and 69 to 54 and 45 under the plain search, and against every opponent in the Greedy family by 15 to 30%: greedy-d2 from 84 and 89 to 58 and 63 on the two seeds. Retention and the Edax rows did not move. The prior on Greedy's wiping reply, read on one fixed set of fatal positions, doubled from about 4% to 8.5% and 8.2%, the first movement of that number in any run, and the plain search at 50 simulations escapes 44% and 32% of the control's fatal positions where the Step 3b seeds escape 21% and 22%. The random-playout row did not move: 31 and 38 wipeouts against 36 and 37. The policy head learned the positions it was shown, against the family of opponents that produces them, and not the rule.

## 1. What was run

| run | seed | games | of which external | training rows | of which value-only | wall time | network's score in the external games | wiped out in the external games |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| step3-Bboth-s1 (Step 3b) | 1337 | 12,000 | 2,400 | 684,605 | 58,461 | 1.32 h | 93.8% | 6.0% |
| step3-Bboth-s2 (Step 3b) | 1338 | 12,000 | 2,400 | 684,173 | 58,201 | 1.29 h | 92.6% | 6.9% |
| **step5-Bboth-sp-s1** | 1337 | 12,000 | 2,400 | 685,099 | **0** | 2.1 h | 94.2% | 5.3% |
| **step5-Bboth-sp-s2** | 1338 | 12,000 | 2,400 | 684,688 | **0** | 2.1 h | 94.1% | 5.4% |

Same games, same row counts; the 58,000 rows that were value-only in Step 3b carry a policy target here. The Step 5 runs trained on the cluster's RTX 6000 node with 14 self-play workers, the Step 3b runs on an L40S with 22, so the wall times are not comparable. The searches at the opponent's positions cost 285 to 289 seconds of self-play per generation on that node; the [Step 4 runs](step4-results.md) on the same node, with the terminal check instead, cost 460 to 463.

The network was wiped out in 5.3% and 5.4% of its external games, a little under the Step 3b seeds' 6.0% and 6.9%; the per-generation rates range from 1.7% to 11.7% on both, and the difference is within what 120 games per generation resolve. The games were played the same way; the design expected the rate unchanged and it is, to within noise.

**Prediction 1 held in substance and missed its threshold.** The searched-row count (`scripts/step5_searched_rows.py`, `research/results/step5/searched-rows.json`) finds 126 and 130 opponent-to-move rows at which some legal move takes every disc of the network's; at 105 and 112 of them, 83% and 86%, the stored target puts more than half its mass on such a move, and the mean mass on wiping moves is 0.74 on both seeds. The design asked for nine in ten. The search found the reply in five of six rows rather than nine of ten: at 200 simulations with a 4% prior and root noise, the wiping child is not always visited. The signal reached the training data at most of the positions that matter, and the rest of the design is readable with that discount.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. Every column is the plain search at 50 simulations. The control and the Step 3b columns are the Step 3 and 3b numbers, not re-run.

| row | opponent | control120 | B-both-s1 | B-both-s2 | B-both-sp-s1 | B-both-sp-s2 |
|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 10.0% [6.6, 14.9], 9 wo | 14.8% [10.5, 20.3], 2 wo | 11.8% [8.0, 16.9], 6 wo | 11.2% [7.6, 16.4], 8 wo | 8.5% [5.4, 13.2], 7 wo |
| **held out** | Edax L5 | 29.8% [23.8, 36.4], 0 wo | 26.2% [20.6, 32.8], 2 wo | 37.0% [30.6, 43.9], 1 wo | 31.2% [25.2, 38.0], 2 wo | 35.2% [29.0, 42.1], 3 wo |
| **held out** | Edax L3 | 72.0% [65.4, 77.8], 0 wo | 68.0% [61.2, 74.1], 0 wo | 64.8% [57.9, 71.0], 1 wo | 71.8% [65.1, 77.5], 1 wo | 70.5% [63.8, 76.4], 1 wo |
| **held out** | rollout-r3000 | 81.0% [75.0, 85.8], 34 wo | 80.8% [74.7, 85.6], 36 wo | 80.8% [74.7, 85.6], 37 wo | 84.5% [78.8, 88.9], 31 wo | 80.5% [74.5, 85.4], 38 wo |
| retention | gen60 | 62.5% [55.6, 68.9], 0 wo | 61.8% [54.9, 68.2], 0 wo | 62.3% [55.4, 68.7], 0 wo | 64.5% [57.7, 70.8], 0 wo | 67.8% [61.0, 73.8], 0 wo |
| seen | Greedy | 67.5% [60.7, 73.6], **65 wo** | 69.0% [62.3, 75.0], **61 wo** | 65.5% [58.7, 71.7], **69 wo** | 72.5% [65.9, 78.2], **54 wo** | 77.5% [71.2, 82.7], **45 wo** |
| seen | Minimax-d4 | 99.0% [96.4, 99.7], 0 wo | 96.2% [92.6, 98.1], 0 wo | 97.5% [94.3, 98.9], 0 wo | 97.5% [94.3, 98.9], 0 wo | 98.0% [95.0, 99.2], 0 wo |
| seen family | greedy-corner | 63.5% [56.6, 69.9], 72 wo | 60.5% [53.6, 67.0], 77 wo | 65.0% [58.2, 71.3], 70 wo | 72.5% [65.9, 78.2], **55 wo** | 71.5% [64.9, 77.3], **57 wo** |
| seen family | greedy-d2 | 59.0% [52.1, 65.6], 82 wo | 58.0% [51.1, 64.6], 84 wo | 55.5% [48.6, 62.2], 89 wo | 70.5% [63.8, 76.4], **58 wo** | 68.0% [61.2, 74.1], **63 wo** |
| seen family | noisy-greedy | 73.5% [67.0, 79.1], 52 wo | 78.0% [71.8, 83.2], 44 wo | 79.5% [73.4, 84.5], 41 wo | 82.0% [76.1, 86.7], **36 wo** | 83.5% [77.7, 88.0], **33 wo** |
| seen family | corner | 98.0% [95.0, 99.2], 4 wo | 96.5% [93.0, 98.3], 7 wo | 97.5% [94.3, 98.9], 5 wo | 98.2% [95.3, 99.4], 3 wo | 98.5% [95.7, 99.5], 3 wo |
| seen family | mobility | 91.5% [86.8, 94.6], 17 wo | 91.0% [86.2, 94.2], 18 wo | 95.0% [91.0, 97.3], 10 wo | 92.5% [88.0, 95.4], 14 wo | 96.5% [93.0, 98.3], 7 wo |
| sanity | Random | 100% [98.1, 100], 0 wo | 99.0% [96.4, 99.7], 2 wo | 98.0% [95.0, 99.2], 4 wo | 99.0% [96.4, 99.7], 2 wo | 100% [98.1, 100], 0 wo |

**The Greedy family moved, all of it, on both seeds.** Eight numbers, two seeds by four disc-hungry opponents, and every one fell: Greedy 61 → 54 and 69 → 45; greedy-corner 77 → 55 and 70 → 57; greedy-d2 84 → 58 and 89 → 63; noisy-greedy 44 → 36 and 41 → 33. The greedy-d2 change, 26 fewer wipeouts on each seed, and the greedy-corner change, 22 and 13, are outside the ±10 the design can resolve; the Greedy row's 7 on seed 1 is not on its own, and reads as part of the family. The scores rose with the wipeouts: greedy-d2 from 58.0% and 55.5% to 70.5% and 68.0%.

**Nothing else moved.** The generation-60 row reads 64.5% and 67.8% against 61.8% and 62.3%, up if anything; the Edax rows and Minimax are within seed noise; the corner and mobility players and Random are where they were. This is what separates the result from group D's ([Step 3 results](step3-results.md), section 6): D cut the Greedy wipeouts to 44 and paid with half its strength against Edax and the generation-60 checkpoint. Step 5 cut them to 45 and 54 and paid nothing the scorecard can see.

**The held-out row did not move.** The random-playout search, the opponent that attacks the same weakness knowing nothing, wiped the Step 5 networks out 31 and 38 times against the Step 3b seeds' 36 and 37 and the control's 34. The design's fourth prediction asked the row to move in proportion to Greedy's; it did not. Section 7 reads this.

**With the check on at evaluation**, B-both-sp-s1 scores 95.5% against Greedy with 9 wipeouts and 93.5% against the random-playout search with 11, with paired openings; the check adds on these weights about what it added on the control (12 and 12).

## 3. The predictions, one by one

1. **The search finds the reply where it exists.** Held in substance, missed the threshold: 83% and 86% of the rows with a wiping move carry a target mostly on one, against the nine in ten asked; the training wipeout rate stayed near Step 3b's.
2. **The prior on the wiping reply rises above 10% on both seeds.** Missed the threshold, by less than it looks. At each network's own wipeouts the prior reads 5.7% and 6.4% (section 4, first table); on the control's fixed positions it reads 8.5% and 8.2% against the Step 3b seeds' 4.2% and 3.9% (second table). It doubled, for the first time in any run, and stopped short of 10%.
3. **The value head does not move.** Held: −0.97 and −0.98 at C, +0.97 and +0.98 at P on the fixed positions; −0.98 and −0.99, +0.96 and +0.97 at the networks' own.
4. **Greedy wipeouts between 35 and 60 on both seeds, the random-playout row moving in proportion.** Half held: Greedy 54 and 45, inside the range; the random-playout row 31 and 38, unmoved.
5. **The Edax rows, retention and calibration within seed noise of B-both.** Held for the scorecard rows; calibration in section 6.

## 4. The head probe

Two readings. The first is the probe as run in Steps 3b to 4: each network at the fatal positions of its own evaluation wipeouts against Greedy. That selects, for each network, the positions it has not learned: wherever training raised the prior enough for the search to escape, the position is no longer a wipeout and drops out of the sample, so a network that learned something is read where it did not. The second reading, added for this step (`--positions-from control120`), reads every network on the control's 63 fatal positions, so the heads are compared on the same boards. `scripts/probe_heads.py`; `research/results/step5/head-probe.json` and `head-probe-fixed.json`.

**At each network's own wipeouts:**

| network | wipeouts probed | raw value at P | raw value at C, for Greedy (exact: +1) | prior on Greedy's wiping reply | prior on the move played | prior on all safe moves | plain search escapes | checked search escapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control120 | 63 | +0.99 | −0.99 | 3.6% | 36% | 44% | 0% | 95% |
| B-both-s1 | 60 | +0.93 | −0.98 | 4.6% | 29% | 46% | 0% | 97% |
| B-both-s2 | 66 | +0.96 | −0.99 | 4.4% | 28% | 35% | 0% | 98% |
| B-both-tc-s1 (Step 4) | 57 | +0.99 | −1.00 | 3.6% | 27% | 38% | 0% | 100% |
| B-both-tc-s2 (Step 4) | 51 | +0.99 | −1.00 | 4.1% | 30% | 41% | 0% | 100% |
| **B-both-sp-s1** | 50 | +0.96 | −0.98 | **5.7%** | 28% | 49% | 0% | 88% |
| **B-both-sp-s2** | 43 | +0.97 | −0.99 | **6.4%** | 26% | 38% | 0% | 93% |

**On the control's 63 fatal positions, every network:**

| network | raw value at P | raw value at C, for Greedy | prior on Greedy's wiping reply | prior on the control's fatal move | prior on all safe moves | plain search escapes | policy-only search escapes | checked search escapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control120 | +0.99 | −0.99 | 3.6% | 36% | 44% | 0% | 5% | 95% |
| B-both-s1 | +0.95 | −0.99 | 4.2% | 29% | 49% | 21% | 22% | 97% |
| B-both-s2 | +0.96 | −0.99 | 3.9% | 27% | 56% | 22% | 24% | 97% |
| B-both-tc-s1 (Step 4) | +1.00 | −1.00 | 3.9% | 26% | 57% | 29% | 30% | 97% |
| B-both-tc-s2 (Step 4) | +1.00 | −1.00 | 3.8% | 28% | 56% | 24% | 27% | 97% |
| **B-both-sp-s1** | +0.97 | −0.97 | **8.5%** | 28% | 56% | **44%** | **43%** | 97% |
| **B-both-sp-s2** | +0.98 | −0.98 | **8.2%** | 29% | 52% | **32%** | **33%** | 97% |

The control escapes none of its own fatal positions by construction. Every other network escapes some of them, because it is a different network reading the same board: the Step 3b and Step 4 seeds escape 21 to 29%, which is the baseline for "a network of this recipe on positions it did not itself fall into". The Step 5 seeds escape 44% and 32%, and the policy-only search, which has nothing but the prior to go on, escapes 43% and 33% against the others' 22 to 30%: the escapes come from the prior. The prior on the wiping reply is the number every earlier run left between 3.6% and 4.6%; here it reads 8.5% and 8.2%. The value at C, read on the same positions, is −0.97 and −0.98, as everywhere.

So the chain closes where the Step 2 analysis said it would. The search at P expands C and, with the prior on the wiping reply at 8% instead of 4%, reaches the finished position below it on some of its visits; where it does, it reads −1 and avoids the move. Doubling the prior roughly doubled the plain search's escape rate on these positions; it did not make the search reliable, and at 50 simulations a prior of 8% still goes unvisited often. That is the partial effect the scorecard shows.

## 5. The training rows

The Step 3b row count on the Step 5 shards, with the opponent's rows identified by the `opponent_to_move` column. `research/results/step5/lesson-rows.json`.

| run | rows | value-only rows | opponent to move, network ≤3 discs | of which the opponent won (the lesson) | of which the network won | network to move, ≤3 discs | of which it lost | of which it won |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step3-Bboth-s1 | 684,605 | 58,461 | 230 | 77 | 153 | 4,573 | 307 | 4,266 |
| step3-Bboth-s2 | 684,173 | 58,201 | 235 | 83 | 152 | 4,171 | 311 | 3,858 |
| **step5-Bboth-sp-s1** | 685,099 | 0 | 248 | **56** | 192 | 4,603 | 204 | 4,399 |
| **step5-Bboth-sp-s2** | 684,688 | 0 | 230 | **56** | 174 | 4,577 | 231 | 4,345 |

The same data as Step 3b, with the value-only column gone: every row now trains both heads. The lesson rows at the opponent's position, 56 and 56, are in the Step 3b range and, by the searched-row count, carry a target on the wiping reply five times in six. These 56 rows per run, among 58,000 opponent-to-move rows and 685,000 in all, moved the prior at unseen positions of the same kind from 4% to 8%.

## 6. Value calibration

As in Step 3b. `scripts/value_calibration.py`, `research/results/step5/value-calibration.json`.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against the Greedy family | against Random |
|---|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.89, won 0.95, gap 0.09 | est. 0.95, won 0.93, gap 0.06 | est. 0.96, won 1.00, gap 0.04 |
| B-both-s1 | est. 0.52, won 0.52, gap 0.02 | est. 0.58, won 0.49, gap 0.09 | est. 0.58, won 0.62, gap 0.06 | est. 0.89, won 0.94, gap 0.07 | est. 0.94, won 0.93, gap 0.06 | est. 0.95, won 1.00, gap 0.05 |
| B-both-s2 | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.50, gap 0.08 | est. 0.59, won 0.62, gap 0.06 | est. 0.90, won 0.94, gap 0.08 | est. 0.94, won 0.94, gap 0.06 | est. 0.95, won 1.00, gap 0.04 |
| **B-both-sp-s1** | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.51, gap 0.08 | est. 0.58, won 0.65, gap 0.07 | est. 0.90, won 0.95, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **B-both-sp-s2** | est. 0.51, won 0.52, gap 0.02 | est. 0.59, won 0.51, gap 0.09 | est. 0.56, won 0.68, gap 0.12 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.95, won 1.00, gap 0.05 |

Unchanged, as the design expected: the value head saw the same rows with the same labels as in Step 3b. Seed 2 undersells itself against the generation-60 checkpoint by 12 points, which is the retention row's 67.8% showing up as outcomes the value head did not expect; the self-play and held-out columns are where every self-play network sits.

## 7. What this establishes

- **The policy head can carry the one-ply lesson the value head could not.** Three forms of training on the value target left the prior on the wiping reply at 4% and the wipeouts where they were. One form of training on the policy target, a search run at the position the search reads, doubled the prior and cut the wipeouts against every disc-hungry opponent by a third, on both seeds, at no cost in strength or retention. The difference between the routes is not the data, which is the same games, but which head is given a target at which position.
- **The lesson was learned as positions, not as the rule.** The Greedy family, four opponents that drive the network into the same low-disc states, all improved; the random-playout search, which finds the same hole from a different direction and produces different boards, did not. A prior that rose from 4% to 8% at positions like the ones seen did not rise at positions unlike them. The remedy reaches what its training data covered.
- **The partial effect is a search-budget effect on top of a learning effect.** At 50 simulations an 8% prior is visited on some searches and not others; where it is, the exact answer one ply below is read and the move avoided. Section 4's fixed-position table shows the plain search escaping twice as often as the Step 3b seeds, not always. More prior, or more simulations, would convert more of the learned positions into escapes; neither reaches positions the prior does not cover.
- **The cost comparison with group D stands.** The one earlier route that cut the wipeouts, a material term in the value target, paid with calibration and with half the strength against Edax and the earlier generation. This route paid nothing visible, because it changed what the policy head is told at 8.5% of the rows and nothing about the value target.

**What it does not establish.** It does not say whether the effect saturates: whether twice the external games, or a weight on the searched rows, would carry the prior past 10% and the wipeouts below 30, or whether 45 to 54 is where positions-learned-as-positions levels off. It does not test whether forty generations do better than twenty. It does not reach the held-out opponent, and the claim that the policy head learned "positions, not the rule" rests on that one row and the fixed-position probe; a second knowledge-free opponent would test it. The searched-row count says the search missed the reply in one row of six, so the dose delivered was five sixths of the dose intended. The scorecard resolves about ±6 to 7 points per row and the seeds differ by up to 10; the Greedy family's changes are outside that, the rollout row's and the retention row's are inside it.

## 8. What follows

In order of cost.

1. **A dose curve on the searched rows.** The same recipe with 240 and 480 external games per generation, or the searched opponent rows weighted two and four times in the policy loss; two seeds each at the cheapest point. It answers the one question this step leaves open: whether the prior keeps rising with the data and the wipeouts keep falling, or whether 8% and 45 is a ceiling for learning positions.
2. **Twenty more generations** of the two runs, 1.3 GPU-hours, for the same question along the time axis.
3. **A second knowledge-free opponent**, so that "did not reach the held-out row" rests on more than one row.
4. The search-budget dose curve with and without the check, on the Step 5 weights as well as the control: the learned prior and the exact check are two ways of reaching the same finished position, and the curve says how they trade.

## 9. Reproduction

As for Step 4 (its section 9), with `configs/full8x8_step5_search.yaml`, run names `step5-Bboth-sp-s1` and `-s2`, evaluation names `Bsp-s1` and `Bsp-s2`, the checked pairings under `~/reversi-runs/step5-eval-tc/`, and the probe on `step5-checkpoints.json`:

```bash
python scripts/step3_scorecard.py ~/reversi-runs/step3-eval --checkpoints control120 Bboth-s1 Bboth-s2 Bsp-s1 Bsp-s2 \
    --out research/results/step5/scorecard.json --markdown
python scripts/step3_scorecard.py ~/reversi-runs/step5-eval-tc --checkpoints Bsp-s1 --out research/results/step5/scorecard-checked.json
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol ~/reversi-runs/step3-eval/step5-checkpoints.json \
    --networks control120 Bboth-s1 Bboth-s2 Btc-s1 Btc-s2 Bsp-s1 Bsp-s2 --out research/results/step5/head-probe.json
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol ~/reversi-runs/step3-eval/step5-checkpoints.json \
    --positions-from control120 --networks control120 Bboth-s1 Bboth-s2 Btc-s1 Btc-s2 Bsp-s1 Bsp-s2 \
    --out research/results/step5/head-probe-fixed.json
python scripts/step5_searched_rows.py ~/reversi-runs/step5-Bboth-sp-s1 ~/reversi-runs/step5-Bboth-sp-s2 \
    --from-generation 121 --out research/results/step5/searched-rows.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step5-Bboth-sp-s1 ~/reversi-runs/step5-Bboth-sp-s2 \
    --from-generation 121 --out research/results/step5/lesson-rows.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval --checkpoints control120 Bboth-s1 Bboth-s2 Bsp-s1 Bsp-s2 \
    --self-play Bsp-s1=~/reversi-runs/step5-Bboth-sp-s1 --self-play Bsp-s2=~/reversi-runs/step5-Bboth-sp-s2 \
    --out research/results/step5/value-calibration.json
```
