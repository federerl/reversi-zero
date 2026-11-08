# Step 6 — Results: The Dose Curve on the Searched Opponent Rows

**Applies to:** the [Step 6 design](step6-design.md); every number below is read against the predictions in its section 4
**Runs:** `step6-Bsp240-s1`, `-s2` (`configs/full8x8_step6_search240.yaml`) and `step6-Bsp480-s1`, `-s2` (`configs/full8x8_step6_search480.yaml`), twenty generations from `control120`; `step5-Bboth-sp-s1`, `-s2` continued from generation 140 to 160 under the Step 5 config; compared with the Step 5 and Step 3b seeds, not re-run
**Date:** November 7, 2026
**Data:** `research/results/step6/` (scorecard, fixed-position head probe, searched-row counts, row counts, per-generation training metrics, calibration)

**Verdict: the held-out row moved. The policy head learns more than the positions it was shown, given enough of them, and the Step 5 reading "positions, not the rule" was a reading of one dose.** Doubling the searched opponent rows cut the random-playout wipeouts from Step 5's 31 and 38 to 24 and 22; quadrupling them, to 24 and 17; twenty more generations at the Step 5 dose, to 27 and 21. Every new checkpoint sits below every earlier one on that row, where every earlier run had left it between 31 and 38. The prior on Greedy's wiping reply rose with the dose, from 8.5% and 8.2% to 11.1% and 9.9% at 240 and 12.0% and 11.9% at 480, and the plain search's escape rate on the control's fatal positions rose from 44% and 32% to 65% and 67%. The curve is concave: the second doubling of the rows bought less than the first on every measure, and at 480 external games it bought it with the network's own strength, the generation-60 retention row falling from 64.5% and 67.8% to 44.8% on both seeds and the Edax rows with it. The design's claim-side prediction, that the random-playout row would not move at any dose, failed in the direction it said would change the contribution's first section, and that section changes.

## 1. What was run

Six checkpoints. The dose is the number of external games per generation, with the self-play games reduced one for one so that every group trains on about 600 games' worth of rows; the Step 5 runs are the curve's first point and were not re-run.

| run | seed | self-play + external games per generation | training rows in the twenty generations | of which searched opponent rows | wall time | network's score in the external games | wiped out in the external games |
|---|---|---:|---:|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5, generations 121 to 140) | 1337 | 480 + 120 | 685,099 | 58,000 | 2.1 h | 94.2% | 5.3% |
| step5-Bboth-sp-s2 (Step 5) | 1338 | 480 + 120 | 684,688 | 58,000 | 2.1 h | 94.1% | 5.4% |
| **step6-Bsp240-s1** | 1337 | 360 + 240 | 677,199 | 118,000 | 1.55 h | 95.3% | 4.2% |
| **step6-Bsp240-s2** | 1338 | 360 + 240 | 677,644 | 118,273 | 1.59 h | 95.5% | 4.1% |
| **step6-Bsp480-s1** | 1337 | 120 + 480 | 664,015 | 239,039 | 1.65 h | 95.9% | 3.3% |
| **step6-Bsp480-s2** | 1338 | 120 + 480 | 664,062 | 239,185 | 1.58 h | 96.1% | 3.4% |
| **step5-Bboth-sp-s1, generations 141 to 160** | 1337 | 480 + 120 | 687,992 | 60,202 | 1.52 h | 97.1% | 2.5% |
| **step5-Bboth-sp-s2, generations 141 to 160** | 1338 | 480 + 120 | 687,082 | 59,648 | 1.55 h | 96.2% | 3.3% |

The groups are matched on rows to within 3%: the 480-game group has slightly fewer, because an opponent position with a single legal move is not recorded. All six Step 6 jobs ran on the cluster's L40S node with 22 self-play workers; the Step 5 runs ran on the RTX 6000 node, so their wall time is not comparable. On the L40S node, self-play took 219 and 223 seconds per generation at 120 external games (the continuations), 221 and 229 at 240, and 240 and 228 at 480: raising the external share from a fifth to four fifths of the games costs a few percent of self-play time, because each external game replaces a self-play game and costs about as many searches.

**Prediction 1 held in substance, and the dose arrived short.** The searched-row count (`scripts/step5_searched_rows.py`, `research/results/step6/searched-rows-dose.json` and `searched-rows-continued.json`) finds the opponent-to-move rows at which some legal move takes every disc of the network's, and reads the stored policy target there:

| run | searched opponent rows | of which at a fatal position | target mostly on a wiping move | mean target mass on wiping moves |
|---|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5) | 58,000 | 126 | 105 (83%) | 0.74 |
| step5-Bboth-sp-s2 (Step 5) | 58,000 | 130 | 112 (86%) | 0.74 |
| **step6-Bsp240-s1** | 118,000 | 203 | 180 (89%) | 0.78 |
| **step6-Bsp240-s2** | 118,273 | 198 | 170 (86%) | 0.75 |
| **step6-Bsp480-s1** | 239,039 | 318 | 269 (85%) | 0.76 |
| **step6-Bsp480-s2** | 239,185 | 323 | 289 (89%) | 0.79 |
| **step5-Bboth-sp-s1, generations 141 to 160** | 60,202 | 60 | 56 (93%) | 0.82 |
| **step5-Bboth-sp-s2, generations 141 to 160** | 59,648 | 80 | 74 (92%) | 0.83 |

The search found the reply in the same five rows of six as in Step 5, so the rows that arrived carry the lesson. Fewer arrived than the design expected: about 200 and 320 fatal rows per run against the 250 and 500 a proportional count gives, and the network's wipeout rate in the external games, which the design expected to stay near 5 to 7%, fell with the dose, to 4.2% and 4.1% at 240 and 3.3% and 3.4% at 480. The design said that if the rate changed with dose the groups would differ in more than the dose, and to read that first. The per-generation series (`research/results/step6/training.json`) says what changed it. Every run starts generation 121 from the same network and is wiped out in 9 to 11% of its first external games; in every run the rate then falls, to 3.3% and 3.7% over the last five generations at the Step 5 dose, 3.7% and 2.8% at 240, and 1.8% and 2.0% at 480; the 480 group's fatal rows per generation go from 53 and 33 at generation 121 to 10 and 10 at generation 140. The network learns during the run to stop walking into the positions, and learns it faster the more searched rows it gets. The groups differ in their wipeout rate because of the dose, not beside it, and the right x-axis for the curve is the number of fatal rows actually delivered: 126 and 130, then 203 and 198, then 318 and 323. The continued runs add 60 and 80 fatal rows in their second twenty generations, for 186 and 210 in all, which places them beside the 240 group on that axis.

The dose is self-limiting. A searched row at a fatal position exists only when the network walks into one, and the first thing the rows teach it is not to. That is also why the second twenty generations of the Step 5 runs delivered half the rows of the first twenty.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. Every column is the plain search at 50 simulations. The Step 5 columns are the Step 5 numbers, not re-run; the control and Step 3b columns are in the [Step 5 results](step5-results.md), section 2, and in `research/results/step6/scorecard.md` beside these.

| row | opponent | B-both-sp-s1 (Step 5) | B-both-sp-s2 (Step 5) | Bsp240-s1 | Bsp240-s2 | Bsp480-s1 | Bsp480-s2 | Bsp160-s1 | Bsp160-s2 |
|---|---|---|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 11.2% [7.6, 16.4], 8 wo | 8.5% [5.4, 13.2], 7 wo | 5.2% [2.9, 9.3], 5 wo | 13.0% [9.0, 18.4], 5 wo | 4.5% [2.4, 8.3], 9 wo | 6.2% [3.7, 10.5], 11 wo | 8.5% [5.4, 13.2], 6 wo | 14.2% [10.1, 19.8], 6 wo |
| **held out** | Edax L5 | 31.2% [25.2, 38.0], 2 wo | 35.2% [29.0, 42.1], 3 wo | 28.5% [22.7, 35.1], 4 wo | 23.8% [18.4, 30.1], 2 wo | **17.2%** [12.6, 23.1], 3 wo | **21.5%** [16.4, 27.7], 2 wo | 31.0% [25.0, 37.7], 1 wo | 35.0% [28.7, 41.8], 2 wo |
| **held out** | Edax L3 | 71.8% [65.1, 77.5], 1 wo | 70.5% [63.8, 76.4], 1 wo | 65.5% [58.7, 71.7], 1 wo | 69.5% [62.8, 75.5], 1 wo | **53.5%** [46.6, 60.3], 0 wo | **56.2%** [49.3, 62.9], 2 wo | 69.0% [62.3, 75.0], 1 wo | 69.8% [63.1, 75.7], 0 wo |
| **held out** | rollout-r3000 | 84.5% [78.8, 88.9], **31 wo** | 80.5% [74.5, 85.4], **38 wo** | 87.5% [82.2, 91.4], **24 wo** | 88.2% [83.1, 92.0], **22 wo** | 87.5% [82.2, 91.4], **24 wo** | 90.0% [85.1, 93.4], **17 wo** | 85.0% [79.4, 89.3], **27 wo** | 88.5% [83.3, 92.2], **21 wo** |
| retention | gen60 | 64.5% [57.7, 70.8], 0 wo | 67.8% [61.0, 73.8], 0 wo | 63.2% [56.4, 69.6], 0 wo | 61.8% [54.9, 68.2], 0 wo | **44.8%** [38.0, 51.7], 0 wo | **44.8%** [38.0, 51.7], 0 wo | 61.0% [54.1, 67.5], 0 wo | 60.5% [53.6, 67.0], 0 wo |
| seen | Greedy | 72.5% [65.9, 78.2], **54 wo** | 77.5% [71.2, 82.7], **45 wo** | 72.5% [65.9, 78.2], **55 wo** | 78.0% [71.8, 83.2], **44 wo** | 77.5% [71.2, 82.7], **45 wo** | 83.5% [77.7, 88.0], **33 wo** | 76.8% [70.4, 82.1], **46 wo** | 75.5% [69.1, 80.9], **48 wo** |
| seen | Minimax-d4 | 97.5% [94.3, 98.9], 0 wo | 98.0% [95.0, 99.2], 0 wo | 98.5% [95.7, 99.5], 0 wo | 94.8% [90.7, 97.1], 0 wo | 94.2% [90.1, 96.7], 1 wo | 95.8% [92.0, 97.8], 0 wo | 98.0% [95.0, 99.2], 0 wo | 97.0% [93.6, 98.6], 0 wo |
| seen family | greedy-corner | 72.5% [65.9, 78.2], 55 wo | 71.5% [64.9, 77.3], 57 wo | 75.0% [68.6, 80.5], 50 wo | 74.5% [68.0, 80.0], 51 wo | 82.5% [76.6, 87.1], **35 wo** | 87.5% [82.2, 91.4], **25 wo** | 71.5% [64.9, 77.3], 55 wo | 80.0% [73.9, 85.0], 40 wo |
| seen family | greedy-d2 | 70.5% [63.8, 76.4], 58 wo | 68.0% [61.2, 74.1], 63 wo | 76.5% [70.2, 81.8], 47 wo | 75.0% [68.6, 80.5], 50 wo | 79.0% [72.8, 84.1], 42 wo | 75.0% [68.6, 80.5], 50 wo | 75.0% [68.6, 80.5], 50 wo | 71.0% [64.4, 76.8], 58 wo |
| seen family | noisy-greedy | 82.0% [76.1, 86.7], 36 wo | 83.5% [77.7, 88.0], 33 wo | 81.0% [75.0, 85.8], 37 wo | 86.5% [81.1, 90.6], 27 wo | 86.0% [80.5, 90.1], 28 wo | 86.5% [81.1, 90.6], 27 wo | 82.5% [76.6, 87.1], 35 wo | 86.5% [81.1, 90.6], 27 wo |
| seen family | corner | 98.2% [95.3, 99.4], 3 wo | 98.5% [95.7, 99.5], 3 wo | 98.0% [95.0, 99.2], 4 wo | 99.5% [97.2, 99.9], 1 wo | 98.5% [95.7, 99.5], 3 wo | 98.5% [95.7, 99.5], 3 wo | 97.5% [94.3, 98.9], 5 wo | 98.0% [95.0, 99.2], 4 wo |
| seen family | mobility | 92.5% [88.0, 95.4], 14 wo | 96.5% [93.0, 98.3], 7 wo | 95.5% [91.7, 97.6], 9 wo | 96.0% [92.3, 98.0], 8 wo | 97.5% [94.3, 98.9], 5 wo | 96.5% [93.0, 98.3], 6 wo | 97.5% [94.3, 98.9], 5 wo | 94.5% [90.4, 96.9], 10 wo |
| sanity | Random | 99.0% [96.4, 99.7], 2 wo | 100% [98.1, 100], 0 wo | 99.5% [97.2, 99.9], 1 wo | 100% [98.1, 100], 0 wo | 99.0% [96.4, 99.7], 2 wo | 100% [98.1, 100], 0 wo | 100% [98.1, 100], 0 wo | 99.5% [97.2, 99.9], 1 wo |

**The held-out row moved, on every new checkpoint.** The random-playout search, the opponent that finds the same hole with no knowledge of Greedy and on different boards, wiped out the control 34 times, the Step 3b seeds 36 and 37, and the Step 5 seeds 31 and 38. It wipes out the 240 group 24 and 22 times, the 480 group 24 and 17, and the continued runs 27 and 21. Taken one pair at a time, the Step 5 seed 1 to 240 seed 1 change (31 to 24) is inside what 200 games resolve; the seed 2 changes (38 to 22 and 17) are outside it. Pooled, the five earlier checkpoints lose 176 of 1,000 games with no discs and the six new ones 135 of 1,200, 17.6% against 11.2%, which is outside the noise by any reading. The scores rose with it, from 84.5% and 80.5% to 87.5% and 90.0% at 480.

**The Greedy family kept falling, with the second doubling worth more than the first, except for Greedy itself.** Summed over the four disc-hungry opponents, the wipeouts per seed go from 203 and 198 (Step 5) to 189 and 172 (240) to 150 and 135 (480), and the continued runs read 186 and 173. Greedy's own row did not move at 240 (55 and 44 against 54 and 45) and moved at 480 on one seed (45 and 33); greedy-corner went 55 and 57, then 50 and 51, then 35 and 25; greedy-d2 58 and 63, then 47 and 50, then 42 and 50; noisy-greedy 36 and 33, then 37 and 27, then 28 and 27.

**The cost shows at 480 and not at 240.** At 240 the generation-60 row reads 63.2% and 61.8% against Step 5's 64.5% and 67.8%, Edax L3 65.5% and 69.5% against 71.8% and 70.5%, and Edax L7 5.2% and 13.0% against 11.2% and 8.5%: within the ±6 to 7 points a row resolves and the 10 the seeds differ by, with one exception, Edax L5 on seed 2 at 23.8% against 35.2%. At 480, with the network's own games a fifth of the data, every strength row fell beyond that: the generation-60 row to 44.8% on both seeds, 20 points, so the 480 networks lose to a checkpoint 80 generations behind them; Edax L5 to 17.2% and 21.5%, Edax L3 to 53.5% and 56.2%, Edax L7 to 4.5% and 6.2%, and Minimax-d4 to 94.2% and 95.8% from 97.5% and 98.0%. This is the price group D paid in Step 3 for a comparable cut in the Greedy wipeouts, and it arrives at the same place by a different route: D changed what the value head was told, the 480 group changed how much of its data is its own play.

**The continued runs read like the 240 group on the Greedy and rollout rows and paid a little on retention.** At generation 160 the Step 5 runs wipe out against the random-playout search 27 and 21 times, against Greedy 46 and 48, and against the generation-60 checkpoint score 61.0% and 60.5% against 64.5% and 67.8% at generation 140, a fall of 3 and 7 points that is at the edge of the noise and in the same direction on both seeds.

## 3. The predictions, one by one

1. **The dose arrives: fatal rows scale with the external games, the target mostly on the wiping reply in five rows of six, the wipeout rate in the external games near 5 to 7%.** Held in substance: the rows arrived, 203 and 198 and 318 and 323, carrying the target at 85 to 89%. The rate did not stay near 5 to 7%; it fell with the dose, and section 1 reads why: the rows teach the network to stop producing them, so the dose is read as rows delivered, not games played.
2. **The prior rises with the dose on both seeds; 240 passes 10%, 480 passes 14%.** Held in direction, missed both thresholds. On the control's fixed positions the prior reads 8.5% and 8.2% (Step 5), 11.1% and 9.9% (240), 12.0% and 11.9% (480); 240 passes 10% on one seed and misses it by 0.1 on the other, 480 does not reach 14% on either.
3. **The Greedy family keeps falling: Greedy at 240 below Step 5's on both seeds, at 480 below 35 on both; greedy-d2 and greedy-corner follow.** Failed for Greedy's own row, held for the family. Greedy reads 55 and 44 at 240 against 54 and 45, unchanged, and 45 and 33 at 480, one seed below 35. The family sum fell at every dose, from about 200 to about 180 to about 140 per seed, with greedy-corner carrying most of the change at 480.
4. **The random-playout row does not move: within 26 to 39 on every new checkpoint.** Failed downward, which the design named as the outcome that changes the claim. Five of the six new checkpoints are below 26 (24, 22, 24, 17, 21) and the sixth reads 27; the 480 group is below 25 on both seeds, the condition the design wrote for "the rule is learnable from searched positions and Step 5 was short of data". Section 7.
5. **The cost shows at 480, not at 240: the generation-60 row falls beyond seed noise at 480, the Edax rows may follow.** Held, and larger than written: 20 points on the retention row and 11 to 16 on Edax L3 and L5 at 480, nothing beyond noise at 240 except one Edax L5 seed.
6. **Twenty more generations continue the fall by less than half the first twenty's change: prior up, Greedy down, random-playout unchanged.** Failed in both directions. The prior rose by 2.9 and 3.4 points, more than half of the first twenty generations' 4.9 and 4.6; the Greedy row did not fall (46 and 48 against 54 and 45); the random-playout row, predicted unchanged, fell to 27 and 21. The time axis gave the same reading as the dose axis, which section 7 takes up.

## 4. The head probe

The fixed-position reading only: every network on the control's 63 fatal positions against Greedy, so the heads are compared on the same boards (`scripts/probe_heads.py --positions-from control120`, `research/results/step6/head-probe-fixed.json`). P is the network's fatal position, C the position one ply later with Greedy to move; "escapes" is the share of the 63 positions at which the named search, run at P, does not play the fatal move. The control and Step 3b rows are the Step 5 readings.

| network | raw value at P | raw value at C, for Greedy (exact: +1) | prior on Greedy's wiping reply, median | positions where that prior is under 5% | prior on the control's fatal move | prior on all safe moves | plain search escapes, 50 sims | policy-only search escapes, 50 sims | plain search escapes, 256 sims | checked search escapes, 50 sims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control120 | +0.99 | −0.99 | 3.6% | 65% | 36% | 44% | 0% | 5% | 56% | 95% |
| B-both-s1 (Step 3b) | +0.95 | −0.99 | 4.2% | 57% | 29% | 49% | 21% | 22% | 63% | 97% |
| B-both-s2 (Step 3b) | +0.96 | −0.99 | 3.9% | 59% | 27% | 56% | 22% | 24% | 63% | 97% |
| B-both-sp-s1 (Step 5) | +0.97 | −0.97 | 8.5% | 29% | 28% | 56% | 44% | 43% | 92% | 97% |
| B-both-sp-s2 (Step 5) | +0.98 | −0.98 | 8.2% | 35% | 29% | 52% | 32% | 33% | 90% | 97% |
| **Bsp240-s1** | +0.97 | −0.98 | **11.1%** | 17% | 27% | 58% | **60%** | 63% | 97% | 97% |
| **Bsp240-s2** | +0.98 | −0.98 | **9.9%** | 16% | 29% | 57% | **46%** | 48% | 97% | 97% |
| **Bsp480-s1** | +0.97 | −0.99 | **12.0%** | 10% | 27% | 59% | **65%** | 60% | 95% | 97% |
| **Bsp480-s2** | +0.98 | −0.98 | **11.9%** | 6% | 26% | 63% | **67%** | 62% | 98% | 97% |
| **Bsp160-s1** | +0.98 | −0.99 | **11.4%** | 14% | 27% | 57% | **49%** | 48% | 97% | 97% |
| **Bsp160-s2** | +0.98 | −0.98 | **11.6%** | 14% | 26% | 60% | **57%** | 59% | 97% | 97% |

Three things to read off it.

The prior keeps rising and the rise slows: 3.6%, then 8.3% at 130 fatal rows, 10.5% at 200, 12.0% at 320 (seed means). The share of positions where the prior is still under 5%, the positions a 50-simulation search will not reach, goes from 65% to 32% to 17% to 8%: the dose is filling in the tail, and at 480 nearly every one of the control's fatal positions has a prior the search can find. The policy-only search, which has nothing but the prior, escapes as often as the plain search at every dose, so the escapes still come from the prior, as in Step 5.

The value head does not move at any dose: +0.97 to +0.98 at P and −0.97 to −0.99 at C on every network, where the exact value at C is +1 for Greedy. The 480 group's value loss in training fell from 0.47 to 0.25, because four fifths of its games are wins against weak opponents whose results are easy to predict; none of that reached the fatal positions.

At 256 simulations, the plain search escapes 95 to 98% of the positions on every new network, against 90 to 92% for Step 5 and 56% for the control: a prior of 11% is enough for a search five times the evaluation budget to find the finished position almost every time. The checked search at 50 simulations escapes 97% on every network, as it does on the control.

## 5. The training rows

The Step 3b row count on the Step 6 shards (`scripts/step3b_lesson_rows.py`, `research/results/step6/lesson-rows-dose.json` and `lesson-rows-continued.json`), with the opponent's rows identified by the `opponent_to_move` column. The lesson rows are the ones with the opponent to move, the network holding at most three discs from ply 16 on, and the opponent winning.

| run | rows | opponent to move, network ≤3 discs | of which the opponent won (the lesson) | of which the network won | network to move, ≤3 discs | of which it lost | of which it won |
|---|---:|---:|---:|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5) | 685,099 | 248 | 56 | 192 | 4,603 | 204 | 4,399 |
| step5-Bboth-sp-s2 (Step 5) | 684,688 | 230 | 56 | 174 | 4,577 | 231 | 4,345 |
| **step6-Bsp240-s1** | 677,199 | 444 | **77** | 367 | 9,253 | 292 | 8,961 |
| **step6-Bsp240-s2** | 677,644 | 414 | **75** | 339 | 8,911 | 298 | 8,613 |
| **step6-Bsp480-s1** | 664,015 | 753 | **107** | 646 | 18,172 | 470 | 17,701 |
| **step6-Bsp480-s2** | 664,062 | 824 | **108** | 716 | 17,916 | 495 | 17,420 |
| **step5-Bboth-sp-s1, generations 141 to 160** | 687,992 | 179 | **12** | 167 | 4,650 | 43 | 4,607 |
| **step5-Bboth-sp-s2, generations 141 to 160** | 687,082 | 200 | **12** | 188 | 4,547 | 56 | 4,491 |

The lesson rows by this definition scale with the dose as the searched-row count does, 56 to about 76 to about 108, and the look-alikes scale faster: at 480 the window holds 18,000 rows in which the network has at most three discs and wins, against 470 in which it loses, a ratio of nearly forty to one where Step 3b's was fourteen. That is the value head's situation, and section 4 shows it did not move. The policy head's situation is different: its 108 lesson rows carry a target on one specific move, and the 17,000 look-alikes carry targets on other moves at other positions, which do not contradict it. The continued runs show the self-limiting dose from the other side: 12 lesson rows in twenty generations, and 43 and 56 rows of the network losing with few discs where the first twenty generations had 204 and 231.

## 6. Value calibration

As in Step 5. `scripts/value_calibration.py`, `research/results/step6/value-calibration.json`.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against the Greedy family | against Random |
|---|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.89, won 0.95, gap 0.09 | est. 0.95, won 0.93, gap 0.06 | est. 0.96, won 1.00, gap 0.04 |
| B-both-s1 | est. 0.52, won 0.52, gap 0.02 | est. 0.58, won 0.49, gap 0.09 | est. 0.58, won 0.62, gap 0.06 | est. 0.89, won 0.94, gap 0.07 | est. 0.94, won 0.93, gap 0.06 | est. 0.95, won 1.00, gap 0.05 |
| B-both-s2 | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.50, gap 0.08 | est. 0.59, won 0.62, gap 0.06 | est. 0.90, won 0.94, gap 0.08 | est. 0.94, won 0.94, gap 0.06 | est. 0.95, won 1.00, gap 0.04 |
| B-both-sp-s1 (Step 5) | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.51, gap 0.08 | est. 0.58, won 0.65, gap 0.07 | est. 0.90, won 0.95, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| B-both-sp-s2 (Step 5) | est. 0.52, won 0.52, gap 0.02 | est. 0.59, won 0.51, gap 0.09 | est. 0.56, won 0.68, gap 0.12 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.95, won 1.00, gap 0.05 |
| **Bsp240-s1** | est. 0.52, won 0.52, gap 0.01 | est. 0.59, won 0.48, gap 0.11 | est. 0.56, won 0.63, gap 0.10 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.96, gap 0.05 | est. 0.95, won 1.00, gap 0.04 |
| **Bsp240-s2** | est. 0.52, won 0.52, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.56, won 0.62, gap 0.09 | est. 0.91, won 0.95, gap 0.07 | est. 0.95, won 0.97, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **Bsp480-s1** | est. 0.54, won 0.53, gap 0.01 | est. 0.57, won 0.42, gap 0.16 | est. 0.53, won 0.45, gap 0.14 | est. 0.92, won 0.94, gap 0.05 | est. 0.96, won 0.97, gap 0.04 | est. 0.96, won 1.00, gap 0.04 |
| **Bsp480-s2** | est. 0.54, won 0.54, gap 0.01 | est. 0.58, won 0.45, gap 0.14 | est. 0.52, won 0.45, gap 0.13 | est. 0.93, won 0.96, gap 0.06 | est. 0.96, won 0.97, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **Bsp160-s1** | est. 0.51, won 0.51, gap 0.01 | est. 0.60, won 0.50, gap 0.10 | est. 0.58, won 0.61, gap 0.06 | est. 0.90, won 0.96, gap 0.09 | est. 0.95, won 0.96, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **Bsp160-s2** | est. 0.52, won 0.52, gap 0.02 | est. 0.60, won 0.53, gap 0.07 | est. 0.57, won 0.61, gap 0.07 | est. 0.91, won 0.95, gap 0.08 | est. 0.96, won 0.96, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |

Unchanged at 240 and on the continued runs; at 480 the value head overestimates. The 480 networks rate their positions against the held-out opponents at 57% and 58% and win 42% and 45%, a gap of 0.16 and 0.14 where every other network's is 0.07 to 0.11, and the same against the generation-60 checkpoint, 0.14 and 0.13 against 0.06 to 0.12. That is the retention and Edax losses of section 2 read from the value head's side: it rates those positions as the Step 5 networks did and no longer wins them. The self-play column is where every network sits, and the Greedy-family columns are the same at every dose, so the searched rows did not touch the calibration; the displaced self-play did.

## 7. What this establishes

- **The policy head generalises past the positions it was shown, once it has seen enough of them.** Step 5's one dose moved the Greedy family and not the held-out row, and the results read that as "positions, not the rule". With 200 to 320 fatal rows instead of 130, the held-out row moved on every checkpoint, by about a third, and the six new checkpoints pooled lose with no discs in 11% of their games against the random-playout search where the five earlier ones lose in 18%. The same hole on different boards is reached by the same prior. The Step 5 reading is withdrawn as a ceiling and kept as a description of what 130 rows buy: the family first, the rule later.
- **The curve is concave on every measure, and the dose is self-limiting.** The first doubling of the rows took the prior from 8.3% to 10.5% and the second to 12.0%; the plain search's escape rate from 38% to 53% to 66%; the random-playout wipeouts from 35 to 23 to 21. Each row teaches the network, among other things, not to produce the next one: the wipeout rate in the external games falls within every run, and the 480 group ends its twenty generations producing ten fatal rows a generation where it began with forty. Rows delivered, not games played, is the dose, and it cannot be raised in proportion to the games.
- **The held-out row moves less than the family and later.** At the dose where the family's wipeouts had fallen by a third (Step 5), the held-out row had not moved; at the dose where the family has fallen by another third, the held-out row has fallen by its first third. A prior learned from positions reaches the positions nearest the ones seen first and the farther ones as the tail fills in; the fixed-position probe's "under 5%" column is that tail, and it closes from 65% to 8% across the curve.
- **The price arrives at 480 and it is the network's own strength.** Twenty points on the retention row and 11 to 16 on the Edax rows, with no cost at 240. The mechanism is not the searched rows but what they displaced: at 480 external games the network's own self-play is a fifth of its training data, and the value head's loss fell by half because most of what it now predicts is the outcome of games against weak opponents. The useful point on the curve is 240: most of the held-out gain, nearly all of the prior gain, and nothing measurable paid.
- **Twenty more generations at the Step 5 dose read like the 240 group.** The continued runs delivered 186 and 210 fatal rows in all, about what the 240 group delivered in twenty, and read like it on the prior (11.4% and 11.6%), the held-out row (27 and 21) and the Greedy family (186 and 173), with a 3 to 7 point fall on the retention row that the 240 group does not show. Along either axis, the quantity that moved the held-out row is the count of searched fatal rows the network has seen.
- **The exact check still does what no dose did.** At 480 the plain search leaves 17 and 24 wipeouts on the held-out row and 33 and 45 against Greedy, having paid a fifth of the network's strength; the check at 50 simulations on the unretrained control leaves 12 and 12 and pays nothing. The learned route reaches the rule partway and at a price that rises with the dose; the calculated route reaches it at once.

**What it does not establish.** Two seeds per point, 200 games per row, and the per-pair differences on the held-out row are inside the noise for the smaller changes: the result rests on the direction being the same on all six new checkpoints and on the pooled contrast. It does not say where the curve levels off, because the self-limiting dose means no run of this recipe can deliver many more than 320 fatal rows in twenty generations; a weight on the searched rows in the policy loss would raise the signal per row without changing the games and is the next thing to try. It does not separate the time axis from the dose axis cleanly, since the continued runs saw twenty more generations of everything, not only of the searched rows. The held-out row is one opponent, and "the rule" is read from one row and one probe; a second knowledge-free opponent is still the test of the word.

## 8. What follows

In order of cost.

1. **A weight on the searched opponent rows in the policy loss**, two and four, at the Step 5 dose. It is the only way to raise the signal per fatal row without raising the external share, and the 480 group shows the external share is what costs. It asks whether the curve continues past 12% and 21 when the rows are not the limit.
2. **A second knowledge-free opponent**, so that the transfer result rests on more than one row.
3. **The search-budget dose curve with and without the check** on the 240 weights as well as the control: at a prior of 11% the plain search at 256 simulations escapes 97% of the probed positions, and the curve says at what budget the learned prior and the exact check meet.

## 9. Reproduction

As for Step 5 (its section 9), with the Step 6 configs, run names `step6-Bsp240-s1`, `-s2`, `step6-Bsp480-s1`, `-s2`, the continuations `step5-Bboth-sp-s1`, `-s2` trained to generation 160, evaluation names `Bsp240-s1`, `Bsp240-s2`, `Bsp480-s1`, `Bsp480-s2`, `Bsp160-s1`, `Bsp160-s2`, and the probe on `step6-checkpoints.json`, which is `step5-checkpoints.json` with the six new entries:

```bash
python scripts/step3_scorecard.py ~/reversi-runs/step3-eval \
    --checkpoints control120 Bboth-s1 Bboth-s2 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bsp160-s1 Bsp160-s2 \
    --out research/results/step6/scorecard.json --markdown
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol ~/reversi-runs/step3-eval/step6-checkpoints.json \
    --positions-from control120 \
    --networks control120 Bboth-s1 Bboth-s2 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bsp160-s1 Bsp160-s2 \
    --out research/results/step6/head-probe-fixed.json
python scripts/step5_searched_rows.py ~/reversi-runs/step6-Bsp240-s1 ~/reversi-runs/step6-Bsp240-s2 \
    ~/reversi-runs/step6-Bsp480-s1 ~/reversi-runs/step6-Bsp480-s2 \
    --from-generation 121 --out research/results/step6/searched-rows-dose.json
python scripts/step5_searched_rows.py ~/reversi-runs/step5-Bboth-sp-s1 ~/reversi-runs/step5-Bboth-sp-s2 \
    --from-generation 141 --out research/results/step6/searched-rows-continued.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step6-Bsp240-s1 ~/reversi-runs/step6-Bsp240-s2 \
    ~/reversi-runs/step6-Bsp480-s1 ~/reversi-runs/step6-Bsp480-s2 \
    --from-generation 121 --out research/results/step6/lesson-rows-dose.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step5-Bboth-sp-s1 ~/reversi-runs/step5-Bboth-sp-s2 \
    --from-generation 141 --out research/results/step6/lesson-rows-continued.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval \
    --checkpoints control120 Bboth-s1 Bboth-s2 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bsp160-s1 Bsp160-s2 \
    --self-play Bsp240-s1=~/reversi-runs/step6-Bsp240-s1 --self-play Bsp240-s2=~/reversi-runs/step6-Bsp240-s2 \
    --self-play Bsp480-s1=~/reversi-runs/step6-Bsp480-s1 --self-play Bsp480-s2=~/reversi-runs/step6-Bsp480-s2 \
    --self-play Bsp160-s1=~/reversi-runs/step5-Bboth-sp-s1 --self-play Bsp160-s2=~/reversi-runs/step5-Bboth-sp-s2 \
    --out research/results/step6/value-calibration.json
```

The six training jobs went through `slurm/train.sbatch` with `CONFIG`, `RUN_ID` and `GENERATIONS` (140 for the dose groups, 160 for the continuations) and `EXTRA_ARGS="--set seed=1338"` for the second seeds; the 78 evaluation pairings through `slurm/cpu.sbatch`, one core each, as for Step 5.
