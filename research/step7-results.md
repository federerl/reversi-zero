# Step 7 — Results: A Weight on the Searched Opponent Rows

**Applies to:** the [Step 7 design](step7-design.md); every number below is read against the predictions in its section 5
**Runs:** `step7-Bsp-w2-s1`, `-s2` (`configs/full8x8_step7_weight2.yaml`) and `step7-Bsp-w4-s1`, `-s2` (`configs/full8x8_step7_weight4.yaml`), twenty generations from `control120`; compared with the Step 5 and Step 6 seeds, not re-run
**Date:** November 8, 2026
**Data:** `research/results/step7/` (scorecard, fixed-position head probe, searched-row counts, row counts, per-generation training metrics, calibration)

**Verdict: the dose curve was the policy signal. Counting the searched opponent rows two and four times in the policy loss, with the games exactly Step 5's, reproduced the Step 6 curve point for point, and paid nothing.** At weight 2 the random-playout row reads 25 and 27 wipeouts where the 240-game group read 24 and 22 and Step 5 read 31 and 38; the Greedy family's wipeouts summed over four opponents read 172 and 164 where the 240 group read 189 and 172; the share of the control's fatal positions at which the prior on the wiping reply is still under 5% reads 22% and 21% where the 240 group read 17% and 16%. At weight 4 the family reads 153 and 160 and the under-5% share 11% and 6%, which are the 480 group's readings, 150 and 135 and 10% and 6%, with the network's own play still four fifths of its data: the generation-60 row reads 59% and 58% instead of the 480 group's 45%, the Edax rows do not move, and the calibration gap stays where Step 5 left it. The quantity that lines the two routes up is the fatal rows delivered times the weight on them: 130, 220 to 230, 200, 340, 320. The held-out row moved at both weights to where the 240 group sits and, on either axis, no further.

## 1. What was run

Four checkpoints. The games, opponents, searches, value targets and replay window are Step 5's; the one change is `train.opponent_policy_weight`, which multiplies the policy loss of every row recorded with the opponent to move by 2 or 4 and divides the batch's policy loss by the sum of the weights instead of the row count. The Step 5 runs are the weight-1 point and were not re-run.

| run | seed | weight | self-play + external games per generation | training rows in the twenty generations | of which searched opponent rows | wall time | network's score in the external games | wiped out in the external games |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5) | 1337 | 1 | 480 + 120 | 685,099 | 58,586 | 2.1 h | 94.2% | 5.3% |
| step5-Bboth-sp-s2 (Step 5) | 1338 | 1 | 480 + 120 | 684,688 | 58,341 | 2.1 h | 94.1% | 5.4% |
| **step7-Bsp-w2-s1** | 1337 | 2 | 480 + 120 | 685,379 | 58,760 | 1.51 h | 95.1% | 4.6% |
| **step7-Bsp-w2-s2** | 1338 | 2 | 480 + 120 | 685,446 | 58,691 | 1.52 h | 94.8% | 4.8% |
| **step7-Bsp-w4-s1** | 1337 | 4 | 480 + 120 | 686,329 | 59,096 | 1.52 h | 96.0% | 3.6% |
| **step7-Bsp-w4-s2** | 1338 | 4 | 480 + 120 | 686,638 | 58,875 | 1.53 h | 96.3% | 3.5% |

The groups are matched on games exactly and on rows to within 0.3%. The four jobs ran on the cluster's L40S node with 22 self-play workers, where self-play took 216 to 219 seconds per generation, the same as the Step 5 continuations on that node (219 and 223): the weight is applied in the trainer and costs nothing at play time. The Step 5 runs ran on the RTX 6000 node, so their wall time is not comparable. The value loss at generation 140 is 0.567 and 0.555 at weight 2 and 0.569 and 0.551 at weight 4, against 0.570 and 0.553 in Step 5: the value head was trained on the same rows with the same targets and reached the same place.

**Prediction 1 held in substance, and the weight did what the dose did to the games.** The searched-row count (`scripts/step5_searched_rows.py`, `research/results/step7/searched-rows.json`) finds the opponent-to-move rows at which some legal move takes every disc of the network's, and reads the stored policy target there:

| run | searched opponent rows | of which at a fatal position | target mostly on a wiping move | mean target mass on wiping moves |
|---|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5) | 58,586 | 126 | 105 (83%) | 0.74 |
| step5-Bboth-sp-s2 (Step 5) | 58,341 | 130 | 112 (86%) | 0.74 |
| **step7-Bsp-w2-s1** | 58,760 | 110 | 99 (90%) | 0.79 |
| **step7-Bsp-w2-s2** | 58,691 | 116 | 97 (84%) | 0.73 |
| **step7-Bsp-w4-s1** | 59,096 | 86 | 79 (92%) | 0.82 |
| **step7-Bsp-w4-s2** | 58,875 | 84 | 71 (85%) | 0.76 |

The search found the reply in the same five rows of six, so the rows that arrived carry the lesson. The weight-2 runs delivered 110 and 116 fatal rows, inside the design's 90 to 140; the weight-4 runs delivered 86 and 84, just under it. The per-generation series (`research/results/step7/training.json`) says why. Every run starts generation 121 from the same network and the weight applies from that generation's training on. The Step 5 runs produced 49 and 48 fatal rows in their first five generations and 20 and 22 in their last five; the weight-2 runs 40 and 40, then 12 and 15; the weight-4 runs 42 and 41, then 8 and 7. The wipeout rate in the external games falls within every run, from 6.7 to 7.0% over the first five generations to 2.0% and 2.5% at weight 2 and 1.3% and 1.2% at weight 4, below the 480 group's 1.8% and 2.0%. The design expected fewer rows if the weight taught the avoidance faster, and it did: each weighted row teaches the network not to produce the next one more firmly than an unweighted row does, so the weight, like the dose, limits the rows it is applied to. The design's window of 2 to 4% for the final rate was written from the Step 5 and 6 runs and the weight-4 runs end below it; the games themselves are Step 5's in every other respect, the score in them 95 to 96% as in Step 6.

The number that lines the Step 6 and Step 7 groups up is the fatal rows delivered times the weight on them, which is how many times a fatal row is counted in the policy loss over the run: 126 and 130 in Step 5; 220 and 232 at weight 2 against 203 and 198 at 240 games; 344 and 336 at weight 4 against 318 and 323 at 480 games. The rest of this document reads the two routes against each other on that axis.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. Every column is the plain search at 50 simulations. The Step 5 and Step 6 columns are those steps' numbers, not re-run; the 480 columns and the control are in the [Step 6 results](step6-results.md), section 2, and in `research/results/step7/scorecard.md` beside these.

| row | opponent | B-both-sp-s1 (Step 5) | B-both-sp-s2 (Step 5) | Bsp240-s1 (Step 6) | Bsp240-s2 (Step 6) | Bw2-s1 | Bw2-s2 | Bw4-s1 | Bw4-s2 |
|---|---|---|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 11.2% [7.6, 16.4], 8 wo | 8.5% [5.4, 13.2], 7 wo | 5.2% [2.9, 9.3], 5 wo | 13.0% [9.0, 18.4], 5 wo | 8.0% [5.0, 12.6], 7 wo | 13.0% [9.0, 18.4], 7 wo | 12.0% [8.2, 17.2], 8 wo | 12.8% [8.8, 18.1], 7 wo |
| **held out** | Edax L5 | 31.2% [25.2, 38.0], 2 wo | 35.2% [29.0, 42.1], 3 wo | 28.5% [22.7, 35.1], 4 wo | 23.8% [18.4, 30.1], 2 wo | 24.2% [18.8, 30.6], 1 wo | 25.2% [19.7, 31.7], 3 wo | 33.5% [27.3, 40.3], 2 wo | 33.0% [26.9, 39.8], 3 wo |
| **held out** | Edax L3 | 71.8% [65.1, 77.5], 1 wo | 70.5% [63.8, 76.4], 1 wo | 65.5% [58.7, 71.7], 1 wo | 69.5% [62.8, 75.5], 1 wo | 66.5% [59.7, 72.7], 1 wo | 77.5% [71.2, 82.7], 0 wo | 73.0% [66.5, 78.7], 0 wo | 68.0% [61.2, 74.1], 0 wo |
| **held out** | rollout-r3000 | 84.5% [78.8, 88.9], **31 wo** | 80.5% [74.5, 85.4], **38 wo** | 87.5% [82.2, 91.4], **24 wo** | 88.2% [83.1, 92.0], **22 wo** | 87.5% [82.2, 91.4], **25 wo** | 85.2% [79.7, 89.5], **27 wo** | 87.0% [81.6, 91.0], **26 wo** | 87.8% [82.5, 91.6], **22 wo** |
| retention | gen60 | 64.5% [57.7, 70.8], 0 wo | 67.8% [61.0, 73.8], 0 wo | 63.2% [56.4, 69.6], 0 wo | 61.8% [54.9, 68.2], 0 wo | 63.7% [56.9, 70.1], 0 wo | 61.0% [54.1, 67.5], 0 wo | 59.2% [52.3, 65.8], 0 wo | 58.0% [51.1, 64.6], 0 wo |
| seen | Greedy | 72.5% [65.9, 78.2], **54 wo** | 77.5% [71.2, 82.7], **45 wo** | 72.5% [65.9, 78.2], **55 wo** | 78.0% [71.8, 83.2], **44 wo** | 74.5% [68.0, 80.0], **51 wo** | 78.5% [72.3, 83.6], **43 wo** | 79.5% [73.4, 84.5], **41 wo** | 78.0% [71.8, 83.2], **44 wo** |
| seen | Minimax-d4 | 97.5% [94.3, 98.9], 0 wo | 98.0% [95.0, 99.2], 0 wo | 98.5% [95.7, 99.5], 0 wo | 94.8% [90.7, 97.1], 0 wo | 95.0% [91.0, 97.3], 0 wo | 98.5% [95.7, 99.5], 0 wo | 97.2% [93.9, 98.8], 0 wo | 97.0% [93.6, 98.6], 0 wo |
| seen family | greedy-corner | 72.5% [65.9, 78.2], 55 wo | 71.5% [64.9, 77.3], 57 wo | 75.0% [68.6, 80.5], 50 wo | 74.5% [68.0, 80.0], 51 wo | 79.5% [73.4, 84.5], **41 wo** | 82.5% [76.6, 87.1], **35 wo** | 78.5% [72.3, 83.6], **43 wo** | 78.2% [72.0, 83.4], **43 wo** |
| seen family | greedy-d2 | 70.5% [63.8, 76.4], 58 wo | 68.0% [61.2, 74.1], 63 wo | 76.5% [70.2, 81.8], 47 wo | 75.0% [68.6, 80.5], 50 wo | 72.0% [65.4, 77.8], 56 wo | 71.0% [64.4, 76.8], 58 wo | 78.5% [72.3, 83.6], 43 wo | 75.5% [69.1, 80.9], 49 wo |
| seen family | noisy-greedy | 82.0% [76.1, 86.7], 36 wo | 83.5% [77.7, 88.0], 33 wo | 81.0% [75.0, 85.8], 37 wo | 86.5% [81.1, 90.6], 27 wo | 88.0% [82.8, 91.8], 24 wo | 86.0% [80.5, 90.1], 28 wo | 87.0% [81.6, 91.0], 26 wo | 88.0% [82.8, 91.8], 24 wo |
| seen family | corner | 98.2% [95.3, 99.4], 3 wo | 98.5% [95.7, 99.5], 3 wo | 98.0% [95.0, 99.2], 4 wo | 99.5% [97.2, 99.9], 1 wo | 98.5% [95.7, 99.5], 2 wo | 99.0% [96.4, 99.7], 2 wo | 97.5% [94.3, 98.9], 5 wo | 98.5% [95.7, 99.5], 3 wo |
| seen family | mobility | 92.5% [88.0, 95.4], 14 wo | 96.5% [93.0, 98.3], 7 wo | 95.5% [91.7, 97.6], 9 wo | 96.0% [92.3, 98.0], 8 wo | 97.0% [93.6, 98.6], 6 wo | 96.0% [92.3, 98.0], 8 wo | 93.5% [89.2, 96.2], 13 wo | 97.0% [93.6, 98.6], 6 wo |
| sanity | Random | 99.0% [96.4, 99.7], 2 wo | 100% [98.1, 100], 0 wo | 99.5% [97.2, 99.9], 1 wo | 100% [98.1, 100], 0 wo | 99.5% [97.2, 99.9], 1 wo | 99.5% [97.2, 99.9], 1 wo | 100% [98.1, 100], 0 wo | 100% [98.1, 100], 0 wo |

**The held-out row moved to where the dose had moved it, in the pooled series.** The random-playout search wiped out the control 34 times, the Step 3b seeds 36 and 37 and the Step 5 seeds 31 and 38; it wipes out the weight-2 networks 25 and 27 times and the weight-4 networks 26 and 22, where the 240 group read 24 and 22 and the 480 group 24 and 17. Every one of the four new checkpoints sits at or below 27. The checkpoints trained without searched rows, the Step 3 continued control among them, spread from 21 to 39 on this row (Step 6 results, section 4), so the per-pair readings overlap and the row is read pooled. The four new checkpoints lose 100 of 800 games with no discs, 12.5%; the nine checkpoints trained without searched rows lose 286 of 1,800, 15.9%, the Step 5 pair 69 of 400, 17.3%, and the Step 6 groups 135 of 1,200, 11.2%, beside them. The second doubling bought nothing on this row that the resolution can see, on either axis: weight 2 to weight 4 goes from 52 to 48 wipeouts in 400 games, 240 to 480 games from 46 to 41.

**The Greedy family fell as it fell with the dose, and at weight 4 as far as at 480.** Summed over the four disc-hungry opponents, the wipeouts per seed read 172 and 164 at weight 2 against the 240 group's 189 and 172, and 153 and 160 at weight 4 against the 480 group's 150 and 135; Step 5 read 203 and 198. Greedy's own row reads 51 and 43 at weight 2, unchanged from Step 5's 54 and 45 as it was at 240, and 41 and 44 at weight 4; greedy-corner went 55 and 57 to 41 and 35 and 43 and 43; noisy-greedy 36 and 33 to 24 and 28 and 26 and 24; greedy-d2 58 and 63 to 56 and 58 and 43 and 49.

**Nothing the 480 group paid was paid here.** At weight 4 the generation-60 row reads 59.2% and 58.0%, Edax L5 33.5% and 33.0%, Edax L3 73.0% and 68.0%, Edax L7 12.0% and 12.8% and Minimax-d4 97.2% and 97.0%; the 480 group, at the same place on the family and the prior, read 44.8% on both seeds, 17.2% and 21.5%, 53.5% and 56.2%, 4.5% and 6.2%, and 94.2% and 95.8%. The Edax rows at weight 4 are within seed noise of Step 5's on every line, and above the 240 group's on L5. The generation-60 row at weight 4 is 5 and 10 points under Step 5's 64.5% and 67.8%, and 3 and 4 under the control's 62.5%; the same direction on both seeds and inside the ±6 to 7 points a row resolves on one of them, which is what the Step 5 continuations showed at generation 160 (61.0% and 60.5%). Section 6 reads the value head on those games and finds it calibrated. At weight 2 the Edax L5 row reads 24.2% and 25.2% against 31.2% and 35.2%, 7 and 10 points, which the weight-4 group does not show and which reads as the seed noise the Step 6 results saw on the same row at 240 (23.8% on one seed).

## 3. The predictions, one by one

1. **The games are Step 5's: the external wipeout rate starts near 9 to 11% and ends between 2 and 4%, the fatal rows number 90 to 140 per run with the target on the wiping reply in five rows of six.** Held in substance. The games are Step 5's in kind and count, the rows carry the target at 84 to 92%, and the weight-2 runs delivered 110 and 116 fatal rows. The weight-4 runs delivered 86 and 84 and ended their external games at a 1.2 to 1.3% wipeout rate, below the window, because the weighted rows teach the avoidance faster, which the design allowed for. The groups differ from Step 5 in the rows they delivered because of the weight, not beside it, and section 1 reads the two routes on the axis that accounts for it.
2. **The prior follows the policy-loss share: weight 2 above Step 5 on both seeds and at or above 10%, weight 4 above weight 2 on both seeds and at or above 12%.** Held for weight 2 in direction, half held on the thresholds; not held for weight 4 over weight 2. On the control's fixed positions the median prior reads 9.8% and 11.1% at weight 2 against Step 5's 8.5% and 8.2%, one seed past 10%; at weight 4 it reads 11.5% and 10.6%, above weight 2 on one seed and under 12% on both. The median is a noisy reading of the prior; the share of positions where the prior is still under 5%, which is what a 50-simulation search can and cannot reach, moved in order on every seed: 29% and 35%, then 22% and 21%, then 11% and 6%, and those are the 240 and 480 groups' readings (17% and 16%, 10% and 6%). Section 4.
3. **The held-out row follows the policy-loss share: random-playout wipeouts under 30 at weight 2 on both seeds, under 25 at weight 4 on both.** Held at weight 2 (25 and 27); at weight 4 one seed reads 26, one point over the threshold, and the other 22. The claim-side reading the design wrote for this outcome stands: the dose worked through the policy signal at the opponent's positions, and the signal can be raised without the games. The second half of the prediction, that weight 4 would read below weight 2 as the 480 group read below the 240 group, is not supported: 26 and 22 against 25 and 27, and 24 and 17 against 24 and 22 in Step 6, are the same reading within the resolution. What the design offered as the alternative, the row staying within Step 5's 26 to 39 while the prior rose, did not happen: all four checkpoints are at or below 27 and the pooled contrast is outside the noise.
4. **No cost at either weight: the generation-60 row, the Edax rows and the calibration gap within seed noise of Step 5.** Held. The Edax rows at both weights and the generation-60 row at weight 2 are within noise; the calibration gap on the held-out and retention games is 0.04 to 0.12 on every new network where the 480 group read 0.13 to 0.16 (section 6). The generation-60 row at weight 4 is the one reading to watch, 59.2% and 58.0%, inside noise on one seed and at its edge on the other, in the same direction on both; it is 14 points above the 480 group's. The explanation the prediction was written to test, that the policy head pays at its own positions for what it carries at the opponent's, is not supported at a policy-loss share of 27%; the 480 cost was the displaced self-play.
5. **The Greedy family falls with the weight: at or below Step 5's 203 and 198 at weight 2, at or below the 240 group's 189 and 172 at weight 4.** Held: 172 and 164, then 153 and 160.
6. **The value head does not move: above +0.9 at the network's fatal position and below −0.9 one ply later, at both weights.** Held: +0.98 and +0.97 at P and −0.99 and −0.98 at C at weight 2, +0.98 and +0.98 and −0.98 and −0.99 at weight 4.

## 4. The head probe

The fixed-position reading: every network on the control's 63 fatal positions against Greedy, so the heads are compared on the same boards (`scripts/probe_heads.py --positions-from control120`, `research/results/step7/head-probe-fixed.json`). P is the network's fatal position, C the position one ply later with Greedy to move; "escapes" is the share of the 63 positions at which the named search, run at P, does not play the fatal move. The control, Step 5 and Step 6 rows are the Step 6 readings.

| network | raw value at P | raw value at C, for Greedy (exact: +1) | prior on Greedy's wiping reply, median | positions where that prior is under 5% | prior on the control's fatal move | prior on all safe moves | plain search escapes, 50 sims | policy-only search escapes, 50 sims | plain search escapes, 256 sims | checked search escapes, 50 sims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control120 | +0.99 | −0.99 | 3.6% | 65% | 36% | 44% | 0% | 5% | 56% | 95% |
| B-both-sp-s1 (Step 5) | +0.97 | −0.97 | 8.5% | 29% | 28% | 56% | 44% | 43% | 92% | 97% |
| B-both-sp-s2 (Step 5) | +0.98 | −0.98 | 8.2% | 35% | 29% | 52% | 32% | 33% | 90% | 97% |
| Bsp240-s1 (Step 6) | +0.97 | −0.98 | 11.1% | 17% | 27% | 58% | 60% | 63% | 97% | 97% |
| Bsp240-s2 (Step 6) | +0.98 | −0.98 | 9.9% | 16% | 29% | 57% | 46% | 48% | 97% | 97% |
| Bsp480-s1 (Step 6) | +0.97 | −0.99 | 12.0% | 10% | 27% | 59% | 65% | 60% | 95% | 97% |
| Bsp480-s2 (Step 6) | +0.98 | −0.98 | 11.9% | 6% | 26% | 63% | 67% | 62% | 98% | 97% |
| **Bw2-s1** | +0.98 | −0.99 | **9.8%** | 22% | 31% | 58% | **51%** | 51% | 97% | 97% |
| **Bw2-s2** | +0.97 | −0.98 | **11.1%** | 21% | 28% | 54% | **56%** | 56% | 97% | 97% |
| **Bw4-s1** | +0.98 | −0.98 | **11.5%** | 11% | 28% | 58% | **57%** | 54% | 98% | 97% |
| **Bw4-s2** | +0.98 | −0.99 | **10.6%** | 6% | 30% | 55% | **51%** | 56% | 97% | 97% |

Three things to read off it.

The tail closes with the weight as it closed with the dose, and in the same amounts. The share of positions where the prior is under 5% goes 32% (Step 5, seed mean), 21% at weight 2, 9% at weight 4; along the dose axis it went 32%, 17%, 8%. The median prior goes 8.3%, 10.4%, 11.1% here and 8.3%, 10.5%, 12.0% there. The plain search's escape rate at 50 simulations goes 38%, 53%, 54% here and 38%, 53%, 66% there: the one place the weight-4 reading sits under the 480 reading, with the two weight-4 seeds at 57% and 51% against 65% and 67%. The policy-only search escapes as often as the plain search on every network, so the escapes still come from the prior.

The value head does not move: +0.97 to +0.98 at P and −0.98 to −0.99 at C on every network, where the exact value at C is +1 for Greedy. The weight touches the policy term only, and the value head's readings at the fatal positions are the Step 5 readings, as its loss in training is.

At 256 simulations the plain search escapes 97 to 98% of the positions on every new network, as on every Step 6 network; the checked search at 50 escapes 97%, as it does on the control.

## 5. The training rows

The Step 3b row count on the Step 7 shards (`scripts/step3b_lesson_rows.py`, `research/results/step7/lesson-rows.json`), with the opponent's rows identified by the `opponent_to_move` column. The lesson rows are the ones with the opponent to move, the network holding at most three discs from ply 16 on, and the opponent winning.

| run | rows | opponent to move, network ≤3 discs | of which the opponent won (the lesson) | of which the network won | network to move, ≤3 discs | of which it lost | of which it won |
|---|---:|---:|---:|---:|---:|---:|---:|
| step5-Bboth-sp-s1 (Step 5) | 685,099 | 248 | 56 | 192 | 4,603 | 204 | 4,399 |
| step5-Bboth-sp-s2 (Step 5) | 684,688 | 230 | 56 | 174 | 4,577 | 231 | 4,345 |
| **step7-Bsp-w2-s1** | 685,379 | 233 | **46** | 187 | 4,509 | 168 | 4,341 |
| **step7-Bsp-w2-s2** | 685,446 | 228 | **44** | 184 | 4,634 | 213 | 4,421 |
| **step7-Bsp-w4-s1** | 686,329 | 235 | **32** | 203 | 4,716 | 114 | 4,602 |
| **step7-Bsp-w4-s2** | 686,638 | 252 | **33** | 219 | 4,765 | 147 | 4,615 |

The lesson rows by this definition fall with the weight, 56 to 45 to 32 per seed, as the fatal searched rows do and for the same reason: the network walks into fewer of the positions. The look-alikes do not fall, since the network still holds few discs and wins in 4,300 to 4,600 rows per run. That is the value head's situation, with a ratio of lesson to look-alike rows that the weight makes worse, from about 80 to one to about 140 to one, and section 4 shows the value head where it was. The policy head's situation is the other way: 85 fatal rows counted four times over is the signal of 340, and the look-alikes carry targets on other moves at other positions, which do not contradict it.

## 6. Value calibration

As in Steps 5 and 6. `scripts/value_calibration.py`, `research/results/step7/value-calibration.json`.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against the Greedy family | against Random |
|---|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.89, won 0.95, gap 0.09 | est. 0.95, won 0.93, gap 0.06 | est. 0.96, won 1.00, gap 0.04 |
| B-both-sp-s1 (Step 5) | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.51, gap 0.08 | est. 0.58, won 0.65, gap 0.07 | est. 0.90, won 0.95, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| B-both-sp-s2 (Step 5) | est. 0.52, won 0.52, gap 0.02 | est. 0.59, won 0.51, gap 0.09 | est. 0.56, won 0.68, gap 0.12 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.95, gap 0.05 | est. 0.95, won 1.00, gap 0.05 |
| Bsp240-s1 (Step 6) | est. 0.52, won 0.52, gap 0.01 | est. 0.59, won 0.48, gap 0.11 | est. 0.56, won 0.63, gap 0.10 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.96, gap 0.05 | est. 0.95, won 1.00, gap 0.04 |
| Bsp240-s2 (Step 6) | est. 0.52, won 0.52, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.56, won 0.62, gap 0.09 | est. 0.91, won 0.95, gap 0.07 | est. 0.95, won 0.97, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| Bsp480-s1 (Step 6) | est. 0.54, won 0.53, gap 0.01 | est. 0.57, won 0.42, gap 0.16 | est. 0.53, won 0.45, gap 0.14 | est. 0.92, won 0.94, gap 0.05 | est. 0.96, won 0.97, gap 0.04 | est. 0.96, won 1.00, gap 0.04 |
| Bsp480-s2 (Step 6) | est. 0.54, won 0.54, gap 0.01 | est. 0.58, won 0.45, gap 0.14 | est. 0.52, won 0.45, gap 0.13 | est. 0.93, won 0.96, gap 0.06 | est. 0.96, won 0.97, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **Bw2-s1** | est. 0.52, won 0.52, gap 0.01 | est. 0.60, won 0.48, gap 0.12 | est. 0.60, won 0.64, gap 0.06 | est. 0.90, won 0.94, gap 0.08 | est. 0.95, won 0.96, gap 0.05 | est. 0.96, won 1.00, gap 0.04 |
| **Bw2-s2** | est. 0.51, won 0.52, gap 0.02 | est. 0.59, won 0.52, gap 0.07 | est. 0.57, won 0.61, gap 0.06 | est. 0.90, won 0.96, gap 0.08 | est. 0.95, won 0.96, gap 0.05 | est. 0.95, won 1.00, gap 0.05 |
| **Bw4-s1** | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.53, gap 0.06 | est. 0.57, won 0.59, gap 0.07 | est. 0.90, won 0.96, gap 0.07 | est. 0.95, won 0.97, gap 0.04 | est. 0.95, won 1.00, gap 0.04 |
| **Bw4-s2** | est. 0.51, won 0.52, gap 0.01 | est. 0.60, won 0.52, gap 0.08 | est. 0.55, won 0.58, gap 0.04 | est. 0.91, won 0.96, gap 0.08 | est. 0.95, won 0.97, gap 0.04 | est. 0.96, won 1.00, gap 0.04 |

Unchanged at both weights. The gap against the held-out opponents is 0.06 to 0.12 and against the generation-60 checkpoint 0.04 to 0.07, where Step 5 read 0.08 and 0.09 and 0.07 and 0.12, and the 480 group 0.16 and 0.14 and 0.14 and 0.13. The weight-4 networks rate their positions against the generation-60 checkpoint at 57% and 55% and win 59% and 58%: the value head knows what those games are worth, which is what separates the retention reading of section 2 from the 480 group's, where the network rated them at 53% and won 45%. The self-play estimate sits at 0.51 to 0.52 on every new network, as in Step 5, where the 480 group's sat at 0.54 because most of its games were wins against weak opponents.

## 7. What this establishes

- **The dose curve was the policy signal at the opponent's positions, and it can be raised without the games.** Weighting the searched rows in the policy loss, with the games, the rows and the value targets exactly Step 5's, lands on the Step 6 curve at every reading: the prior's tail, the plain search's escapes, the Greedy family and the held-out row at weight 2 are the 240 group's; the tail and the family at weight 4 are the 480 group's. The Step 6 results could not say whether the held-out row had moved because the policy head saw more fatal positions or because the whole network trained on different games. It moved here with the games unchanged, on 110 and 116 distinct fatal positions counted twice, as far as it moved on 200 distinct ones counted once.
- **The axis the two routes share is the weighted count of fatal rows.** Rows delivered times the weight on them reads 130, 226, 340 along the weight axis and 130, 200, 320 along the dose axis, and the measures that follow it follow it on both: the under-5% share 32%, 21%, 9% against 32%, 17%, 8%; the family sum 200, 168, 157 against 200, 181, 143. The weight reduces the rows as the dose did, because every counted lesson teaches the network not to produce the next, so neither axis can be raised in proportion to what is spent on it; but the weight is spent on nothing, where the dose is spent on the network's own games.
- **The held-out row reads the same from both axes, and the second doubling of the signal did not move it again.** Weight 2 reads 25 and 27, weight 4 26 and 22, 240 games 24 and 22, 480 games 24 and 17: one reading at 22 to 27 reached from both directions, with one seed at 17, and a pooled one, since the checkpoints trained without searched rows spread from 21 to 39 on this row and overlap it. Whether that is where this signal stops or where 85 to 200 distinct positions stop is the one question the two axes together leave open, and the one run that answers it is the weight on the 240 games.
- **The 480 cost was the displaced self-play and nothing else.** The design named one other explanation, that the policy head would carry less at its own positions when asked to carry more at the opponent's. At a policy-loss share of 27%, between the 240 and 480 groups' 17% and 35%, the Edax rows are Step 5's, the calibration gaps are Step 5's, the value loss is Step 5's, and the generation-60 row is 14 points above the 480 group's. The recipe the dose curve pointed at was 240 external games; the recipe it should have pointed at is a weight of 2 to 4 on the searched rows at 120, which reaches the same place with four fifths of the games the network's own.
- **The exact check still does what no weight did.** At weight 4 the plain search leaves 22 and 26 wipeouts on the held-out row and 41 and 44 against Greedy; the check at 50 simulations on the unretrained control leaves 12 and 12. The learned route now reaches the rule at a fifth of the external share and for nothing measurable; it reaches it partway, and the calculated route reaches it at once.

**What it does not establish.** Two seeds per weight, 200 games per row, and the per-pair differences on the held-out row are inside the noise, as in Step 6; the result rests on all four checkpoints reading 22 to 27 and on the pooled contrast, since the checkpoints trained without searched rows spread from 21 to 39 on that row (Step 6 results, section 4). The weight is applied as a weighted mean, the sum of weight times loss over the sum of the weights, so the policy term keeps its scale and the share of it that the network's own rows carry shrinks as the weight grows: at weight 4 a self-play row carries about four fifths of the policy gradient it carries at weight 1. The retention and Edax rows did not move, so nothing visible came of that, but a cost read at a larger weight carries it. Weight 4 against weight 2 is inside the noise on every row but the family and the prior's tail. The generation-60 row at weight 4 is 5 and 10 points under Step 5 in the same direction on both seeds; the calibration says the value head is not the reason, and a third seed or a longer run would say whether the reading is real. The weight was tested at the Step 5 dose only, so the result says the signal suffices for what the dose did, not that distinct positions add nothing on top of it. The held-out row is one opponent.

## 8. What follows

In order of cost.

1. **The weight on the 240 games**, weight 2, two seeds: the two axes combined. If the held-out row goes below 22 the reading at 22 to 27 was the signal's limit at 100 to 200 distinct positions and distinct positions add to it; if it stays, it is where this prior stops reaching the random-playout search's boards at this search budget, and the next question is the search-budget curve.
2. **A second knowledge-free opponent**, so that the transfer result rests on more than one row.
3. **The search-budget dose curve with and without the check** on the weight-2 weights as well as the control: at a prior of 10 to 11% the plain search at 256 simulations escapes 97% of the probed positions, and the curve says at what budget the learned prior and the exact check meet.

## 9. Reproduction

As for Step 6 (its section 9), with the Step 7 configs, run names `step7-Bsp-w2-s1`, `-s2`, `step7-Bsp-w4-s1`, `-s2`, evaluation names `Bw2-s1`, `Bw2-s2`, `Bw4-s1`, `Bw4-s2`, and the probe on `step7-checkpoints.json`, which is `step6-checkpoints.json` with the four new entries:

```bash
python scripts/step3_scorecard.py ~/reversi-runs/step3-eval \
    --checkpoints control120 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bw2-s1 Bw2-s2 Bw4-s1 Bw4-s2 \
    --out research/results/step7/scorecard.json --markdown
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol ~/reversi-runs/step3-eval/step7-checkpoints.json \
    --positions-from control120 \
    --networks control120 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bw2-s1 Bw2-s2 Bw4-s1 Bw4-s2 \
    --out research/results/step7/head-probe-fixed.json
python scripts/step5_searched_rows.py ~/reversi-runs/step7-Bsp-w2-s1 ~/reversi-runs/step7-Bsp-w2-s2 \
    ~/reversi-runs/step7-Bsp-w4-s1 ~/reversi-runs/step7-Bsp-w4-s2 \
    --from-generation 121 --out research/results/step7/searched-rows.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step7-Bsp-w2-s1 ~/reversi-runs/step7-Bsp-w2-s2 \
    ~/reversi-runs/step7-Bsp-w4-s1 ~/reversi-runs/step7-Bsp-w4-s2 \
    --from-generation 121 --out research/results/step7/lesson-rows.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval \
    --checkpoints control120 Bsp-s1 Bsp-s2 Bsp240-s1 Bsp240-s2 Bsp480-s1 Bsp480-s2 Bw2-s1 Bw2-s2 Bw4-s1 Bw4-s2 \
    --self-play Bsp240-s1=~/reversi-runs/step6-Bsp240-s1 --self-play Bsp240-s2=~/reversi-runs/step6-Bsp240-s2 \
    --self-play Bsp480-s1=~/reversi-runs/step6-Bsp480-s1 --self-play Bsp480-s2=~/reversi-runs/step6-Bsp480-s2 \
    --self-play Bw2-s1=~/reversi-runs/step7-Bsp-w2-s1 --self-play Bw2-s2=~/reversi-runs/step7-Bsp-w2-s2 \
    --self-play Bw4-s1=~/reversi-runs/step7-Bsp-w4-s1 --self-play Bw4-s2=~/reversi-runs/step7-Bsp-w4-s2 \
    --out research/results/step7/value-calibration.json
```

The four training jobs went through `slurm/train.sbatch` with `CONFIG`, `RUN_ID` and `GENERATIONS=140`, and `EXTRA_ARGS="--set seed=1338"` for the second seeds; the 52 evaluation pairings through `slurm/cpu.sbatch`, one core each, as for Steps 5 and 6.
