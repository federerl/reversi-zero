# Step 8 — The Two Axes Combined, and the Weight Pushed Past Them

**Applies to:** the open question the [Step 7 results](step7-results.md) leave in their section 7: the held-out random-playout row reaches 22 to 27 wipeouts from both directions, by more searched positions (Step 6) and by a weight on the same positions (Step 7), and the second doubling on either axis did not move it again; whether that is where the signal stops or where the distinct positions stop
**Depends on:** the [Step 7 design](step7-design.md) (the weight), the [Step 6 design](step6-design.md) (the dose) and the [Step 3 design](step3-design.md) (the matching rule and scorecard); nothing in any of them changes here
**Date:** November 8, 2026, written before any run
**Code:** none; `configs/full8x8_step8_search240_weight2.yaml` is the Step 6 240-game config with `train.opponent_policy_weight: 2`, and `configs/full8x8_step8_weight8.yaml` is the Step 5 config with `train.opponent_policy_weight: 8`

## 1. The question

Steps 6 and 7 put six points on one curve. Along the dose axis the fatal searched rows went 130, 200, 320 per run; along the weight axis the rows delivered times the weight went 130, 226, 340. The prior's tail, the plain search's escapes and the Greedy family followed that weighted count on both axes, and the held-out row followed it once: from 31 and 38 wipeouts at 130 to 22 to 27 at 200 to 230, and then to 22 to 26 at 320 to 340, with one seed of the 480-game group at 17. The second doubling, by games or by weight, bought nothing on that row that 200 games resolve.

Two readings of that are open, and they point to different next steps. The first is that the row is set by the weighted signal and the signal saturates near a weighted count of 300: more of it, however bought, buys nothing, and what the random-playout search still finds, about one game in ten, is what this prior at this search budget cannot reach. The second is that the signal is not the limit but the positions are: at 85 to 200 distinct fatal positions the prior has learned the boards near them, and the boards the random-playout search reaches that are not near any of them need positions nearer to them, which only more games supply, and the 480 group did not show it because it was paying for its games with the rest of the network.

Two groups separate the readings. The first applies both axes at once, 240 external games and a weight of 2: about the 240 group's distinct positions, each counted twice, for a weighted count near 350 at an external share of two fifths, where Step 6 found no cost. If distinct positions are what the row needs, this group has twice the weight-4 group's at the same weighted count and should read below it. The second doubles the weight again with the games Step 5's: a weight of 8 on about 70 distinct positions, for a weighted count near 500 at the Step 5 external share. If the signal is not saturated, this group has the most of it at no cost in games and should read below both Step 7 groups; if the signal is saturated it reads where they did.

| group | external games | weight | share of the policy loss from the opponent's rows | distinct fatal positions, expected | weighted count, expected |
|---|---:|---:|---:|---:|---:|
| B-both-sp (Step 5) | 120 | 1 | 8.5% | 126, 130 | 126, 130 |
| B-both-sp-240 (Step 6) | 240 | 1 | 17% | 203, 198 | 203, 198 |
| B-both-sp-480 (Step 6) | 480 | 1 | 35% | 318, 323 | 318, 323 |
| B-both-sp-w2 (Step 7) | 120 | 2 | 16% | 110, 116 | 220, 232 |
| B-both-sp-w4 (Step 7) | 120 | 4 | 27% | 86, 84 | 344, 336 |
| **B-both-sp-240-w2** | 240 | 2 | 29% | 140 to 190 | 280 to 380 |
| **B-both-sp-w8** | 120 | 8 | 43% | 55 to 80 | 440 to 640 |

The weight-8 group also tests the one explanation of a cost that Step 7 left open. At a policy-loss share of 27% the policy head paid nothing at its own positions for what it carried at the opponent's; at 43% it may, and if it does the weight route has a ceiling of its own that the games route does not share.

## 2. Groups and matching

| group | config | self-play games | external games | weight | seeds |
|---|---|---:|---:|---:|---|
| **B-both-sp-240-w2** | `full8x8_step8_search240_weight2.yaml` | 360 | 240 | 2 | 1337, 1338 |
| **B-both-sp-w8** | `full8x8_step8_weight8.yaml` | 480 | 120 | 8 | 1337, 1338 |

Both are twenty generations from `control120` with its replay window; the opponents, searches, value targets, trainer and window are Step 5's. The first group's games are the 240 group's and its loss is the weight-2 group's; it is read against both and against the weight-4 and 480 groups, which it matches on the weighted count. The second group's games are Step 5's and it is read against the Step 7 groups, which it matches on everything but the weight.

## 3. Evaluation

The frozen Step 1 protocol on the frozen roster for the four new checkpoints at generation 140: 200 games per pairing, 50 simulations, seed 20261025, the plain search, all thirteen rows. The fixed-position head probe on the control's 63 fatal positions, the searched-row count, the Step 3b row count and the calibration check, as in Steps 5 to 7. The scorecard is read with the Step 5, 6 and 7 columns beside the new ones.

## 4. Predictions, written before the runs

The numbers these are read against. At a weighted count of 320 to 340: random-playout wipeouts 26 and 22 (weight 4), 24 and 17 (480 games); Greedy family sums 153 and 160, 150 and 135; share of the control's fatal positions with a prior on the wiping reply under 5%, 11% and 6%, 10% and 6%; median prior 11.5% and 10.6%, 12.0% and 11.9%; generation-60 row 59.2% and 58.0% at weight 4, 44.8% on both seeds at 480. Step 5: random-playout 31 and 38, family 203 and 198, generation-60 64.5% and 67.8%, Edax L5 31.2% and 35.2%.

1. **The rows arrive, and fewer of them.** Both groups learn the avoidance faster than any earlier group, so the external wipeout rate ends below 2% and the fatal rows number 140 to 190 per run in the 240-game group and 55 to 80 in the weight-8 group, with the target mostly on the wiping reply in five rows of six. The weighted count lands near 350 in the first group and 450 to 650 in the second. If the rows fall outside those ranges the groups are not where the table puts them and the rest is read against where they are.
2. **The prior's tail is closed in both groups.** The under-5% share at or below 11% on every seed, where the weight-4 and 480 groups read 6 to 11%; the median prior at or above 10.5% on every seed. The plain search at 50 simulations escapes at least half of the probed positions.
3. **The held-out row.** This is the claim-side prediction and it is written as the three readable outcomes, with the one expected named. *Saturation*: both groups read between 17 and 27 on every seed, and pooled over 400 games per group neither is below 40 wipeouts; the weighted signal is the variable and it stops near 300, so the remaining tenth of the random-playout games is what this prior cannot reach at 50 simulations, and the next question is the search-budget curve. *Positions*: the 240-game group reads below 22 on both seeds and the weight-8 group does not; the row needs distinct positions past what the weight supplies, and more games at a share that costs nothing is the recipe. *Signal*: the weight-8 group reads below 22 on both seeds; the signal was not saturated, the second doubling in Step 7 was too small to show it, and the weight route has further to go. Expected: saturation. Four checkpoints at a weighted count of 320 to 340 read 17 to 26, and nothing in Steps 6 or 7 distinguishes the 480 group's 17 from the others' 22 to 26.
4. **No cost in the 240-game group; the weight-8 group is the test.** The 240-game group's generation-60 row, Edax rows and calibration gap are within seed noise of the 240 group's (63.2% and 61.8% on the retention row). For the weight-8 group the retention row above 58% on both seeds and the Edax rows within noise of Step 5's is the expected reading; if the generation-60 row falls below 55% on both seeds with the calibration gap unchanged, the policy head is paying at its own positions and the weight has a ceiling of its own, at a share between 27% and 43%.
5. **The Greedy family.** The 240-game group's family sum at or below 150 and 135 on both seeds, the weight-8 group's at or below 153 and 160; Greedy's own row below 45 on every seed.
6. **The value head does not move.** Above +0.9 at the network's fatal position and below −0.9 one ply later on the fixed positions, in both groups.

## 5. Cost

Four runs of twenty generations, about 1.5 GPU-hours each on the L40S node by the Step 6 and 7 timings; 52 evaluation pairings on CPU, of which the four random-playout pairings take about six hours each; the probe, the row counts and the calibration check. Two days including the write-up.

## 6. How a run is launched

As in the Step 3 design, section 7:

```bash
cp -r runs/control runs/step8-Bsp240-w2-s1
reversi train -c configs/full8x8_step8_search240_weight2.yaml --run-id step8-Bsp240-w2-s1 --generations 140 --device cuda \
    --set selfplay.n_workers=22
# the second seed adds --set seed=1338; the weight-8 group uses full8x8_step8_weight8.yaml and run ids step8-Bsp-w8-s1, -s2
```

On the cluster the runs go through `slurm/train.sbatch` with `CONFIG`, `RUN_ID`, `GENERATIONS` and, for the second seeds, `EXTRA_ARGS="--set seed=1338"`. Evaluation names `B240w2-s1`, `B240w2-s2`, `Bw8-s1`, `Bw8-s2`.
