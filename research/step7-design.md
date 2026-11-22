# Step 7 — A Weight on the Searched Opponent Rows

**Applies to:** the open question the [Step 6 results](step6-results.md) leave in their section 8: whether the policy head's gain from the searched opponent rows can be raised without raising the external share of the games, which is what cost the network its strength at 480
**Depends on:** the [Step 5 design](step5-design.md) (the recipe) and the [Step 3 design](step3-design.md) (the matching rule and scorecard); nothing in either changes here
**Date:** November 8, 2026, written before any run
**Code:** `train.opponent_policy_weight` (`src/reversi/config.py`, `src/reversi/train/loss.py`, `src/reversi/train/trainer.py`, the `opponent_to_move` column carried through `src/reversi/data/replay.py`), `configs/full8x8_step7_weight2.yaml`, `configs/full8x8_step7_weight4.yaml`

## 1. The question

Step 6 raised the dose of searched opponent rows by raising the external games per generation, and three things moved together: the policy head's prior on the wiping reply, the held-out random-playout row, and, at 480 games, the network's strength against everything, because its own self-play had become a fifth of its data. The dose curve cannot say which of the first two came from the policy head seeing more rows and which from the whole network training on a different distribution; at 480 the second effect is large enough to swamp any reading of the first.

A weight on the searched rows in the policy loss separates them. The games are Step 5's exactly, 480 self-play and 120 external per generation, the same opponents, searches, value targets and replay window; the one change is that a row recorded with the opponent to move counts two or four times in the policy loss. The value head sees nothing different. The data distribution is Step 5's. Only the policy head's share of attention at the opponent's positions changes, by the same factor that the dose changed it:

| group | external share of the games | share of the policy loss from the opponent's rows |
|---|---:|---:|
| B-both-sp (Step 5) | 20% | 8.5% |
| B-both-sp-240 (Step 6) | 40% | 17% |
| B-both-sp-480 (Step 6) | 80% | 35% |
| **B-both-sp-w2** | 20% | 16% |
| **B-both-sp-w4** | 20% | 27% |

The weight-2 group matches the 240 group's policy-loss share and the weight-4 group falls between 240 and 480, with the games of the 120 group. If the prior and the held-out row follow the policy-loss share, they will land near the 240 and 480 readings, and the retention and Edax rows will not move, because nothing the value head or the body sees has changed in kind. If they do not follow it, the dose worked through something the weight does not supply, and the obvious candidate is coverage: the 240 group searched 200 distinct fatal positions where a weight of two counts 130 distinct ones twice. That is the second thing the step can learn, and it decides whether the route is limited by signal or by data.

The weight also tests the one explanation of the 480 cost that the Step 6 results did not rule out: that the policy head itself, asked to carry more at the opponent's positions, carried less at its own. If weight 4 costs retention with the data unchanged, that explanation stands; if it does not, the cost at 480 was the displaced self-play and nothing else.

## 2. What the weight does

In the trainer the policy loss is the mean over a batch of the cross-entropy between the search's visit distribution and the policy head's output, with value-only rows masked out of the mean. With the weight, each row's cross-entropy is multiplied by the weight if the row was recorded with the opponent to move and by one otherwise, and the sum is divided by the sum of the weights rather than the count. The loss stays a mean on the same scale, so its balance against the value term is unchanged; what changes is that the opponent's rows make up a larger share of it, as they would if each appeared in the data two or four times over. Rows without a policy target stay masked whatever their weight.

The weight is a training-time setting. It changes no shard, no search and no game; the replay window built from Step 5's generations 101 to 120 is read with the same rows and the weight applies from the first training step of generation 121.

## 3. Groups and matching

| group | config | external games | weight on the opponent's rows | seeds |
|---|---|---:|---:|---|
| B-both-sp (Step 5, not re-run) | `full8x8_step5_search.yaml` | 120 | 1 | 1337, 1338 |
| B-both-sp-240 (Step 6, not re-run) | `full8x8_step6_search240.yaml` | 240 | 1 | 1337, 1338 |
| **B-both-sp-w2** | `full8x8_step7_weight2.yaml` | 120 | 2 | 1337, 1338 |
| **B-both-sp-w4** | `full8x8_step7_weight4.yaml` | 120 | 4 | 1337, 1338 |

The two new configs are the Step 5 config with `train.opponent_policy_weight` set and nothing else. Each run is twenty generations from `control120` with its replay window; the games, opponents, search budget, noise, temperature, trainer and window are Step 5's. The groups are matched on games and on rows, exactly, since the rows are the same kind in the same number; they differ in the policy loss only.

## 4. Evaluation

The frozen Step 1 protocol on the frozen roster for the four new checkpoints at generation 140: 200 games per pairing, 50 simulations, seed 20261025, the plain search, all thirteen rows. The fixed-position head probe on the control's 63 fatal positions, the searched-row count, the Step 3b row count and the calibration check, as in Steps 5 and 6. The scorecard is read with the Step 5 and Step 6 columns beside the new ones.

## 5. Predictions, written before the runs

The numbers these are read against. Step 5 (weight 1, 120 games): prior on the wiping reply on the fixed positions 8.5% and 8.2%; random-playout wipeouts 31 and 38; Greedy-family wipeouts summed over the four disc-hungry opponents 203 and 198; generation-60 row 64.5% and 67.8%; Edax L5 31.2% and 35.2%; fatal searched rows 126 and 130. Step 6 at 240 games: prior 11.1% and 9.9%; random-playout 24 and 22; family 189 and 172; generation-60 63.2% and 61.8%. At 480: prior 12.0% and 11.9%; random-playout 24 and 17; family 150 and 135; generation-60 44.8% on both.

1. **The games are Step 5's.** The network's wipeout rate in the external games starts near 9 to 11% at generation 121 and ends between 2 and 4%, as in every Step 5 and 6 run; the fatal searched rows number between 90 and 140 per run, with the target mostly on the wiping reply in five rows of six. Fewer rows than Step 5's are expected if the weight teaches the avoidance faster, as the dose did. If the games differ in any other way, the groups are not matched and that is read first.
2. **The prior follows the policy-loss share.** On the fixed positions, weight 2 reads above Step 5 on both seeds and at or above 10%, where the 240 group read 11.1% and 9.9%; weight 4 reads above weight 2 on both seeds and at or above 12%, where the 480 group read 12.0% and 11.9%. Held with more confidence for the direction than for the thresholds.
3. **The held-out row follows the policy-loss share.** Random-playout wipeouts at weight 2 below 30 on both seeds, where the 240 group read 24 and 22; at weight 4 below 25 on both, where the 480 group read 24 and 17. This is the claim-side prediction. If it holds, the dose worked through the policy signal at the opponent's positions, and the signal can be raised without the data. If the prior rises (prediction 2) and the row stays within Step 5's 26 to 39, the weight re-learns the same positions harder and the held-out row needs distinct positions, which only the games supply: the route is limited by coverage, and the contribution's first section says so.
4. **No cost at either weight.** The generation-60 row, the Edax rows and the calibration gap stay within seed noise of Step 5 at both weights: the value head sees the same rows with the same targets, the body sees the same positions. If weight 4 costs retention, the policy head's capacity at its own positions is part of the price and the 480 cost was not all displacement.
5. **The Greedy family falls with the weight.** The family sum at weight 2 at or below Step 5's 203 and 198 on both seeds; at weight 4 at or below the 240 group's 189 and 172.
6. **The value head does not move.** Value at the network's fatal position above +0.9 and at the position one ply later below −0.9 on the fixed positions, at both weights.

The readable outcomes. Predictions 2, 3 and 4 hold: the policy signal at the opponent's positions is the mechanism, it can be raised without raising the external share, and the recipe for the report is weight 4 at 120 games rather than 480 games. Prediction 3 fails with 2 holding: coverage is the limit, the Step 6 curve's x-axis is distinct fatal positions, and the way forward is more distinct positions at a fixed external share, which the self-limiting dose makes hard. Prediction 4 fails at weight 4: the policy head pays at its own positions for what it learns at the opponent's, and the weight has a ceiling of its own. Prediction 2 fails: the weighted rows did not move the prior that the same rows unweighted did, which would need a check of the implementation before anything else is read. Prediction 1 fails: fixed first.

## 6. Cost

Four runs of twenty generations, about 1.5 GPU-hours each on the L40S node by the Step 6 timings, since the games and searches are Step 5's; 52 evaluation pairings on CPU, of which the four random-playout pairings take about six hours each; the probe, the row counts and the calibration check. Four to five days including the write-up.

## 7. How a run is launched

As in the Step 3 design, section 7:

```bash
cp -r runs/control runs/step7-Bsp-w2-s1
reversi train -c configs/full8x8_step7_weight2.yaml --run-id step7-Bsp-w2-s1 --generations 140 --device cuda \
    --set selfplay.n_workers=22
# the second seed adds --set seed=1338; the weight-4 group uses full8x8_step7_weight4.yaml and run ids step7-Bsp-w4-s1, -s2
```

On the cluster the runs go through `slurm/train.sbatch` with `CONFIG`, `RUN_ID`, `GENERATIONS` and, for the second seeds, `EXTRA_ARGS="--set seed=1338"`. Evaluation names `Bw2-s1`, `Bw2-s2`, `Bw4-s1`, `Bw4-s2`.
