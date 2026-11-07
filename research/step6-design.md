# Step 6 — The Dose Curve on the Searched Opponent Rows

**Applies to:** the open question the [Step 5 results](step5-results.md) leave in their section 7: whether the prior keeps rising and the wipeouts keep falling with more searched rows, and whether the held-out row ever moves
**Depends on:** the [Step 5 design](step5-design.md) (the recipe) and the [Step 3 design](step3-design.md) (the matching rule and scorecard); nothing in either changes here
**Date:** November 7, 2026, written before any run
**Code:** none; `configs/full8x8_step6_search240.yaml` and `configs/full8x8_step6_search480.yaml` are the Step 5 config with the external share raised

## 1. The question

Step 5 gave the policy head a target at the opponent's positions and the prior on Greedy's wiping reply doubled, from 4% to 8%, with 126 and 130 searched rows at fatal positions per run. The wipeouts against every disc-hungry opponent fell by a third; the random-playout row, the knowledge-free opponent that finds the same hole on different boards, did not move. The reading was "the policy head learned the positions it was shown, not the rule".

That reading rests on one dose. Two things could be true of it. The prior may keep rising with the number of searched rows, and the wipeouts keep falling, until the held-out row moves too, in which case the rule is learnable and Step 5 was short of data. Or the prior may rise and the Greedy family keep improving while the random-playout row stays where it is, in which case "positions, not the rule" is a ceiling the data cannot lift, and the exact check is the only thing that reaches the held-out opponent. The design cannot tell these apart with one dose; it can with three.

A third question rides along: whether twenty more generations of the Step 5 runs continue the fall, which asks the same thing along the time axis.

## 2. Groups and matching

The dose is the number of external games per generation. Step 3 matched its groups on training examples rather than games; with both sides of an external game recorded and searched, an external game yields about as many rows as a self-play game, so the self-play games are reduced one for one and every group trains on about 600 games' worth of rows per generation.

| group | config | self-play games | external games | share of rows from external games | seeds |
|---|---|---:|---:|---:|---|
| B-both (Step 3b, not re-run) | `full8x8_step3_both.yaml` | 480 | 120 | ~8.5%, value-only | 1337, 1338 |
| B-both-sp (Step 5, not re-run) | `full8x8_step5_search.yaml` | 480 | 120 | ~8.5%, searched | 1337, 1338 |
| **B-both-sp-240** | `full8x8_step6_search240.yaml` | 360 | 240 | ~17% | 1337, 1338 |
| **B-both-sp-480** | `full8x8_step6_search480.yaml` | 120 | 480 | ~35% | 1337, 1338 |
| **B-both-sp, generations 141 to 160** | `full8x8_step5_search.yaml` | 480 | 120 | ~8.5% | the Step 5 runs, continued |

The opponents, their alternation, the recording and searching rules, the search budget, noise, temperature, trainer and replay window are Step 5's. Each new group is twenty generations from `control120` with its replay window. The 480-game group is the point at which the network's own self-play games are a fifth of the data; it is included because the curve needs a point where the dose is large enough to show a ceiling if there is one, and its retention row is where the cost of that would show.

The searched rows cost one search each; at 480 external games the search work per generation is about a third more than Step 5's and is reported.

## 3. Evaluation

The frozen Step 1 protocol on the frozen roster for all six new checkpoints (the four dose runs at generation 140, the two continued runs at generation 160): 200 games per pairing, 50 simulations, seed 20261025, the plain search, all thirteen rows. The fixed-position head probe on the control's 63 fatal positions for every checkpoint, the searched-row count, the Step 3b row count and the calibration check, as in Step 5. The scorecard is read with the Step 5 and Step 3b columns beside the new ones, so the curve has its zero and its first point.

## 4. Predictions, written before the runs

The Step 5 numbers these are read against: prior on the wiping reply on the fixed positions 8.5% and 8.2%; Greedy wipeouts 54 and 45; greedy-d2 58 and 63; random-playout 31 and 38; generation-60 retention 64.5% and 67.8%; searched rows at fatal positions 126 and 130, with the target mostly on the wiping reply in 83% and 86% of them.

1. **The dose arrives.** The searched rows at fatal positions scale with the external games, about 250 and 500 per run, with the target mostly on the wiping reply in the same five rows of six, and the network's wipeout rate in the external games stays near 5 to 7%. If the rate changes with dose, the groups differ in more than the dose and that is read first.
2. **The prior rises with the dose.** On the fixed positions, the prior on the wiping reply is higher at 240 than at 120 and higher at 480 than at 240, on both seeds; 240 passes 10% and 480 passes 14%. Held with less confidence at the second threshold than the first.
3. **The Greedy family keeps falling.** Greedy wipeouts at 240 below Step 5's on both seeds, and at 480 below 35 on both; greedy-d2 and greedy-corner follow.
4. **The random-playout row does not move at any dose.** It stays within 26 to 39 on every new checkpoint. This is the claim-side prediction and the one that matters: if the row falls with the dose, with 480 below 25 on both seeds, the rule is learnable from searched positions and Step 5 was short of data, and the contribution's section 1 changes.
5. **The cost shows at 480, not at 240.** At 240 the generation-60 row and the Edax rows stay within seed noise of Step 5. At 480, with the network's own games a fifth of the data, the generation-60 row falls by more than seed noise; the Edax rows may fall with it. Reported as the price of the dose, as group D's price was.
6. **Twenty more generations continue the fall by less than the first twenty did.** At generation 160 the prior is above the generation-140 reading and the Greedy wipeouts below it, by less than half the change from generation 120 to 140; the random-playout row is unchanged.

The readable outcomes. Predictions 2, 3 and 4 hold: the curve rises, the family improves, the held-out row does not, and "positions, not the rule" is a measured ceiling, which is the strongest form of the Step 5 reading. Prediction 4 fails downward: the policy head learns the rule given enough searched positions, the exact check is no longer the only route to the held-out opponent, and the claim narrows to the value head alone. Prediction 2 fails with 3 holding: the family improves without the prior at the probed positions moving, and the mechanism needs re-reading. Prediction 5 fails at 480 with no cost: the external share can be raised freely, which would be worth knowing. Prediction 1 fails: the groups are not comparable as doses, fixed first.

## 5. Cost

Four runs of twenty generations and two continuations of twenty, about 1.3 to 1.8 GPU-hours each by the Step 5 timings; 78 evaluation pairings on CPU, of which the six random-playout pairings take several hours each; the probe, the row counts and the calibration check. About a week including the write-up.

## 6. How a run is launched

As in the Step 3 design, section 7:

```bash
cp -r runs/control runs/step6-Bsp240-s1
reversi train -c configs/full8x8_step6_search240.yaml --run-id step6-Bsp240-s1 --generations 140 --device cuda \
    --set selfplay.n_workers=22
# the second seed adds --set seed=1338; the 480 group uses full8x8_step6_search480.yaml and run ids step6-Bsp480-s1, -s2

# the continuations reuse the Step 5 run directories and raise the target
reversi train -c configs/full8x8_step5_search.yaml --run-id step5-Bboth-sp-s1 --generations 160 --device cuda \
    --set selfplay.n_workers=22
```

On the cluster the runs go through `slurm/train.sbatch` with `CONFIG`, `RUN_ID`, `GENERATIONS` and, for the second seeds, `EXTRA_ARGS="--set seed=1338"`. The two Step 6 configs are the Step 5 config with `games_per_generation` and `opponent_games` changed and nothing else.
