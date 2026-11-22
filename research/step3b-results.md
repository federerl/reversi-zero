# Step 3b — Results: Training the Value Head Where the Search Reads It Changes Nothing

**Applies to:** the follow-up pre-registered in [step3b-design.md](step3b-design.md), to the [Step 3 results](step3-results.md), section 10, item 1
**Trained:** November 1, 2026, on the cluster, one L40S and 22 self-play workers per run, the code of `configs/full8x8_step3_both.yaml` and `selfplay.record_opponent_positions` at commit `b1d96f8`
**Evaluated:** November 1, 2026, on the cluster's CPU partition: 26 pairings of 200 games, one job each, the same frozen protocol as Step 3.
**Reports:** [`results/step3b/scorecard.json`](results/step3b/scorecard.json) (every pairing's score, interval and wipeout count, the Step 3 control and group B included), [`results/step3b/head-probe.json`](results/step3b/head-probe.json), [`results/step3b/lesson-rows.json`](results/step3b/lesson-rows.json), [`results/step3b/value-calibration.json`](results/step3b/value-calibration.json). The arena reports and recorded games are under `~/reversi-runs/step3-eval/` on the cluster and are not committed.
**Verdict: the null the design named, and the reason behind it.** The opponent's positions were recorded and trained on, about 2,900 of them per generation. At the positions the search reads when it walks into a wipeout, the value head did not move: −0.98 and −0.99 where the exact answer is +1, against −0.98 for Step 3's group B and −0.99 for the control. The wipeouts against Greedy did not fall (61 and 69 against the control's 65). Section 5 counts the training rows and finds why: in twenty generations the trainer saw about 80 rows that carry the lesson, and more than four thousand that look the same to a value head and say the opposite. The Step 3 explanation, "the value head was never trained where the search reads it", is closed. The weakness is not a gap in the training data; it is a one-move tactic that a value target cannot carry and a 50-simulation search reading a 4% prior does not look at.

## 1. What was run

Two runs, each a copy of the control run at generation 120 continued for twenty generations under `configs/full8x8_step3_both.yaml`: 480 self-play games and 120 games against Greedy and Minimax-d4 per generation, both sides of each external game recorded, the opponent's positions as value-only rows. Everything else is Step 3's group B.

| run | seed | games | of which external | training rows | of which value-only | gradient steps | wall time | network's score in the external games | wiped out in the external games |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step3-Bboth-s1 | 1337 | 12,000 | 2,400 | 684,605 | 58,461 (8.5%) | 16,000 | 1.32 h | 93.8% | 6.0% |
| step3-Bboth-s2 | 1338 | 12,000 | 2,400 | 684,173 | 58,201 (8.5%) | 16,000 | 1.29 h | 92.6% | 6.9% |

The row counts are 1.5% below Step 3's (694,800 and 695,793 for the B seeds), inside the design's intent that the groups be matched on data and on gradient steps. An external game with both sides recorded yields slightly fewer rows than a self-play game; the opponent's forced moves are skipped as the network's are. The value-only share is 8.5% of the rows, a little under the design's estimate of a tenth.

The in-training wipeout rate against the two opponents was flat, as it was for group B: 5.8% and 7.5% in generation 121, 5.0% and 4.2% in generation 140, 6.0% and 6.9% over the twenty generations, against 7.4% and 6.8% for the B seeds. Those games are played at 200 simulations with exploration noise, where the search already finds the escape about half the time (Step 3 results, section 5), so the rate is a weak reading; the evaluation at 50 simulations is the one that counts.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. The control and Step 3's group B are the Step 3 numbers, not re-run.

| row | opponent | control120 | B-s1 | B-s2 | B-both-s1 | B-both-s2 |
|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 10.0% [6.6, 14.9], 9 wo | 9.8% [6.4, 14.7], 4 wo | 13.8% [9.7, 19.2], 5 wo | 14.8% [10.5, 20.3], 2 wo | 11.8% [8.0, 16.9], 6 wo |
| **held out** | Edax L5 | 29.8% [23.8, 36.4], 0 wo | 29.5% [23.6, 36.2], 3 wo | 40.2% [33.7, 47.2], 3 wo | 26.2% [20.6, 32.8], 2 wo | 37.0% [30.6, 43.9], 1 wo |
| **held out** | Edax L3 | 72.0% [65.4, 77.8], 0 wo | 68.5% [61.8, 74.5], 3 wo | 68.5% [61.8, 74.5], 1 wo | 68.0% [61.2, 74.1], 0 wo | 64.8% [57.9, 71.0], 1 wo |
| **held out** | rollout-r3000 | 81.0% [75.0, 85.8], 34 wo | 88.0% [82.8, 91.8], 21 wo | 80.5% [74.5, 85.4], 39 wo | 80.8% [74.7, 85.6], 36 wo | 80.8% [74.7, 85.6], 37 wo |
| retention | gen60 | 62.5% [55.6, 68.9], 0 wo | 57.8% [50.8, 64.4], 0 wo | 60.8% [53.8, 67.3], 0 wo | 61.8% [54.9, 68.2], 0 wo | 62.3% [55.4, 68.7], 0 wo |
| seen | Greedy | 67.5% [60.7, 73.6], **65 wo** | 73.5% [67.0, 79.1], **53 wo** | 68.0% [61.2, 74.1], **64 wo** | 69.0% [62.3, 75.0], **61 wo** | 65.5% [58.7, 71.7], **69 wo** |
| seen | Minimax-d4 | 99.0% [96.4, 99.7], 0 wo | 99.5% [97.2, 99.9], 0 wo | 98.5% [95.7, 99.5], 0 wo | 96.2% [92.6, 98.1], 0 wo | 97.5% [94.3, 98.9], 0 wo |
| seen family | greedy-corner | 63.5% [56.6, 69.9], 72 wo | 64.5% [57.7, 70.8], 71 wo | 65.5% [58.7, 71.7], 69 wo | 60.5% [53.6, 67.0], 77 wo | 65.0% [58.2, 71.3], 70 wo |
| seen family | greedy-d2 | 59.0% [52.1, 65.6], 82 wo | 58.0% [51.1, 64.6], 84 wo | 61.0% [54.1, 67.5], 78 wo | 58.0% [51.1, 64.6], 84 wo | 55.5% [48.6, 62.2], 89 wo |
| seen family | noisy-greedy | 73.5% [67.0, 79.1], 52 wo | 74.0% [67.5, 79.6], 51 wo | 67.0% [60.2, 73.1], 66 wo | 78.0% [71.8, 83.2], 44 wo | 79.5% [73.4, 84.5], 41 wo |
| seen family | corner | 98.0% [95.0, 99.2], 4 wo | 98.0% [95.0, 99.2], 4 wo | 98.5% [95.7, 99.5], 3 wo | 96.5% [93.0, 98.3], 7 wo | 97.5% [94.3, 98.9], 5 wo |
| seen family | mobility | 91.5% [86.8, 94.6], 17 wo | 91.0% [86.2, 94.2], 18 wo | 94.0% [89.8, 96.5], 12 wo | 91.0% [86.2, 94.2], 18 wo | 95.0% [91.0, 97.3], 10 wo |
| sanity | Random | 100% [98.1, 100], 0 wo | 98.5% [95.7, 99.5], 3 wo | 99.0% [96.4, 99.7], 2 wo | 99.0% [96.4, 99.7], 2 wo | 98.0% [95.0, 99.2], 4 wo |

Every row sits where the control and group B sit, inside the seed noise Step 3 measured (10 points on one row between two control seeds). The one row that leans the same way on both seeds, noisy-greedy at 78.0% and 79.5% with 44 and 41 wipeouts against the control's 73.5% and 52, is a 5-point lean on a row where the two A seeds of Step 3 differed by 4 points and 8 wipeouts, and the Greedy row beside it does not share it. The random-playout row, the held-out opponent that attacks the same weakness Greedy does, is the control's number to within a game on both seeds: 36 and 37 wipeouts against 34.

## 3. The predictions, one by one

The design fixed five predictions before the runs (section 4 there).

**1. "The probe moves where Step 3's did not: the value at the position with Greedy to move rises from −0.98 toward +1, while the prior on the wiping reply stays near 4 to 5%."** False in the half that mattered. The prior stayed at 4.6% and 4.4%, as predicted. The value at the position with Greedy to move is −0.98 and −0.99: it did not rise at all. Section 4 has the table. The design said that if the value does not move, the recording rule was not the reason for group B's null, and that is what is reported.

**2. "Wipeouts against Greedy fall sharply, to well under half of 53 to 65."** False: 61 and 69.

**3. "The random-playout row moves."** False: 80.8% on both seeds against the control's 81.0%, with 36 and 37 wipeouts against 34. Group B's two seeds spanned 21 to 39 wipeouts on this row; both B-both seeds land in the middle of that range.

**4. "The Edax rows do not move beyond seed noise."** True: 14.8% and 11.8% against L7, 26.2% and 37.0% against L5, 68.0% and 64.8% against L3, all inside the range the control and B seeds span.

**5. "Retention and calibration hold."** Retention holds: 61.8% and 62.3% against the generation-60 checkpoint, against the control's 62.5% and group B's 57.8% and 60.8%, with no wipeouts on any of them. Calibration holds too (section 6): on every row the two B-both networks are within two points of the control's estimate and gap. Prediction 5 is the one that came true in full, and it says only that the change cost nothing.

This is the third of the three readable outcomes the design listed: prediction 1 fails and the rest is flat, "the null, and closes the recording-rule explanation."

## 4. The head probe

The Step 3 probe, re-run on the B-both checkpoints: every wipeout each suffered against Greedy in the evaluation, at the network's last move, in the positions where a surviving move existed. The Step 3 rows are repeated for comparison.

| network | wipeouts probed | raw value at the fatal position | one ply later, raw value for Greedy (exact: +1) | prior on Greedy's wiping reply | prior on the move played | prior on all safe moves |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 63 | +0.99 | −0.99 | 3.6% | 36% | 44% |
| B-s1 | 52 | +0.84 | −0.98 | 4.6% | 25% | 44% |
| B-s2 | 63 | +0.85 | −0.98 | 4.3% | 30% | 49% |
| **B-both-s1** | 60 | +0.93 | **−0.98** | 4.6% | 29% | 46% |
| **B-both-s2** | 66 | +0.96 | **−0.99** | 4.4% | 28% | 35% |

Positions with the opponent to move were training examples in these runs, about 2,900 of them per generation, and at the fatal ones the value head reads exactly as the control does. At the network's own position one ply earlier the value moved less than group B's did, +0.93 and +0.96 against +0.84 and +0.85, which is the dose: B-both played half as many external games as B, so it recorded about half as many of the network's own fatal positions (section 5).

The search-budget result is unchanged too. At 50 simulations the recorded search replays the fatal move in every probed position; with the prior flattened the value head alone finds the escape in 100% of them; with the value zeroed the prior alone finds it in 7% and 9%; at 256 simulations the recorded search finds it in 62% and 59%. Six checkpoints in Step 3 and two more here read the same way.

## 5. Why the value head did not move: the lesson was outnumbered

The design assumed that the position with Greedy to move and the network holding a disc or two has "the clearest value target in the game". It does, in the fatal case. The replay data say how often that case occurs among the positions that look like it. [`scripts/step3b_lesson_rows.py`](../scripts/step3b_lesson_rows.py) reads every shard a run produced from generation 121 on and counts the rows past the opening (ply 16 or later) in which the network holds at most three discs, split by who is to move and by how the game ended.

| run | rows | value-only rows | **opponent to move**, network holds ≤3 discs | of which the opponent won (the lesson) | of which the network won | **network to move**, holds ≤3 discs | of which it lost | of which it won |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step3-A-s1 (pure self-play) | 694,569 | 0 | 0 | 0 | 0 | 51 | 5 | 45 |
| step3-B-s1 | 694,800 | 0 | 0 | 0 | 0 | 8,987 | 767 | 8,219 |
| step3-B-s2 | 695,793 | 0 | 0 | 0 | 0 | 8,586 | 575 | 8,011 |
| step3-Bboth-s1 | 684,605 | 58,461 | 230 | **77** | 153 | 4,573 | 307 | 4,266 |
| step3-Bboth-s2 | 684,173 | 58,201 | 235 | **83** | 152 | 4,171 | 311 | 3,858 |

The 77 and 83 are the fatal rows that pass both filters, three discs or fewer and ply 16 or later. The training log counts every external game the network ended with no discs: 144 and 166 of them (6.0% and 6.9% of 2,400), and each such game has exactly one row with the opponent to move at the fatal position. The rows the filters leave out are wipeouts from four or more discs, or inside the opening. The lesson the trainer saw is therefore between the two counts, about 150 rows rather than 80, against the same thousands of look-alikes. The ratio, not the count, is what the argument below rests on.

Three things stand out.

**Self-play never produces the position.** In twenty generations of pure self-play, 51 rows of nearly 700,000 have the mover down to three discs past the opening. Two copies of the same network do not reduce each other to a disc or two; the state the wipeout starts from is one the network reaches only against an opponent that plays for discs. This is the sharpest form of the Step 2 finding that self-play never prepares the network for the loss, and it says why Step 3's group B could reasonably have been expected to help: it put nine thousand such positions into the window where self-play had put fifty.

**In that position the network usually wins.** Of group B's nine thousand network-to-move rows with three discs or fewer, 91% and 93% are labelled as wins. Of B-both's opponent-to-move rows with the network at three discs or fewer, two thirds are labelled as wins for the network. Holding few discs in the middle of an Othello game is not a losing state; it is often a strong one, since the side with fewer discs has more moves and the discs it does have are hard to flank. The value head's +0.99 in those positions is the right answer for most of them, and the training data told it so thousands of times.

**The lesson is a one-move tactic, not a state.** What separates the 77 fatal rows from the 153 beside them is not how many discs the network holds but whether one of the opponent's legal moves happens to flank every disc it holds. That is a fact about one reply, read off the board by a one-ply search in a few microseconds; it is not a property a value head generalises from "few discs, late, opponent to move", because the rows with that property are labelled 2 to 1 the other way. The hundred-odd rows carrying the lesson sit in a window of 400,000, beside more than four thousand that look the same and say the opposite, and the network did what a smooth function fit by gradient descent does with an exception that small: it kept the rule. That is the explanation the counts support, not one the experiment tested: no run weighted the lesson rows up, and no probe asked the value head to separate the fatal rows from the ones beside them (section 8, item 3).

This also settles what Step 3's group B did and did not learn. Its value at the network's own fatal position fell from +0.99 to +0.84: nine thousand rows at that state, 8% of them losses, moved the estimate by the amount the mixture implies, and the half dose of B-both moved it half as far. Neither run moved the position one ply later, because the rows there were fewer and the mixture there is the same. The value head learned exactly what the data said. The data said the position is fine.

## 6. Value calibration

The check the Step 3 design asked for after any training change: every raw value the network produced during the evaluation games, before search, binned by its value and set against how the game ended, both on a win-probability scale. "Gap" is the count-weighted mean absolute difference between a bin's mean estimate and its mean outcome over ten bins. The self-play column is the network's own last generation of training positions, 20,000 sampled, with the result those games reached. The control and group B rows are recomputed from the same recorded games as Step 3's section 8 and agree with it to a point of rounding.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against Random |
|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.89, won 0.95, gap 0.09 | est. 0.96, won 1.00, gap 0.04 |
| B-s1 | est. 0.59, won 0.60, gap 0.02 | est. 0.60, won 0.50, gap 0.10 | est. 0.57, won 0.58, gap 0.06 | est. 0.91, won 0.96, gap 0.07 | est. 0.96, won 1.00, gap 0.04 |
| B-s2 | est. 0.60, won 0.60, gap 0.02 | est. 0.61, won 0.53, gap 0.08 | est. 0.60, won 0.61, gap 0.06 | est. 0.91, won 0.95, gap 0.06 | est. 0.96, won 1.00, gap 0.04 |
| B-both-s1 | est. 0.52, won 0.52, gap 0.02 | est. 0.58, won 0.49, gap 0.09 | est. 0.58, won 0.62, gap 0.06 | est. 0.89, won 0.94, gap 0.07 | est. 0.95, won 1.00, gap 0.05 |
| B-both-s2 | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.50, gap 0.08 | est. 0.59, won 0.62, gap 0.06 | est. 0.90, won 0.94, gap 0.08 | est. 0.95, won 1.00, gap 0.04 |

Nothing moved. The B-both networks are calibrated on their own training positions to within a point, overrate themselves against the held-out opponents by the same 8 to 10 points as the control and group B, and are within two points of the control on every other row. The one visible difference is in the self-play column's mean outcome, 0.52 against group B's 0.60, and it is a property of the data rather than of the network: group B recorded only the network's side of its external games, which it won 92% of the time, so a fifth of its training positions carried a result near +1; B-both recorded both sides, so each external game contributed a win and a loss, and the mean result of the training set went back to even. The value head's estimate followed in both cases, which is one more reading of section 5: the head learned the average of what it was shown.

## 7. What this establishes

- **The recording rule was not the reason for group B's null.** Recording the opponent's positions trained the value head at the position the search reads, and the value there did not change on either seed. The explanation Step 3 proposed in its section 5 is closed by the test it asked for. The test ran at half the external dose: B-both played 120 external games per generation against group B's 240 (design, section 2), and the probe at the network's own position moved half as far (+0.93 and +0.96 against +0.84 and +0.85), which is what the dose implies; the position one ply later did not move at either dose.
- **The wipeout is not a data-coverage problem.** It is a one-ply tactic in a state that is usually favourable. A value target of any kind reaches that state through the game result, and the game result in that state says, most of the time, that it is fine. The material target of group D moved the value there by brute force, by labelling every low-disc position as bad, and paid for it with every other position's calibration (Step 3 results, section 6).
- **The weakness sits in the search, and the fix does too.** At 50 simulations the exact answer, a terminal position one ply below the child, is never looked at because the prior on the wiping reply is 4% and the child's value says there is nothing to see. The value-head-only search finds the escape in every probed position on every checkpoint, which means the information is already present one ply down; what is missing is a reason to expand that node. A one-ply check for terminal replies at the root's children, or a search budget that reaches them anyway, closes the hole by construction. That is an engineering change to the agent, not an experiment about opponent diversity, and it is outside the Step 3 question.
- **Opponent exposure did produce the states.** Nine thousand low-disc positions against fifty is the one clear thing the opponent mix changed in the training data. The mix did its job at the level of the data; the lesson inside those states was too rare to learn by value regression.

**What it does not establish.** The scorecard resolves about ±6 to 7 points per row and the seeds differ by up to 10, so "did not move" means "did not move by more than that" throughout; the noisy-greedy row leans five points up on both seeds (78.0% and 79.5%, 44 and 41 wipeouts, against the control's 73.5% and 52), inside that resolution. It does not test a larger external share or a longer run; the count in section 5 scales with both, but so does the count of look-alike rows, and the ratio is set by the game, not by the share. It does not test weighting the lesson rows up in the loss, which would be the data-side counterpart of group D's target and would likely inherit its cost in a weaker form. It says nothing about adaptive adversaries; the opponents were fixed.

## 8. What follows

The Step 3 series has now tested the two data-side routes to the Step 2 weakness, an opponent mix and a different target, plus the one-variable repair the first route pointed to. The mix changed nothing that transfers; the target transferred at too high a price; the repair reached the position and found nothing to learn there. The next step is the Phase 1 report ([outline](phase1-report-outline.md)), with this document as its last results section. If any further run is made before the report, it is one of these, in order of cost:

1. **The search-side fix as a measurement, not a remedy.** Add the one-ply terminal check to the evaluation search only, and re-run the Greedy and random-playout rows for the control. If the wipeouts vanish, the agent's weakness is fully characterised: a search-depth artefact, present in every checkpoint, that no amount of self-play training touches. That is a clean finding for the report and costs a few CPU hours.
2. **A second D seed**, to put an interval on the one signal-only result.
3. **Weighting value-only lesson rows**, only if the report needs to say whether the data-side route is closed outright; the expectation from section 5 is a smaller version of group D's trade-off.

## 9. Reproduction

On the cluster, with the two runs under `~/reversi-runs/` and the patch of PR #64 applied in the working copy:

```bash
cp -r ~/reversi-runs/control ~/reversi-runs/step3-Bboth-s1
slurm/submit_chain.sh configs/full8x8_step3_both.yaml step3-Bboth-s1 140 1
EXTRA_ARGS="--set seed=1338" slurm/submit_chain.sh configs/full8x8_step3_both.yaml step3-Bboth-s2 140 1

# evaluation: one job per pairing, as in Step 3 results section 11, with Bboth-s1 and Bboth-s2 as the entrants
python scripts/step3_scorecard.py ~/reversi-runs/step3-eval --markdown --out research/results/step3b/scorecard.json
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol <checkpoints map> \
    --networks Bboth-s1 Bboth-s2 --out research/results/step3b/head-probe.json
python scripts/step3b_lesson_rows.py ~/reversi-runs/step3-Bboth-s1 ~/reversi-runs/step3-Bboth-s2 \
    ~/reversi-runs/step3-B-s1 ~/reversi-runs/step3-B-s2 ~/reversi-runs/step3-A-s1 \
    --from-generation 121 --out research/results/step3b/lesson-rows.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval \n    --self-play control120=~/reversi-runs/control --self-play B-s1=~/reversi-runs/step3-B-s1 \n    --self-play B-s2=~/reversi-runs/step3-B-s2 --self-play Bboth-s1=~/reversi-runs/step3-Bboth-s1 \n    --self-play Bboth-s2=~/reversi-runs/step3-Bboth-s2 --out research/results/step3b/value-calibration.json
```

The checkpoints map is the Step 3 one with two entries added, `Bboth-s1` and `Bboth-s2`, pointing at each run's `gen_00140.pt`. The scorecard and calibration scripts list the Step 3 checkpoints; for these tables their `CHECKPOINTS` tuple was `control120, B-s1, B-s2, Bboth-s1, Bboth-s2`.
