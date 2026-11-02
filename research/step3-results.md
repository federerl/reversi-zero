# Step 3 — Results: Playing Greedy in Training Does Not Transfer, and Barely Takes

**Applies to:** Step 3 of the [experiment design proposal](experiment-design-proposal-2026-10.md), run as [step3-design.md](step3-design.md) pre-registered it, on the roster frozen in [held-out-opponents.md](held-out-opponents.md)
**Trained:** October 28–29, 2026, on the cluster, one L40S and 22 self-play workers per run, commit `0d699f4`
**Evaluated:** October 29, 2026, on the cluster's CPU partition: 78 pairings of 200 games, one job each
**Reports:** [`results/step3/scorecard.json`](results/step3/scorecard.json) (every pairing's score, interval and wipeout count), [`results/step3/head-probe.json`](results/step3/head-probe.json), [`results/step3/value-calibration.json`](results/step3/value-calibration.json). The 78 arena reports and 15,600 recorded games (76 MB) are under `~/reversi-runs/step3-eval/` on the cluster and are not committed; the tables below are produced from them by the scripts named in section 11.
**Gate C: a controlled negative result.** The training ran as specified. Group B, which played Greedy and Minimax-d4 for a fifth of its training examples, is no better against any held-out opponent than the control, beyond what one seed shows and the other does not, and is barely better against Greedy itself. The head probe says why, and the reason is a property of how the external games were recorded, not of opponent diversity as such. The random-playout search, a held-out opponent with no knowledge of any kind, turns out to exploit the Step 2 weakness as Greedy does; group B does not fix that either, and group D's material target cuts it almost in half while costing about half the network's strength against Edax.

## 1. What was run

Five runs, each a copy of the control run at generation 120 continued for 20 generations under its group's config. The second seed of group D was not run; one seed was enough to read its cost.

| run | group | seed | games | of which against Greedy or Minimax-d4 | training positions | gradient steps | wall time | network's score in the external games | wiped out in the external games |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| step3-A-s1 | A, control | 1337 | 12,000 | 0 | 694,569 | 16,000 | 1.45 h | | |
| step3-A-s2 | A, control | 1338 | 12,000 | 0 | 694,805 | 16,000 | 1.49 h | | |
| step3-B-s1 | B, opponent mix | 1337 | 14,400 | 4,800 | 694,800 | 16,000 | 1.33 h | 92.0% | 7.4% |
| step3-B-s2 | B, opponent mix | 1338 | 14,400 | 4,800 | 695,793 | 16,000 | 1.33 h | 92.8% | 6.8% |
| step3-D-s1 | D, signal only | 1337 | 12,000 | 0 | 693,276 | 16,000 | 1.48 h | | |

The example counts match to within a fifth of a percent, as the design intended: an external game records only the network's moves, so 480 self-play games plus 240 external games yield as many positions as 600 self-play games. Every run trained for the same 16,000 gradient steps on batches of 1,024. The replay window holds about twelve generations, so by generation 140 every position the trainer sampled was produced under the group's own recipe.

Group B's external games were played with the self-play search settings (200 simulations, exploration noise, a temperature for the first fifteen plies). In those games the network was wiped out in 7% of its games against the two opponents together, every generation from 121 to 140, with no downward trend: the rate in generation 140 (5.8% and 7.9% for the two seeds) is the rate in generation 121 (6.3% and 8.3%). Section 5 returns to this.

The evaluation is the frozen Step 1 protocol: 200 games per pairing, 100 seeded 4-ply openings each played with both colours, 50 simulations per move, no exploration noise, seed 20261025. Each pairing is its own two-entrant arena job with every game recorded, so the random-playout opponent, slower than every other entrant by an order of magnitude (its pairings took two and a half to six hours each), runs beside the cheap pairings rather than after them.

## 2. The scorecard

Scores are the checkpoint's wins plus half its draws over 200 games, with 95% Wilson intervals; "wo" is the number of those games the checkpoint lost with no discs left. `control120` is the checkpoint every run started from.

| row | opponent | control120 | A-s1 | A-s2 | B-s1 | B-s2 | D-s1 |
|---|---|---|---|---|---|---|---|
| **held out** | Edax L7 | 10.0% [6.6, 14.9], 9 wo | 14.8% [10.5, 20.3], 6 wo | 8.0% [5.0, 12.6], 8 wo | 9.8% [6.4, 14.7], 4 wo | 13.8% [9.7, 19.2], 5 wo | 4.0% [2.0, 7.7], 7 wo |
| **held out** | Edax L5 | 29.8% [23.8, 36.4], 0 wo | 30.0% [24.1, 36.7], 1 wo | 29.0% [23.2, 35.6], 1 wo | 29.5% [23.6, 36.2], 3 wo | 40.2% [33.7, 47.2], 3 wo | 17.2% [12.6, 23.1], 1 wo |
| **held out** | Edax L3 | 72.0% [65.4, 77.8], 0 wo | 73.2% [66.7, 78.9], 2 wo | 63.2% [56.4, 69.6], 0 wo | 68.5% [61.8, 74.5], 3 wo | 68.5% [61.8, 74.5], 1 wo | 58.5% [51.6, 65.1], 2 wo |
| **held out** | rollout-r3000 | 81.0% [75.0, 85.8], 34 wo | 85.8% [80.2, 89.9], 26 wo | 79.8% [73.6, 84.7], 38 wo | 88.0% [82.8, 91.8], 21 wo | 80.5% [74.5, 85.4], 39 wo | 90.5% [85.6, 93.8], 19 wo |
| retention | gen60 | 62.5% [55.6, 68.9], 0 wo | 65.0% [58.2, 71.3], 0 wo | 61.3% [54.3, 67.7], 0 wo | 57.8% [50.8, 64.4], 0 wo | 60.8% [53.8, 67.3], 0 wo | 40.8% [34.2, 47.7], 0 wo |
| seen | Greedy | 67.5% [60.7, 73.6], **65 wo** | 67.0% [60.2, 73.1], **66 wo** | 68.5% [61.8, 74.5], **63 wo** | 73.5% [67.0, 79.1], **53 wo** | 68.0% [61.2, 74.1], **64 wo** | 78.0% [71.8, 83.2], **44 wo** |
| seen | Minimax-d4 | 99.0% [96.4, 99.7], 0 wo | 97.8% [94.6, 99.1], 0 wo | 96.5% [93.0, 98.3], 0 wo | 99.5% [97.2, 99.9], 0 wo | 98.5% [95.7, 99.5], 0 wo | 96.5% [93.0, 98.3], 0 wo |
| seen family | greedy-corner | 63.5% [56.6, 69.9], 72 wo | 72.0% [65.4, 77.8], 56 wo | 66.5% [59.7, 72.7], 67 wo | 64.5% [57.7, 70.8], 71 wo | 65.5% [58.7, 71.7], 69 wo | 74.5% [68.0, 80.0], 51 wo |
| seen family | greedy-d2 | 59.0% [52.1, 65.6], 82 wo | 57.0% [50.1, 63.7], 86 wo | 52.0% [45.1, 58.8], 96 wo | 58.0% [51.1, 64.6], 84 wo | 61.0% [54.1, 67.5], 78 wo | 69.0% [62.3, 75.0], 62 wo |
| seen family | noisy-greedy | 73.5% [67.0, 79.1], 52 wo | 81.5% [75.5, 86.3], 37 wo | 77.5% [71.2, 82.7], 45 wo | 74.0% [67.5, 79.6], 51 wo | 67.0% [60.2, 73.1], 66 wo | 86.0% [80.5, 90.1], 28 wo |
| seen family | corner | 98.0% [95.0, 99.2], 4 wo | 95.0% [91.0, 97.3], 10 wo | 97.0% [93.6, 98.6], 5 wo | 98.0% [95.0, 99.2], 4 wo | 98.5% [95.7, 99.5], 3 wo | 99.0% [96.4, 99.7], 2 wo |
| seen family | mobility | 91.5% [86.8, 94.6], 17 wo | 94.0% [89.8, 96.5], 12 wo | 93.5% [89.2, 96.2], 12 wo | 91.0% [86.2, 94.2], 18 wo | 94.0% [89.8, 96.5], 12 wo | 94.5% [90.4, 96.9], 11 wo |
| sanity | Random | 100% [98.1, 100], 0 wo | 99.5% [97.2, 99.9], 1 wo | 99.0% [96.4, 99.7], 2 wo | 98.5% [95.7, 99.5], 3 wo | 99.0% [96.4, 99.7], 2 wo | 99.5% [97.2, 99.9], 1 wo |

**How much is seed.** The two control seeds are the noise floor: the same recipe, the same starting point, different random numbers. They differ by 10 points against Edax L3 (73.2% against 63.2%), 7 against Edax L7, 9 against greedy-corner, 5 against greedy-d2 and 4 against gen60, with the wipeout counts against Greedy at 66 and 63. A difference between groups smaller than that, on one row, is not a difference.

**The control did not move.** Twenty more generations of the control recipe leave every row where `control120` had it, within seed noise. This is the Step 1 finding again (sixty generations did not touch the wipeouts) at a shorter range, and it is what makes the other groups readable: whatever moved, moved because of the recipe.

## 3. The predictions, one by one

The design fixed five predictions before the runs (section 5 there). Three of five failed.

**1. "Group B's wipeouts against Greedy fall sharply, to a handful per 200 games."** False. 65 became 53 for one seed and 64 for the other, against 66 and 63 for the two control seeds. The one-seed drop of 12 is at the edge of what seeds produce. Group B's score against Greedy rose by 6 points for one seed and not at all for the other.

**2. "Against the seen-family opponents, B improves in proportion to how closely each resembles Greedy."** False, because nothing improved: greedy-corner 64.5% and 65.5% against the control's 63.5%, noisy-greedy 74.0% and 67.0% against 73.5%, with wipeout counts within a few of the control's.

**3. The held-out rows.** Of the three outcomes the design named, the second is the closest, with a qualification: B improved on the seen rows only slightly, and on the held-out rows not at all. Against Edax L7, L5 and L3 the two B seeds sit inside the range of the two A seeds, except B-s2's 40.2% against Edax L5, which B-s1 (29.5%) does not share. Against the random-playout search, the held-out opponent that attacks the same weakness (section 4), B-s1 scores 88.0% with 21 wipeouts and B-s2 80.5% with 39; the control scored 81.0% with 34, and the A seeds 85.8% with 26 and 79.8% with 38. The B seeds sit inside the A seeds' range on an opponent B never saw. Retention: B's scores against gen60, 57.8% and 60.8%, are 2 to 5 points below the control's 62.5% and below both A seeds (61.3% and 65.0%), inside every interval but on the low side twice; a small cost is not excluded.

**4. Group D.** Half right. D cut the wipeouts against Greedy from 65 to 44, a third, which is the amount the ownership head managed (72 to 35 and to 47 in Step 2), and the cut carries across the Greedy family: 72 to 51 against greedy-corner, 82 to 62 against greedy-d2, 52 to 28 against noisy-greedy. Its retention row fell, as predicted, but by more than "a measurable amount": 62.5% to 40.8% against gen60, 29.8% to 17.2% against Edax L5, 10.0% to 4.0% against L7, 72.0% to 58.5% against L3. Whether D would move the held-out rows was left open; it moved them both ways. Against the random-playout search it was wiped out 19 times, below the control's 34 and both A seeds' 26 and 38, and scored 90.5% against their 81.0%, 85.8% and 79.8%, the one held-out gain in the experiment; against Edax it lost at every level. The signal-only route is the costlier one by a wide margin.

**5. The head probe.** Half right, and the half that was wrong is the finding of this step. Section 5.

## 4. The random-playout search finds the same hole, knowing nothing

`rollout-r3000` is a tree search with no network and no hand-written knowledge: every prior equal, each leaf judged by one random playout to the end of the game, 3,000 playouts a move. It is on the roster as the held-out opponent furthest from Greedy by construction (roster, section 2), and the calibration there put it at even against Minimax-d4. The networks beat it four games in five, and lose to it the way they lose to Greedy.

| network | score vs rollout-r3000 | losses | of which wipeouts | moves at the wipeout: shortest / median / longest | raw value at the last move | a surviving move existed at the last move, against any reply |
|---|---:|---:|---:|---|---:|---:|
| control120 | 81.0% | 38 | **34** | 8 / 17.5 / 33 | +0.99 | 29 of 34 |
| A-s1 | 85.8% | 28 | **26** | 9 / 20 / 35 | +0.99 | 21 of 26 |
| A-s2 | 79.8% | 40 | **38** | 6 / 16 / 27 | +0.99 | 35 of 38 |
| B-s1 | 88.0% | 24 | **21** | 5 / 14 / 33 | +0.80 | 19 of 21 |
| B-s2 | 80.5% | 39 | **39** | 5 / 15 / 35 | +0.77 | 33 of 39 |
| D-s1 | 90.5% | 19 | **19** | 7 / 17 / 25 | +0.26 | 16 of 19 |

Nearly every loss is a wipeout, half of them within fifteen to eighteen moves of the opening, with the network's raw value at its last move at +0.99 while it holds one disc, and a surviving move available in six of seven cases even if the opponent could choose any legal reply. This is the Step 2 pattern to the digit, from an opponent that has never heard of disc counts. A random playout from a position where one side holds one or two discs ends, most of the time, with that side wiped out a move or two later, so the random-playout search rates wiping moves as near-certain wins and steers toward them. Greedy has the same preference by construction; the random-playout search has it because the preference is correct. Step 2 section 8 concluded that the loss is "not specific to Greedy" by showing it against two other hand-written disc-hungry opponents; this is the same conclusion from an opponent with no design at all, and it is what makes the row usable: it is the one held-out opponent on the roster that measures the weakness Step 3 set out to fix.

Read that way, the row says what the Greedy row said. The A seeds were wiped out 26 and 38 times, the B seeds 21 and 39: no separation between the groups, and a 12-game spread within each. Group D was wiped out 19 times, below both A seeds and the control. The transfer question the design asked has an answer for D and a null for B, and the null has the same cause as B's null against Greedy itself (section 5).

## 5. Why group B did not learn: the value head was never trained where the search reads it

The Step 2 probe (failure analysis, section 7) is re-run on every wipeout each checkpoint suffered against Greedy, at the network's last move, in the positions where a surviving move existed. Two readings matter: the raw value at the network's own position, and the raw value one ply later, from Greedy's side, after the move the network actually played. That second position is the one the search uses to judge the move: when the search considers a move, it asks the network about the position the move leads to, from the opponent's point of view. There, the exact answer is +1: Greedy is about to win on the spot.

| network | wipeouts probed | raw value at the fatal position | one ply later, raw value for Greedy (exact: +1) | prior on Greedy's wiping reply | prior on the move played | prior on all safe moves |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 63 | +0.99 | −0.99 | 3.6% | 36% | 44% |
| A-s1 | 66 | +0.98 | −0.99 | 4.5% | 38% | 37% |
| A-s2 | 61 | +0.98 | −0.98 | 2.9% | 39% | 33% |
| B-s1 | 52 | **+0.84** | **−0.98** | 4.6% | 25% | 44% |
| B-s2 | 63 | **+0.85** | **−0.98** | 4.3% | 30% | 49% |
| D-s1 | 42 | +0.28 | −0.37 | 6.0% | 33% | 42% |

The prediction was that B's raw value at the fatal position would fall well below +0.99 while the prior on the wiping reply changed little. The prior did not change. The value at the network's own position did fall, from +0.99 to +0.84 and +0.85: the 4,800 external games reached the network. But one ply later, at the position where Greedy is to move, nothing moved: −0.98 against the control's −0.99, the sign still backwards, on both seeds.

That position was never a training example. The design records only the network's moves in an external game, on the grounds that the opponent's moves have no search behind them to learn a policy from (section 3 of the design). The value target is a different matter: a position with Greedy to move and the network one disc from the board has the clearest value target in the game, and it is exactly the position the search consults. In self-play both sides are the network, so both sides' positions are recorded and the value head is trained at every node the search will later visit. In B's external games, half the nodes on the search's path were positions it never trained on, and the half left out was the half that carries the lesson.

So the value head learned something about the positions it was shown: the network-to-move position one ply earlier now reads +0.84 instead of +0.99. It could not learn what it was not shown, and the search reads the position it was not shown. The weak movement at the position it *was* shown is consistent with how few examples carried the lesson: at a 7% wipeout rate, each seed put about 340 wipeout games into the window over twenty generations, and only the last one or two positions of each are the fatal ones, a few hundred examples in a window of 400,000.

This also explains the flat in-training wipeout rate in section 1. The external games were played at 200 simulations with exploration noise, which is why the rate there (7%) is a quarter of the evaluation's rate at 50 simulations (26% to 33%), as Step 2's budget experiment predicts; it stayed flat because the search that avoids the wipeout reads a value head that the external games did not touch where it counts.

**The search-budget result holds on every checkpoint.** The same positions searched three ways, as recorded, with the prior flattened (value head only) and with the value zeroed (policy head only):

| network | positions | recorded, 50 sims | value only, 50 | policy only, 50 | recorded, 256 sims | value only, 256 | policy only, 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control120 | 63 | 0% | 98% | 5% | 56% | 100% | 51% |
| A-s1 | 66 | 0% | 100% | 3% | 59% | 100% | 58% |
| A-s2 | 61 | 0% | 100% | 3% | 48% | 100% | 46% |
| B-s1 | 52 | 0% | 100% | 12% | 62% | 100% | 63% |
| B-s2 | 63 | 0% | 100% | 8% | 59% | 100% | 57% |
| D-s1 | 42 | 0% | 100% | 2% | 79% | 100% | 79% |

The cells are the share of positions in which the search picks a surviving move. At 50 simulations the value head alone finds the escape in every position and the policy head alone almost never does; at 256 the recorded search finds it half to three quarters of the time. Step 2's reading stands for all six networks: the prior is what lets the loss through at 50 simulations, and the value head sees the disaster one ply too late to matter.

## 6. Group D: what the material term bought and what it cost

D's value target is half the game's result and half the position's disc share. The probe shows the target doing what it was designed to do: the raw value at the fatal position is +0.28 instead of +0.99, and one ply later, from Greedy's side, −0.37 instead of −0.99. Neither has the right sign (the exact answers are about −1 and +1), but both moved by two thirds of the scale, and the search at 256 simulations now finds the escape in 79% of the positions against the control's 56%. The wipeouts against Greedy fell by a third, and against every opponent in the Greedy family.

The cost shows in the calibration check (section 8) before it shows in the scorecard. D's raw value is no longer a win probability: against Greedy and Minimax-d4 its mean estimate is 0.69 on the probability scale where the games were won 95% of the time, and against Random 0.77 where every game was won. The share term is near zero in most positions, so half of every target is near zero, and the value head's output is compressed toward the middle. A search reading a compressed value head has a weaker signal to separate good moves from bad ones everywhere, not only near wipeouts, and that is the retention loss: 62.5% to 40.8% against the generation-60 checkpoint, and about half the score against Edax L5 and L7.

The ownership head's cut of the same size (Step 2, section 5) came without this cost, because it added sixty-four targets beside the value target rather than diluting it. The lesson of D is that where the signal goes matters as much as what it says.

## 7. Edax wipeouts are a different loss

The wipeout descriptors are run against Edax L7 as well as Greedy, with every legal reply treated as a possible Edax move, which makes "avoidable" a lower bound. The 39 wipeouts the six checkpoints suffered against Edax L7 look nothing like the Greedy wipeouts: at the network's seventh move before the end it still held 10 to 19 discs and its searched value was already −1.00; in 37 of the 39 no surviving line was available at the last, second-last or third-last move, and in the other two only at one of those moves. These are endgames lost to a stronger engine that happens to finish by taking everything, not the Step 2 blind spot. The Edax wipeout counts in the scorecard are reported because the design says to report them; they do not measure the weakness Step 3 is about.

## 8. Value calibration

Every recorded network estimate from the evaluation games, binned by its value and set against how the game ended, both on a win-probability scale. "Gap" is the count-weighted mean absolute difference between a bin's mean estimate and its mean outcome over ten bins. The self-play column is the network's own last generation of training positions, 20,000 sampled, with the result those games reached; for group B a fifth of those positions come from its external games, which is why its mean outcome there is 0.60 rather than 0.50.

| network | self-play positions | against held-out opponents | against gen60 | against Greedy and Minimax-d4 | against Random |
|---|---|---|---|---|---|
| control120 | est. 0.51, won 0.51, gap 0.01 | est. 0.59, won 0.50, gap 0.09 | est. 0.57, won 0.63, gap 0.08 | est. 0.90, won 0.95, gap 0.10 | est. 0.96, won 1.00, gap 0.05 |
| A-s1 | est. 0.52, won 0.51, gap 0.02 | est. 0.60, won 0.53, gap 0.08 | est. 0.58, won 0.65, gap 0.08 | est. 0.89, won 0.94, gap 0.09 | est. 0.96, won 1.00, gap 0.04 |
| A-s2 | est. 0.51, won 0.51, gap 0.02 | est. 0.59, won 0.47, gap 0.12 | est. 0.58, won 0.61, gap 0.04 | est. 0.89, won 0.94, gap 0.09 | est. 0.96, won 1.00, gap 0.04 |
| B-s1 | est. 0.59, won 0.60, gap 0.02 | est. 0.60, won 0.50, gap 0.10 | est. 0.57, won 0.58, gap 0.06 | est. 0.91, won 0.96, gap 0.07 | est. 0.96, won 1.00, gap 0.04 |
| B-s2 | est. 0.60, won 0.60, gap 0.02 | est. 0.61, won 0.53, gap 0.08 | est. 0.60, won 0.61, gap 0.06 | est. 0.91, won 0.95, gap 0.07 | est. 0.96, won 1.00, gap 0.04 |
| D-s1 | est. 0.48, won 0.51, gap 0.11 | est. 0.51, won 0.44, gap 0.17 | est. 0.47, won 0.41, gap 0.12 | est. 0.69, won 0.95, gap 0.26 | est. 0.77, won 1.00, gap 0.23 |

On their own training positions the five self-play networks are calibrated to within two points. Every one of them overrates itself against the held-out opponents by 7 to 12 points of win probability; against Edax alone it expects an even game and gets a third. That is the same for A and B, so B did not pay for its opponent exposure with overconfidence, which was the cost the design said to watch for. Against gen60 the control and the A seeds undersell themselves by 4 to 7 points and the B seeds are calibrated to within a point. D is the exception everywhere, in the direction section 6 describes: it undersells itself by a quarter against opponents it beats, because its output is no longer trying to be a probability.

## 9. What this establishes

- **Twenty generations of a 20% opponent mix do not transfer.** Not to Edax at any level, not to the random-playout search that attacks the very weakness the mix was meant to fix, and not within the Greedy family. The held-out rows are flat to within seed noise, on two seeds.
- **They barely take, either.** The seen-opponent gain the design expected as a matter of course (prediction 1) did not arrive. The mechanism is identified: the external games never trained the value head at opponent-to-move positions, and those are the positions the search reads. This is a property of the recording rule, and it is testable with one change.
- **The negative result is controlled.** The control seeds bound the noise, the example counts and gradient steps match, every run started from the same weights and the same replay window, and the evaluation is the frozen protocol with the same openings per pairing.
- **A better target moves the value head where opponent games could not, and the gain transfers, at a price.** D reached the opponent-to-move position because its target applies to every position in the window, both sides included; its wipeout cut shows against Greedy, against the Greedy family and against the random-playout search it never saw. It paid with a value head that is no longer a probability and with about half its strength against Edax and the generation-60 checkpoint.
- **The weakness itself is opponent-general.** A search with no knowledge at all, judging positions by random playouts, wipes the self-play networks out in a sixth to a fifth of their games and loses nearly every other game. Step 2's "any disc-hungry opponent" is wider than it sounded: any opponent that judges a position by what actually happens in it finds the hole.
- **The Step 2 mechanism holds on six checkpoints.** The value head alone escapes every wipeout at 50 simulations; the policy head alone escapes almost none; more search closes half the gap.

**What it does not establish.** It does not show that opponent diversity cannot transfer; it shows that this way of adding it did not reach the part of the network that needed it. It does not show that a larger share, a longer run or Edax in the mix would do better; the share and length were fixed in advance, and the mechanism points at the recording rule, not the amount. It says nothing about adaptive adversaries; the opponents were fixed. And group D is one seed.

## 10. What follows

Two one-variable follow-ups fall out of section 5, in order of cost:

1. **Record both sides of an external game for the value target.** The opponent's positions get the game's result as their value target and no policy target (the policy loss is masked where there is no visit distribution). Nothing else about group B changes. If the wipeouts against Greedy then fall as prediction 1 expected, the Step 3 question can be asked properly: a network that has learned the seen opponents is the precondition for asking whether that learning transfers, and this experiment never produced one.
2. **A smaller material weight for D**, 0.1 or 0.2 in place of 0.5, to find whether the wipeout cut survives at a cost the retention row can bear; or the material term as a third head beside the value head, which is what the ownership head's result suggests.

Group C (Edax in the mix) stays deferred. Edax does not produce the positions the mechanism needs, and section 7 shows its wipeouts are a different loss.

## 11. Reproduction

On the cluster, from `~/reversi-zero`, with the five runs under `~/reversi-runs/`:

```bash
# one job per pairing, 78 in all; the two entrants and the output name change
sbatch --cpus-per-task=1 --mem=6G slurm/cpu.sbatch uv run reversi arena --suite custom \
    -e B-s1=$HOME/reversi-runs/step3-B-s1/checkpoints/gen_00140.pt -e edax-l7 \
    --games 200 --simulations 50 --workers 1 --seed 20261025 --record-games \
    --out $HOME/reversi-runs/step3-eval/B-s1-vs-edax-l7.json

python scripts/step3_scorecard.py ~/reversi-runs/step3-eval --out research/results/step3/scorecard.json --markdown
python scripts/analyze_wipeouts.py ~/reversi-runs/step3-eval --opponent greedy
python scripts/analyze_wipeouts.py ~/reversi-runs/step3-eval --opponent edax-l7 --replies any
python scripts/probe_heads.py ~/reversi-runs/step3-eval --protocol <checkpoints map> \
    --out research/results/step3/head-probe.json
python scripts/value_calibration.py ~/reversi-runs/step3-eval --self-play B-s1=~/reversi-runs/step3-B-s1 ... \
    --out research/results/step3/value-calibration.json
```

The checkpoints map is a JSON file with a `checkpoints` object of evaluation name to checkpoint path, in the shape of [`eval-protocol-frozen.json`](eval-protocol-frozen.json).
