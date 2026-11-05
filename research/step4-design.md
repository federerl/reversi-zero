# Step 4 — The Terminal Check in Self-Play

**Applies to:** the first new experiment named in the [November 2 agenda](meetings/2026-11-02-advisor-meeting-agenda.md), section 4, and the strongest remaining test of the claim in [what this study contributes](contribution.md), section 1
**Depends on:** the [Step 3 design](step3-design.md) (groups, matching rule, scorecard), the [Step 3b design](step3b-design.md) (the B-both recipe this run continues) and the [Step 3c check](step3c-terminal-check.md); nothing in any of them changes here
**Date:** November 3, 2026, written before any run
**Code:** `mcts.terminal_check` in a run's YAML (`src/reversi/config.py`, passed through by `SearchConfig.for_selfplay`), `configs/full8x8_step4_check.yaml`, the `wipeout_rate` self-play counter

## 1. The question

Step 3c switched the check on at evaluation only, on fixed weights, and the wipeouts against Greedy fell from 65 to 12. The network learned nothing; the search read one ply further. The study's claim rests on that: the weakness is a one-move tactic, and the two training routes tried so far, opponent exposure (Step 3) and training the value head where the search reads (Step 3b), could not teach it.

Both of those routes put the *states* into the training data and left the *lesson* for the heads to infer from game results. The check in self-play does something stronger. Every position in every training game is searched with the exact answer to "can a side win on the spot" available one ply down, so the search that writes the policy target has already seen the tactic, and the game then goes the way the exact answer says. The training data are produced by a player that does not fall for the wipeout. If the heads can learn the tactic from any data this training loop can make, these are the data.

So the experiment is a test that could go either way, and either way it is a result. If the heads learn the tactic here, the plain search at evaluation escapes the wipeouts on its own, and the claim narrows: the tactic is learnable from exactly resolved search, just not from opponent exposure. If they do not, the data-side route is closed in its strongest form, and the report's third claim stands with one more exclusion behind it.

## 2. Why the check goes on top of the opponent mix, not on pure self-play

A test of whether the network can learn a lesson needs the positions that carry it to occur in training. In pure self-play they barely do: Step 2 counted a position with one or two discs after the opening about once per hundred games, and the Step 3b row count found 51 network-to-move positions with three discs or fewer in a whole group A replay window of 694,569 rows. Twenty generations of pure self-play with the check on would resolve about a hundred such positions exactly, in seven hundred thousand rows, and a null result could not be told apart from "the network never saw the data".

The Step 3 mix exists to make those positions. Greedy flips as many discs as it can every move, so the network against it spends long stretches with few discs; the B-both runs had 4,573 and 4,171 network-to-move rows at three discs or fewer, ninety times group A's count. That is the stimulus. The check goes on top of the B-both recipe, and the comparison is against the two B-both seeds from Step 3b, which already have the scorecard, the probe, the calibration check and the row counts. One key differs.

## 3. What the check changes in the training games

The Step 2 probe names two positions in a wipeout: **P**, where the network holds a few discs and is to move, and one of its moves *m* lets the opponent take everything; and **C**, the position after *m*, with the opponent to move and the wiping reply available.

- **At P**, whether the opponent is the network itself or Greedy, the search opens C and the check reads the wiping reply exactly: C is worth −1 from P's side, however the network rates it. The visits move away from *m* whenever a safe move exists, so the stored policy target at P moves away from *m*, and the game continues; the value target at P is the result of that continued game. When no safe move exists, every child is proven and the game is lost: the row at P carries −1, which is the truth about that position.
- **At C with Greedy to move**, nothing is written beyond what Step 3b wrote: Greedy does not search, so there is no policy target, and the value-only row (the result from Greedy's side) exists only if the game reaches C. The check makes that rarer, since the network now avoids C when it can.
- **At C with the network to move**, in the self-play games, the wiping child is finished and backs up +1 from its first visit; the check makes the first visit certain, nothing more. These positions are rare either way.
- **At every other node**, the same rule: where a move ends the game, its exact result enters the search one ply earlier than the plain search would reach it.

The net effect on the data: the network is wiped out far less often in the external games (Step 3b's rate was 6.0% and 6.9% of them); the thousands of rows at P-like positions carry policy targets that avoid the fatal move; and the value-only rows at C, which carried the lesson in Step 3b (77 and 83 of them against about 150 that said the opposite and four thousand look-alikes one ply up), become fewer still. The value target is unchanged, the check adds no rows, and no position is labelled by the check itself.

That is why the prediction for the value head is a null: its lesson rows go down, not up. What the check adds is on the policy side, and in quantity: a few thousand rows per run where the target at a low-disc position avoids the move that allows a wipeout. Whether the policy head can turn those into a rule, "when holding few discs, do not play the move after which one reply takes everything", when the rule turns on one specific reply and not on anything visible in the board's features, is the open question. The value head failed the same kind of question in Step 3b; the policy head has more rows to work with here than the value head ever had.

## 4. Groups and matching

| group | config | search during self-play | games per generation |
|---|---|---|---|
| B-both (Step 3b, two seeds, not re-run) | `configs/full8x8_step3_both.yaml` | plain, 200 simulations | 480 self-play + 120 external, both sides recorded |
| **B-both-tc** (two seeds) | `configs/full8x8_step4_check.yaml` | the check on, 200 simulations | the same |

The two configs differ in one key, `mcts.terminal_check`. Everything else is the Step 3b recipe: twenty generations continued from `control120` with its replay window, seeds 1337 and 1338, the same opponents and mix, the same search budget, noise, temperature and trainer. The groups are matched on games and, within noise, on training examples; a game with the check on yields about as many rows as without it, though the external games may run a few plies longer now that fewer of them end early by wipeout. The check costs one `rules.apply` per legal move at every opened node and no network calls; self-play wall-clock is expected to rise by a tenth to a third and is reported.

## 5. Evaluation

The frozen Step 1 protocol on the frozen roster, exactly as for Step 3: 200 games per pairing, 100 seeded openings in both colours, 50 simulations, no exploration noise, seed 20261025, all thirteen scorecard rows.

The scorecard is measured **with the plain search**. `SearchConfig.for_evaluation` leaves the check off whatever the run's YAML says, so the arena plays the trained weights without it; that is the transfer question, whether the plain search now escapes on its own. One extra column, B-both-tc seed 1 with the check on at evaluation, on the Greedy and random-playout rows only, confirms the check still adds what it added on the control.

Also re-run on both final checkpoints: the head probe (value at P, value at C, prior on the wiping reply at C, prior on the move played at P, the share of prior on safe moves at P, escape rate of the plain and the checked search at 50 simulations), the value calibration check, and the Step 3b row count (`scripts/step3b_lesson_rows.py`), so the lesson and look-alike counts sit beside the B-both ones.

The training side is measured from the self-play metrics log, which Step 3b already wrote: `opponent_wipeout_rate`, the share of external games the network ended with no discs, per generation. The new `wipeout_rate` counter (all games, either side) is logged beside it.

## 6. Predictions, written before the runs

The B-both numbers these are read against, from the Step 3b results: Greedy wipeouts 61 and 69 per 200 games, random-playout wipeouts 36 and 37, the network wiped out in 6.0% and 6.9% of the external training games, prior on the wiping reply at C 4.6% and 4.4%, value at C −0.98 and −0.99, value at P +0.93 and +0.96, prior on the move played at P 29% and 28%.

1. **The training games change.** The network's wipeout rate in the external games falls from 6 to 7% to under 1% from the first generation on. This is the check that the switch reached the workers; if it fails, the run is stopped and nothing below is read.
2. **The value head does not move.** At the recorded fatal positions, the value at C stays below −0.9 and the value at P stays above +0.9. The lesson rows at C are fewer than in Step 3b, and the look-alikes are unchanged.
3. **The policy head moves at P, not at C.** The prior on the fatal move at P falls below 20% for both seeds, since thousands of targets there now avoid it; the prior on the wiping reply at C stays within 2 to 6%, since nothing trains the policy head at C.
4. **The plain search still loses most of the same games.** Greedy wipeouts for B-both-tc stay above 45 for both seeds; the random-playout row stays within 26 to 39; the Edax rows and the generation-60 retention row stay within seed noise of B-both. The reasoning is section 3's last paragraph: the rule the policy head would need turns on one reply, not on the board's features. This is the prediction held with the least confidence, and it is the one that matters.
5. **With the check on at evaluation, B-both-tc reads like `control120-tc`:** about 12 wipeouts against Greedy and 94% score, about 12 against the random-playout search.
6. **No cost.** Retention, the Edax rows and the self-play calibration gap stay where B-both has them. The training differs only in what the search resolves exactly, and the value target is unchanged.

The readable outcomes. All six hold: the data-side route is closed in its strongest form, and the report says so. Prediction 4 fails, with Greedy wipeouts below 30 for both seeds under the plain search: the heads learned the tactic from exactly resolved search, the claim narrows to "learnable from exact search, not from opponent exposure", and the probe says which head carried it; prediction 3 holding with 4 failing points at the policy head. Prediction 3 fails with 4 holding: the policy head learned the positions it saw and not the rule, which is the policy-side twin of the Step 3b value result. Prediction 1 fails: an implementation fault, fixed before anything else is run. Prediction 6 fails: the check in self-play has a strength cost, reported as such; it would need its own explanation, since the value target did not change.

## 7. Cost

Two runs of twenty generations, about 1.3 GPU-hours each by the Step 3b timings plus the check's overhead; 26 evaluation pairings on CPU, of which the two random-playout pairings take several hours each, plus two checked pairings; the probe, the calibration check and the row count. About a week including the write-up.

## 8. How a run is launched

As in the Step 3 design, section 7, with the Step 4 config:

```bash
cp -r runs/control runs/step4-Bboth-tc-s1       # checkpoints up to gen_00120 and the replay window
reversi train -c configs/full8x8_step4_check.yaml --run-id step4-Bboth-tc-s1 --generations 140 --device cuda \
    --set selfplay.n_workers=22
```

The second seed adds `--set seed=1338`. On the cluster the runs go through `slurm/submit_chain.sh`, with the seed passed as `EXTRA_ARGS="--set seed=1338"`. The training loop records the changed config hash in the run's `meta.json`, as it did for every Step 3 group. `configs/full8x8_step4_check.yaml` is `configs/full8x8_step3_both.yaml` with one key added and nothing else, so the diff is the experiment.
