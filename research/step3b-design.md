# Step 3b — Recording the Opponent's Positions

**Applies to:** the follow-up proposed in [Step 3 results](step3-results.md), section 10, item 1
**Depends on:** the [Step 3 design](step3-design.md) (groups, matching rule, scorecard) and the [frozen roster](held-out-opponents.md); nothing in either changes here
**Date:** October 31, 2026, written before any run
**Code:** `selfplay.record_opponent_positions` (`src/reversi/selfplay/opponents.py`), `configs/full8x8_step3_both.yaml`

## 1. The one change

Step 3's group B played a fifth of its training data against Greedy and Minimax-d4 and recorded, from those games, only the positions the network moved from. The head probe (Step 3 results, section 5) found that the value head learned at those positions (+0.99 to +0.84 at the position before the fatal move) and not one ply later, at the position with Greedy to move (−0.98 before and after, where the exact answer is +1). That later position is the one the search reads when it judges the fatal move, and it was never a training example.

Group B-both plays the same mix and also records the positions the opponent moves from. Each such position is labelled with the game's result from the opponent's side, as every position is labelled from its mover's side, and carries no policy target: no search ran there, so there is no visit distribution to learn, and the loss skips the policy term on that row. A fatal position with Greedy to move in a game the network lost by wipeout is therefore labelled +1, which is what the search needs to read there.

The policy head is deliberately not trained at these positions. The alternative, a one-hot target on the move the opponent played, would raise the prior on Greedy's wiping reply, but the policy head is one function of the board with no notion of who is to move: trained to output Greedy's move where Greedy moved, it would prefer Greedy's move where the network moves too. If the value head is right one ply down, the search does not need the prior to find the danger: the child position itself reads as lost.

## 2. Groups and matching

| group | games per generation | recorded from external games | config |
|---|---|---|---|
| A (Step 3) | 600 self-play | — | `configs/full8x8.yaml` |
| B (Step 3) | 480 self-play + 240 external | network's positions | `configs/full8x8_step3_mix.yaml` |
| **B-both** | 480 self-play + 120 external | both sides' positions, the opponent's value-only | `configs/full8x8_step3_both.yaml` |

Step 3 matched groups on training examples, not games. An external game that records both sides yields about as many examples as a self-play game, so B-both plays 120 external games rather than 240 to keep the total near 600 self-play games' worth and the external share at a fifth. Of the 120, 60 are against Greedy and 60 against Minimax-d4, 30 as each colour. Everything else is the Step 3 recipe: twenty generations continued from `control120` with its replay window, two seeds (1337 and 1338), the same search budget, noise and trainer.

The groups A and B are not re-run; B-both is compared against the Step 3 checkpoints on the same scorecard.

## 3. Evaluation

The frozen Step 1 protocol on the frozen roster, exactly as for Step 3: 200 games per pairing, 100 seeded openings in both colours, 50 simulations, no exploration noise, seed 20261025. The scorecard rows are unchanged (held out: Edax L7/L5/L3, random-playout search; retention: generation 60; seen: Greedy, Minimax-d4; seen family; sanity). The head probe and the value calibration check are re-run on both final checkpoints.

## 4. Predictions, written before the runs

1. **The probe moves where Step 3's did not.** At the recorded fatal positions, the value at the position with Greedy to move rises from −0.98 toward +1. The prior on the wiping reply stays within a few points of 4 to 5%, because nothing trains the policy head there. If the value does not move, the recording rule was not the reason for group B's null, and that is reported.
2. **Wipeouts against Greedy fall sharply**, from the Step 3 range of 53 to 65 per 200 games to well under half of that. This is Step 3's first prediction, now with the mechanism in place, and it is the seen-opponent result, not evidence of transfer.
3. **The random-playout row moves.** It is the one held-out opponent that attacks the same weakness; its wipeouts (34 for the control, 21 and 39 for group B) fall. This is the transfer result the design asks about.
4. **The Edax rows do not move** beyond seed noise. Learning where the value head was blind about low disc counts says nothing about Edax's endgame, and a gain there would need another explanation.
5. **Retention and calibration hold.** The value target is still the game result, so the generation-60 row stays near the control's 62.5% and the self-play calibration gap stays at or under 0.02. A fall in retention comparable to group D's would mean the extra examples cost general strength, and is reported as such.

The readable outcomes: predictions 1 to 3 hold and 4 to 5 hold is the result Step 3 was designed to find, reached by the recording rule. Prediction 1 holds and 2 fails means the value head learned the position and the search at 50 simulations still does not act on it, which points at the prior after all. Prediction 1 fails and the rest is flat is the null, and closes the recording-rule explanation.

## 5. Cost

Two runs of twenty generations, about 1.5 GPU-hours each on an L40S by the Step 3 timings; 26 evaluation pairings on CPU, of which the two random-playout pairings take several hours each. About a week including the probe and the calibration check.

## 6. How a run is launched

As in the Step 3 design, section 7, with `configs/full8x8_step3_both.yaml` in place of the mix config. The checkpoint and replay window are copied from `control120`; the second seed adds `--set seed=1338`.
