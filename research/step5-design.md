# Step 5 — Searching the Opponent's Positions

**Applies to:** the open question the [Step 4 design](step4-design.md) names in its section 3 and the Step 4 result answers: the value head cannot carry the one-move lesson, and the check in self-play removes the lesson rows rather than adding them. This step gives the *policy* head the lesson, at the position where the search reads it.
**Depends on:** the [Step 3 design](step3-design.md) (groups, matching rule, scorecard) and the [Step 3b design](step3b-design.md) (the B-both recipe this run continues); nothing in either changes here
**Date:** November 4, 2026, written before any run
**Code:** `selfplay.search_opponent_positions` (`src/reversi/selfplay/opponents.py`, `worker.py`, `game_batch.py`), the `opponent_to_move` shard column (`src/reversi/data/schema.py`), `configs/full8x8_step5_search.yaml`

## 1. The question

Every training route so far has tried to teach the **value head** that the position with Greedy to move and the network holding a disc or two is lost. Step 3 never trained it there; Step 3b did, with value-only rows, and it did not move; Step 4 had the search resolve those positions exactly during training, and that made the rows rarer, not more numerous, because the network stopped walking into them.

What the plain search needs one ply down is not, in the first place, the value. The Step 2 head experiment showed that a search with the value head alone escapes every wipeout, and the Step 3c probe shows why the full search does not: the **prior** on the opponent's wiping reply is 4%, so at 50 simulations the reply is never expanded, and the exact −1 that sits one ply below is never read. Raise that prior and the plain search reads the terminal position on its first visit to the child, whatever the value head says about it.

No run has ever trained the policy head at those positions, because no search ever ran there: in an external game the opponent moves, and the opponent does not search. This step runs the network's own search at each recorded opponent position, exactly as at the network's own positions, and stores its visit distribution as the row's policy target. The opponent still plays its own move. At a fatal position the search, with 200 simulations and root noise, finds the game-ending reply and its target is concentrated on it; the row then says, from the opponent's seat, "the move here is the one that takes everything".

The question is whether the policy head learns that from the rows it gets, in a way that reaches the positions the evaluation games produce. If it does, the plain search at evaluation escapes the wipeouts without any change to the search, and the claim narrows to "the tactic is learnable by the policy head from searched positions, not by the value head from results". If it does not, the policy head joins the value head on the list of things that cannot carry a one-move exception, and the claim stands on both heads.

## 2. What the change does to the training data

In the Step 3b runs, each external game produced about thirty rows with the opponent to move, value-only, about 2,900 per generation. Those rows now carry a policy target as well. Nothing else changes: the games are played by the same two sides making the same decisions, the value target is still the game result from the mover's side, the self-play games are untouched. The row count per generation is the same as Step 3b's; what changes is that 8.5% of the window's rows now train the policy head where before they trained only the value head.

The rows that carry the lesson are the ones at a fatal position with the opponent to move: about 150 per run by the training log's wipeout count (Step 3b results, section 5, as corrected), at which the search's target puts most of its mass on the wiping reply. They sit among some 60,000 other opponent-to-move rows whose targets are ordinary moves. The policy head's task is the same shape as the value head's was in Step 3b, a rare exception among look-alikes, with two differences in its favour: the target is a specific move rather than a scalar average, and the move that takes every disc has a visible signature on the board. Whether that is enough is what the run measures.

Each searched opponent position costs one search; an external game now costs about twice the network calls it did, so self-play wall-clock rises by about a fifth of a fifth, a few percent, and is reported.

## 3. Groups and matching

| group | config | at the opponent's positions | games per generation |
|---|---|---|---|
| B-both (Step 3b, two seeds, not re-run) | `configs/full8x8_step3_both.yaml` | recorded, value-only | 480 self-play + 120 external |
| **B-both-sp** (two seeds) | `configs/full8x8_step5_search.yaml` | recorded, searched, policy target stored | the same |

The two configs differ in one key, `selfplay.search_opponent_positions`. Everything else is the Step 3b recipe: twenty generations continued from `control120` with its replay window, seeds 1337 and 1338, the same opponents and mix, the same search budget, noise, temperature and trainer, the terminal check off everywhere. The groups are matched on games and rows.

The check stays off because Step 4 showed what it does to the data: it removes the fatal positions by steering the network away from them, and this step needs those positions to occur. The search at the opponent's position does not need the check to find the wiping reply: the root is searched with 200 simulations and exploration noise, the reply's child is a finished position whose exact value enters the search on its first visit, and from then on it draws the visits.

## 4. Evaluation

The frozen Step 1 protocol on the frozen roster, exactly as for Step 3: 200 games per pairing, 100 seeded openings in both colours, 50 simulations, no exploration noise, seed 20261025, all thirteen scorecard rows, measured with the plain search. The head probe on both final checkpoints (value at the network's fatal position, value at the position one ply later with Greedy to move, prior on the wiping reply there, prior on the move played at the network's position, escape rates of the plain and the checked search), the value calibration check, and the Step 3b row count, which now reads the opponent's rows from the `opponent_to_move` column.

One training-side measurement is added, from the replay shards: among the opponent-to-move rows at which some legal move takes every one of the network's discs, the share whose policy target puts more than half its mass on such a move. It says whether the search found the reply at the positions that matter, which prediction 1 needs before anything else is read.

## 5. Predictions, written before the runs

The B-both numbers these are read against, from the Step 3b results: Greedy wipeouts 61 and 69 per 200 games, random-playout wipeouts 36 and 37, the network wiped out in 6.0% and 6.9% of the external training games, prior on the wiping reply at the Greedy-to-move position 4.6% and 4.4%, value there −0.98 and −0.99, value at the network's own fatal position +0.93 and +0.96.

1. **The search finds the reply where it exists.** At the opponent-to-move rows where a move takes every disc of the network's, the stored target puts more than half its mass on such a move in at least nine rows out of ten. The network's wipeout rate in the external games stays at 6 to 7%, since nothing about how the games are played has changed. If either fails, the implementation is at fault and nothing below is read.
2. **The prior moves, for the first time.** At the probed fatal positions with Greedy to move, the prior on the wiping reply rises above 10% on both seeds. Every earlier run left it between 2.9% and 4.6%; this is the first run with a training signal there.
3. **The value head does not move.** The value at the Greedy-to-move position stays below −0.9 and the value at the network's own position stays above +0.9. The value target is unchanged and the row counts are Step 3b's.
4. **The wipeouts fall by less than half.** Greedy wipeouts for B-both-sp land between 35 and 60 on both seeds, and the random-playout row moves in proportion. The reasoning: the prior rises at positions like the ones seen, but a prior learned from a hundred and fifty rows reaches some fatal positions and not others, and the plain search at 50 simulations needs it high at the one that matters in each game. This is the prediction held with the least confidence.
5. **The Edax rows, the generation-60 retention row and the calibration gap stay within seed noise of B-both.** The policy head is trained on more rows of the same games; the value head sees the same rows as Step 3b.

The readable outcomes. Predictions 1, 2 and 3 hold and 4 lands in its range: the policy head learns the positions it was shown and the lesson generalises partway, and the report says the policy route is open and dose-limited, with the dose curve (more external games, or weighting the searched rows) as the next step. Prediction 4 fails downward, Greedy wipeouts below 30 on both seeds: the policy head learned the rule, the tactic is learnable from searched positions, and the claim narrows to the value head. Prediction 2 fails, the prior stays under 6%: the policy head did not learn even the positions it was shown, the strongest null, and the claim stands on both heads. Prediction 1 fails: implementation, fixed first. Prediction 5 fails: the searched rows cost strength, reported as such.

## 6. Cost

Two runs of twenty generations, about 1.3 GPU-hours each by the Step 3b timings plus a few percent for the extra searches; 26 evaluation pairings on CPU, of which the two random-playout pairings take several hours each; the probe, the calibration check and the row counts. About a week including the write-up.

## 7. How a run is launched

As in the Step 3 design, section 7, with the Step 5 config:

```bash
cp -r runs/control runs/step5-Bboth-sp-s1       # checkpoints up to gen_00120 and the replay window
reversi train -c configs/full8x8_step5_search.yaml --run-id step5-Bboth-sp-s1 --generations 140 --device cuda \
    --set selfplay.n_workers=22
```

The second seed adds `--set seed=1338`. On the cluster the runs go through `slurm/submit_chain.sh`, with the seed passed as `EXTRA_ARGS="--set seed=1338"`. `configs/full8x8_step5_search.yaml` is `configs/full8x8_step3_both.yaml` with one key added and nothing else, so the diff is the experiment.
