# Experiments

One entry per run: what was expected, what was changed, what happened, and what
was decided as a result. Including the runs that did not work out.

**Training loss is never cited as evidence of strength anywhere in this
document.** A falling loss shows the network is learning to predict its own
search; whether it plays better is a separate question, answered by playing.

---

## Run 1 — `20260827-030939-full8x8-60cfdda-s1337`

**Question.** Can the pipeline take randomly initialised weights and produce an
8×8 agent that beats a depth-4 alpha-beta searcher?

**Setup.** 8×8, 6 residual blocks × 64 channels (458,696 parameters), 200
simulations per move, 600 games per generation, 6 worker processes, 24 games in
flight each. RTX A1000 laptop GPU, 4 GB.

**Scale.** 60 generations, 36,000 self-play games, 9.1 hours of self-play,
~9 minutes per generation.

### Result: yes, decisively

Bradley–Terry ratings from a round robin of 28 pairings, 30 colour-balanced
games each, 4-ply seeded opening book, anchored at Random = 0:

| agent | Elo | 95% bootstrap interval |
|---|---|---|
| **generation 60** | **+877** | [+774, +1028] |
| generation 40 | +855 | [+747, +1018] |
| generation 20 | +758 | [+659, +898] |
| generation 5 | +547 | [+467, +686] |
| Minimax-d4 | +523 | [+434, +653] |
| Minimax-d2 | +334 | [+250, +459] |
| Greedy | +313 | [+220, +468] |
| Random | 0 | — |

Generation 60 beat Minimax-d4 **30 games to nothing**. Its rating interval lies
entirely above generation 20's, which is the strict form of "later is stronger" —
two point estimates in the right order would not have been enough.

The baselines are in the same fit rather than measured separately, so the scale
means something: the agent crossed from below Minimax-d4 to roughly 350 Elo above
it.

### The finding that was not expected

**The agent got *worse* against Greedy as it got better against everything else.**

| generation | vs Greedy | vs Minimax-d4 |
|---|---|---|
| 5 | **90%** | 57% |
| 20 | 70% | 87% |
| 40 | 70% | 97% |
| 60 | **63%** | **100%** |

Its score against Greedy fell by 27 points over the same 55 generations in which
its score against Minimax-d4 rose by 43. At generation 60 the Greedy result is
not even decisively above chance: 63%, 95% CI [46%, 78%].

This is non-transitive — Minimax-d4 beats Greedy 97%, our agent beats Minimax-d4
100%, and yet our agent barely beats Greedy.

**Reading.** Self-play produces a narrow training distribution. The agent only
ever sees positions arising from its own play, and as that play refines, the
distribution *narrows further*. Minimax also plays sensibly, so its positions stay
familiar. Greedy plays badly in a specific alien way — taking the most discs
available every move — producing lopsided positions the agent increasingly never
meets. Training made it better in-distribution and further from Greedy's
distribution at the same time.

There is a supporting signal: against Greedy the colour split is 90% as black
against 50% as white, an asymmetry that appears against no other opponent.

**Not yet established.** Whether this is specifically about Greedy's strategy or a
general fragility against unfamiliar play. Testing it needs an opponent that is
bad in a *different* way — a "random with occasional good moves" agent would
separate the two explanations.

### The second finding: it plateaued

| comparison | score | decisive? |
|---|---|---|
| gen 20 vs gen 40 | 38.3% [23.2%, 56.1%] | no |
| gen 40 vs gen 60 | 56.7% [39.2%, 72.6%] | no |
| gen 20 vs gen 60 | 20.0% [9.5%, 37.3%] | yes |

Generations 40 and 60 are indistinguishable at this sample size. Twenty
generations of training — three hours — produced no demonstrable gain.

The training curve points at the same thing from a different direction: **the
value head stopped improving at generation 5** and never moved again.

| generation | policy loss | value loss |
|---|---|---|
| 1 | 2.481 | 0.800 |
| 5 | 1.945 | **0.644** |
| 20 | 1.592 | 0.643 |
| 40 | 1.308 | 0.647 |
| 60 | **1.216** | **0.661** |

The policy kept learning throughout. The value head converged in five generations
and then drifted very slightly worse over the next fifty-five. A network that
predicts moves well but cannot say who is winning is exactly the shape of agent
that plays reasonable openings and misjudges endgames.

**Candidate explanations, untested and in the order worth testing:**

1. **The value loss is under-weighted.** `value_loss_weight` is 1.0 while the
   policy term is roughly twice as large in magnitude for most of training, so
   the shared trunk may be optimising mostly for the policy. Cheapest to test: one
   config change and one run.
2. **The value head is too small.** One 1×1 convolution to a single plane, then
   64 hidden units. That is a very narrow channel through which to express
   "who is winning".
3. **The target is genuinely noisy.** With 15 plies of temperature sampling and
   exploration noise, the same opening position can lead to either outcome, so
   some irreducible error is expected. This would predict a floor, which is what
   we see — but not one reached this early.

### Decisions taken

* **`full8x8` was reprofiled from measurements**, not estimates. The planned 2,500
  games at 300 simulations would have taken 2.5 hours per generation on this
  hardware — three generations a night, not enough to show a curve. See
  `bench/results/worker-sweep.json`.
* **200 simulations is probably too conservative.** The measurement behind that
  choice was itself confounded (too few games per worker, so batches collapsed);
  the corrected figure says 300 simulations would fit in ~17.5 minutes per
  generation. Left at 200 here because changing mid-run makes generations
  incomparable. **Worth raising for run 2.**
* **The plateau is the next thing to investigate**, starting with the value loss
  weight.

### Caveats on these numbers

* **30 games per pairing.** Wide intervals — the table marks four pairings as not
  decisive, and those should not be read as results.
* **One run, one seed.** Nothing here separates "training works" from "this seed
  worked". A second run at a different seed would.
* **Ratings are relative to these baselines.** The scale is anchored at Random and
  shaped by Minimax-d4's strength; it is not comparable to published Othello Elo.

---

## Run 2 — the value head — `20260902-062601-full8x8-value4-7b96ef1-s1337`

**Result: the change made the agent weaker. The hypothesis is refuted.**

**Question.** Run 1's value head stopped improving at generation 5 and never moved
again, while the policy kept learning for fifty-five more generations. Does the
plateau lift if the value term is weighted more heavily?

**Change.** `train.value_loss_weight` from 1.0 to 4.0, and nothing else. Comparing
the two resolved configs field by field: of 45 settings, exactly two differ — that
one and the profile's name. Same seed (1337), same self-play cost per generation.

**Scale.** 53 generations, 31,800 self-play games, stopped early because the
answer was already available at a matched generation.

### The prediction, registered before the run

| if | then |
|---|---|
| value loss < 0.60 **and** it beats run 1 with non-overlapping intervals | the weight was the constraint |
| value loss < 0.60 but no strength gain | the value head learned more and it did not matter |
| value loss stays near 0.64 | capacity or irreducible noise; next test is a wider head |
| **policy loss rises materially** | **4.0 is too aggressive and the trunk was starved of the policy signal** |

**The fourth row is what happened.**

### What the losses did

At generation 53, the two terms moved in opposite directions:

| | run 1 | run 2 | change |
|---|---|---|---|
| value loss | 0.6524 | **0.6327** | 3% better |
| policy loss | 1.2436 | **1.3724** | **10.4% worse** |

The value head did improve, and it improved at every generation, not just this
one — run 1 drifted from 0.6298 at generation 13 to 0.6605 at generation 60,
while run 2 stayed roughly flat around 0.625. So reweighting genuinely stopped the
value head degrading.

It never came close to the 0.60 the prediction asked for. The best it reached was
0.6132, at generation 4.

### What it cost, measured by playing

Both runs have a checkpoint at generation 53, so the comparison is exact: same
generation, same games played, same self-play compute. 200 colour-balanced games,
4-ply opening book, no exploration noise.

| | score for run 2 | 95% Wilson interval | record |
|---|---|---|---|
| generation 53, run 2 vs run 1 | **42.0%** | [35.4%, 48.9%] | 77W 109L 14D |

The interval lies entirely below 50%, so this is a real difference rather than
noise. It corresponds to roughly **56 Elo weaker**.

### Reading

**A better value head is not worth a worse policy.** Weighting the value term four
times as heavily bought a 3% improvement in predicting who is winning and paid
10.4% in predicting which move to play. One trunk feeds both heads, so the second
number is what the first one cost. Since move selection is what actually plays the
game, the trade was bad — and losing 56 Elo is the price.

**This eliminates the cheapest explanation for the plateau.** The value head is
not underweighted. It can be made to stop degrading, and doing so does not make
the agent stronger.

### Decisions taken

* **`value_loss_weight` stays at 1.0.** `configs/full8x8_value4.yaml` is kept so
  the result can be reproduced, not because it should be used.
* **The next test is capacity, not weighting.** The value head is one 1×1
  convolution to a single plane, then 64 hidden units — a narrow channel through
  which to express "who is winning". If it cannot represent the answer, no loss
  weight will make it learn one. Widening it, or giving it its own layers instead
  of competing for the shared trunk, is the experiment run 2 points at.
* **Raising simulations to 300 is now clearly a separate question.**
  `configs/full8x8_sims300.yaml` is ready. More search improves the *policy*
  target; it does not touch the value target, which is the game's final result.
  Run 2 gives no reason to expect it to lift the plateau, and a reason to test it
  on its own rather than alongside another change.

### Caveats

* **One matched comparison, at generation 53.** A second pairing at generation 40
  was started and not completed; the conclusion rests on the deeper of the two.
* **Same seed, one run each.** This separates "weighting the value term hurts"
  from "this seed was unlucky" no better than run 1 separated "training works"
  from "this seed worked".
* **200 games gives about ±7%.** Enough to place the interval below 50%, not
  enough to pin the gap to closer than roughly ±25 Elo.

---

## Run 3 — the control — `control` (CSSE Slurm cluster)

**Result: run 1 was representative to within about 25 Elo, and the plateau is the
recipe's.** The prediction table below was written before the first generation
finished; the result sections after it were written on 2026-09-04 once the run
had been rated.

**Question.** Run 1 is one run with one seed on one laptop. How much of its curve
is the recipe, and how much is noise? And what does a generation cost on the
cluster hardware that every later run will use?

**Change.** None. The profile is `full8x8.yaml` exactly as run 1 used it. Two
things differ and neither is a hyperparameter: the machine (one NVIDIA L40S and
22 self-play worker processes, against a laptop RTX A1000 and 6), and the seed,
which is the profile's default 1337 as in run 1, so the games differ only through
the worker split and floating-point order on different hardware.

**Scale.** 120 generations as a chain of 24-hour jobs (`slurm/submit_chain.sh`),
72,000 self-play games. Run 1 stopped at 60; this one runs on so the plateau run 1
showed between generations 40 and 60 is seen a second time, or not.

### The prediction, registered before the run

| if | then |
|---|---|
| generation 60 rates within run 1's 95% interval (774–1028) against the same baselines | run 1 was representative; a single seed is enough to compare recipes against |
| generation 60 rates clearly above or below run 1 | seed and hardware noise is large, and every recipe comparison in this document needs a second seed before it is believed |
| the value loss again stops improving near generation 5 and drifts upward | the plateau is the recipe's, not the laptop's |
| generations 60 to 120 rate above generation 60 with non-overlapping intervals | run 1 stopped early; the plateau was an artefact of stopping |

The wall-clock per generation is recorded as the cluster baseline that E1's cost
is compared against.

### What it cost

120 generations in 8 hours 15 minutes on one L40S with 22 self-play workers:
3.5 minutes of self-play and about 4.1 minutes in total per generation, against
9 minutes on the laptop. Same games, same simulations; the hardware is the whole
difference.

### The losses reproduced run 1

At every matched generation the two runs are within a few hundredths of each
other. The value loss sits at 0.63 to 0.66 for both, from generation 5 to the
end. Run 1's value plateau was not the laptop's doing.

| generation | run 1 value / policy | control value / policy |
|---|---|---|
| 10 | 0.633 / 1.851 | 0.630 / 1.834 |
| 30 | 0.655 / 1.411 | 0.644 / 1.365 |
| 53 | 0.652 / 1.244 | 0.655 / 1.220 |
| 60 | 0.661 / 1.216 | 0.655 / 1.213 |
| 120 | — | 0.651 / 1.096 |

### Strength: indistinguishable from run 1 at generation 60

Both generation-60 checkpoints were entered in one tournament with the baselines
and run 4's checkpoints (`docs/ratings/runs-1-3-4-matched-generations.json`: 11
entrants, 100 colour-balanced games per pairing, 4-ply openings, 50 simulations,
Bradley–Terry anchored at Random = 0, bootstrap intervals). Ratings are
comparable **within** that table only; a different field gives a different scale,
which is why these numbers are smaller than the +877 of run 1's original table.

| entrant | Elo | 95% interval |
|---|---|---|
| run 1, generation 60 | 636 | [590, 687] |
| control, generation 60 | 649 | [602, 697] |
| control, generation 120 | 709 | [664, 760] |

Head to head, run 1's generation 60 scored **49.0%** against the control's
(47W 49L 4D, Wilson interval [39.4%, 58.7%]). The first row of the prediction
table is what happened: one seed on one laptop was a fair sample of the recipe.

### Generations 60 to 120: a little, then flat

The control's own cross-generation table (`docs/ratings/run3-control-crossgen.json`,
same protocol, 6 generations plus 4 baselines):

| checkpoint | Elo | 95% interval |
|---|---|---|
| generation 120 | 836 | [779, 907] |
| generation 100 | 835 | [780, 898] |
| generation 114 | 827 | [771, 892] |
| generation 65 | 792 | [737, 854] |
| generation 35 | 754 | [700, 815] |
| generation 5 | 485 | [438, 539] |
| minimax-d4 | 452 | [406, 509] |

Generation 120 scored 58% against generation 60 in the combined table and 57%
against generation 65 here; generations 100, 114 and 120 are within 10 Elo of one
another and split their games 49–51. So the second sixty generations bought
perhaps 50 Elo and then nothing. The fourth prediction row, a decisive gain from
running on, is **not met**. The plateau run 1 showed between 40 and 60 is real; it
sits nearer generation 100 on this longer run, and the recipe does not climb past
it at this self-play budget.

### The 1000-game matches

A 100-game pairing cannot see a 55% effect. So the two questions above were
replayed as single pairings of 1000 games each, split across 62 CPU processes
(same protocol otherwise: colour-balanced, 4-ply seeded openings, 50 simulations,
no exploration noise; `docs/ratings/head-to-head-1000.json`):

| pairing | score | 95% Wilson | record |
|---|---|---|---|
| control gen 60 vs run 1 gen 60 | **53.4%** | [50.4%, 56.5%] | 526W 457L 17D |
| control gen 120 vs control gen 60 | **55.1%** | [52.0%, 58.2%] | 521W 419L 60D |

Both intervals exclude 50%, barely in the first case. The control's generation 60
is a little stronger than run 1's, by about 24 Elo; the same recipe, on different
hardware with a different worker split, does not land on the same agent. That is
the size of the noise floor for a single-seed comparison on this project, and
the number every later recipe comparison has to clear before it means anything.

The second row says the sixty extra generations were worth about 36 Elo. Real,
small, and finished by generation 100, where the cross-generation table goes flat.

### Decisions taken

* Run 1 stands as a representative result, with one qualification: two instances
  of the same recipe differ by about 25 Elo here, and by 48 Elo in the pair runs 5
  and 7 form. A recipe change that shows less than about 50 Elo against a single
  control has not shown anything; it has earned a second seed.
* 120 generations is the length for the capacity experiments, because that is
  where this recipe stops improving. Anything a change buys after that is the
  change's, not the extra generations'.
* The value plateau is the recipe's own. The next thing to try on it is a change
  to the value head's *input*, not its weight (run 2) or the network's size
  (run 4): the ownership head of E2.


---

## Run 4 — E1, capacity — `e1-10x128` (CSSE Slurm cluster)

**Result: the value head learned more, and the agent is stronger at a matched
generation, by about 30 to 40 Elo. That is real at 1000 games, and it is about the
size of the noise between two instances of the same recipe.** The prediction table
below was written before the first generation finished; the result sections after
it were written on 2026-09-04 once the run had been rated.

**Question.** Run 2 concluded that the next thing to test is capacity, not
weighting. Does a network with about 6.5 times the parameters learn a stronger
agent from the same self-play budget?

**Change.** `net.n_blocks` from 6 to 10 and `net.channels` from 64 to 128, and
nothing else. About 2.97 M parameters against run 1's 458,696, and roughly 6.7
times the arithmetic per position. Self-play cost is left identical: 200
simulations per move, 600 games per generation. So generation *N* here has seen
exactly as much play as generation *N* of runs 1 and 3, and the comparison at a
matched generation is exact. The bigger network costs more wall-clock per
generation; that is reported alongside, not hidden.

**Fallback.** If the cluster bench shows a generation would exceed twenty minutes
(the wall-clock signal arrives fifteen minutes before the limit, and the current
generation must finish inside that window), the run is cancelled before it has
produced anything and resubmitted as `full8x8_e1_8x96.yaml`: 8 blocks of 96
channels, about 1.34 M parameters. The prediction below applies unchanged.

**Scale.** 120 generations, 72,000 games, as a chain of 24-hour jobs.

### The prediction, registered before the run

Compared against run 3, the control, which is the same recipe on the same
hardware:

| if | then |
|---|---|
| at generations 40 and 60, the Bradley–Terry interval lies entirely above the control's, **and** value loss falls below 0.60 by generation 30 | capacity was the constraint on both heads |
| strength rises with non-overlapping intervals but value loss does not fall below 0.60 | capacity helped the policy only; the value head's problem is something else |
| value loss falls below 0.60 but strength does not rise | the value head learned more and it did not matter — run 2's second row, reached by a different route |
| neither moves | capacity is not the constraint at this self-play budget; the next test is more simulations or more games per generation (`full8x8_sims300.yaml`), not a bigger network |

Run 1 never went below 0.64 on the value loss. Its policy loss was still falling
at generation 60.

### What it cost

The bench (`bench/results/cluster-nvidia-l40s-*.json`) said the network would not
be where the time goes, and the run agreed: 5.4 minutes of self-play per
generation against the control's 3.5, for 6.7 times the arithmetic per position.
The tree search in Python is the bottleneck, and the GPU is mostly idle either
way. 120 generations took 12 hours 30 minutes on one L40S.

### The value head learned more

E1's value loss fell below 0.60 at generation 10 and stayed there until the last
twenty generations, where it drifted back up. The control never got below 0.62.
The policy losses finished identical.

| generation | control value / policy | E1 value / policy |
|---|---|---|
| 10 | 0.630 / 1.834 | 0.599 / 1.837 |
| 30 | 0.644 / 1.365 | **0.584** / 1.368 |
| 60 | 0.655 / 1.213 | 0.591 / 1.183 |
| 100 | 0.643 / 1.097 | **0.571** / 1.111 |
| 120 | 0.651 / 1.096 | 0.620 / 1.096 |

So the loss half of the first prediction row is met: below 0.60 by generation 30,
comfortably. The value plateau at 0.65 *was* partly a capacity limit. The late
drift upward, from 0.571 at generation 100 to 0.620 at 120, is unexplained here
and is the first thing to look at in the metrics before drawing on it.

### Strength at matched generations: ahead, but the intervals overlap

From the same combined tournament as run 3's section (100 games per pairing, 50
simulations, one Bradley–Terry fit):

| generation | control Elo [95%] | E1 Elo [95%] | E1 vs control, head to head |
|---|---|---|---|
| 40 | 633 [589, 683] | 630 [585, 679] | 53.0% [43.3%, 62.5%] |
| 60 | 649 [602, 697] | 689 [644, 738] | 55.0% [45.2%, 64.4%] |
| 120 | 709 [664, 760] | 759 [712, 809] | 58.5% [48.7%, 67.7%] |

Nothing at generation 40. Forty to fifty Elo at 60 and at 120, in the same
direction both times, with every head-to-head interval still spanning 50%. The
first prediction row asked for intervals *entirely* above the control's; they
overlap by about 35 Elo at both generations, so that row is **not met** on this
evidence. A 100-game pairing has a margin of about ±10 percentage points and
cannot resolve a 55% effect either way; the section after this one is the longer
match that can.

E1's own cross-generation table (`docs/ratings/run4-e1-10x128-crossgen.json`) shows
the same shape as the control's: a rise to generation 100 and a plateau after it
(generations 100, 114, 120 at 891, 892, 905).

### The 1000-game matches

The same two matched pairings, replayed as 1000 games each across 62 CPU
processes (`docs/ratings/head-to-head-1000.json`):

| pairing | score for E1 | 95% Wilson | record | about |
|---|---|---|---|---|
| E1 gen 60 vs control gen 60 | **55.6%** | [52.5%, 58.7%] | 533W 421L 46D | +39 Elo |
| E1 gen 120 vs control gen 120 | **54.1%** | [51.1%, 57.2%] | 523W 440L 37D | +29 Elo |

Both intervals exclude 50%. The bigger network is stronger at a matched
generation, and the direction did not change between 60 and 120.

### Reading

Against the prediction table: the value-loss half of the first row is met, and
the strength half is met on the 1000-game evidence but not in the form it was
written, which asked for Bradley–Terry intervals that do not overlap at 100 games
per pairing. Recorded as **row 1, weakly**: capacity was a constraint on both
heads, and lifting it bought 30 to 40 Elo.

Two things keep that from being a headline. First, run 3 measured the noise
between two instances of the *same* recipe at about 24 Elo, so E1's gain is
roughly one noise-width, established only because 1000 games were played.
Second, E1 plateaus at the same generation the control does, around 100, so more
capacity moved the ceiling up a little rather than removing it. At 6.7 times the
arithmetic per position, that is an expensive 35 Elo.

### Decisions taken

* The 1.0 network is chosen on strength per browser-millisecond, not on this
  table alone. E1's generation 120 is the current best checkpoint, and it costs a
  WebGPU path to serve; that trade is measured in `docs/web-app.md` before it is
  taken.
* The value head's ceiling is not capacity alone: E1's value loss fell to 0.57
  and the agent gained 35 Elo, not 150. The next experiment changes what the
  value head is *asked to predict* (E2, the ownership head), not how big it is.
* Every future recipe comparison plays at least 1000 games at a matched
  generation, or reports that it cannot tell.
* The late rise in E1's value loss (0.571 at generation 100 to 0.620 at 120) is
  written down and unexplained. It did not cost strength that the arena can see.



---

## Run 5 — E2, the ownership head — `e2-ownership` (CSSE Slurm cluster)

**Result: the largest gain of any change so far, for 65 parameters and no extra
self-play cost: +45 Elo at generation 60 and +74 at generation 120 over the
control, both decisive at 1000 games. And it did not happen the way the
prediction said it would.** The prediction table below was written before the run
was submitted; the result sections after it were written on 2026-09-05 once the
run had been rated.

**Question.** Runs 1 and 3 showed the value head stops improving at generation 5
and sits at a loss of 0.65 for the rest of the run. Run 2 showed that weighting
its one target more heavily starves the policy. Run 4 showed that a bigger
network lowers the value loss to 0.57 but buys only 30 to 40 Elo, so capacity is
not the whole story. What if the problem is the *target*? Does giving the value
side sixty-four answers per position instead of one make a stronger agent?

**Change.** `net.ownership: true` and `train.ownership_loss_weight: 1.0`, and
nothing else about the run 1 recipe (`configs/full8x8_e2_ownership.yaml`).
The network grows a third head, a 1×1 convolution off the trunk that predicts
for every square who owns it when the game ends: +1 the player to move, −1 the
opponent, 0 empty. The final disc margin that decides `z` is the sum of those 64
numbers, so the head is asked for the same answer as the value head plus where it
comes from. The loss gains a mean-squared-error term on it, weighted like the
value term. The idea is KataGo's (Wu, *Accelerating Self-Play Learning in Go*,
arXiv:1902.10565), where auxiliary ownership and score targets were among the
largest single gains in learning efficiency.

The head is never consulted at play time. `forward` still returns a policy and a
value; the search, the export and the browser see a network of exactly the shape
run 3 produced. Only the trainer sees the third output.

**Scale.** 120 generations, 72,000 games, one L40S, chained 24-hour jobs. 6×64
network, 200 simulations, 600 games per generation: the same self-play budget as
runs 1, 3 and 4, so generation *N* is comparable across all of them.

### The prediction, registered before the run

Compared against run 3, the control, at matched generations 60 and 120, in
1000-game head-to-head matches (the protocol run 4 settled on):

| if | then |
|---|---|
| E2 beats the control at both generations with intervals excluding 50%, **and** by more than run 4's 30 to 40 Elo | the target was the constraint, and it is a cheaper lever than capacity |
| E2 beats the control decisively but by about what run 4 gained | the ownership head is worth about as much as a 6.7× bigger network at a fraction of the cost; E1 and E2 are then combined for the 1.0 network |
| value loss falls below 0.60 but E2 does not beat the control | the head made a better predictor and not a better player; the value head is not what limits play at this search budget |
| E2 is weaker than the control | the extra term competes with the policy after all, as run 2's weight did; halve the weight and try once more |

Value loss below 0.60 by generation 30 is expected in every row but the last; run
1 and run 3 never went below 0.62.

### What it cost

Nothing measurable. 3.5 minutes of self-play per generation, the same as the
control; 120 generations in 8 hours 20 minutes on one L40S. The head is 65
weights.

### The losses did not do what the prediction expected

| generation | control value / policy | E2 value / policy | E2 ownership |
|---|---|---|---|
| 10 | 0.630 / 1.834 | 0.651 / 1.793 | 0.881 |
| 30 | 0.644 / 1.365 | 0.640 / 1.347 | 0.865 |
| 60 | 0.655 / 1.213 | 0.646 / 1.168 | 0.867 |
| 100 | 0.643 / 1.097 | 0.640 / 1.111 | 0.868 |
| 120 | 0.651 / 1.096 | 0.649 / 1.124 | 0.865 |

The value loss tracks the control's at 0.65 from start to finish. It never came
near the 0.60 that every winning row of the prediction table assumed. The
ownership term fell from 1.08 to 0.87 in the first fifteen generations and then
sat at 0.865 for the remaining hundred.

### Why the ownership term sits at 0.87: the target is mostly unpredictable

A diagnostic (`docs/ratings/ownership-diag-e2-g120.json`): the generation-120
head scored on the 34,720 positions of its own last generation, grouped by how
far into the game each position is, against two reference predictors. "All
empty" says every square ends up unowned; on a finished board every square is
occupied, so its error is 1.0 by construction. "Board as is" says every disc
stays the colour it is now and every empty square stays empty.

| moves into the game | head | all empty | board as is |
|---|---|---|---|
| 0–9 | 0.996 | 1.000 | 1.137 |
| 10–19 | 0.988 | 1.000 | 1.293 |
| 20–29 | 0.970 | 1.000 | 1.444 |
| 30–39 | 0.923 | 1.000 | 1.564 |
| 40–49 | 0.801 | 1.000 | 1.527 |
| 50–59 | 0.452 | 1.000 | 1.018 |

For the first thirty moves the head is no better than "all empty". And "board as
is" is the *worst* predictor at every stage, worse than saying nothing: discs flip
so much in Reversi that a square's current colour is anti-informative about its
final colour. That is the difference from Go, where the idea comes from and where
territory settles early. In Reversi, who owns a square is close to unknowable
until the last ten moves, and the head learned exactly the part that is knowable.

### Strength: decisively better anyway

1000-game head-to-head matches against the control at matched generations
(`docs/ratings/head-to-head-e2-1000.json`; colour-balanced, 4-ply seeded
openings, 50 simulations, no exploration noise):

| pairing | score for E2 | 95% Wilson | record | about |
|---|---|---|---|---|
| E2 gen 60 vs control gen 60 | **56.4%** | [53.3%, 59.4%] | 539W 412L 49D | +45 Elo |
| E2 gen 120 vs control gen 120 | **60.5%** | [57.4%, 63.4%] | 578W 369L 53D | +74 Elo |

Both intervals exclude 50% by a wide margin. At generation 120 the gap is three
times the 24 Elo noise floor run 3 measured between two instances of the same
recipe, and about double what run 4 bought with a network 6.7 times the size.
E2's own cross-generation table (`docs/ratings/run5-e2-ownership-crossgen.json`)
has the same shape as the others, a climb to generation 100 and a plateau after
it, at generations 100, 114 and 120 rated 984, 956 and 968 against Random.

### Reading

The first prediction row is what happened on strength: E2 beats the control at
both generations with intervals excluding 50%, and by more than run 4's 30 to 40
Elo. The row's *mechanism* was wrong. It assumed the gain would come through a
better value predictor, visible as a lower value loss. The value loss did not
move, and the ownership target turned out to be mostly noise until the endgame.

What the head did instead was shape the trunk. The trunk is the part of the
network both heads read from, and every position now pushes it toward features
that say something about the endgame, which the single win/loss number does not.
The policy head, reading a better trunk, got better. The evidence for that
reading is indirect: E2's policy loss is not lower than the control's either, so
"better" here means the policy's *mistakes* changed in a way the search converts
into wins, not that the network matches its own search more closely. That is a
statement about play, and play is what was measured.

Two cautions. This is one seed. Run 3 established that two instances of a recipe
differ by about 24 Elo, and the +74 at generation 120 is comfortably past that,
but the +45 at generation 60 is not far past it. And the gain is at 50
simulations; the difficulty ladder plays at 16 to 800.

### Against Edax: from level 5 to about level 6

The 0.0 agent (run 1, generation 60, 256 simulations) was statistically even with
Edax at level 5 and lost to level 6 and above. The same protocol on E2's
generation 120 (`docs/ratings/edax-e2-g120.json`; 80 colour-balanced games per
level, 4-ply seeded openings, 256 simulations, Edax 4.6 built from source on the
cluster, book off, one thread):

| Edax level | score for E2 | 95% Wilson | record |
|---|---|---|---|
| 5 | **71.2%** | [60.5%, 80.0%] | 55W 21L 4D |
| 6 | 59.4% | [48.4%, 69.5%] | 45W 30L 5D |
| 7 | 41.2% | [31.1%, 52.2%] | 32W 46L 2D |
| 8 | 21.9% | [14.2%, 32.1%] | 15W 60L 5D |

Level 5, where the 0.0 agent scored 53%, is now beaten decisively. Level 6 is
probably beaten and level 7 probably not, but both of those intervals include
50%, so at 80 games the honest statement is "somewhere between level 6 and
level 7", up from "level 5". Level 8 is clearly out of reach. Eighty games per
level is the 0.0 protocol and is kept for comparability; 320 games would settle
levels 6 and 7 and is cheap on the cluster.

### Decisions taken

* The ownership head is part of the 1.0 recipe. It costs nothing measurable and
  is the biggest single gain found.
* The next run combines it with E1's capacity, E1+E2, 10×128 with the head, to
  see whether the two gains add. Its prediction is registered before it starts.
* The "value loss below 0.60" expectation is retired as a proxy for a better
  agent on this project. Runs 4 and 5 together show it is neither necessary (E2)
  nor sufficient (E1, which lowered it to 0.57 for 35 Elo).
* The diagnostic's "board as is" row is worth remembering when anyone proposes a
  feature plane or target built on current disc ownership: in Reversi, the board
  as it stands is a bad guide to the board as it ends.


---

## Play-time sweep: `c_puct`

**Result: 1.5 stays. More exploration hurts at this search budget; less does not
measurably help.**

**Question.** `c_puct` is the constant that trades exploring untried moves against
exploiting the best-looking one during search. This project has used 1.5 since day
4 without measuring it, and `references.md` recorded that as a gap. The papers
give no number for a board this size or a budget this small.

**Setup.** No training. Run 4's generation-120 network entered one tournament four
times, at `c_puct` 1.0, 1.5, 2.5 and 4.0, alongside Random, Greedy, Minimax-d2 and
Minimax-d4: 8 entrants, 28 pairings, 200 colour-balanced games each, 4-ply
openings, 50 simulations, one Bradley–Terry fit anchored at Random = 0
(`docs/ratings/cpuct-sweep-e1-g120.json`; 93 minutes on 64 CPU cores).

| `c_puct` | Elo | 95% interval |
|---|---|---|
| 1.0 | 762 | [723, 805] |
| **1.5** | 750 | [712, 795] |
| 2.5 | 695 | [655, 737] |
| 4.0 | 620 | [583, 658] |
| minimax-d4 | 384 | [350, 417] |

Head to head, 1.0 scored 52.2% against 1.5 (99W 90L 11D, interval [45.4%,
59.1%]): no difference the games can see. 1.5 beat 2.5 with 64.5% and 4.0 with
70.5%, both decisive.

### Reading

At 50 simulations there is not much budget to spend on exploring, and the network
at generation 120 has good enough first impressions that spending it on the
second- and third-best moves costs more than it finds. That is the direction the
pseudocode's own schedule points: its coefficient starts at 1.25 and only grows
with visit count, so at small budgets it is near the bottom of the range tested
here. Whether 1.0 is better than 1.5 at 800 simulations, where the Max difficulty
level plays, was not tested and is a different question; the difficulty ladder is
recalibrated on the 1.0 network anyway, and that is the place to ask it.

### Decisions taken

* `c_puct` stays at 1.5 for training, the arena, the difficulty levels and the
  browser. Changing it to 1.0 would be a change without evidence.
* The sweep protocol, one network entered several times with different search
  settings via `reversi arena --entrant NAME=PATH@SIMS;c_puct=…`, is the way any
  play-time constant is tuned on this project from here on.

---

## Run 6 — E1+E2, capacity and the ownership head together — `e12-10x128-ownership`

**Result: the gains do not add by generation 120. The big network with the head
is far ahead at generation 60 (+73 Elo over E2 alone) and level with it at 120
(+13 Elo, interval spanning 50%). The 1.0 network is the small one with the head,
and the browser does not need WebGPU to run it.** The prediction table below was
written before the run was submitted; the result sections after it were written
on 2026-09-06 once the run had been rated.

**Question.** Run 4 gained 30 to 40 Elo from a 10×128 network. Run 5 gained 45 to
74 Elo from an ownership head on the 6×64 network. Do the two gains add?

**Change.** Both changes applied to the run 1 recipe, and nothing else
(`configs/full8x8_e12_10x128_ownership.yaml`): `net.n_blocks: 10`,
`net.channels: 128`, `net.ownership: true`, `train.ownership_loss_weight: 1.0`.
Self-play cost is unchanged, so generation *N* is comparable to generation *N* of
runs 1, 3, 4 and 5.

**Why they might add.** The two changes act on different things: E1 gives the
trunk more room, E2 gives it a richer training signal. Run 5's reading was that
the ownership target shaped the trunk; a bigger trunk should be able to use the
same signal at least as well.

**Why they might not.** Run 4 lowered the value loss to 0.57 with the plain value
target. If the bigger network was already extracting from the win/loss number
what the ownership target adds, the head has nothing left to contribute.

**Scale.** 120 generations, one L40S, chained 24-hour jobs; about 12.5 hours at
run 4's pace.

### The prediction, registered before the run

Against run 5 (E2 alone) at matched generations 60 and 120, 1000-game head-to-head
matches:

| if | then |
|---|---|
| E1+E2 beats E2 at both generations with intervals excluding 50%, by 30 Elo or more | the gains add; the 1.0 network is 10×128 with the head, and the browser needs WebGPU to run it at full strength |
| E1+E2 beats E2 decisively but by less than 20 Elo | the gains mostly overlap; the 1.0 network is 6×64 with the head, which the browser already runs at full speed, and the WebGPU path is for later networks rather than this one |
| no decisive difference either way | same decision as the row above; capacity is not the lever at this self-play budget once the trunk is fed properly |
| E1+E2 loses to E2 | the bigger trunk overfits the 64-square target; halve the ownership weight before growing the network again |

### What it cost

5.3 minutes of self-play per generation against E2's 3.5, the same ratio run 4
showed against the control. 120 generations in 12 hours 31 minutes on one L40S.

### The losses

| generation | E1 (10×128) value / policy | E2 (6×64 + head) value / policy | E1+E2 value / policy / ownership |
|---|---|---|---|
| 10 | 0.599 / 1.837 | 0.651 / 1.793 | 0.605 / 1.876 / 0.858 |
| 30 | 0.584 / 1.368 | 0.640 / 1.347 | 0.605 / 1.348 / 0.859 |
| 60 | 0.591 / 1.183 | 0.646 / 1.168 | 0.598 / 1.164 / 0.864 |
| 100 | 0.571 / 1.111 | 0.640 / 1.111 | 0.580 / 1.069 / 0.864 |
| 120 | 0.620 / 1.096 | 0.649 / 1.124 | 0.595 / 1.055 / 0.864 |

The value loss sits where E1's did, the ownership term where E2's did, and the
policy loss ends lowest of any run. None of that is evidence of strength; it is
recorded because it is consistent with each change doing what it did alone.

### Strength: ahead early, level by the end

1000-game head-to-head matches against run 5 (E2 alone) at matched generations
(`docs/ratings/head-to-head-e12-1000.json`; colour-balanced, 4-ply seeded openings,
50 simulations, no exploration noise):

| pairing | score for E1+E2 | 95% Wilson | record | about |
|---|---|---|---|---|
| E1+E2 gen 60 vs E2 gen 60 | **60.3%** | [57.2%, 63.3%] | 581W 375L 44D | +73 Elo |
| E1+E2 gen 120 vs E2 gen 120 | 51.9% | [48.9%, 55.0%] | 484W 445L 71D | +13 Elo |

At generation 60 the first prediction row holds, and by a wide margin: the big
network with the head is 73 Elo ahead of the small one with the head, twice E1's
own gain over the control. At generation 120 the third row holds: the interval
spans 50%, and 13 Elo is well inside the 24 Elo noise floor. The small network
caught up.

Against Edax, same protocol as runs 1 and 5 (`docs/ratings/edax-e12-g120.json`):

| Edax level | E1+E2 gen 120 | E2 gen 120 (run 5) |
|---|---|---|
| 5 | 74.4% [63.8%, 82.7%] | 71.2% |
| 6 | 52.5% [41.7%, 63.1%] | 59.4% |
| 7 | 42.5% [32.3%, 53.4%] | 41.2% |
| 8 | 29.4% [20.5%, 40.1%] | 21.9% |

The same picture from the outside: between level 6 and level 7, indistinguishable
from run 5 at 80 games per level.

One thing the cross-generation table (`docs/ratings/run6-e12-crossgen.json`)
adds: E1+E2 does not show the plateau at generation 100 that every other run has.
Generations 100, 114 and 120 rate 956, 988 and 1006, still rising, where run 5's
were flat at 984, 956, 968. The intervals overlap, so this is a hint and not a
finding. If it is real, the big network with the head would pull ahead of the
small one given more generations than 120, at 1.5 times the self-play cost per
generation.

### Reading

The two changes are not independent levers. The bigger trunk learns faster per
generation, reaching at 60 what the small trunk with the head reaches at 120, and
then the small one catches up. Per game of self-play the big network is more
efficient; per minute of GPU it is not, because each of its generations costs 1.5
times as much. At a matched budget of 120 generations they land in the same place.

For the question the run was submitted to answer, which network to ship, the
third prediction row decides: **the 1.0 network is 6×64 with the ownership head**,
run 5's recipe. It plays at the same strength as the big one at generation 120,
its checkpoint is 1.8 MB against 12 MB, and the browser already runs it at full
speed in WebAssembly. The WebGPU inference path, which the roadmap made a
requirement on the assumption that the bigger network would be needed, is no
longer required for 1.0; it stays on the list for a later network.

### Decisions taken

* The 1.0 recipe is `configs/full8x8_e2_ownership.yaml`: 6×64, ownership head,
  weight 1.0, everything else as run 1. The 1.0 checkpoint is chosen among run 5
  and its replicates (runs 7 and 8) once those are rated.
* The WebGPU path moves from "required for 1.0" to "stretch" in the roadmap. The
  hours it would have taken go to the difficulty ladder and the game features.
* The big network's late climb was checked: its generation 120 beat its
  generation 100 by **54.9%** over 1000 games ([51.8%, 58.0%], 534W 436L 30D,
  `docs/ratings/head-to-head-e12-late-1000.json`), about +34 Elo, decisive. The
  small network's generations 100 and 120 split 50–50. So E1+E2 was still
  improving when the run stopped, and the small network was not. That does not
  change the 1.0 decision, which is about a matched budget, but it makes a longer
  E1+E2 run the natural candidate for a 1.1 network, served through the WebGPU
  path. Registered here as the next capacity question, not started.

---

## Run 7 — E2, second seed — `e2-ownership-seed2` (CSSE Slurm cluster, gebru)

**Result: the head helps, and run 5 was a lucky seed. The second seed beats the
control decisively at both generations, by +25 and +33 Elo, against run 5's +45
and +74. Two seeds of the same recipe differ by 48 Elo at generation 120, twice
the noise floor run 3 measured.** The prediction table below was written before
the first generation finished; the result sections after it on 2026-09-06.

**Question.** Run 5's +74 Elo over the control at generation 120 is the largest
gain in the project and rests on one seed. Run 3 measured about 24 Elo of noise
between two instances of the same recipe. How much of run 5 is the head, and how
much is the seed?

**Change.** None to the recipe. `configs/full8x8_e2_ownership.yaml` with `seed`
overridden to 2024 on the command line (run 5 used the profile's default, 1337).
The run is on gebru, an RTX 6000 with 14 self-play workers, rather than gus with
22; run 3 showed that the hardware and worker split change the games without
changing the recipe, and this run measures that noise along with the seed's.

**Scale.** 120 generations, one GPU, about 11 hours at gebru's pace.

### The prediction, registered before the run

Against run 3 (the control) at matched generations 60 and 120, 1000-game matches:

| if | then |
|---|---|
| the second seed beats the control decisively at both generations, within about 25 Elo of run 5's +45 and +74 | the gain is the head's; run 5's numbers stand and go in the README as "about +60 Elo, two seeds" |
| it beats the control decisively but by clearly less than run 5 | the head helps, and run 5 was a lucky seed; report the mean of the two seeds and say so |
| it does not beat the control decisively | run 5 was mostly seed; the head's gain is inside the noise and the README claims only what both seeds support |

Also: run 5 against run 7 directly at generation 120. Two seeds of the same recipe
should split near 50%; a decisive result either way is the noise floor for this
recipe, to set beside run 3's 24 Elo.

### What it cost

4.1 minutes of self-play per generation on an RTX 6000 with 14 workers, against
3.5 on the L40S with 22. 120 generations in 10 hours 27 minutes.

### The losses

Indistinguishable from run 5's at every matched generation: value 0.64 to 0.66
throughout, ownership 0.87, policy within 0.04. Nothing in the curves tells the
two seeds apart.

### Strength

1000-game matches (`docs/ratings/head-to-head-e2s2-1000.json`):

| pairing | score for run 7 | 95% Wilson | record | about |
|---|---|---|---|---|
| run 7 gen 60 vs control gen 60 | **53.6%** | [50.6%, 56.7%] | 521W 448L 31D | +25 Elo |
| run 7 gen 120 vs control gen 120 | **54.8%** | [51.7%, 57.8%] | 527W 432L 41D | +33 Elo |
| run 7 gen 120 vs run 5 gen 120 | **43.2%** | [40.2%, 46.3%] | 395W 530L 75D | −48 Elo |

Both intervals against the control exclude 50%, so the head's gain is real on a
second seed. It is less than half of run 5's. And the third row is the one to
remember: two runs of the *same* recipe, differing only in the seed and the node
they ran on, are 48 Elo apart at generation 120, decisively. Run 3 put that
noise at about 24 Elo from one pair of runs; with a second pair the honest range
is 25 to 50 Elo.

### Reading

The second prediction row is what happened: the head helps, and run 5 drew a good
seed. Reported as the recipe's effect, the ownership head is worth about **+45 Elo
at generation 120, the mean of two seeds (+74 and +33)**, with the two seeds far
enough apart that a third would move the mean by 10 or 15 Elo either way.

The noise floor matters beyond this run. Run 4's capacity gain of 30 to 40 Elo
was a single-seed comparison and now sits inside the range two seeds of one recipe
can span. Run 6's +73 at generation 60 is outside it; its +13 at 120 was never
claimed. The README carries what both seeds support and nothing that one seed
produced alone.

### Decisions taken

* The ownership head stays in the 1.0 recipe: three runs with it (5, 7 and 8) all
  beat the control decisively at generation 120, and no run without it does that
  at the same cost.
* Run 5's generation 120 remains the 1.0 checkpoint candidate. It is the best of
  three seeds, so its +74 over the control is a best-of-three number and is said
  to be one; its strength against Edax is measured directly and does not depend
  on how it was chosen.
* A claim of less than about 50 Elo from a single pair of runs is not a claim on
  this project. It is a hint that buys a second seed.

---


---

## Run 8 — E2, ownership weight halved — `e2-ownership-w05` (CSSE Slurm cluster, gebru)

**Result: no better and no worse than the second seed at full weight. Nothing at
generation 60, +26 Elo over the control at 120. The weight does not matter over a
factor of two, as far as one run can tell; 1.0 stays.** The prediction table
below was written before the first generation finished; the result sections after
it on 2026-09-06.

**Question.** Run 5 used an ownership weight of 1.0, the first value tried. Does
the weight matter?

**Change.** `train.ownership_loss_weight` overridden to 0.5 on the command line;
everything else as run 5. Same node and worker count as run 7.

### The prediction, registered before the run

Against run 3 (the control) at matched generations 60 and 120, 1000-game matches:

| if | then |
|---|---|
| within about 25 Elo of run 5 | the head's gain is robust to the weight over a factor of two; 1.0 stays because it is what was measured most |
| clearly less than run 5 | the weight matters and 1.0 was on the low side; 2.0 is the next thing to try |
| clearly more than run 5 | 1.0 was too much and the term was competing with the policy after all; 0.5 becomes the recipe's value |

### Strength

1000-game matches (`docs/ratings/head-to-head-e2w05-1000.json`):

| pairing | score for run 8 | 95% Wilson | record | about |
|---|---|---|---|---|
| run 8 gen 60 vs control gen 60 | 50.0% | [46.9%, 53.1%] | 471W 471L 58D | 0 |
| run 8 gen 120 vs control gen 120 | **53.7%** | [50.6%, 56.8%] | 504W 430L 66D | +26 Elo |

The losses match runs 5 and 7 at every matched generation, with the ownership term
a little higher (0.877 against 0.865), as a smaller weight would predict.

### Reading

Against run 5, which the table was written to compare with, this is the second
row: clearly less. But run 7 has since shown that run 5 was the lucky seed of the
recipe, and against run 7's +33 the halved weight's +26 is well inside the noise.
Read against the two-seed mean, the first row holds: the head's gain is robust to
the weight over a factor of two, and 1.0 stays because it is the value with the
most games behind it. The third row's question, whether the term competes with the
policy, is answered no: halving it did not help.

### Decisions taken

* `train.ownership_loss_weight` stays at 1.0.
* Weight 2.0 is not worth a run on this evidence. Halving changed nothing that
  1000 games could see, so doubling is unlikely to either, and the GPU time is
  better spent on a longer run of the big network with the head (run 6's late
  climb).

---


---

## Play-time sweep: simulations — how much of the strength is search?

**Result: search is the largest lever in the project. The same network that is
even with Edax level 7 at 256 simulations beats level 8 decisively at 3200, and
each doubling of the budget is worth roughly one Edax level up to level 8.**

**Question.** Every rating so far is at 50 or 256 simulations per move, and the
browser's strongest level searches 800 with a two-second cap. Recipe changes have
moved the agent by tens of Elo. How much does the search budget move it?

**Setup.** No training. Run 5's generation 120, the 1.0 candidate, at 50, 256,
800, 1600 and 3200 simulations per move against Edax levels 6 to 9: 80
colour-balanced games per cell, 4-ply seeded openings, no exploration noise, one
thread each side (`docs/ratings/sims-curve-e2-g120.json`; 1,600 games, 71 minutes
on 64 CPU cores).

| simulations | vs level 6 | vs level 7 | vs level 8 | vs level 9 |
|---|---|---|---|---|
| 50 | 24.4% [16%, 35%] | 6.9% [3%, 15%] | 8.8% [4%, 17%] | 11.9% [6%, 21%] |
| 256 | 51.2% [40%, 62%] | 41.9% [32%, 53%] | 17.5% [11%, 27%] | 17.5% [11%, 27%] |
| 800 | 75.0% [65%, 83%] | 58.1% [47%, 68%] | 41.2% [31%, 52%] | 40.0% [30%, 51%] |
| 1600 | 81.9% [72%, 89%] | 76.9% [67%, 85%] | 48.1% [38%, 59%] | 38.1% [28%, 49%] |
| 3200 | **91.2%** [83%, 96%] | **73.8%** [63%, 82%] | **68.1%** [57%, 77%] | 37.5% [28%, 48%] |

### Reading

Read down a column. Against level 6 the agent goes from losing three games in
four at 50 simulations to winning nine in ten at 3200. Against level 8 it goes
from 9% to 68%. Every doubling from 256 to 3200 buys about one Edax level, up to
level 8. Level 9 is the exception: 40%, 38%, 38% at 800, 1600 and 3200, flat
within the intervals, so from 800 simulations up something other than search
depth decides those games. That is the first place an exact endgame solver would
show, and it is what the `sims300` question in `configs/full8x8_sims300.yaml` is
really about.

Two calibrations of the earlier numbers. The 256-simulation cell against level 6
reads 51% here and 59% in run 5's own Edax table, same checkpoint, different
seeds: at 80 games the interval is about ±11 points, and both readings sit inside
it. And the browser's "Max" level, 800 simulations capped at two seconds, reaches
about 560 simulations on a fast laptop and fewer on a phone, so what the website
serves is closer to the 256 row than the 800 row. The network is capable of
level 8. The browser search is not letting it show that.

### Decisions taken

* The strength ceiling of the shipped agent is set by play-time search, not by
  the network. Making the browser search faster per simulation (batching several
  leaves per network call, which the `Evaluator` interface already allows) is
  worth more than any recipe change measured so far and goes into the 1.0 web
  work in place of the WebGPU hours run 6 freed.
* An exact endgame solver for the last empties is the second lever, and the one
  that addresses the level-9 plateau. It is hand-written and will be labelled as
  such wherever the agent is described.
* Every future strength claim names its simulation budget, and the model card's
  Edax line becomes a row of this table rather than one number.

---

## Calibration: are the four difficulty levels actually different opponents?

**Run:** 2026-08-31, `models/reversi-8x8-gen60.pt`, 21 pairings × 300 games,
8 CPU workers, 5.9 hours. Evidence: `docs/difficulty_calibration.json`.

**Question.** The interface offers Casual, Club, Strong and Max. Those labels
promise that moving up a rung gets you a harder game. The promise rested on an
argument — more search should play better — rather than on a measurement, which
is the one user-facing claim in this project still doing that.

**Setup.** One checkpoint, four levels, differing only in how much they search
and how they choose. Rated against each other and against the three frozen
baselines, all in a single Bradley–Terry fit anchored at random play = 0.
Colour-balanced with a seeded 4-ply opening book, no exploration noise.

Using **one** network is the point. If the rungs separate, the separation comes
from the method rather than from four different models.

### Result: S15 met, with room to spare

| Level | Simulations | Elo | 95% interval | Gap to the rung below |
|---|---:|---:|---|---:|
| Casual | 16 | 431 | 400 – 465 | — |
| Club | 64 | 623 | 588 – 664 | **+192** |
| Strong | 256 | 891 | 845 – 943 | **+267** |
| Max | 800 | 1053 | 1001 – 1115 | **+163** |

For scale, on the same run: Minimax depth-4 rates +358, Greedy +262, Random 0.

All five conditions hold. Ratings rise strictly; every adjacent gap clears the
80-Elo bar by at least double; no adjacent pair of intervals overlaps; Casual
beats Random 97.8% with a Wilson lower bound of 95.4%; and the guardrail held
over 500 of Casual's own moves, worst drop 0.326 against a limit of 0.35.

That last number is the interesting one — 0.326 against a limit of 0.35 says the
guardrail is *binding*, not decorative. Casual really does walk up to the edge of
what it is allowed to play.

### What the measurement caught

The calibration failed on its first run, at 97 guardrail violations in 500 moves.
Two separate causes, and only the second was in the shipped code.

**The check was wrong.** It took the best value over *every* move, including
moves the search never visited — whose value is a placeholder `0.0` rather than
an estimate. In a losing position where every searched move scores −0.8, that
placeholder becomes the "best" and every choice looks like a 0.8 shortfall. 97
violations became 1.

**The remaining one was real.** `_acceptable` truncated to the top three moves by
visit count *before* applying the guardrail, so the guard compared against the
best of those three rather than the best move available. A strong move that
happened to be searched less went invisible, and a weaker one passed a guard it
should have failed. Rare — one move in 500 — and a genuine deviation from the
criterion, which is worded against the best move available.

The telling detail: the TypeScript port used in the browser already applied the
two filters in the correct order. The two implementations had silently diverged,
and nothing except measuring this would have shown it. Both now apply the
guardrail first and narrow by visit count second.

### Honest limits

* **The report was reassembled.** The run played all 21 pairings and then crashed
  building its result object — `TournamentResult` was imported for type checking
  only and did not exist at runtime. The ratings here are refitted from the
  pairing scores the run logged, which is the whole input the fit needs, so
  nothing was replayed. Per-colour splits and game lengths were lost with the
  process.
* **One checkpoint.** These ratings describe generation 60. A weaker network
  would likely show narrower gaps, since a bad move costs less when every move is
  bad.
* **Not comparable to the cross-generation table.** That run anchored at Random
  through a different set of pairings, so the two scales agree on the anchor and
  on nothing else. Max at +1053 here and generation 60 at +877 there are not
  the same measurement.

### Reported from play: the easy levels feel inconsistent

A player using **generation 5 at Club** described the agent as sometimes
unbeatable, sometimes ordinary, and sometimes losing badly — across games at a
single setting. That is a real effect with three separate causes, none of them a
bug, and none of them measured.

**Casual and Club deliberately do not play their best move.** They sample:

| level | simulations | temperature | top-k | guard | plays |
|---|---|---|---|---|---|
| Casual | 16 | 0.8 | 3 | 0.35 | samples among 3 |
| Club | 64 | 0.35 | 2 | 0.20 | samples between 2 |
| Strong | 256 | 0.0 | — | 0.05 | best move |
| Max | 800 | 0.0 | — | 0.0 | best move |

Sampling is what makes a weak level *weak* rather than *stupid* — the value
guardrail keeps the alternatives reasonable, so it plays a second-best move rather
than throwing away a corner. But a level that picks between two moves plays a
different game every time, and at 64 simulations the second choice is sometimes
nearly as good and sometimes clearly worse. **Strong and Max are deterministic and
should not show this at all**, which is a prediction worth testing rather than an
assumption.

**The ratings describe a mean and say nothing about spread.** Club is +623 over
300 games per pairing. That is an average. Nothing in the calibration measured
how much one game differs from the next, so "how consistent is this level" is a
question this project has never asked. Two levels with identical ratings and very
different variances would be indistinguishable in the report and obviously
different to play.

**The combination that was played has never been rated.** Calibration ran on
`reversi-8x8-gen60.pt` — one checkpoint, four levels. The interface lets the
opponent generation and the thinking-time level be chosen independently, so
generation 5 at Club is a combination no measurement covers. The two numbers on
screen come from two different round robins: generation 5's +547 from the
cross-generation table, Club's +623 from the calibration, and those scales agree
on the anchor and on nothing else. Their combination is not +547, not +623, and
not any number this repository has produced.

**For 1.0.** Three things would close this, in order of how much they buy:

1. Measure the *spread*, not just the mean — the same pairing repeated, reporting
   the distribution of results rather than one rating. A level whose games vary
   wildly is a different product from one that does not, and right now the report
   cannot tell them apart.
2. Rate the combinations that the interface actually offers, or stop offering the
   ones that were never rated. Four levels times six opponents is 24 combinations
   presented as if each were characterised; four were.
3. Reconsider whether sampling is the right way to make a level weak. Fewer
   simulations lowers strength without adding variance; temperature lowers it by
   adding variance. The current levels change both at once, so their separate
   contributions are unknown.


---

## Measuring the spread: is a level the same opponent twice?

**Registered 2026-09-07, before the run.** This is item 1 of the three above.
`reversi spread models/reversi-8x8-gen60.pt`, one command, on the cpu partition.

The complaint is about variation, and the calibration only ever reported means.
"Inconsistent" turns out to name two different things, so both are measured.

**Within a game: the drop distribution.** After each search, the difference in
value between the move the level played and the best move the search actually
examined. The guardrail is the ceiling on that difference and the calibration
already checks the ceiling holds — 500 moves, worst drop 0.326 against a limit of
0.35, no violations. What no report has shown is the shape underneath. A level
whose drops are almost always zero but occasionally 0.3 is two opponents wearing
one name, and that is exactly what "sometimes unbeatable, sometimes losing badly"
would feel like from the other side of the board.

**Between games: the dispersion of block scores.** Each level plays
`minimax-d4` in 15 blocks of 20 games, and the variance between block scores is
compared against the variance chance alone would produce.

Two protocol choices are worth stating, because either one done the other way
would produce a wrong answer that looks fine:

*The unit is an opening pair, not a game.* Every opening is played twice with the
colours swapped, so the two games of a pair are strongly anti-correlated —
whoever the opening favours tends to win one and lose the other. Treating those
as independent draws understates the chance variance, and the measurement would
then report overdispersion that is an artefact of the protocol rather than a
property of the level. Pairs are independent of each other; games within a pair
are not.

*The reference opponent sits inside the ladder, at +358.* A level that wins 99%
of its games has almost no variance left to measure, and no amount of blocking
recovers information the scoreline never contained. Against `random` or `greedy`
the two strongest levels would sweep and the measurement would be empty. Depth-4
minimax keeps Casual near 60% and Club near 82%, which is where the statistic has
power.

**Subjects and controls.** Only Casual and Club sample their move; Strong and Max
play the most-visited survivor of the guard at temperature 0, so in a given
position they always answer the same way. Whatever dispersion *they* show is
contributed by the opening book and the opponent, not by the level — which makes
them the control for the protocol itself. They are measured at a quarter of the
games and a quarter of the moves, because Max searches fifty times as deep as
Casual and matching the sample sizes would spend nine tenths of the compute on
the two levels nobody has complained about.

### The prediction, registered before the run

| if | then |
|---|---|
| Casual and Club overdispersed, Strong and Max near 1.0 | sampling is the cause, and the fix is the choice rule rather than the search budget |
| **all four near 1.0, but Casual's drop distribution has a heavy tail** | **the variation is within games, not between them — a player meets it move by move and the scoreline averages it away** |
| Casual and Club overdispersed *and* Strong and Max overdispersed too | the opening book or the reference opponent is contributing, and the protocol needs fixing before the levels can be judged |
| all four near 1.0 and every drop distribution tight | the reported effect is not in the agent; look at the interface, the thinking-time cap, or expectation |

The second row is the one to expect. A block of 20 games averages away a great
deal, and a level with a 0.35 guard has room to play one materially worse move
per game without that ever showing up in a win rate — while a person playing it
would notice every one of them.

**No pass mark.** Unlike the calibration, this run has no criterion to meet.
Criterion S15 asks the levels to be *separated*, which is a claim about means and
is already demonstrated. There is no agreed figure for how consistent a
difficulty level ought to be, and inventing one before seeing the first
measurement would be a guess written down as a standard. What the numbers are
for is deciding item 3: whether temperature is the right way to make a level
weak, or whether fewer simulations at temperature 0 would give the same strength
with less variance.

**Cost.** About 1,700 games and 5,000 searched positions, dominated by Max at 800
simulations. Roughly three hours on eight laptop cores; well under an hour on 64.

```bash
sbatch slurm/cpu.sbatch uv run reversi spread models/reversi-8x8-gen60.pt \
    --out runs/calibration/difficulty_spread.json --workers @CPUS@
```

Evidence lands in `docs/difficulty_spread.json`.

---

## The release tournament: putting two runs on one scale

**Result: the 1.0 network is +131 Elo over the one the site serves, and the
prediction's expected row was wrong. Subtracting ratings from two tournaments
understated the gap by 40 points, because the anchor is too far away from either
network to bridge on.** The prediction table below was registered on 2026-09-07
before the run; the result sections after it were written the same evening from
jobs 7105 and 7107.

Nothing about the agent changes here. This exists because a number the project
wants to state cannot currently be stated.

**The problem.** The 1.0 network is run 5's generation 120. The network the site
serves is run 1's generation 60. Their ratings come from different tournaments:

| | rated in | its own table says |
|---|---|---:|
| run 5, generation 120 | `docs/ratings/run5-e2-ownership-crossgen.json` | +968 |
| run 1, generation 60 | `docs/crossgen.json` | +877 |

Subtracting those gives +91 and means nothing. Two Bradley-Terry fits anchored at
random play agree on the anchor and on nothing else, and the size of the
disagreement is measurable right here: depth-4 minimax is a frozen opponent that
played in both, and it rates **+523** in run 1's table and **+503** in run 5's.
Greedy rates +313 and +324. So the two scales differ by roughly 20 points on
opponents that did not change at all, and the +91 could reasonably be anything
from about +70 to about +110 before any other source of error.

That is not good enough for a release. It is also not good enough for the web
app, whose difficulty ladder sorts levels by measured rating and therefore cannot
mix two tables.

**The run.** One round robin containing run 5's checkpoints, the four frozen
baselines, **and** run 1's generation 60 as an entrant. Every pairing played,
one fit, one scale. 100 games per pairing at 50 simulations, 4 opening plies --
the same protocol as run 5's own table, so the two are comparable in method as
well as in field.

Then a dedicated 1000-game match on the one pairing that carries the release
claim, because that is this project's standard for a two-way strength statement
and the round robin gives that pairing only 100 games.

Run 1's generation 60 has to travel to the cluster to do this: run 1 was trained
on a laptop before the cluster existed, so its checkpoint exists nowhere else.
`slurm/push_model.sh` is that direction, which `fetch_run.sh` did not cover.

### The prediction, registered before the run

| if | then |
|---|---|
| **run 5's gen 120 beats run 1's gen 60 by 60-110 Elo, interval excluding 0** | **the naive subtraction was about right, the ownership head and the longer run both paid, and the release is a straightforward upgrade** |
| it beats it by more than 110 | one of the two tables was flattering its own entrants more than the frozen baselines suggested; worth understanding before quoting either historical number again |
| it beats it by less than 60 | most of the apparent gain was the difference between two fits, and the honest release note is a smaller number than the tables implied |
| it does not beat it decisively | the 1.0 choice was made on a within-run comparison that does not survive contact with the network actually shipped, and the release should wait |

The first row is what to expect. Run 5's generation 120 beat *its own control* by
+74 at 1000 games, and run 1's recipe is that control's recipe at half the
generations, so a gain in the 60-110 band is the arithmetic working out.

### What the result decides

**Which generations become levels.** Run 5's rated generations will be roughly
05, 35, 65, 100, 114 and 120, and its own table puts the top three at +984, +956
and +968 -- three networks a player cannot tell apart. Publishing all six would
give the ladder three top rungs that differ by noise, which is the same mistake
the difficulty calibration exists to prevent. So the tournament rates six and the
release publishes the subset that is actually separated, chosen from the fit
rather than in advance. On run 5's own numbers that is likely 05, 35, 65 and 120,
which with random and greedy gives six levels whose smallest gap is larger than
the smallest gap in today's ladder (22 Elo, between run 1's generations 40 and
60).

**What the release note may claim.** One number, from the 1000-game match, on one
scale, with its interval.

**Nothing about the ladder's ordering is decided by hand.** `ladder.ts` sorts by
measured rating, so the levels renumber themselves from whatever this produces.

### Cost

55 pairings at 100 games is about 5,500 games; run 5's 45-pairing table took 14.5
minutes on the cpu partition. The 1000-game match adds three pairings. Under an
hour in total.

```bash
# once, from the laptop -- run 1's checkpoint has to reach the cluster
slurm/push_model.sh models/reversi-8x8-gen60.pt <you>@slurm.csse.rose-hulman.edu

# on the cluster, from ~/reversi-zero
sbatch slurm/cpu.sbatch uv run reversi arena --suite crossgen \
    --run-id e2-ownership --max-checkpoints 6 \
    --entrant "run1-gen60=$HOME/reversi-models/reversi-8x8-gen60.pt" \
    --games 100 --simulations 50 --workers @CPUS@ \
    --out docs/ratings/release-one-scale.json \
    --notes "1.0 release: run 5 and the network 0.0 shipped, one fit"

sbatch slurm/cpu.sbatch uv run reversi arena --suite custom \
    -e "gen120=$HOME/reversi-runs/e2-ownership/checkpoints/gen_00120.pt" \
    -e "run1-gen60=$HOME/reversi-models/reversi-8x8-gen60.pt" \
    -e random \
    --games 1000 --simulations 50 --workers @CPUS@ \
    --out docs/ratings/release-head-to-head-1000.json \
    --notes "1.0 release: the shipped network against the one it replaces"
```

*As registered. The second command does not work as written: 1000 games needs 500
distinct 4-ply openings and only 244 exist. It ran at `--games 480`. See "The
1000-game match could not be run" below.*

The entrant name for run 1's checkpoint must not begin with `gen` followed by
something that is not a number. The web manifest treats a `genNN` entrant as a
playable model and parses the digits after the prefix, so `gen60-run1` would
crash the manifest build while `run1-gen60` becomes a rated reference row -- which
is what it should be, since the ladder's levels are one run's progression rather
than a mixture of two.

Evidence lands in `docs/ratings/release-one-scale.json` and
`docs/ratings/release-head-to-head-480.json`. The second file is named 480
rather than 1000 for a reason worth its own section below.

### Result: +131 Elo, on one scale, decisive

11 entrants, 55 pairings, 100 games each, 5,500 games in 21 minutes on 64 cores.

| entrant | Elo | 95% interval |
|---|---:|---|
| gen100 | 962.2 | 904.5 - 1031.3 |
| **gen120** | **959.0** | 902.7 - 1029.0 |
| gen114 | 946.1 | 887.4 - 1015.4 |
| gen65 | 881.1 | 824.6 - 944.5 |
| **run1-gen60** | **828.3** | 771.9 - 893.3 |
| gen35 | 811.5 | 755.6 - 875.2 |
| gen05 | 529.3 | 478.7 - 584.6 |
| *minimax, depth 4* | *486.0* | *435.6 - 542.5* |
| *greedy* | *355.2* | *295.5 - 419.2* |
| *minimax, depth 2* | *327.9* | *276.4 - 385.7* |
| *random* | *0.0* | - |

**gen120 - run1-gen60 = +130.7**, and the intervals do not overlap: 902.7 against
893.3, by 9 points. In the same tournament that pairing scored 74.0%
[64.6%, 81.6%] over 100 games.

The second prediction row is what happened: more than 110, so one of the two
original tables was flattering its own entrants relative to the frozen baselines.

### Why the naive subtraction was low

Every entrant moved when re-measured, and one moved much further than the rest.

| entrant | in its own table | on one scale | moved |
|---|---:|---:|---:|
| run 1, gen 60 | 877.3 | 828.3 | **-49.0** |
| run 5, gen 35 | 839.4 | 811.5 | -27.9 |
| run 5, gen 05 | 554.4 | 529.3 | -25.0 |
| run 5, gen 100 | 984.3 | 962.2 | -22.0 |
| run 5, gen 65 | 899.4 | 881.1 | -18.3 |
| minimax, depth 4 | 502.8 | 486.0 | -16.8 |
| run 5, gen 114 | 956.2 | 946.1 | -10.1 |
| run 5, gen 120 | 967.9 | 959.0 | -9.0 |

Run 1's generation 60 fell furthest, and two things about its original
measurement explain it. That table used **30 games per pairing** against run 5's
100, so its intervals were roughly twice as wide. And in that field it was the
strongest entrant, with nothing above it: a Bradley-Terry fit places the top
entrant using only its wins over weaker opponents, which bounds it from below and
barely from above. Here it has five stronger networks above it and is pinned from
both sides.

**The lesson is about which opponent to bridge on.** Both networks played
depth-4 minimax in their own tables, so the gap to that frozen opponent is a
second way to compare them without a shared fit:

| method | predicts |
|---|---:|
| subtract the ratings, both anchored at random = 0 | +90.7 |
| bridge on depth-4 minimax, the closest shared opponent | +110.5 |
| **measured in one fit** | **+130.7** |

Bridging on the nearest frozen opponent beat bridging on the anchor, and neither
was good enough. The reason is visible in the score sheet: depth-2 minimax scored
**0.0%** against gen35, gen100 and gen114, and random scores near nothing against
anything strong. A pairing that saturates carries almost no information about
where the winner sits, so an anchor 900 points below the entrants being compared
is a very long lever with nothing on the end of it. For a cross-tournament
comparison, prefer the shared opponent closest in strength -- and prefer one fit
to either.

### The 1000-game match could not be run, and 244 is why

The registered plan called for 1000 games on the pairing carrying the release
claim. That failed in fifteen seconds:

```
only found 244 usable openings of 500 asked for on a 8x8 board after 50000 attempts.
```

A match draws one opening per pair of games and plays it twice with the colours
swapped, so 1000 games needs 500 distinct openings. **There are only 244 usable
distinct 4-ply openings on an 8x8 board.** 488 games is the ceiling at this
opening depth, and the match ran at 480.

This is worth recording because it means the project's own standard -- a thousand
games or it is not a claim -- was never reachable through `reversi arena` at 4-ply
openings. The five earlier 1000-game head-to-heads got there by *chunking*: fifty
separate 20-game matches, each drawing ten openings, which reuses openings across
chunks. That is a defensible thing to do and it is not the same protocol, and
until now nothing said so.

Two ways out, neither taken yet: draw openings with replacement once the distinct
supply is exhausted, stating that in the report; or use deeper openings, where far
more distinct positions exist. Registered as an open question about the tool, not
about the agent.

### Result: the direct match agrees, and the small field does not

480 games, 4-ply openings, the same 50 simulations.

| | score | record |
|---|---:|---|
| gen120 vs run1-gen60 | **70.3%** [66.1%, 74.2%] | 328W 133L 19D |
| gen120 vs random | 99.8% [98.8%, 100.0%] | 479W 1L 0D |
| run1-gen60 vs random | 99.0% [97.6%, 99.6%] | 475W 5L 0D |

The advantage holds with either colour -- 66.5% as black, 74.2% as white -- which
is the cheap check that a result this size is a real strength difference and not
a perspective bug. Contract C1 says the network never sees colour; a gap that
appeared with one colour and not the other would say otherwise.

70.3% is **+150 Elo** as a pairwise estimate, with the score interval mapping to
+116 to +184. The round robin's pooled answer of +131 sits inside that, and the
round robin's own 100-game pairing gave 74.0%, so the tighter 480-game figure
lands between the two. All three measures exclude zero comfortably.

**The rating fit from this match should not be quoted**, and it is instructive
why. With three entrants it reports gen120 at +972 [858, 1172] and run1-gen60 at
+821 [701, 1016] -- intervals that overlap heavily, on 1,440 games. The field is
the problem, not the sample: both networks beat the only other entrant about 99%
of the time, so nothing in the data locates either of them against the anchor. A
big sample through a saturated pairing buys precision on the score and none on
the rating. **The score is the statistic here; the rating is not.** The +131 from
the 11-entrant fit is the number to quote.

### An efficiency finding, for whoever runs the next one

The round robin played 5,500 games in 21 minutes. This match played 1,440 games
in 66 minutes. The parallelism in `round_robin_parallel` is one process per
*pairing*, so a three-entrant field uses three of the 64 cores allocated and the
rest idle. Small fields belong on fewer cores, or the games within a pairing need
splitting -- which is what the chunked head-to-head scripts were doing without
saying so.

### Decisions taken

* **The 1.0 network is run 5's generation 120**, confirmed rather than merely
  chosen: +959 on the release scale, +131 over the network the site serves, both
  measures decisive.
* **generation 100 is not better.** It rates +962 to gen120's +959, and their
  head-to-head is 49.0% [39.4%, 58.7%]. gen100, gen114 and gen120 are one
  network as far as this measurement can tell, so the pre-registered choice
  stands and the table must not be read as ranking them.
* **The published ladder is gen05, gen35 and gen120**, plus the two rule-only
  baselines. Five levels, and every adjacent pair has non-overlapping intervals:

  | level | plays as | Elo | gap below |
  |---|---|---:|---:|
  | 1 | random | 0.0 | - |
  | 2 | greedy | 355.2 | +355 |
  | 3 | gen05 | 529.3 | +174 |
  | 4 | gen35 | 811.5 | +282 |
  | 5 | gen120 | 959.0 | +148 |

  Adding gen65 breaks it: at +881 it overlaps gen35 below and gen120 above.

* **The ladder that ships today is worse than this by the same test.** Measured
  at 30 games per pairing, run 1's intervals are wide enough that *only*
  random-to-greedy is separated -- greedy/gen05, gen05/gen20, gen20/gen40 and
  gen40/gen60 all overlap. Six nominal levels, one distinguishable step. Five
  separated levels is the better product and the more honest one.
* **A sixth level goes in the gen05-to-gen35 gap**, which is 282 Elo wide and the
  only place a rung would clearly separate. Generations 10 through 30 are on disk;
  rating one of them in the same fit is the follow-up.
* **run1-gen60 stays a rated reference row, not a level.** At +828 it overlaps
  both gen35 and gen65, so it is not a distinguishable rung, and a ladder's levels
  are one run's progression rather than a mixture of two.

---

## External check: how does the agent compare to Edax?

**Run:** 2026-09-01, generation 60 at the `strong` setting (256 simulations),
against [Edax 4.6](https://github.com/abulmo/edax-reversi) at a range of levels.
Colour-balanced with the arena's 4-ply opening book, Edax's own book disabled.

**Question.** Every number in this repository is the agent measured against
baselines written in this repository. The scale is anchored at a random player we
wrote and shaped by a minimax we wrote, which is internally consistent and says
nothing about the agent's standing outside it. Edax is the reference Othello
engine — open source, and downloadable by anyone who doubts the result.

### Result

| Opponent | Our score | Record |
|---|---:|---|
| Edax level 1 | 100.0% | 20–0 |
| Edax level 2 | 95.0% | 19–1 |
| Edax level 3 | 100.0% | 20–0 |
| Edax level 4 | 90.0% | 18–2 |
| **Edax level 5** | **53.1%** | **42–37–1** |
| Edax level 6 | 35.0% | 7–13 |
| Edax level 8 | 17.5% | 3–16–1 |
| Edax level 10 | 12.5% | 2–17–1 |
| Edax level 12 | 0.0% | 0–20 |

Levels 1–4 and 6–12 are 20-game scouting runs, enough to locate the crossover and
not enough to quote. **Level 5 was then measured over 80 games: 53.1%, 95%
interval [42.3%, 63.7%].**

**The claim that interval supports:** at 256 simulations a move, generation 60 is
*indistinguishable from Edax at level 5*. Not "beats" — the interval spans even,
and saying otherwise would be reading a point estimate as a result. That is the
honest form of a crossover: the level at which neither side is measurably ahead.

### What it does and does not say

The agent learned Othello from nothing but self-play, with no human games, no
opening book and no hand-written evaluation, in nine hours on a laptop GPU. It
plays a recognised engine's five-ply search to a draw. That is the first
statement about this agent that does not depend on anything else in this
repository.

It is also a modest level. Edax's default is 21, and at level 12 it wins every
game. A 458k-parameter network trained for nine hours was never going to trouble
that, and a comparison that only reported the loss would have been worth little —
which is why the experiment searched for the crossover rather than playing one
match at full strength.

### Honest limits

* **`strong`, not `max`.** 256 simulations rather than 800, because 800 costs 1.7
  seconds a move and the scouting sweep would have taken a day. `max` would place
  somewhat higher; how much is unmeasured.
* **Edax levels are not an Elo scale.** Level 5 is a search depth, not a rating.
  "Level with Edax 5" locates the agent against a reproducible reference; it does
  not convert to a published Othello Elo.
* **Not in the Bradley–Terry table.** These are head-to-head matches, not a round
  robin, so Edax does not appear in `docs/crossgen.json`. Folding it in would put
  every rating in this project on an externally anchored scale, and is the
  obvious next step.
* **The scouting rows are 20 games** — roughly ±20 points. They locate the
  crossover and should not be quoted as measurements.

### Getting Edax

**On Linux, build it.** The v4.6 release's Linux binary is linked against
glibc 2.38 and exits at once on anything older (Debian 12 ships 2.36) with
`version 'GLIBC_2.38' not found`, which the adapter reports as "edax-l5 is no
longer running". Building from source takes a minute:

```bash
git clone --depth 1 --branch v4.6 https://github.com/abulmo/edax-reversi.git edax-src
mkdir -p edax-src/bin && cd edax-src/src
make build ARCH=x86-64 OS=linux CC=gcc COMP=gcc
cp ../bin/lEdax-x86-64 <repo>/tools/edax/
```

`tools/edax/data/eval.dat` still comes from the release tarball. The `bin`
directory has to exist before the build, and the Makefile defaults to clang
unless `CC=gcc` is given.

Not in this repository: it is a binary and a 14 MB evaluation table, and the rule
that keeps checkpoints out of git applies to it too.

```bash
gh release download v4.6 --repo abulmo/edax-reversi   --pattern '*MS-windows*' --dir tools/edax
# then unzip it there, so that tools/edax/data/eval.dat exists
```

`tools/edax/` is gitignored. The adapter finds it by convention and, when it is
absent, says how to fetch it rather than failing obscurely; the tests that need
it skip.
