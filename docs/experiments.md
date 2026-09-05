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
  of the same recipe differ by about 25 Elo. A recipe change that shows less than
  that against a single control has not shown anything.
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

## Run 6 — E1+E2, capacity and the ownership head together — `e12-10x128-ownership`

**Result: pending.** This section was written before the run was submitted.

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

Not in this repository: it is a binary and a 14 MB evaluation table, and the rule
that keeps checkpoints out of git applies to it too.

```bash
gh release download v4.6 --repo abulmo/edax-reversi   --pattern '*MS-windows*' --dir tools/edax
# then unzip it there, so that tools/edax/data/eval.dat exists
```

`tools/edax/` is gitignored. The adapter finds it by convention and, when it is
absent, says how to fetch it rather than failing obscurely; the tests that need
it skip.
