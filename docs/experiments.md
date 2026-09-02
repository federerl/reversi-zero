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

## Planned: Run 2 — the value head — `configs/full8x8_value4.yaml`

**Written before the run, so the prediction cannot be adjusted afterwards.**

**Question.** Run 1's value head stopped improving at generation 5 and never
moved again, while the policy kept learning for fifty-five more generations. Does
the plateau lift if the value term is weighted more heavily?

**Change.** `train.value_loss_weight` from 1.0 to 4.0. Nothing else. The two
resolved configs were compared field by field: of 45 settings, exactly two
differ — that one, and the profile's name.

**Why this and not more simulations.** Raising `n_simulations` to 300 is also
worth doing, and run 1's config records that the reduction to 200 rested on a
confounded benchmark. But it aims at the wrong target for *this* plateau. More
simulations improve the **policy** target — a better-searched move distribution.
The value target is not produced by the search at all; it is the game's final
result. More search does not make it easier to predict, and the value head is
what stopped moving. `configs/full8x8_sims300.yaml` is prepared for run 3.

**Same seed (1337) and same self-play cost**, so generation 60 here compares
directly against generation 60 of run 1 — same games, same simulations, same
wall clock, one different number.

**Predictions, in advance:**

| if | then |
|---|---|
| value loss < 0.60 **and** this run's gen 60 beats run 1's with non-overlapping intervals | the weight was the constraint |
| value loss < 0.60 but no strength gain | the value head learned more and it did not matter — the plateau is elsewhere |
| value loss stays near 0.64 | capacity or irreducible noise; next test is a wider value head |
| policy loss rises materially | 4.0 is too aggressive and the trunk was starved of the policy signal |

The third and fourth rows are the ones worth watching. A negative result here is
still a result: it eliminates the cheapest explanation and points at the head's
size, which is the more expensive thing to change.

**To run it:**

```bash
python -m reversi.cli train --config configs/full8x8_value4.yaml --generations 60
```

About nine minutes per generation, so roughly nine hours. If it is stopped early,
resume with the run id it printed — `--generations` is a total, not an increment,
so the same 60 means "finish the 60", not "do 60 more".

```bash
python -m reversi.cli train --config configs/full8x8_value4.yaml --generations 60 --run-id <the id>
```

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
