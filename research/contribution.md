# What This Study Contributes, and What It Does Not

**Purpose:** the Step 1 to 3c results are in; this page says which of them carry the study's claim, which are known results restated in a small environment, and what the claim should be when the report and any paper are written. It is the positioning the [Phase 1 report outline](phase1-report-outline.md) section 8 and the [November 2 agenda](meetings/2026-11-02-advisor-meeting-agenda.md) section 2 rest on.
**Sources:** the [Step 2 analysis](step2-failure-analysis.md), the [Step 3](step3-results.md), [Step 3b](step3b-results.md) and [Step 3c](step3c-terminal-check.md) results, and the [literature reviews](lit-reviews/README.md).

## 1. The claim in one sentence

An opponent-specific weakness of a self-play agent can be *tactical* rather than *strategic*: a short, exactly calculable sequence of moves, not a misjudgement of the position. For that kind, the remedies the literature recommends, more diverse training opponents and more training data at the positions where the search reads, are shown by controlled experiment not to work; a cheap exact search does; and the reason the learned remedies fail can be counted in the training data.

## 2. What is not new

Three things in these results are known, and the report should say so before a reader does.

- **A one-move look-ahead for game-ending replies.** Resolving what can be resolved exactly before asking an evaluator is standard practice in classical game programs, and AlphaZero-style search also scores a finished position exactly when it reaches one. The Step 3c rule reaches it one move earlier. A result whose headline is "a terminal check raised the score against Greedy from 67.5% to 94.0%" has no claim in it.
- **Self-play agents can be beaten by simple policies.** Wang et al. (2023) showed it against KataGo at superhuman strength, with a trained adversary. The Step 1 and 2 results are the same phenomenon in an environment small enough to take apart, which is their use, not their novelty.
- **Self-play overfits to itself.** Lanctot et al. (2017) named it; Czempin and Gleave (2022) proposed population-based training as the remedy. That the training distribution is the agent's own is the starting point here, not a finding.

## 3. What is new

Four things, in the order of how much weight they can carry.

**A causal chain measured at every link, not inferred at any.** Step 1 reproduced the pattern under a frozen protocol and found that every loss to Greedy is a wipeout. Step 2 located it in the value head (one disc rated a near-certain win) and in the prior (the wiping reply at 4%), and showed with a knowledge-free held-out opponent that the hole is not Greedy's. Step 3 tested the data-side remedy the literature points to, opponent exposure, with pre-registered predictions, and it failed. Step 3b tested the one-variable repair Step 3 proposed, training the value head exactly where the search reads it, and it failed too, and the training rows were counted to say why. Step 3c tested the search-side alternative, found it works, and read the residue. Most exploitability work stops at "the hole exists and an adversary can be trained to find it". Following one hole from the game record to the training rows and back, and testing each repair by elimination, is the part no prior work on this question has done.

**A countable reason why value regression cannot learn a tactical exception.** Step 3b, section 5: in twenty generations of training with the opponent's positions recorded, the trainer saw between about 80 and 170 rows carrying the lesson (few discs, opponent to move, opponent won; the count depends on where the disc and ply filters are drawn, Step 3b results, section 5) and more than four thousand rows that look the same to a value head and say the opposite, because a side with few discs in the middle of an Othello game usually does win. The value head learned the average, and the average is right. This is not a fact about Othello. Any evaluator trained by regression on game results will learn the typical outcome of a state and miss an exception that depends on one reply, when the exception is outnumbered twenty to one by states it is assumed unable to tell apart. That assumption is what the counts support; no run has tested it (section 5 below). The count makes the failure quantitative: it says how rare the lesson was relative to its look-alikes, and it predicts (section 5 below) when the same failure will recur.

**A split in the literature's remedy.** Opponent diversity is recommended against opponent-specific weakness in general. Step 3 shows what it does and does not do here: it put nine thousand low-disc positions into a training window where self-play had put fifty, so it reached the state; the lesson inside the state was not learnable by value regression, so it did not take. The recommendation is right for weaknesses of judgement, where seeing more positions of a kind changes the estimate of that kind. It does not reach weaknesses of calculation, where the estimate of the kind is already right and the loss turns on one move. Stating the limit of a remedy, with the experiment that finds it, is worth more than one more case where the remedy worked.

**The method.** A frozen evaluation protocol audited before any experiment; a held-out roster with a written admission rule and a measured similarity check; a held-out opponent with no learned or hand-written knowledge that reproduces the weakness independently; three experiments with predictions written before the runs, two of which failed and are reported as such. For a one-term undergraduate study this is the standard a reader will notice first.

## 4. Tactical and strategic weaknesses, and the horizon between them

The words come from chess, where they are used the way they are used here.

A **tactical** weakness is a loss that turns on a short, concrete sequence of moves whose outcome can be calculated exactly: here, one reply that takes every disc and ends the game. The position before it is not misjudged in general; it is misjudged in one specific line. Calculation finds the line, and finds it faster than any amount of learned judgement, because the line's outcome is a fact about the rules.

A **strategic** weakness is a misjudgement of what a position is worth that no calculation of affordable length corrects: overvaluing material, undervaluing mobility or corners, trading the wrong thing for the wrong thing over forty moves. The loss arrives slowly, there is no single refuting line to find, and a deeper search helps only in proportion to how much of the long game it can see. Improving judgement means improving the evaluator, which means learning.

The two are not a binary. What separates them is the **horizon**: how many moves an exact calculation has to read before it can confirm the outcome without asking the evaluator. The wipeout here has a horizon of one move. A capturing race in Go has a horizon of the whole race, often tens of moves, but it is still a calculation: counting liberties and reading out the fill is something a beginner can do exactly, and classical Go programs did it with dedicated life-and-death and capturing-race readers rather than with their general evaluation. A positional misjudgement has no horizon at which calculation settles it.

Where a weakness sits on that axis decides what fixes it, and the fix differs from one end to the other.

| | horizon of one move (here) | horizon of a long forced line | no finite horizon |
|---|---|---|---|
| what is wrong | one calculable reply is never looked at | one calculable line is never read to the end, and the evaluator misjudges every position along it | the estimate of a kind of position is off |
| what fixes it | exact resolution at the node (Step 3c) | a dedicated exact reader for that kind of line, or a search budget that reaches the end despite the evaluator | a better evaluator: more or different data, a different target |
| what opponent diversity does | reaches the state, cannot teach the reply (Steps 3, 3b) | teaches the evaluator about such lines, if enough of them with enough variety are seen; the adversary then finds the next one | teaches the kind, if the new opponents produce it |
| what the value target can carry | not the exception, when its look-alikes outnumber it (Step 3b, section 5) | the average over such positions, which is right most of the time and wrong in the race | the shift in the average, which is the point |
| cost of the fix | microseconds a position, no network calls | a solver per tactic, or search that scales with the line's length | GPU hours, and a calibration or strength cost elsewhere (Step 3 group D) |

The test that places a weakness on the axis is the one Step 2 and Step 3c applied: leave the learned evaluator alone, add exact calculation of a chosen depth, and ask whether the loss goes. The Step 2 budget series (50, 256, 800 simulations: wipeouts fall by a third to a half per step and never to zero) looked, on its own, like a long horizon or none. The one-ply check at 50 simulations, which removes 82 to 88% of the wipeouts and more than 256 simulations do, says the search was not short of depth in general but of one specific ply that the prior kept it from spending.

The cyclic attack on KataGo (Wang et al. 2023) reads as the middle column. The decisive error is one move, failing to start filling liberties in a race the network has misjudged, and the correct move is exactly calculable, so the weakness is tactical in kind. But the horizon is the length of the race, the value network misjudges every position along it, and the policy network does not propose the fill, so search receives no help from either head: the adversary still wins 72% of games against KataGo at ten million visits. The weakness here is the same kind at the shortest possible horizon, which is why one cheap rule removes it and why the result can be placed by experiment. The general statement both support: a learned evaluator with a generic search inherits the tactical blind spots that classical programs covered with dedicated exact readers, and the cost of recovering each one by search alone grows with its horizon. The terminal check is the smallest such reader.

## 5. What would make the claim stronger

In order of cost.

- **The lesson-to-look-alike ratio as a predictor.** The replay data of every Step 3 and 3b run are on the cluster. For each run, count the rows that carry the wipeout lesson and the rows that match them on everything a value head can see, and set the ratio against whether that run's value head moved at the fatal position (the probe says it did not, for every run). One ratio and one outcome per run is a weak test with seven runs; the stronger form varies the ratio on purpose, by weighting the lesson rows in the loss, and predicts the value head moves once the ratio crosses a threshold. This turns the count in Step 3b from an explanation into a measurable quantity.
- **A search-budget dose curve with and without the check.** Four budgets, both settings, the Greedy and rollout rows. CPU only. It makes "one ply, not depth in general" a curve rather than two points.
- **A second tactic, or a second game.** One wipeout rule in one game is one example. A second one-move tactic with the same signature (a usually favourable state, one refuting reply, a lesson outnumbered in the data) in 6×6 Othello or another game would make section 3's second point a regularity instead of an observation. This is the Phase 2 item that most changes what the claim can be.

## 6. What this means for the report

- The headline is not the 94%. It is the sentence in section 1, with the chain in section 3 as its evidence and the split in section 4 as its interpretation.
- The two nulls are results, not setbacks: each excludes a remedy, and the exclusion is what places the weakness on the tactical side.
- The discussion should state the limits plainly: one game, one tactic, one starting checkpoint, two seeds per training group, no adaptive adversary, and a repair that is standard practice whose contribution is where it was shown to be the only working one, not that it works.
