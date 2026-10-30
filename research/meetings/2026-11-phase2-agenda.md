# Meeting with Dr. Boutell — after Step 3

**Early November 2026 · Lifei Liu · follow-up to the late-October meeting**

**Purpose:** Step 3, the pre-registered opponent-diversity experiment, has finished and reads as a controlled negative result. This agenda sets out what it found, whether the finding is a property of the algorithm, the plan for the Phase 1 report, and how far the work stands from a publishable paper, so that the scope of Phase 2 can be decided.

Every number below comes from `research/step3-results.md` (PR #62) and the result files under `research/results/step3/`; the tables can be regenerated with the scripts merged in PR #61. The report plan is `research/phase1-report-outline.md` (PR #63).

---

## 1. Where Step 3 landed

Five runs of twenty generations each, continued from the control recipe's generation 120 on one L40S (1.3 to 1.5 GPU-hours per run): two seeds of pure self-play (group A), two seeds with a fifth of the training examples from games against Greedy and Minimax-d4 (group B), one seed with a value target that is half game result and half disc share (group D). Every final checkpoint and the starting checkpoint then played the frozen thirteen-opponent roster: 78 pairings of 200 games, all recorded.

| finding | evidence |
|---|---|
| **The opponent mix did not transfer and barely took.** | Group B's wipeouts against Greedy itself went from 65 to 53 on one seed and 64 on the other. Its held-out rows (Edax L3/L5/L7, random-playout search) and its seen-family rows are inside the two control seeds' range. Three of the five pre-registered predictions failed. |
| **The reason is how the external games were recorded, not opponent diversity as such.** | The head probe shows the value head moved at the network's own position (+0.99 to +0.84) but not one ply later, at the position with Greedy to move, which is the position the search reads when it judges a move (−0.98 before and after; the exact answer is +1). Only the network's moves were recorded from external games, so that position was never a training example. The prior on Greedy's wiping reply is unchanged at 4 to 5%. |
| **The weakness is opponent-general.** | The random-playout search, a held-out opponent with no network and no hand-written knowledge, wipes out the self-play networks in a sixth to a fifth of its games with the Step 2 signature (one disc, raw value +0.99, a surviving move available). It is the one held-out opponent that measures the weakness directly. |
| **A changed value target removes the weakness, seen and unseen, at a steep cost.** | Group D cut wipeouts against Greedy, the Greedy family, and the random-playout search (34 to 19). It lost about half its strength against Edax (L5 29.8% to 17.2%, L7 10% to 4%) and against the generation-60 checkpoint (62.5% to 40.8%), and its value head is no longer a probability. The ownership head got a comparable cut in Step 2 without that cost. |
| **Calibration is unchanged.** | Every self-play network is calibrated on its own positions (gap at most 0.02) and overrates itself by 7 to 12 points against Edax, groups A and B alike. |

Gate C, as written: "advance to a write-up only if the training ran as specified and either shows a clear held-out improvement without unacceptable regression, or a controlled negative result." The second branch holds.

## 2. Is this a limitation of the algorithm?

The question is whether the weakness belongs to policy-value network plus MCTS, or to something else. The evidence separates two things.

**What belongs to self-play as a source of training data, and which the algorithm inherits:**

- Self-play is policy iteration on the agent's own distribution of positions. It guarantees nothing about positions another opponent forces. Wang et al. (2023) showed this against KataGo, the same algorithm at superhuman strength; the Step 2 and Step 3 results are the same phenomenon in an environment small enough to take apart.
- The value head extrapolates with full confidence (+0.99) into positions it has never seen. A learned evaluator has no notion of its own ignorance unless it is given one.
- PUCT's exploration is gated by the policy prior, and the prior was trained only on the agent's own moves. In self-play the opponent is the agent, so the prior never learns what a disc-hungry opponent does; at 50 simulations the search never expands the wiping reply.

**What does not belong to the algorithm, with direct evidence:**

- The search can see the escape when the prior is removed: value-only search escapes 98 to 100% of recorded wipeouts (Step 2, section 7). The information is in the network; the prior hides it. More simulations also reduce the losses.
- The same network and the same search, with a different training target (the ownership head in Step 2, the material target in Step 3), halve the wipeouts. Capacity is not the limit; the training signal is.
- Group B's null is a recording rule, one line of code, not a property of the method.

The accurate statement for the report: this is the known limitation of a learned evaluator trained on self-play alone, inherited by policy-value MCTS but not created by it. Edax is free of it not because alpha-beta beats MCTS but because its evaluation was tuned on diverse games and its endgame search is exhaustive. The contribution is to have traced the limitation to a specific belief, a specific data gap, and a specific recording rule, and to have shown which repairs work and what they cost.

**To discuss:** is this the right framing for the report's discussion section, and for the literature positioning (Wang et al. 2023, Lanctot et al. 2017 on self-play overfitting, Czempin and Gleave 2022 on population training as the remedy)?

## 3. The Phase 1 report

The outline in PR #63 plans twelve to fifteen pages: introduction, background for a reader with one course of RL, related work, method (frozen protocol, held-out roster, pre-registration), three results sections (Steps 1 to 3), discussion, next experiments, reproducibility, appendices. Eight of the thirteen tables and figures exist; five plots are still to draw from committed JSON. Target remains mid-November.

**To discuss:**

- Whether a draft of the three results sections should come first, before the rest is written.
- Format: Markdown rendered to PDF, or LaTeX.
- Whether the one cheap follow-up (section 4, item 1) should be run now so that it can appear in the results, or be listed as future work only.

## 4. From a report to a paper

A rough placement: the work is perhaps 60% of the way to a workshop paper and about a third of the way to a main-conference paper. The gap divides into what a reviewer would demand before reading further and what would make the paper worth citing.

### 4.1 Required for any paper

| gap | why it blocks | cost |
|---|---|---|
| **The central intervention is a null explained by a design flaw.** | The first review comment will be "fix the recording rule and rerun." The corrected group (B′: both sides of every external game recorded for the value target) is the paper's main experiment, not an option. The probe says exactly what should move: the child value at the opponent-to-move position. | Two seeds, about 3 GPU-hours, plus 26 evaluation pairings on CPU; about a week including analysis. |
| **Too few seeds.** A and B have two, D has one. | Two seeds of one recipe differed by 48 rating points earlier in the project; B's two seeds differed by 21 versus 39 wipeouts against the random-playout search. Effect and noise cannot be separated at n = 2. | Three to four seeds per group (A, B, B′, D): about eight more runs, about 12 GPU-hours, plus evaluation. |
| **One starting checkpoint, twenty generations.** | Every group is a continuation of `control120`. The conclusion could be a property of that one set of weights. | One B′ run of 120 generations from scratch, about 8 to 9 GPU-hours, or a second starting checkpoint. |
| **No comparison with the known remedy.** | The literature's answer to opponent-specific weakness is population-based training (Czempin and Gleave 2022). Without it, "fixed opponents do not transfer" has no reference point. | The most expensive item: implementing a population or league in the self-play loop, two to three weeks of hands-on time, then training. |

### 4.2 What would make the paper worth citing

- **Group C: Edax in the training mix.** The question asked on October 22 and the natural contrast "is a mature engine a better teacher than hand-written opponents?" Needs the Edax-per-move smoke test first.
- **Turn the mechanism into a measurable quantity.** The story is "a data gap produces a confident wrong belief." If the frequency and win rate of low-disc positions in the replay window predict exploitability, that is a result that outlives Othello.
- **Dose curves instead of single points.** The mix share was 20% only; the material weight 0.5 only; the search budget three points. Each should be a curve.
- **An adaptive adversary.** All current opponents are fixed. Wang et al. trained the attacker; a paper that uses only fixed opponents has to say so, and reviewers will ask.
- **A second environment**, even 6×6 Othello, as the research plan's Phase 2 already lists.
- **The two secondary held-out opponents** the roster planned but did not obtain: Egaroucid or a published n-tuple evaluator, and a human-imitation policy from WTHOR games.

### 4.3 Already at paper standard

The frozen protocol and pre-registration; the held-out roster's admission rule and similarity check; the Step 2 dissection (value head versus policy head, the training-data statistics, the independent reproduction of the weakness by a knowledge-free search); the reproducibility layout. These are what make the report defensible as it stands.

## 5. Proposed Phase 2 scope

Target a workshop paper, not a main conference. Do the first three rows of section 4.1 (B′, seeds, a from-scratch run): under 30 GPU-hours in total, with hands-on time as the bottleneck. Decide separately on the population baseline, since it is the one item that opens a main-conference route and the one that costs weeks.

Order: B′ first, because it is cheap, it is the paper's main experiment, and if it lands before mid-November it closes the report instead of being future work. Seeds second. Everything else after the report.

## 6. Decisions needed

1. Is the framing in section 2 (a known limitation of self-play-trained evaluators, dissected rather than discovered) the right one for the report and for positioning against the literature?
2. Run B′ now, in parallel with writing, so that it can land in the report? Or report first, then run?
3. Report format and whether a results-only draft is wanted first.
4. Phase 2 target: workshop paper with the three required items, or the main-conference route with the population baseline?
5. Which of the section 4.2 items, if any, is worth doing before the report deadline? Proposal: none; they all go after.
6. Is Group C (Edax in the mix) in Phase 2 or out?
