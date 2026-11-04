# Meeting with Dr. Boutell — the Step 3 series is finished

**November 2, 2026 · Lifei Liu · follow-up to the late-October meeting**

**Purpose:** the pre-registered opponent-diversity experiment (Step 3) and its two follow-ups (Step 3b, Step 3c) have all run. Together they turn a null result into a located fault: the Greedy weakness is not something the network fails to know, it is something a 50-simulation search fails to look at, and one exact rule in the search removes most of it on every checkpoint without training. This agenda sets out the chain of evidence, the explanation it supports, what changes in the Phase 1 report, and what Phase 2 should be.

This agenda supersedes the early-November one written after Step 3 alone, which proposed "fix the recording rule and rerun" as the paper's main experiment. That rerun is Step 3b, and it came back null. Every number below is in `research/step3-results.md` (PR #62), `research/step3b-results.md` (PR #64) and `research/step3c-terminal-check.md`, with the JSON behind each table under `research/results/`.

| | | |
|---|---|---|
| 1 | The chain: Step 3, 3b, 3c, in one table | ~15 min |
| 2 | What the fault is, and what it is not | ~10 min |
| 3 | The Phase 1 report: what changes in the outline | ~10 min |
| 4 | From a report to a paper, revised | ~10 min |
| 5 | Decisions | ~5 min |

---

## 1. The chain

All three steps start from the same checkpoint, the control recipe at generation 120, and are evaluated under the same frozen protocol: 200 games per pairing, 50 simulations per move, fixed seed, every game recorded. The Step 2 weakness they chase: against Greedy, every loss is a wipeout; the network holds one to three discs, its value head calls the position a near-certain win, Greedy's reply takes every disc, and the search never expanded that reply because the policy head gave it a 4% prior.

| step | what was changed | pre-registered expectation | what happened | wipeouts vs Greedy (control: 65) |
|---|---|---|---|---|
| **3** (two seeds each) | a fifth of training games played against Greedy and Minimax-d4, twenty generations | wipeouts fall sharply, held-out rows move | neither; the value at the position the search reads was unchanged (−0.98, exact answer +1) | 53, 64 |
| **3** group D (one seed) | value target half game result, half disc share | removes the weakness at a cost | removed most of it, seen and unseen (rollout row 34 → 19), and lost half its strength against Edax and its own earlier generation | 44 |
| **3b** (two seeds) | same mix, but the opponent's positions also recorded, value-only, so the value head is trained exactly where the search reads it | the value there rises toward +1 and wipeouts fall | the value did not move at all (−0.98, −0.99); nothing else moved; the training data held about 80 rows carrying the lesson against four thousand look-alikes saying the opposite | 61, 69 |
| **3c** (no training) | one rule in the evaluation search: before trusting the network's value for a position, play each legal move and see whether one ends the game with a win | if the wipeouts vanish, the weakness is a search-depth artefact | 65 → 12 against Greedy, score 67.5% → 94.0%; the Greedy family 59–74% → 94–97%; the held-out rollout row 81.0% → 93.2%, 34 → 12 wipeouts; Edax and retention unchanged; on the Step 2 fatal positions the checked search escapes 95–98% of the time on all five checkpoints, the plain search 0% | **12** |

The random-playout search, the one held-out opponent that attacks the same weakness without any hand-written knowledge: control 81.0% with 34 wipeouts; Step 3b seeds 80.8% with 36 and 37; with the check 93.2% with 12. The rule was never shown this opponent; it transfers because it reads the board rather than learning from games.

Gate C as written ("a clear held-out improvement without unacceptable regression, or a controlled negative result") is met on the second branch by Steps 3 and 3b, and the controlled negative now comes with the mechanism measured directly rather than inferred.

## 2. What the fault is, and what it is not

The late-October framing was "a value-head belief learned from a self-play data gap, inherited by policy-value MCTS." Steps 3b and 3c sharpen it in two places.

**It is not a data-coverage gap that more data closes.** Step 3b put the fatal positions into the training set, about 2,900 opponent-to-move positions per generation, and the value head learned exactly what the data said: that a position with few discs, late in the game, opponent to move, is usually fine for the side with few discs. It is. Two thirds of those rows are wins for the network. What separates the 80 fatal rows from the rest is not the state but whether one specific reply happens to flank every disc. That is a fact about one move, read off the board in microseconds, and it is not a property a value head can generalise from, because the rows that share every feature it can see are labelled 2 to 1 the other way. Group D moved the value by brute force, relabelling every low-disc position as bad, and paid in calibration and in strength everywhere else.

**It is a search-depth artefact, and an exact rule removes it.** The answer sits one move below a position the search already visits. The prior hides the reply; the value head says there is nothing to look for; 50 simulations are not enough to override both. A look-ahead that costs no network calls reads the answer and removes 82 to 88% of the wipeouts against every disc-hungry opponent, on every checkpoint, with no training and no effect on the rest of the game. The twelve games that remain are two further search effects (section 5 of the Step 3c document): the first-play-urgency rule keeps re-visiting moves already proven lost, and one trap was set two moves out.

The accurate statement for the report's discussion: the limitation belongs to learning an evaluator from game results by self-play, and to a search whose exploration is gated by a prior learned the same way. It does not belong to the architecture's capacity (the value-only search finds the escape) or to the opponent mix as such (the mix produced the states; the lesson inside them is not learnable by value regression). What makes it visible in Othello and not in Go is the wipeout rule: a game can end from the middle of the board in one move, so a one-ply tactic has terminal consequences. The repair is the standard one from classical game programs, resolve what can be resolved exactly before asking the evaluator, and the contribution is to have shown, by elimination, that it is the only one of the three routes that works at a cost worth paying.

**To discuss:** is "dissected rather than discovered" still the right register, now that the dissection ends in a fix? The known-limitation framing (Wang et al. 2023 for the exploitability; Lanctot et al. 2017 for self-play overfitting) still holds; the new part is that the exploit here is a tactic the search can solve, not a strategic hole.

## 3. The Phase 1 report

The outline in PR #63 planned twelve to fifteen pages with three results sections (Steps 1 to 3). Two things change.

- **Results III grows two subsections.** 7.9 Step 3b: the one-variable follow-up and its null, with the lesson-row count as the explanation (one table). 7.10 Step 3c: the check, the scorecard with and without it, the probe's fourth column, the twelve residual games (two tables). The abstract's last sentence changes from "the next experiment is to record both sides" to the finding itself.
- **The discussion's third claim changes.** From "the reason is the recording rule, not opponent diversity as such" to "the lesson is a one-move tactic the value target cannot carry and the search budget does not reach; it is recoverable by exact search, not by training." Claims (i) and (ii), the value-head belief and its opponent-generality, stand.

Every table in the outline exists as Markdown in the results documents, and so do the three added here; four of the five figures (the recording diagram, the wipeout move-number histogram, the scorecard dot plot, the reliability curves) are still to draw from committed JSON. Target remains mid-November. Writing order as planned: results sections first.

**To discuss:**

- Whether a draft of sections 5 to 7 (the three results sections, now with 7.9 and 7.10) should go to the advisor before the rest is written.
- Format: Markdown rendered to PDF, or LaTeX.
- Whether the Step 3c residual repairs (exclude proven-lost moves from the final choice; a two-ply check at the root) are worth one more evaluation day before the report, so that the "95%" becomes a measured ceiling or does not. Proposal: no; list them as next experiments.

## 4. From a report to a paper, revised

The late-October estimate was 60% of the way to a workshop paper. The gap table changes.

| gap, as stated in October | status now |
|---|---|
| "The central intervention is a null explained by a design flaw; fix the recording rule and rerun." | **Resolved.** The rerun is Step 3b; the null stands with the flaw removed, and Step 3c says why. The paper's main result is no longer an intervention that should have worked; it is a located fault and a repair that does. |
| Too few seeds (A and B two, D one). | Still true for the training groups. Less of a block than it was: the headline result (Step 3c) is a measurement on fixed weights, exact where it fires, and it reads the same on all five checkpoints, so seed noise is not what it rests on. The D result still needs a second seed. |
| One starting checkpoint, twenty generations. | Still true. The Step 3c probe spanning five checkpoints from four recipes weakens it; a from-scratch run would settle it. |
| No comparison with the known remedy (population-based training). | Still true, and now a different question: the population remedy addresses strategic exploits, and the fault here is tactical. A reviewer may ask it anyway. |

**New items the results open:**

- **The check in self-play as a training intervention.** Step 3c turned it on only at evaluation. On in self-play, the network would see positions resolved exactly where it now sees its own wrong guess; whether the value head then learns the low-disc positions correctly, and what that does to strength, is a two-seed, 3 GPU-hour experiment with a clean prediction. It is the natural Phase 2 experiment and the one that connects the search finding back to the learning question.
- **The search-budget dose curve.** Step 2 has 50, 256, 800 simulations without the check; the check at 50 beats the plain search at 256. A curve with and without the check, four budgets, is a CPU-only afternoon and makes the "depth, not knowledge" claim quantitative.
- The items listed in October that stand unchanged: Group C (Edax in the mix), an adaptive adversary, a second environment, the two secondary held-out opponents.

## 5. Proposed Phase 2 scope

Target a workshop paper. The main result is now the located fault plus the one repair that works and the two that do not; the training experiments are the elimination, not the headline. Phase 2 in order of cost:

1. The search-budget dose curve with and without the check (CPU, one day).
2. The check in self-play, two seeds, with pre-registered predictions on the fatal-position value, the Greedy row, the rollout row, Edax and retention (about 3 GPU-hours plus evaluation).
3. A second D seed, so the one signal-only result has an interval.
4. Decide separately on the from-scratch run and the population baseline.

## 6. Decisions needed

1. Is the section 2 framing, a search-depth artefact of a one-move tactic, recoverable by exact search and not by training, the right one for the report and for the literature positioning?
2. Report: results-sections draft first, or the whole document at once? Markdown-to-PDF or LaTeX?
3. Do the Step 3c residual repairs go in the report as measurements or as future work? Proposal: future work.
4. Is "the check in self-play" the right Phase 2 headline experiment, or should the population baseline come first despite its cost?
5. Review order for the open pull requests: #62 (Step 3 results), #63 (report outline; its agenda is superseded by this one), #64 (Step 3b), and the Step 3c branch once it is up.
