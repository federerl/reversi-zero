# Meeting with Dr. Boutell — the training route into the policy head, Steps 4 to 8

**November 9, 2026 · Lifei Liu · follow-up to the November 2 meeting**

**Purpose:** at the November 2 meeting the Step 3 series had ended in a located fault: a one-move tactic that the value head cannot learn and that one exact rule in the search removes without training. The week since has run five more experiments on the other half of the question, whether any *training* route reaches the tactic, and one of them does, through the policy head. This agenda sets out that chain, the one correction made to its wording after an audit, what the claim is now, and what the Phase 1 report should say, since writing starts this week.

Every number below is in `research/step4-results.md` to `research/step8-results.md` and the pages they link, with the JSON behind each table under `research/results/`. The [contribution page](../contribution.md) holds the claim as it stands.

| | | |
|---|---|---|
| 1 | The chain since November 2: Steps 4 to 8, in one table | ~15 min |
| 2 | What the claim is now, and one correction | ~10 min |
| 3 | The Phase 1 report: what it says and when | ~10 min |
| 4 | What runs during the writing, and Phase 2 | ~5 min |
| 5 | Decisions | ~5 min |

---

## 1. The chain since November 2

All five steps start from the same checkpoint, the control recipe at generation 120, train twenty more generations, and are evaluated under the frozen protocol of Step 1: 200 games per pairing, 50 simulations a move, fixed seed, every game recorded. Two seeds per group. The reading that matters most is the **held-out random-playout row**: an opponent with no learned or hand-written knowledge that finds the same one-move hole on different boards, so a change on that row is the network learning the tactic rather than memorising the positions of the opponents it trained against.

| step | what was changed | pre-registered expectation | what happened | wipeouts vs Greedy (control: 65) | held-out row (control: 34) |
|---|---|---|---|---|---|
| **4** | the exact rule from Step 3c switched on *inside* the training search, on top of the opponent mix | the value head sees the resolved outcome and learns the low-disc positions | null: the check stopped the wipeouts during training, so the lesson rows left the data; the value head and the prior did not move; self-play 60% slower | 59, 58 | 26, 29 |
| **5** | the network's search run at the *opponent's* positions in the external games, giving the policy head a target where the search reads | the prior on the wiping reply rises, Greedy wipeouts fall | the first training route that moved: prior 4% → 8%, the whole Greedy family down a third on both seeds, at no cost; the held-out row did not move | 54, 45 | 31, 38 |
| **6** | the dose doubled and quadrupled: 240 and 480 external games per generation | the prior keeps rising; the held-out row is the test | the held-out row moved, read pooled over checkpoints; prior to 12%; at 480 games the network's own play is a fifth of its data and it paid 11 to 20 points on strength and retention | 55, 44 (240); 45, 33 (480) | 24, 22; 24, 17 |
| **7** | the Step 5 games unchanged, the searched rows counted 2 and 4 times in the policy loss | the readings follow the signal, not the games | reproduced the Step 6 curve point for point, with nothing paid: the dose worked through the policy signal, the cost was the displaced self-play | 51, 43 (×2); 41, 44 (×4) | 25, 27; 26, 22 |
| **8** | both axes at once (240 games, ×2), and the weight pushed to ×8 at the Step 5 games | ×8: saturation on the held-out row expected; the combined group decides between "the signal stops" and "the distinct positions run out" | ×8 read as expected, 19 and 24, with the Greedy family the lowest of any group and nothing paid at a 43% share of the policy loss; the combined group read 12 and 22, one seed below every earlier checkpoint and the other where the ×4 group sits, which is neither of the design's outcomes | 38, 38 (240, ×2); 35, 34 (×8) | **12**, 22; 19, 24 |

The quantity that lines the two axes up is the number of fatal searched rows times the weight on them, 130 to 700 across the series. The Greedy family keeps falling along it (about 200 per seed at the Step 5 dose, 170 to 180 at the first doubling by either route, 140 to 157 at the second, 129 at weight 8); the held-out row reads 17 to 27 at every point from 200 on, except the one seed at 12.

## 2. What the claim is now, and one correction

**The claim.** The value head cannot be taught a one-move tactic by any of four routes (opponent diversity, the value target at the position the search reads, the exact answer resolved inside the training search, and now the same data with a policy target beside it). The policy head can be, partly, by one route: a search run at the position the search reads. What it carries grows with the weight of that signal in its loss, whether the weight comes from more positions or from counting the same positions more times; it reaches the opponents that produce the positions first and the knowledge-free opponent's boards once the signal is doubled; the gain is concave; the positions limit themselves, because each lesson teaches the network to stop producing the next; and buying the signal with games costs the network's own strength while buying it with a weight in the loss costs nothing, up to a weight of 8. The exact check reaches every board at once and for nothing.

**The correction.** The Step 6 and 7 write-ups first read the held-out row as having moved "on every new checkpoint". An audit on November 8 found that the runs trained *without* any searched rows already spread from 21 to 39 wipeouts on that row (the continued control reads 26 and 38), so no single checkpoint of 200 games separates a dose group from the control. What holds is the pooled contrast: over the nine checkpoints trained without searched rows the row reads 15.9% of games lost with no discs; over the fourteen trained with them, from Step 6 on, 11.1%, a difference nearly five standard errors wide. The documents, the contribution page and the README now say so, in PR #76. The claim's substance is unchanged; the per-checkpoint wording was wrong and is withdrawn. The same audit re-ran five old evaluation pairings at the current code and found them identical game for game, so the comparisons across steps are sound.

**To discuss:** the combined group's 12. It is the lowest plain-search reading on that row of the 26 checkpoints evaluated, below the 17 of the best 480-game seed, and it is one seed; its partner reads 22. Under the 11% rate the series has read since Step 6, a 200-game checkpoint reads 12 or fewer about one time in a hundred, and fourteen such checkpoints have been read. The honest statement is that the combined recipe may reach further on the knowledge-free opponent and one seed cannot say. A third seed is 1.6 GPU-hours plus a six-hour evaluation.

## 3. The Phase 1 report

Writing starts this week; the target is still a complete draft by the end of November. The outline in PR #63 planned three results sections for Steps 1 to 3. It now needs:

- **Results III, Steps 3 to 8 as one chain.** Three nulls into the value head (3, 3b, 4), the exact check (3c), and the policy route (5) with its dose curve (6), its mechanism (7) and its limits (8): one table in the shape of section 1 above, then one subsection per step with its scorecard rows and probe columns. Every table exists as Markdown already.
- **The discussion's claim** as written in section 2, with the tactical-versus-strategic horizon framing from the contribution page and the Go cyclic attack placed on it.
- **Limits, stated plainly:** one game, one tactic, one starting checkpoint, two seeds per group, 200 games per row, one knowledge-free opponent, no adaptive adversary, and a repair that is standard practice whose contribution is where it was shown to be the only working one.
- **Figures:** the recording-rule diagram, the wipeout move-number histogram, the scorecard dot plot, the reliability curves, and one new one, the family and held-out rows against the weighted count of fatal rows across Steps 5 to 8. All from committed JSON.

Proposed order: the results sections first, to the advisor as a partial draft around November 16, the rest by November 30. Markdown rendered to PDF unless LaTeX is wanted.

**To discuss:** whether the report should present Step 8 as the end of the policy-route series or run the third seed first; and whether a reader outside RL needs the probe tables at all, or only the scorecard rows and the one curve.

## 4. What runs during the writing, and Phase 2

Cheap items that need no GPU and no new code, so they can run while the report is written:

1. **The search-budget curve with and without the check** on the control and the weight-2 and combined checkpoints: 50, 100, 256 and 800 simulations, Greedy and held-out rows. CPU only, one day. It makes "one ply, not depth in general" a curve rather than two points.
2. **A third seed of the combined group**, if decided (section 5).

Phase 2, December to January, toward a workshop paper, in order of what most changes the claim:

1. **A second knowledge-free opponent**, so the transfer result rests on more than one row.
2. **A second tactic or a second game** with the same signature (a usually favourable state, one refuting reply, a lesson outnumbered in the data), 6×6 Othello first if the board size is cheap to parameterise.
3. **The lesson-to-look-alike ratio varied on purpose**, by weighting those rows in the value loss, to predict when the value head moves: the Step 3b count turned into a measurable quantity.
4. The population baseline: decide in December whether to run it or state in the limitations that it addresses strategic weaknesses and this one is tactical.

## 5. Decisions needed

1. Is the section 2 claim the right one for the report and for the paper's positioning? In particular, is "partly, by one route, and for nothing up to a weight of 8" the right strength?
2. The combined group's 12: third seed now, or report it as one seed and move on?
3. Report: results sections first as a partial draft around November 16? Markdown to PDF?
4. Review order for the open pull requests: #72 (contribution page after Step 5, Step 6 design), #73 (Step 6 results), #74 (Step 7 code and design), #75 (Step 7 results), #76 (the held-out-row correction and the evaluation recheck), then the Step 8 design and results once they are up. Each is stacked on the one before it.
