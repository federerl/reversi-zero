# Phase 1 report — outline

**Working title:** A One-Move Tactical Weakness in a Self-Play Othello Agent: Where It Lives, Which Head Can Learn It, and What Removes It
**Audience:** the supervising faculty member, then graduate admissions readers who have not seen the repository and have one course of reinforcement learning
**Target:** 18–22 pages plus appendices; a complete draft by November 30, 2026, the results sections first
**Sources:** every number already exists in the Step 0 to 8 documents, `contribution.md`, and the JSON under `research/results/`. The report rearranges them for a reader who starts cold; it adds no new measurements. The bibliography is [`references.md`](references.md), every entry checked against its publisher's record.
**Format:** Markdown in the repository, rendered to PDF with pandoc. The tables are already Markdown; LaTeX would cost two days of layout and buy no content.

Each section below lists: what it says, where the material is, which table or figure carries it, and what is still open. "Step n §k" means section k of `research/stepn-results.md` (or `step3c-terminal-check.md`, `step2-failure-analysis.md`).

---

## 0. Front matter

- Title, author, advisor, date, repository URL and the commit the report describes.
- **Abstract** (≤ 250 words). The one-sentence claim of `contribution.md` §1, unpacked into five sentences: the pattern reproduces; the loss is a one-move tactic the value head rates as a win because self-play taught it to; three ways of training the value head on it fail and the training rows say why; one exact rule in the search removes it for nothing, and sixteen times the search budget does not; the policy head can be taught part of it, in proportion to the weight of the signal and no further than the positions it is shown. Write last.

## 1. Introduction (1.5 pages)

- The observation that started the project: an agent that beats a depth-4 minimax search 96% of the time loses a third of its games to a one-ply greedy player, and every one of those losses is a wipeout, the agent left with no discs (Step 1 headline table).
- Why it matters beyond Othello: Wang et al. (2023) showed a superhuman Go agent beaten by a simple policy, with a trained adversary; here the same shape appears with no adversary at all, in an environment small enough to take apart.
- The questions, in the order the project answered them: does it reproduce under a controlled protocol (Step 1); what is the mechanism (Step 2); can training remove it, and does the removal reach an opponent never trained against (Steps 3 to 8); can exact search remove it instead (Step 3c and its budget curve).
- **Contributions**, five bullets, in the order of `contribution.md` §3:
  1. A causal chain measured at every link: game record → value head → training rows → each repair tested by elimination.
  2. A countable reason why value regression cannot learn a tactical exception: about 80 to 170 lesson rows against more than four thousand look-alikes labelled the other way.
  3. A split in the literature's remedy: opponent diversity reaches the state and cannot teach the reply.
  4. Which head, at which position, carries a one-move fact, and how much of its loss it has to be: the policy head, given a search target where the search reads, in proportion to the signal's weight, bounded by the positions it is shown.
  5. The method: a frozen protocol audited first, a held-out roster with a written admission rule, eight pre-registered experiments with the failures reported as failures.
- One sentence on what is *not* claimed as new: the terminal check is standard practice (Winands et al. 2008); its contribution is where it was shown to be the only working route.

## 2. Background (2.5 pages, for a reader with one course of RL)

- Othello in one paragraph: the 8×8 board, flips, the game ending when neither side can move, and why disc count mid-game is a poor guide to the result (mobility, corners, parity). This is what makes Greedy weak and what makes losing to it surprising. The game is weakly solved (Takizawa 2023); Edax is the reference engine.
- **Wipeout**: a game lost with zero discs. Under the rules a side with no discs cannot move, so the game ends from the middle of the board in one move. This puts a terminal position one ply from ordinary-looking positions, which is the whole reason the weakness exists.
- The AlphaZero loop (Silver et al. 2017, 2018): self-play → positions labelled with the search's move distribution and the game's result → a network with a policy head and a value head → the search uses both. Define the value target *z* and the policy target π. Define the **prior**: the policy head's probability on each move, which decides how many simulations the search spends on it. The search judges a move by the value of the position *after* it, with the opponent to move; this single fact carries Steps 3b, 5 and 7, so it gets a diagram (F1).
- Monte-Carlo tree search in two paragraphs (Coulom 2006; Browne et al. 2012): selection by prior and value, expansion, backup; first-play urgency as the engines use it (no paper; see `references.md`). Exact resolution inside the tree has a lineage (Winands et al. 2008); the one-ply check of Step 3c is its smallest form.
- The opponents: Greedy (most discs flipped), the Greedy family (greedy-corner, greedy-d2, noisy-greedy), Minimax-d4, Edax levels 3/5/7 (search depth, not different networks), the random-playout search (no knowledge of any kind). Define **seen**, **seen family**, **held out** and **retention** as the roster uses them (`held-out-opponents.md` §2, §4).
- Vocabulary for the results: **fatal position** (the network to move, one legal reply for the opponent wipes it out), **the wiping reply**, **lesson row** and **look-alike** (Step 3b §5), **searched opponent row** (Step 5 §5), **weighted count** = fatal rows × weight (Step 8 §1).

## 3. Related work (1.5 pages)

One paragraph each, tied to the section of this report it bears on. Full citations in `references.md`.
- **Exploitable self-play agents.** Lanctot et al. 2017 (self-play overfits to itself), Wang et al. 2023 and Tseng et al. 2025 (the cyclic attack and the failed defences), Lan et al. 2022 (a tactical blind spot found by search, no trained adversary), Timbers et al. 2022 (exploitability as a learned best response; what the held-out roster stands in for). Bears on sections 1, 4.3, 10.
- **Non-transitivity.** Czarnecki et al. 2020: why Greedy rates below Minimax and is the harder opponent. Bears on section 5.
- **Learning from search.** Anthony et al. 2017 (Expert Iteration: the search is the expert), Schrittwieser et al. 2020 (the value target as a learned average). Bears on sections 7 and 9: the one working route gives the policy head a search target at a position the loop never searches.
- **Exact resolution in MCTS.** Winands et al. 2008. Bears on section 8.
- **What the network knows.** McGrath et al. 2022: probing; the background for the head probe and calibration. Bears on sections 4.5, 6.
- **Othello and RL.** van der Ree & Wiering 2013: self-play against a fixed opponent, no search, and the finding that learning from the opponent's moves made things worse, which Step 3b reproduces for the value head and Step 5 reverses for the policy head. Buro 2002 for the classical evaluator. Bears on sections 7, 9.
- **The remedy the literature recommends.** Czempin & Gleave 2022, population-based training. Bears on sections 10, 11.

## 4. Method (3 pages)

### 4.1 The system
- Reversi Zero: 6-block, 64-channel network, three input planes, the training recipe, replay window, 200 simulations in self-play; the `control120` checkpoint every training group starts from; the ownership-head recipe (after Wu 2019) used as a comparison in Step 2. Cite `docs/architecture.md` and the config files rather than restating them.

### 4.2 The frozen evaluation protocol
- 200 games per pairing, 100 seeded 4-ply openings each played in both colours, 50 simulations, no exploration noise, seed 20261025; Wilson 95% intervals; scores from the network's side. From `step0-audit.md` §6 and `eval-protocol-frozen.json`. Same entrant name ⇒ same openings, which is what lets the budget curve (section 8.3) pair with the scorecard.
- Why frozen before any run: the Step 0 audit found the historical 63% rested on 30 games (`step0-audit.md` §1, table: historical vs reproduced).
- The evaluation recheck (`results/eval-recheck/README.md`): five pairings re-run at the final code reproduce game for game.

### 4.3 The held-out roster
- The admission rule ("a different source of knowledge"), the similarity check as run, the resulting thirteen rows grouped held out / retention / seen / seen family / sanity. **Table 1**: the roster (`held-out-opponents.md` §4). Say which secondary opponents were not obtained (Egaroucid, WTHOR imitation) and that the transfer result rests on one knowledge-free row.

### 4.4 Pre-registration
- Every experiment's predictions were committed before its runs: Step 3 (five, three failed), 3b, 4, 5 (two thresholds missed by a little), 6 (the claim-side prediction failed in the direction the design said would change the claim), 7, 8 (the combined group read outside every written outcome on two seeds, settled by a third). One table of experiment / predictions written / held / failed, with the section that reads each. Quote nothing yet; the per-step "predictions one by one" tables go in the results sections.

### 4.5 The four measurements used throughout
- **The scorecard** (`step3_scorecard.py`): score, interval and wipeout count on each of the thirteen rows.
- **The fixed-position head probe** (`probe_heads.py`, `head-probe-fixed.json`): on the control's 63 fatal positions, every checkpoint's raw value at the network's position and at the child with the opponent to move, the prior on the wiping reply (median and under-5% share), and the plain search's escape rate at 50 simulations. This is what makes "the value head did not move" and "the prior doubled" exact statements.
- **Row counts in the replay data** (`step3b_lesson_rows.py`, `step5_searched_rows.py`): lesson rows against look-alikes; searched opponent rows at fatal positions and their target mass on the wiping move.
- **Value calibration** (`value_calibration.py`): reliability curves and the expected gap by row group.

## 5. Results I — the weakness reproduces (Step 1, 1.5 pages)

- **Table 2**: the headline table (rating, vs Greedy, vs Minimax-d4, wipeouts) for gen60, control120, ownership gen120, plus Minimax-d4 and Greedy themselves.
- Every loss to Greedy is a wipeout (§3); colour (§4); the two one-variable comparisons, generation 60 vs 120 and standard vs ownership head (§5).
- **Figure F2**: wipeout move-number histogram. The wipeouts end on moves 20 to 35: a midgame trap, not an endgame loss.
- What this does not show (§7), one short paragraph.

## 6. Results II — the mechanism (Step 2, 3 pages)

Follow the document's order; it already reads as an argument.
- 6.1 What the network believes while it is being wiped out: +0.99 while holding one disc; the won games look the same at the fifth move. **Figure F3** (exists): `figures/wipeout-values.png`.
- 6.2 The losses were avoidable and the search saw the escape.
- 6.3 The training data said this was fine: in a million self-play positions, holding one or two discs after the opening happened about once per hundred games and that side won three quarters of the time. **Table 3**: disc-count bins vs outcome.
- 6.4 More search helps and does not change what goes wrong (256 and 800 simulations). Forward reference to the full curve in 8.3.
- 6.5 Which head lets the wipeout through: value-only vs policy-only search (value-only escapes 98–100%, policy-only 2–12%). **Table 4.**
- 6.6 It is not Greedy: the Greedy family produces the same loss. One table.
- 6.7 The ownership head's effect holds on a second seed.
- Close with what §6 establishes: the belief is in the value head, the gate is the prior, and the data taught both.

## 7. Results III — the value head cannot be taught the lesson (Steps 3, 3b, 4; 4 pages)

Three experiments, each a route into the value head, each a null, each with the reason measured. Open with **Table 5**: the groups across the three steps (name, what changes, seeds, games, GPU hours), from Step 3 §1, 3b §1, 4 §1.

- 7.1 **Opponent diversity (Step 3).** Groups A control, B 20% Greedy/Minimax mix, D material term in the value target. **Table 6**: the scorecard rows that matter (Greedy, family, random-playout, Edax L5, retention) for all five checkpoints; the full 13-row scorecard goes to appendix A. The five predictions, three failed (§3). The random-playout search finds the same hole knowing nothing (§4): the one held-out row that tests the weakness directly. Group D cut the wipeouts seen and unseen and paid with retention 62.5 → 40.8% and Edax L5 29.8 → 17.2% (§6): the hole *can* be closed, so the nulls are not a ceiling.
- 7.2 **Why B did not learn (Step 3 §5).** Only the network's own positions were recorded from external games, so the value head was never trained at the position the search reads. **Table 7**: the head probe, control vs B. Refer to F1.
- 7.3 **The value head trained where the search reads (Step 3b).** The opponent's positions recorded as value-only rows, about 2,900 a generation. The value head read −0.98 and −0.99 at the fatal child where the exact answer is +1, as before (§4). **The count (§5)**: 80 to 170 lesson rows against more than four thousand look-alikes that a value head cannot tell apart and that say the opposite, because a side with few discs in mid-game usually wins. The value head learned the average, and the average is right. **Table 8**: the lesson-row count by run (Steps 3, 3b, 4, 5 to 8 all have it: `lesson-rows.json`).
- 7.4 **The exact answer inside the training search (Step 4).** With the terminal check on in self-play, the network stopped walking into wipeouts during training (0.5% and 0.6% against 6.0% and 6.9%), and learned nothing the plain search could use: Greedy wipeouts 59 and 58 against 61 and 69 and the control's 65; value head and prior unmoved (§4). The row counts show the check *removed* the lesson rows rather than adding them (§5). The strongest form of the data-side route fails for a reason that is visible in the data.
- 7.5 Value calibration across the three steps (Step 3 §8, 3b §6, 4 §6): self-play calibrated for every checkpoint; 7–12 points of overrating against Edax, the same for all; D compressed and no longer a probability. **Figure F7**: reliability curves.
- Close: three routes, three nulls, one measured reason. State it as the result it is.

## 8. Results IV — exact search removes it (Step 3c, 2.5 pages)

- 8.1 **The check (§1).** One rule in the evaluation search: before trusting the network's estimate of a child, look one move ahead for a reply that ends the game. No training, no weights changed. Place it against Winands et al. 2008 in one sentence.
- 8.2 **The scorecard (§2, §3, §4).** Greedy wipeouts 65 → 12 on the control; 82–88% removed against every disc-hungry opponent on every checkpoint from four recipes; nothing else moves (Edax, retention unchanged). The value head's belief is still there; the search stops reading it at the one place it is wrong. **Table 9**: plain vs checked, by checkpoint and row.
- 8.3 **The search-budget curve (§7).** Four budgets (50, 100, 256, 800), plain and checked, two rows, on the control and three policy-route checkpoints, 64 cells. **Figure F6** (exists): `figures/dose-curve.png`. **Table 10**: the wipeout cells. Readings: sixteen times the budget buys no more than one ply of exact reading (control at 800: 16 and 12; check at 50: 10 and 13); the checked curve is nearly flat (10, 8, 7, 2), so what the check leaves is not a budget problem until 800; the trained prior lowers the budget at which the plain search finds the reply from past 800 to about 256; the held-out row reads as the Greedy row at every budget. This section is also where section 9's plateau gets its interpretation: the 17 to 27 on the held-out row is a reading of the 50-simulation budget on these priors, not a ceiling of the networks.
- 8.4 **What remains (§5).** Twelve games: seven are first-play urgency pressing an untried safe move below −1, one is a trap set two moves earlier. The 95% is a property of this rule at this budget, not a ceiling. The two residual repairs are named and not run.
- 8.5 What it does and does not establish (§6): a search-depth artefact, present in every checkpoint, independent of training, local; not a statement that the check belongs in self-play, which Step 4 tested.

## 9. Results V — the policy head can be taught part of it (Steps 5, 6, 7, 8; 5 pages)

The series reads as one curve, so open with **Figure F4** (exists): `figures/chain-weighted-count.png`, the Greedy-family sum and the held-out row against the weighted count of fatal rows, every seed of every group, Steps 5 to 8. Then **Table 11**: the seven training groups (Step 8 §1 table: games split, searched rows, fatal rows, weight, weighted count, GPU hours). The per-step scorecard rows and probe columns go in one wide table each (**Tables 12, 13**), 16 checkpoints, from `results/step8/scorecard.md` and `head-probe-fixed.json`; the per-step "predictions one by one" tables go to appendix B.

- 9.1 **A search target at the opponent's position (Step 5).** The network's search is run at the opponent's positions too, so the policy head gets a target where the search reads. Greedy wipeouts 61 and 69 → 54 and 45; a third fewer against every opponent in the family; the prior on the wiping reply 4% → 8%; no cost in strength, retention or calibration (§2, §4, §6). The held-out random-playout row did not move (31, 38). Escapes traced to the prior with a policy-only search (§4). The partial effect is a search-budget effect on top of a learning effect (§7).
- 9.2 **The dose curve by games (Step 6).** 240 and 480 external games a generation, and twenty more generations at the Step 5 dose. The held-out row moved once the searched rows were doubled: 24 and 22 at 240, 24 and 17 at 480, by a third in the pooled series and on no single checkpoint by an amount 200 games resolve (§2). Prior to 10% and 12%. The gain is concave; the dose limits itself because each lesson teaches the network to stop producing the next (§5); at 480 the network's own play is a fifth of its data and the retention and Edax rows pay 11 to 20 points (§2, §7).
- 9.3 **The same curve by weight (Step 7).** The searched rows counted two and four times in the policy loss, games exactly Step 5's. Reproduces the Step 6 curve point for point: weight 2 reads as 240 games, weight 4 as 480, on the held-out row, the family and the prior's tail, and nothing the 480 group paid is paid (§2, §4). The dose worked through the policy signal; the cost was the displaced self-play. **Figure F5**: the prior's tail (under-5% share) by group, 65% → 8%.
- 9.4 **The limits of the route (Step 8).** Weight 8: the family to its lowest (129, 128) at no cost, the held-out row where weight 4 had it (19, 24 against 26, 22): the signal is not what stops that row. The two axes combined (240 games × weight 2, three seeds): family 157, 145, 164; held-out 12, 22, 24, pooled 58/600 against weight 4's 48/400, 1.1 standard errors: twice the distinct positions at the same weighted count buy nothing either (§2, §7). Past a weighted count of about 300 the held-out row reads 17 to 27 however the signal is bought. Section 8.3 says what that plateau is.
- 9.5 What the series establishes (Step 8 §7, `contribution.md` §3 fourth point): the difference between a null and a result was not the data but which head was given a target at which ply, and how much of its loss that target was. State the limits in the same sentence as the size.

## 10. Discussion (2.5 pages)

- **The claim**, as `contribution.md` §1 states it, in plain sentences: a tactical weakness, a value head that cannot carry it by three routes and a countable reason, an exact check that removes it for nothing and a budget that does not, a policy head that carries it in proportion to the signal and no further than the positions it is shown.
- **Tactical and strategic weaknesses, and the horizon between them** (`contribution.md` §4): the table of what is wrong / what fixes it / what diversity does / what each target can carry / cost, by horizon. The test that places a weakness on the axis: leave the evaluator alone, add exact calculation of a chosen depth, ask whether the loss goes. Place the KataGo cyclic attack (Wang et al. 2023) on the axis as the same kind at a long horizon; Lan et al. 2022 as the nearest published short-horizon case.
- **What is not new** (`contribution.md` §2): the check is standard practice; simple policies beating self-play agents is known; self-play overfits to itself is known. Say it before a reader does.
- **Reading the nulls.** Why they count: pre-registered, the mechanism of each null identified by an independent measurement, and group D plus the policy route show the hole can be closed, so no null is a ceiling effect.
- **Relation to the Go literature.** Fixed opponents are not adaptive adversaries (Tseng et al. 2025); the held-out roster measures something weaker than exploitability (Timbers et al. 2022); population training (Czempin & Gleave 2022) treats weaknesses of judgement, and the one here is a weakness of calculation.
- **Limitations**, stated as facts: one game, one tactic, one starting checkpoint, two seeds per group and three in one, 200 games per row, one knowledge-free opponent, no adaptive adversary, no Edax in training, secondary held-out opponents not obtained, the retention reading shared by the weight-4 and combined groups (2 to 10 points under their parents on five seeds, inside the noise on each) unsettled, and a repair that is standard practice whose contribution is where it was shown to be the only working one.

## 11. Next experiments (1 page)

In order of cost, from Step 8 §8 and `contribution.md` §5:
1. **A second knowledge-free opponent**, so that the transfer result rests on more than one row.
2. **The seeds that would settle the retention reading** of the weight-4 and combined groups.
3. **The two residual repairs of the check** (Step 3c §5): exclude proven-lost moves at the root; a two-ply root check. Expected to take the twelve to five or fewer.
4. **The lesson-to-look-alike ratio as a predictor**: weight the lesson rows in the value loss on purpose and predict when the value head moves. Turns the count of section 7.3 into a measurable quantity.
5. **A second tactic or a second game** (6×6 Othello, or another game with the same signature). The Phase 2 item that most changes what the claim can be.
6. Population-based training as the contrast, or stated as a limitation.

## 12. Reproducibility (0.5 page)

- Repository and commit; the protocol JSON; the scripts that regenerate every table and figure (`step3_scorecard.py`, `probe_heads.py`, `step3b_lesson_rows.py`, `step5_searched_rows.py`, `value_calibration.py`, `analyze_wipeouts.py`, `step3c_dose_curve.py`, `scripts/figures/`); the cluster layout and launch commands (each step's "Reproduction" section); which data are committed (reports, JSON) and which are retained on the cluster only (recorded games, replay windows, checkpoints) and why.
- The one-paragraph disclosure of tools and assistance, as the README's "How this was built" has it.

## Appendices

- A. Full scorecards with intervals and records: Steps 3, 3b, 4, 5 to 8 (`results/step*/scorecard.json`; the Step 8 file has all 16 checkpoints on 13 rows).
- B. The predictions, one by one, for every step, as the results documents state them.
- C. Head-probe readings per checkpoint (`head-probe-fixed.json`).
- D. Lesson-row and searched-row counts per run.
- E. Calibration bins (`value-calibration.json`).
- F. The search-budget curve, all 64 cells with intervals (`results/step3c/dose-curve.json`).
- G. Run configurations: the YAML files as a diff against the control.
- H. The Step 0 audit's checkpoint identity table.

---

## Figures and tables, collected

Figures follow the numbering of the [November 9 agenda](meetings/2026-11-09-advisor-meeting-agenda.md) §4. Every figure is drawn by a script under `scripts/figures/` from committed JSON, with one palette; six exist, F1 and F2 are to draw.

| # | What | Section | Exists? |
|---|---|---|---|
| F1 | The recording rule: how an opponent position becomes a training row, and which head gets a target at it, in Steps 3b, 5 and 7 | 2, 7.2 | to draw (diagram, no data) |
| F2 | Wipeout move-number histogram | 5 | to draw from the Step 1 / Step 2 game records |
| F3 | Raw and search value by move during wipeouts | 6.1 | `figures/wipeout-values.png` |
| F4 | The chain: family sum and held-out row against the weighted count, every seed | 9 | `figures/chain-weighted-count.png` |
| F5 | The prior's tail (under-5% share) and median by group, against the weighted count | 9.3 | `figures/prior-tail.png` |
| F6 | Search-budget curves, plain and checked | 8.3 | `figures/dose-curve.png` |
| F7 | Reliability curves by row group, Steps 3, 3b, 4 and group D | 7.5 | `figures/reliability.png` |
| F8 | Scorecard dot plot: 26 checkpoints on each of the thirteen rows, against the control's interval | 7.1, 9 | `figures/scorecard-dots.png`; makes "nothing else moved" visible |

| # | What | Section | Exists? |
|---|---|---|---|
| T1 | Held-out roster | 4.3 | exists |
| T2 | Step 1 headline | 5 | exists |
| T3 | Disc-count bins vs outcome in self-play | 6.3 | exists |
| T4 | Value-only vs policy-only search | 6.5 | exists |
| T5 | Groups of Steps 3, 3b, 4 | 7 | to assemble from three §1 tables |
| T6 | Step 3 scorecard, the rows that matter | 7.1 | exists |
| T7 | Head probe, control vs B | 7.2 | exists |
| T8 | Lesson rows against look-alikes, every run | 7.3 | to assemble from `lesson-rows.json` files |
| T9 | Plain vs checked scorecard, Step 3c | 8.2 | exists |
| T10 | Budget-curve wipeout cells | 8.3 | exists (Step 3c §7) |
| T11 | The seven policy-route groups | 9 | exists (Step 8 §1) |
| T12 | Policy-route scorecard, 16 checkpoints | 9 | exists (`results/step8/scorecard.md`) |
| T13 | Policy-route head probe, 16 checkpoints | 9 | exists (Step 8 §4) |
| T14 | The horizon table | 10 | exists (`contribution.md` §4) |

## Writing order

1. Sections 7, 8, 9 first: they condense documents that already exist, they are the longest, and they fix the vocabulary everything else uses. Sections 5 and 6 next, same reason.
2. Section 4, then 2, so that every term used in 5 to 9 has been defined once.
3. Sections 10 and 11.
4. Sections 1 and 3.
5. Abstract last.

The results sections (5 to 9) go to the advisor as a partial draft before the rest is written; a reader outside RL can say whether the tables are readable, which is the feedback that matters most.

## Open decisions

- Whether a reader outside RL needs the probe tables (T7, T13) in the body or only the scorecard rows and the two curves (F4, F6), with the probes in appendix C.
- Whether Step 8, with the third seed and the budget curve, is presented as the end of the policy-route series. The numbers say it is.
- How much of the Step 0 audit belongs in the body versus appendix H.
- F8 (the scorecard dot plot) carries sections 7 and 9 at once; whether it replaces T6 and T12 in the body, with the tables in appendix A.
