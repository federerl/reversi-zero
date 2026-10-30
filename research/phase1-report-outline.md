# Phase 1 report — outline

**Working title:** Opponent-Specific Weaknesses in a Self-Play Othello Agent: Reproduction, Mechanism, and a Pre-Registered Intervention
**Audience:** the supervising faculty member, then graduate admissions readers who have not seen the repository
**Target:** 12–15 pages plus appendices, mid-November 2026
**Sources:** every number already exists in `research/step0-audit.md`, `step1-pilot-results.md`, `step2-failure-analysis.md`, `held-out-opponents.md`, `step3-design.md`, `step3-results.md` and the JSON under `research/results/`. The report rearranges them for a reader who starts cold; it adds no new measurements unless the recording-rule follow-up (section 9) finishes in time.

Each section below lists: what it says, where the material is, which table or figure carries it, and what is still open.

---

## 0. Front matter

- Title, author, advisor, date, repository URL and the commit the report describes.
- **Abstract** (≤ 200 words). The one-paragraph version in `step2-failure-analysis.md` section 7, extended by one sentence per Step 3 finding: the mix did not transfer and barely took, because of how external games were recorded; a knowledge-free search exploits the same weakness; a changed value target removes it at a cost in strength. Write last.

## 1. Introduction (1 page)

- The observation that started the project: an agent that beats a depth-4 minimax search 96% of the time loses a third of its games to a one-ply greedy player (Step 1 headline table).
- Why it matters beyond Othello: the Wang et al. (2023) pattern, strong overall and exploitable by one simple policy, here appearing with no trained adversary at all.
- The three questions, in the order the project answered them: does it reproduce under a controlled protocol; what is the mechanism; does training against fixed opponents remove it and does the removal transfer to opponents never trained against.
- **Contributions**, four bullets:
  1. A controlled reproduction with a frozen protocol (Step 1).
  2. A mechanism traced to the value head and the self-play data, with the policy head's share separated (Step 2).
  3. A pre-registered intervention with a held-out roster, read as a controlled negative result, and the recording rule that explains it (Step 3).
  4. A reproducible evaluation setup: scripts, protocol file, per-pairing reports.

## 2. Background (2 pages, for a reader with one course of RL)

- Othello in one paragraph; why disc count mid-game is a poor guide to the result (mobility, corners, parity), which is what makes Greedy weak and what makes the weakness surprising.
- The AlphaZero loop: self-play games → positions labelled with the search's move distribution and the game's result → a network with a policy head and a value head → the search uses both. Define the value target *z* (result from the mover's view) and the fact that the search judges a move by the value of the position *after* it, with the opponent to move. This one fact carries Step 3's explanation, so it gets a diagram here (figure 1).
- The opponents: Greedy (most discs flipped), Minimax-d4 (depth-4 search with a hand-written evaluation), Edax (a mature engine; levels are search depth, not different networks), the random-playout search (no knowledge of any kind). Define **wipeout**: a game lost with zero discs.
- Define **seen**, **seen family**, **held out** and **retention** as the roster uses them (`held-out-opponents.md` sections 2 and 4).

## 3. Related work (1 page)

One paragraph each, tied to a specific result in this report, from `research/lit-reviews/`:
- Wang et al. 2023 and Tseng et al. 2025: exploitability of superhuman Go agents; the framing for Step 1 and the "fixed opponents are not adaptive adversaries" caveat.
- Czarnecki et al. 2020: non-transitivity in real-world games; why Greedy < Minimax < agent by rating while Greedy is the harder opponent.
- McGrath et al. 2022: what the network knows; the background for the head probe and the calibration check.
- van der Ree and Wiering 2013: Othello, self-play versus a fixed opponent, no search, no failure analysis; the gap Step 3 sits in.
- Czempin and Gleave 2022: population training as the Phase 2 contrast to fixed hand-written opponents.

## 4. Method (2 pages)

### 4.1 The system
- Reversi Zero: network size, training recipe, replay window, simulations per move; the `control120` checkpoint all Step 3 groups start from, and the ownership-head recipe used as a comparison in Step 2. Cite `docs/architecture.md` and the config files rather than restating them.

### 4.2 The frozen evaluation protocol
- 200 games per pairing, 100 seeded 4-ply openings played in both colours, 50 simulations, no exploration noise, fixed seed; Wilson 95% intervals; scores from the network's side. From `step0-audit.md` section 6 and `eval-protocol-frozen.json`.
- Why frozen before any run: the Step 0 audit found the historical 63% rested on 30 games (table: historical vs reproduced, `step0-audit.md` section 1).

### 4.3 The held-out roster
- The admission rule ("a different source of knowledge"), the similarity check as run, and the resulting rows. Table 1: the roster with source, role and status (`held-out-opponents.md` section 4). Note which secondary opponents were not obtained (Egaroucid, WTHOR imitation) and say so plainly.

### 4.4 Pre-registration
- Gate A, B, C as written in the proposal; the five Step 3 predictions quoted verbatim before any result is shown (`step3-design.md` section 5). The point to make: the predictions were written down, and three of five failed.

## 5. Results I — the weakness reproduces (Step 1, 1.5 pages)

- Table 2: the headline table (rating, vs Greedy, vs Minimax-d4, wipeouts) for gen60, control120, ownership gen120, plus Minimax-d4 and Greedy themselves.
- Every loss to Greedy is a wipeout (section 3 of the pilot); colour (section 4); the two one-variable comparisons, generation 60 vs 120 and standard vs ownership head (section 5).
- Figure 2: wipeout move number distribution, or disc count over the game for a typical wipeout.
- What this does not show (section 7): keep as a short paragraph.

## 6. Results II — the mechanism (Step 2, 3 pages)

Follow the document's own order, it already reads as an argument:
- 6.1 What the network believes while it is being wiped out: +0.99 while holding one disc. Figure 3: `research/figures/wipeout-values.png` (raw value and search value by move for the losing games).
- 6.2 The losses were avoidable and the search saw the escape.
- 6.3 The training data said this was fine: in a million self-play positions, holding one or two discs after the opening happened about once per hundred games and that side won three quarters of the time. Table 3: disc-count bins vs outcome in the replay window.
- 6.4 More search helps and does not change what goes wrong (256 and 800 simulations).
- 6.5 Which head lets the wipeout through: the value-only vs policy-only search experiment (value-only escapes 98–100%, policy-only 2–12%). Table 4.
- 6.6 It is not Greedy: greedy-corner, greedy-d2, noisy-greedy produce the same loss. One table.
- 6.7 The ownership head's effect holds on a second seed: a changed target, not changed opponents, halves the wipeouts.
- Close with the Gate B statement.

## 7. Results III — the intervention (Step 3, 4 pages)

- 7.1 What was run: the groups table (A control, B mix, D material target; C deferred and why), matched on training examples, two seeds for A and B, one for D; 20 generations from control120; cost in GPU hours. From `step3-results.md` section 1.
- 7.2 The scorecard. Table 5: the full 13 × 6 scorecard, rows grouped held out / retention / seen / seen family / sanity, wipeout counts in the Greedy and rollout rows. Figure 4: the same as a heatmap or dot plot, one panel per row group, control range shaded.
- 7.3 Predictions one by one: a five-row table, prediction / outcome / verdict. Three failed.
- 7.4 The random-playout search finds the same hole knowing nothing. Table 6: score, losses, wipeouts, move at wipeout, raw value at the last move, for all six checkpoints. This is the section that turns "no transfer" into a measured statement, because it is the one held-out row that tests the weakness directly.
- 7.5 Why group B did not learn. Table 7: the head probe, raw value at the network's position vs value at the child with Greedy to move vs prior on the wiping reply, control vs B. The explanation: only the network's own positions were recorded from external games, so the value head was never trained at the position the search reads. Refer back to figure 1.
- 7.6 Group D: the cut (Greedy, family, rollout) and the price (retention 62.5 → 40.8%, Edax L5 29.8 → 17.2%, L7 10 → 4%).
- 7.7 Edax wipeouts are a different loss (value −1 seven moves out; unavoidable endgames). Short.
- 7.8 Value calibration. Figure 5: reliability curves, one panel per checkpoint, self-play vs held-out. Table 8: expected gap by row. Points: self-play calibrated for all; 7–12 points of overrating against Edax, same for A and B; D compressed and no longer a probability.

## 8. Discussion (1.5 pages)

- **What is established**, three claims, each with the evidence that carries it: (i) the weakness is a value-head belief learned from a self-play data gap; (ii) it is opponent-general, since a knowledge-free search finds it; (iii) twenty generations of a 20% hand-written opponent mix neither transfer nor take, and the reason is the recording rule, not opponent diversity as such.
- **Reading the negative result.** Why it counts: predictions were pre-registered, the mechanism of the null is identified by an independent measurement (the probe), and the alternative route (D) shows the hole *can* be closed, so the null is not a ceiling effect.
- **Relation to the Go literature.** This is the "seen-only" outcome the design listed; fixed opponents are not adaptive adversaries; the population approach is the natural Phase 2 contrast.
- **Limitations**, stated as facts: one game; twenty generations from one starting checkpoint; one D seed; no Edax in training; secondary held-out opponents not obtained; wipeout counts against held-out opponents are small numbers on 200 games; D's value head cannot be compared on calibration because it no longer estimates a probability.

## 9. Next experiments (0.5–1 page)

In order of cost, from `step3-results.md` section 10:
1. **Record both sides of external games for the value target.** One variable; the probe says exactly what should move (the child value at the opponent-to-move position). Two seeds, about 1.5 GPU hours each plus the 13-row evaluation. *If finished before the report deadline, this becomes section 7.9 and the abstract's last sentence changes.*
2. A cheaper form of D: smaller material weight, or the material share as a third head so the value head stays a probability.
3. Group C (Edax in the mix), and the population-based contrast, as Phase 2.

## 10. Reproducibility (0.5 page)

- Repository and commit; the protocol JSON; the scripts that regenerate every table (`step3_scorecard.py`, `probe_heads.py`, `value_calibration.py`, `analyze_wipeouts.py`, `analyze_pilot.py`, `scan_replay_disc_counts.py`); the cluster layout and launch commands (`step3-design.md` section 7); which data are committed (reports, JSON) and which are retained on the cluster only (recorded games, replay windows, checkpoints) and why.
- The one-line disclosure of AI assistance, as the README has it.

## Appendices

- A. Full scorecard with intervals and records, from `research/results/step3/scorecard.json`.
- B. Head-probe readings per checkpoint, from `head-probe.json`.
- C. Calibration bins, from `value-calibration.json`.
- D. Run configurations: the three YAML files side by side as a diff.
- E. The Step 0 audit's checkpoint identity table.

---

## Figures and tables, collected

| # | What | Exists? |
|---|---|---|
| Fig 1 | The search reads the child position (opponent to move); the external games labelled only the parent | to draw |
| Fig 2 | Wipeout move-number distribution, Step 1 | data in pilot JSON; to plot |
| Fig 3 | Raw and search value by move during wipeouts | `research/figures/wipeout-values.png` |
| Fig 4 | Scorecard as dot plot, control range shaded | to plot from `scorecard.json` |
| Fig 5 | Reliability curves per checkpoint | to plot from `value-calibration.json` |
| Tab 1 | Held-out roster | exists |
| Tab 2 | Step 1 headline | exists |
| Tab 3 | Disc-count bins vs outcome in self-play | exists |
| Tab 4 | Value-only vs policy-only search | exists |
| Tab 5 | Step 3 scorecard | exists |
| Tab 6 | Rollout row detail | exists |
| Tab 7 | Head probe, control vs B | exists |
| Tab 8 | Calibration gap by row | exists |

## Writing order

1. Sections 5–7 first: they are condensations of documents that already exist, and they fix the vocabulary everything else uses.
2. Section 4, then 2, so that every term used in 5–7 has been defined once.
3. Sections 8 and 9.
4. Sections 1 and 3.
5. Abstract last.

## Open decisions

- Format: Markdown in the repository rendered to PDF, or a LaTeX document. The tables are already Markdown.
- Whether the recording-rule follow-up (9.1) is launched now so that it can land in section 7 by mid-November, or is written up as future work only.
- How much of the Step 0 audit belongs in the body versus appendix E.
- Whether the advisor wants to see a draft of sections 5–7 before the rest is written.
