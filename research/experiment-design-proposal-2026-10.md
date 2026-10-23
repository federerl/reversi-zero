# Research Design Discussion Draft — Reversi-Zero

**For:** Meeting with Dr. Matt Boutell  
**Prepared:** October 18, 2026  
**Updated after advisor discussion:** October 22, 2026  
**Status:** Dr. Boutell considered the overall approach feasible. This revision captures his three follow-up questions; experimental configurations remain proposals, not approved or completed results  
**Working topic:** *Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents*

## Advisor discussion update — October 22, 2026

Dr. Boutell was generally supportive of the research direction. He raised three linked questions:

1. **Generalization after opponent-specific training:** If the agent trains against Greedy and Minimax, will it also become better against *other* agents? Which other agent families should constitute a meaningful unseen-opponent test set?
2. **Edax as a training opponent:** Can the same Edax engine already used for evaluation be integrated into training, so Reversi-Zero learns from games against Edax?
3. **What does an Edax level actually mean?** Clarify its implementation before treating L3/L5/L7 as meaningful differences in opponent strength or diversity.

**Updated emphasis:** Keep the original controlled reproduction and failure analysis as the first experiment. Make held-out generalization the principal outcome of the *subsequent intervention*, if the pilot and feasibility work justify it. Do not confuse improvement against training opponents with transfer to unseen policies.

## 1. Motivation and preliminary evidence

Reversi-Zero is an AlphaZero-style Othello agent trained by self-play, with a policy-value network and PUCT MCTS. Its prior evaluations suggest an unusual opponent-dependent pattern: improvement against a stronger-looking search baseline did not correspond to equally good performance against a simple Greedy policy.

An early evaluation of **Run 1, generation 60** recorded the following (see [existing experiment log](../docs/experiments.md), Run 1):

| Opponent | Gen-60 score | Important qualification |
|---|---:|---|
| Minimax depth-4 | 100% | Small original pairing; not a general guarantee |
| Greedy | 63% | Reported 95% CI: **46%–78%** |
| Greedy, agent as black | ~90% | Historical color-split observation |
| Greedy, agent as white | ~50% | Historical color-split observation |

In that tournament, Minimax-d4 reportedly beat Greedy 97%. These point estimates are surprising relative to a simple one-dimensional ordering of agents, **but do not establish a statistically reliable non-transitive cycle**. The 63% estimate has a wide interval that includes 50%; even with a reproducible matchup pattern, strategy interactions could have several causes.

The repository also contains later control runs and larger tournaments (including 1,000-game comparisons). Those results show that training-run variability and evaluation design matter. Historical metrics from different tournaments, generations, search budgets, or opening distributions should not be combined into one comparison without checking protocol equivalence.

**Motivating observation, not established explanation:** self-play might underexpose the agent to positions induced by unusual opponent strategies. Alternative explanations include color/opening effects, finite-sample variance, MCTS behavior, opponent implementation, and network value errors.

## 2. Lessons from the four papers

| Literature | What it contributes to *this* proposed study |
|---|---|
| Wang et al. (2023), [*Adversarial Policies Beat Superhuman Go AIs*](https://proceedings.mlr.press/v202/wang23g.html) | High overall game strength need not eliminate opponent-specific vulnerabilities; inspect rules, search, and value predictions before asserting a mechanism. A deliberately trained Go adversary is **not equivalent** to our fixed Greedy opponent. |
| Tseng et al. (2025), [*Can Go AIs Be Adversarially Robust?*](https://arxiv.org/abs/2406.12843) | Better performance against attacks used for training does not establish general robustness. Any future opponent-mixing experiment needs **held-out opponents** and matched training budgets. |
| Czarnecki et al. (2020), [*Real World Games Look Like Spinning Tops*](https://arxiv.org/abs/2004.09468) | Strategy interactions may not fit a single ranking, motivating a payoff-matrix view. Our current observations do **not** prove cycles or locate this agent in the proposed spinning-top geometry. |
| McGrath et al. (2022), [*Acquisition of Chess Knowledge in AlphaZero*](https://arxiv.org/abs/2111.09259) | Intermediate activations and linear probes can help study what concepts a trained agent represents. Representation plots and probe accuracy alone cannot establish causes of decisions. |

Our short individual reviews are collected in [literature reviews](lit-reviews/README.md).

## 3. Tentative research questions

**Primary question (Phase 1):**  
**Do AlphaZero-style Reversi agents show reproducible opponent-specific performance differences under controlled evaluation, and which observable factors are associated with the failures?**

Subquestions, in order:

1. **Reproducibility:** Does the historical Greedy-versus-Minimax pattern persist when games use consistent checkpoints, simulation budgets, openings, player colors, and evaluation rules?
2. **Failure characterization:** Are the losses concentrated by color, game phase, corner occupancy, mobility, or other board features? Do the agent's value estimates and MCTS choices diverge from reliable reference assessments?
3. **Advisor-requested generalization experiment (after baseline validation):** If the agent trains with Greedy/Minimax or Edax as opponents, does it improve against **held-out policies not used in training**, compared with an otherwise matched self-play baseline? Are improvements specific to the seen opponents or transferable across opponent types?

**Working hypotheses, not findings:**

- H1: At least one checkpoint shows a stable performance difference between Greedy and other opponents after opening/color balancing.
- H2: A meaningful portion of the effect could be associated with color or recurring game-state features and/or network/search evaluation errors.
- H3 (optional): Opponent diversity could improve held-out generalization, but an effect specific to training opponents would not suffice.

## 4. Experimental plan (sequenced, with stop/go decisions)

### Step 0 — Protocol and data audit (before playing new games)

- Identify exactly which checkpoint files, opponent implementations/versions, and search budgets produced the Run 1 results. Reconcile original pairing logs with the later tournament configuration.
- Verify game rules, tie/pass handling, determinism/random seeds, opening generator, and color-balanced evaluation.
- Confirm that raw logs support replaying moves, extracting board states, tracking network predictions, and storing provenance (commit SHA, model SHA, config, hardware, seed).
- Use [docs/experiments.md](../docs/experiments.md) as the historical source of truth; do not treat the early author's causal interpretation as a tested result.

**Output:** a reproducibility checklist and a frozen evaluation configuration.

### Step 1 — Controlled behavioral evaluation (first experiment)

**Initial proposal for advisor review, subject to runtime estimate:**

| Variable | Proposed setting |
|---|---|
| Checkpoints | Original Run 1 Gen 60 and later 120-generation control checkpoint (verify identities/availability first) |
| Core opponents | Greedy and Minimax-d4; add Random as a sanity check |
| Search budget | Fix the agent's MCTS simulations at one documented value for all core matchups (candidate: 256; first check historical comparability) |
| Opening design | Same seeded openings for each matchup; every opening played in both colors |
| Pilot size | 100 opening pairs **per matchup** = 200 games per checkpoint–opponent pairing; revise after runtime and precision estimate |
| Additional evaluation | Expand targeted, ambiguous pairings rather than running everything at the maximum sample size |
| Primary metrics | Score = (wins + 0.5 × draws)/games; 95% uncertainty intervals; black/white split and opening-paired differences |
| Secondary metrics | Move logs, game length, phase at first large evaluation change, corner control, mobility |

With 2 checkpoints × 2 core opponents × 200 games, the **core pilot is 800 games**, excluding the optional Random sanity checks. This is a *planning estimate*, not a pre-approved sample-size justification. Compute expected runtime and desired confidence-interval width before locking N. Games sharing openings or random seeds may be correlated, so uncertainty analysis should preserve pairing (e.g., opening-pair bootstrap), not automatically assume independent games.

**Interpretation:** Comparing raw win rates against two opponents of different intrinsic strength does not, by itself, prove a blind spot. Look for persistent and especially color-dependent interactions, and compare with the opponents' own head-to-head results.

**Gate A:** If the pattern disappears, document which protocol change explains it and report a reproducibility result. If the pattern persists with acceptable precision, proceed to targeted failure analysis.

### Step 2 — Failure analysis (conditional on Gate A)

1. Select representative winning and losing games using **predefined**, matched criteria (same opening/phase/color when possible), not only striking anecdotes.
2. Plot value-head estimates and MCTS root statistics across critical moves. Mark the earliest sizeable divergence, rather than just the final lost move.
3. Compare decisions with stronger reference evaluation where available (e.g., Edax with recorded settings); keep reference evaluations distinct from perfect-play ground truth.
4. Test inexpensive, targeted alternatives: color and opening dependence first; then MCTS-budget sensitivity (e.g., 64 / 256 / 800 **only on selected pairings/positions**), and board-level descriptors such as corner access and mobility.
5. Record which explanation(s) are supported, contradicted, or still unresolved.

**Gate B:** Identify at least one repeatable, descriptively meaningful failure pattern or establish a credible null result. Avoid causal language until there is a controlled intervention.

### Step 3 — Advisor-requested opponent-diversity intervention (conditional, separate approval)

**New primary question for this step:** *Does adding a small, controlled amount of opponent diversity improve performance against unseen opponents, not merely against the opponents used in training?*

**Candidate training groups**, starting from the **same checkpoint** and matched as closely as possible for learning steps, game counts, and total computational cost:

| Group | Training opponent distribution | Interpretation |
|---|---|---|
| A — control | Continue standard self-play only | Baseline for further training |
| B — heuristic mix | Self-play + Greedy + Minimax-d4 | Does exposure to two different fixed policies transfer? |
| C — engine mix (feasibility dependent) | Self-play + Edax at specified lower levels, e.g., L3/L5 | Is a mature engine useful as a training opponent? |

The *proportions* of self-play vs external-opponent games, learning targets, rollout budgets, and train/test partitions must be agreed upon before running. A smaller A-vs-B pilot is preferred if three full training runs exceed the available budget. No fixed fraction is claimed here.

**Held-out evaluation is essential:** Construct the opponent roster **before training** and never use the designated test policies for training or hyperparameter selection. Suggested categories:

- **Different heuristics:** corner-priority, mobility-based, and a noisy-Greedy policy (to be implemented and validated; these are *proposals*, not existing benchmarks).
- **Search-based controls:** Minimax with a depth not used in training, subject to enough meaningful behavioral difference.
- **External engine:** Edax levels not used during training, with a crucial limitation: Edax L3/L5/L7 share the same underlying evaluator, so L7 is a *held-out search configuration*, **not an independent opponent family**.
- **Sanity checks and strength retention:** Random, seen opponents, other Reversi-Zero checkpoints, and selected stronger engines/levels.

**Evaluation:** Use identical color-swapped opening pairs and agent-side MCTS budgets for all training groups. Report each opponent's score with paired uncertainty, **held-out results separately from seen-opponent results**, and overall strength retention. Predefine one small held-out scorecard instead of selecting only positive matchups post hoc. Include per-opponent effects; an aggregate score can hide regressions.

**Confounders:** Distinguish improvement from simply spending additional compute. Fixed game count alone does not guarantee equal compute when Edax search is expensive. Log games, moves, training updates, Edax search depth/selectivity, nodes (when available), elapsed compute, and random seeds. The Edax-mixed group is a different strength/strategy distribution from the heuristic-mixed group, so a gain alone would not isolate a pure *diversity* effect. More independent seeds and a single-opponent ablation would be needed for stronger causal claims.

**Gate C:** Advance only if the added training is technically feasible and either (a) yields clear held-out improvement without unacceptable regressions or (b) yields an informative controlled negative result. Do not claim robustness against adaptive adversaries based on these fixed opponents.

### Step 3a — Edax-as-training feasibility and level definition (before Group C)

The current project has **evaluated** agents against Edax 4.6, using opening book off, one thread, and color-balanced matchups; **training integration has not yet been verified**. Proposed smoke test:

1. Inspect the existing Edax evaluation adapter and training self-play data flow. Confirm that an external opponent can supply moves under identical game rules, including passes and draws.
2. Run a few complete Edax L3/L5 games in both colors; verify legal moves, deterministic settings, result signs, board perspective, and that the Reversi-Zero player's own MCTS visits are recorded correctly.
3. Explicitly specify how the external-opponent plies contribute to learning data. Do **not** pretend Edax provides MCTS visit-count policy targets from Reversi-Zero. A simple pilot can store the network agent's own search targets, plus final game outcomes, and document whether opponent moves are used only to advance the state.
4. Benchmark total wall time and search-node usage for L3/L5 against ordinary self-play before selecting mixture probabilities.
5. Freeze the Edax executable/version, evaluator files, opening book setting, threads, flags, and randomization policy.

**Verified Edax 4.6 level semantics (from [`src/search.c`](https://github.com/abulmo/edax-reversi/blob/v4.6/src/search.c), `search_global_init()` and `search_set_level()`):** A level is a rule assigning **search depth and search selectivity based on the number of empty squares**, not an Elo rating, distinct trained network, or uniform per-move thinking-time allowance. For **levels 1–10**, `selectivity = 5` (no selectivity), and

```text
depth = min(level, n_empties)        if n_empties <= 2 * level
depth = level                        otherwise
```

The key subtlety is that the search depth **jumps from the fixed level to the number of remaining empties as soon as `n_empties <= 2 * level`**; it is **not** simply `min(level, n_empties)`. **Endgame interpretation:** when at most `2 * level` empty squares remain, the configured depth reaches all remaining empties (subject to search implementation and Othello passes). Examples: L3 searches 3 plies in the early game and has full remaining-empties depth by 6 empties; L5 similarly at 10 empties; L7 at 14; L8 at 16. **Higher levels** use more complex depth/selectivity schedules with selective search; do not extrapolate the L1–L10 rule to all levels. Edax's search uses a specialized evaluator and Principal Variation Search, so “Edax L5” is not equivalent to an ordinary Minimax-d5 opponent.

**Crucial limitation:** Edax levels vary mainly in search settings and are **correlated variants of one engine**, not independently developed styles. They are useful for controlled strength and search comparisons, but evaluation must also include opponents outside the Edax family.

**Gate D:** Only include Edax training in Step 3 if the adapter produces valid training records and runtime is practical.

### Step 4 — Optional interpretability (not necessary for Phase 1)

### Step 4 — Optional interpretability (not necessary for Phase 1)

If failure analysis points to a specific concept, consider comparing internal activations across checkpoints or using linear probes for concepts such as mobility or corner access. t-SNE may be used as an exploratory visualization only, with game-phase, color, and other confounders checked. **Neither visible clusters nor probe accuracy demonstrate causation.**

## 5. Deliverables and timeline

| When (tentative) | Deliverable |
|---|---|
| Advisor discussion (completed, Oct 22) | Overall plan judged feasible; follow-ups on unseen-opponent transfer, Edax training, and Edax level definitions |
| Late October | Audited protocol, pilot evaluations, table of win rates/uncertainty/color splits; smoke-test Edax opponent integration if practical |
| Early November | Targeted evaluation expansion plus annotated failure cases; define and freeze held-out opponent roster before any mixture-training pilot |
| Mid-November target | Short independent-study report with methods, figures, findings (including null findings), limitations, reproducibility instructions, and proposed next experiments |
| Later, if results merit | A-vs-B (possibly A-vs-B-vs-C) opponent-diversity training comparison, with held-out evaluation and compute accounting; multi-seed and broader publication-oriented work afterward |

The mid-November milestone is a **tentative planning target**, not a confirmed deadline from Dr. Boutell. Our priority before graduate applications is a defensible result and a report, not a guaranteed positive finding.

## 6. Open decisions after the October 22 advisor discussion

1. Is the **behavior-first** research question appropriately scoped, or should we narrow it to the Greedy color asymmetry?
2. Which checkpoint and reference opponents provide the fairest, most interpretable pilot? Is 256 MCTS simulations reasonable given existing protocols?
3. What evidence would be sufficient to call the observed behavior an **opponent-specific weakness** rather than variation in opponent strength?
4. Would he prefer a straightforward paired-matchup analysis before any representation analysis or t-SNE?
5. Can the held-out opponent roster be frozen now? Which policies are sufficiently different to test transfer rather than another search depth of the same engine?
6. Is a small A-vs-B training pilot in scope now, or should it remain after reproducibility/failure analysis? Should Group C use Edax L3/L5 only after integration benchmarks?
7. How should computational cost be matched when Edax search is more expensive than Greedy/Minimax?
8. What is the minimum convincing Phase 1 result/report by graduate-application time?
9. Is the current Othello-only scope appropriate for an independent-study report, with broader generalization deferred to a publication-oriented Phase 2?

## 7. Scope boundaries

- Do **not** claim self-play caused the original anomaly without experiments isolating training distribution.
- Do **not** equate the result to Wang/Tseng's trained adversarial attacks; our Greedy baseline is not optimized to exploit the target.
- Do **not** claim a non-transitive cycle from a 63% score versus Greedy.
- Do **not** infer a position in the spinning-top geometry from Edax difficulty levels.
- Do **not** assume that a t-SNE cluster demonstrates a failure mechanism.
- Do **not** treat a gain against seen training opponents as evidence of generalization to unseen opponents.
- Do **not** treat held-out Edax levels as fully independent opponent families; they share engine components.
- Preserve negative findings, full protocols, and uncertainty estimates.

**Proposed next actions after the October 22 discussion:** (1) freeze the original controlled-evaluation protocol, (2) prepare and reserve held-out opponent definitions, (3) run the paired anomaly-reproduction pilot, and (4) do a small Edax training-adapter feasibility test in parallel if cheap. Begin mixture training only after scope, data targets, and compute-accounting rules are agreed.
