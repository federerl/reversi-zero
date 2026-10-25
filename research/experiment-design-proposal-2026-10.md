# Research Design Discussion Draft — Reversi-Zero

**For:** Meeting with Dr. Matt Boutell  
**Prepared:** October 18, 2026  
**Status:** Proposal for discussion; no new experiments have been run or approved  
**Working topic:** *Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents*

## 1. Motivation and preliminary evidence

Reversi-Zero is an AlphaZero-style Othello agent trained by self-play, with a policy-value network and PUCT MCTS. Its prior evaluations suggest an unusual opponent-dependent pattern: improvement against a stronger-looking search baseline did not correspond to equally good performance against a simple Greedy policy.

An early evaluation of **Run 1, generation 60** recorded the following (see [existing experiment log](../docs/experiments.md), Run 1):

| Opponent | Gen-60 score | Important qualification |
|---|---:|---|
| Minimax depth-4 | 100% | Small original pairing; not a general guarantee |
| Greedy | 63% | Reported 95% CI: **46%–78%** |
| Greedy, agent as black | 60% (9 of 15) | From the report; the 90%/50% split once quoted here is withdrawn, see [step0-audit.md](step0-audit.md) |
| Greedy, agent as white | 67% (10 of 15) | From the report |

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
3. **Optional extension, not the first experiment:** Does training with a more diverse set of opponents improve performance against **unseen** opponent strategies, compared with a compute-matched self-play baseline?

**Working hypotheses, not findings:**

- H1: At least one checkpoint shows a stable performance difference between Greedy and other opponents after opening/color balancing.
- H2: A meaningful portion of the effect could be associated with recurring game-state features (a first look suggests early wipeouts, see [step0-audit.md](step0-audit.md) §4), with color, and/or with network/search evaluation errors. No color asymmetry is established; the historical one was withdrawn.
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
| Checkpoints | Three: Run 1 generation 60; generation 120 of the `control` run (Run 1's recipe, trained longer); generation 120 of the `e2-ownership` run (the network 1.0 ships). The middle one separates "more training" from "the ownership head"; identities and hashes in [step0-audit.md](step0-audit.md) |
| Core opponents | Greedy and Minimax-d4; add Random as a sanity check |
| Search budget | 50 simulations per move, the budget every historical number was measured at; see [eval-protocol-frozen.json](eval-protocol-frozen.json) |
| Opening design | Same seeded openings for each matchup; every opening played in both colors |
| Pilot size | 100 opening pairs **per matchup** = 200 games per checkpoint–opponent pairing; revise after runtime and precision estimate |
| Additional evaluation | Expand targeted, ambiguous pairings rather than running everything at the maximum sample size |
| Primary metrics | Score = (wins + 0.5 × draws)/games; 95% uncertainty intervals; black/white split and opening-paired differences |
| Secondary metrics | Move logs, game length, phase at first large evaluation change, corner control, mobility |

With 3 checkpoints × 2 core opponents × 200 games, the **core pilot is 1,200 games**; the full round robin, including the Random sanity checks and the checkpoints against each other, is 3,000. This is a *planning estimate*, not a pre-approved sample-size justification. Compute expected runtime and desired confidence-interval width before locking N. Games sharing openings or random seeds may be correlated, so uncertainty analysis should preserve pairing (e.g., opening-pair bootstrap), not automatically assume independent games.

**Interpretation:** Comparing raw win rates against two opponents of different intrinsic strength does not, by itself, prove a blind spot. Look for persistent and especially color-dependent interactions, and compare with the opponents' own head-to-head results.

**Gate A:** If the pattern disappears, document which protocol change explains it and report a reproducibility result. If the pattern persists with acceptable precision, proceed to targeted failure analysis.

### Step 2 — Failure analysis (conditional on Gate A)

1. Select representative winning and losing games using **predefined**, matched criteria (same opening/phase/color when possible), not only striking anecdotes.
2. Plot value-head estimates and MCTS root statistics across critical moves. Mark the earliest sizeable divergence, rather than just the final lost move.
3. Compare decisions with stronger reference evaluation where available (e.g., Edax with recorded settings); keep reference evaluations distinct from perfect-play ground truth.
4. Test inexpensive, targeted alternatives: color and opening dependence first; then MCTS-budget sensitivity (e.g., 64 / 256 / 800 **only on selected pairings/positions**), and board-level descriptors such as corner access and mobility.
5. Record which explanation(s) are supported, contradicted, or still unresolved.

**Gate B:** Identify at least one repeatable, descriptively meaningful failure pattern or establish a credible null result. Avoid causal language until there is a controlled intervention.

### Step 3 — Small intervention (optional; separate approval)

Only if Steps 1–2 establish an effect and sufficient compute/time remains:

- Compare self-play-only versus a carefully specified mixed-opponent training recipe, ideally from a **shared initial checkpoint**.
- Keep comparable experience/compute, evaluation search budget, and seeds; do not simultaneously modify network architecture or loss weights.
- Include both training opponents (e.g., Greedy) **and held-out** opponents with different behaviors. Predefine the primary held-out metric.
- Multiple independent seeds and an ablation study are desirable for later publication-oriented work; a single-run pilot should be clearly labeled preliminary.

**Gate C:** Consider expanding if generalization improves without an unacceptable loss of overall playing strength; otherwise report negative/limited results.

### Step 4 — Optional interpretability (not necessary for Phase 1)

If failure analysis points to a specific concept, consider comparing internal activations across checkpoints or using linear probes for concepts such as mobility or corner access. t-SNE may be used as an exploratory visualization only, with game-phase, color, and other confounders checked. **Neither visible clusters nor probe accuracy demonstrate causation.**

## 5. Deliverables and timeline

| When (tentative) | Deliverable |
|---|---|
| Next meeting | Agree on primary question, scope, pilot protocol, and checkpoint choice |
| Late October | Audited protocol, pilot evaluations, table of win rates/uncertainty/color splits |
| Early November | Targeted evaluation expansion plus annotated failure cases |
| Mid-November target | Short independent-study report with methods, figures, findings (including null findings), limitations, reproducibility instructions, and proposed next experiments |
| Later, if results merit | Multi-seed studies, opponent-diversity interventions, broader environment comparisons, and potential workshop/conference submission |

The mid-November milestone is a **tentative planning target**, not a confirmed deadline from Dr. Boutell. Our priority before graduate applications is a defensible result and a report, not a guaranteed positive finding.

## 6. Decisions to discuss with Dr. Boutell

1. Is the **behavior-first** research question appropriately scoped, or should we narrow it to the Greedy color asymmetry?
2. Which checkpoint and reference opponents provide the fairest, most interpretable pilot? Is 256 MCTS simulations reasonable given existing protocols?
3. What evidence would be sufficient to call the observed behavior an **opponent-specific weakness** rather than variation in opponent strength?
4. Would he prefer a straightforward paired-matchup analysis before any representation analysis or t-SNE?
5. What is the minimum convincing Phase 1 result/report by graduate-application time, and should opponent-mixing experiments be postponed?
6. Is the current Othello-only scope appropriate for an independent-study report, with broader generalization deferred to a publication-oriented Phase 2?

## 7. Scope boundaries

- Do **not** claim self-play caused the original anomaly without experiments isolating training distribution.
- Do **not** equate the result to Wang/Tseng's trained adversarial attacks; our Greedy baseline is not optimized to exploit the target.
- Do **not** claim a non-transitive cycle from a 63% score versus Greedy.
- Do **not** infer a position in the spinning-top geometry from Edax difficulty levels.
- Do **not** assume that a t-SNE cluster demonstrates a failure mechanism.
- Preserve negative findings, full protocols, and uncertainty estimates.

**Proposed first action after advisor feedback:** freeze the evaluation protocol and run the small paired pilot **before** starting any new model training.
