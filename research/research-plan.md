# Reversi-Zero Independent Study — Research Plan

**Updated:** 2026-10-22  
**Supervisor:** Dr. Matt Boutell (approved)  
**Status:** Advisor reviewed the initial experiment direction and considered it feasible; follow-up questions concern generalization to unseen opponents, Edax training feasibility, and the definition of Edax levels.

**Next:** [Updated Research Design and Advisor Feedback — October 22](experiment-design-proposal-2026-10.md) (initial anomaly evaluation, proposed held-out generalization intervention, Edax 4.6 level semantics, and feasibility checks)

## Working Topic

**Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents**

The project examines whether AlphaZero-style self-play-trained Reversi agents exhibit reproducible, opponent-specific failure patterns despite strong overall playing ability. It is **not yet established** that self-play causes such weaknesses. The initial Greedy-versus-minimax win-rate contrast is a motivation to investigate, not proof of a blind spot or non-transitivity.

### Candidate questions
1. Under controlled evaluation, do opponent-specific weaknesses persist across openings, colors, seeds, and search budgets?
2. Are failures associated with network value/policy estimates, MCTS decisions, or particular game states?
3. **Advisor follow-up:** If trained with Greedy and Minimax, will the agent improve against *held-out* opponents? Compare compute-matched self-play, heuristic-mixed training, and (if feasible) Edax-mixed training; distinguish performance on seen opponents from genuine transfer.
4. Can Edax 4.6 be integrated into training, and how do its level-specific depth/selectivity settings affect training cost and the interpretation of experiments?

## Milestones

### Phase 1 — Faculty-supervised independent study (before master's applications)
- Complete targeted literature reviews and discuss the refined question with Dr. Boutell.
- Audit evaluation protocol: opponent versions, MCTS budgets, openings, side swaps, seeds, game counts, and confidence intervals.
- Re-evaluate existing checkpoints against Greedy, minimax, Edax, and selected controls.
- Analyze failures through game logs and network predictions; define held-out opponent policies before training experiments.
- Audit Edax 4.6's levels and smoke-test using its moves in training episodes. Treat Edax level variants as a *related engine family*, not independent strategies.
- Consider a modest, compute-accounted heuristic-opponent training comparison if evidence, runtime and advisor scope permit.
- Deliver a reproducible evaluation setup and a research report describing positive *or null* findings, limitations, and future directions.
- **Tentative goal:** mid-November 2026 (not yet confirmed as an advisor deadline).

### Phase 2 — Publication-oriented extension (if initial findings support it)
- Independent training seeds and more model checkpoints.
- Controlled opponent-mixing or population-training ablations with comparable training compute, including self-play vs Greedy+Minimax and (if feasible) an Edax-opponent group.
- Hold-out opponents, out-of-distribution openings, and cross-checkpoint/compute-budget evaluation.
- Value calibration and targeted representation analysis (t-SNE for exploration only, not causal proof).
- Consider another environment for generality if warranted; evaluate workshop/conference suitability with advisor.

## Literature

### Core papers currently being reviewed
- Wang et al. (2023), **Adversarial Policies Beat Superhuman Go AIs**. ICML. https://proceedings.mlr.press/v202/wang23g.html
- Tseng et al. (2025), **Can Go AIs Be Adversarially Robust?** AAAI. https://arxiv.org/abs/2406.12843
- Czarnecki et al. (2020), **Real World Games Look Like Spinning Tops**. https://arxiv.org/abs/2004.09468
- van der Maaten & Hinton (2008), **Visualizing Data using t-SNE**. JMLR. https://www.jmlr.org/papers/v9/vandermaaten08a.html
- McGrath et al. (2022), **Acquisition of Chess Knowledge in AlphaZero**. PNAS. https://pmc.ncbi.nlm.nih.gov/articles/PMC9704706/

### Newly suggested Othello-specific papers
1. Li et al. (2023), **Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task** (Othello-GPT), ICLR. https://arxiv.org/abs/2210.13382  
   **Why read:** Demonstrates how Othello can serve as a manageable testbed for studying learned internal representations. Relevant if this project pursues value-network or representation analysis; note that Othello-GPT is a sequence model, not an AlphaZero-style RL agent.
2. van der Ree & Wiering (2013), **Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play**. IEEE ADPRL, Singapore, pp. 108–115. https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/paper-othello.pdf  
   **Why read:** Prior Othello-specific work comparing self-play and learning against a fixed opponent. Reviewed in [Paper 5](lit-reviews/paper-5-vanderree-2013.md): each run trains on one source of games, with no search and no failure analysis, so mixed-opponent training and its transfer remain open.

### Additional related direction
- Czempin & Gleave (2022), **Reducing Exploitability with Population Based Training**. ICML AdvML workshop. https://arxiv.org/abs/2208.05083. Reviewed in [Paper 6](lit-reviews/paper-6-czempin-2022.md): the exploitability framing for the Greedy result, and the population defence as a Phase 2 contrast to fixed hand-written opponents.

## Scope and claims
- Othello is substantially smaller than Go, but supports inexpensive controlled experiments, strong reference engines, and detailed failure analysis.
- A single model and environment can support an informative undergraduate study; broad RL claims and publication-level generalization would need stronger evidence.
- Do not present the preliminary Greedy/minimax result as evidence of self-play overfitting until alternatives have been tested.
