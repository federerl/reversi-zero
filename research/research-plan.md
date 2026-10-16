# Reversi-Zero Independent Study — Research Plan

**Updated:** 2026-10-12  
**Supervisor:** Dr. Matt Boutell (approved)  
**Status:** Literature review; research question remains provisional

## Working Topic

**Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents**

The project examines whether AlphaZero-style self-play-trained Reversi agents exhibit reproducible, opponent-specific failure patterns despite strong overall playing ability. It is **not yet established** that self-play causes such weaknesses. The initial Greedy-versus-minimax win-rate contrast is a motivation to investigate, not proof of a blind spot or non-transitivity.

### Candidate questions
1. Under controlled evaluation, do opponent-specific weaknesses persist across openings, colors, seeds, and search budgets?
2. Are failures associated with network value/policy estimates, MCTS decisions, or particular game states?
3. If the pattern is reproducible, does increasing training-opponent diversity improve performance against *held-out* opponents without sacrificing general playing strength?

## Milestones

### Phase 1 — Faculty-supervised independent study (before master's applications)
- Complete targeted literature reviews and discuss the refined question with Dr. Boutell.
- Audit evaluation protocol: opponent versions, MCTS budgets, openings, side swaps, seeds, game counts, and confidence intervals.
- Re-evaluate existing checkpoints against Greedy, minimax, Edax, and selected controls.
- Analyze failures through game logs and network predictions; consider a modest intervention only if evidence and time permit.
- Deliver a reproducible evaluation setup and a research report describing positive *or null* findings, limitations, and future directions.
- **Tentative goal:** mid-November 2026 (not yet confirmed as an advisor deadline).

### Phase 2 — Publication-oriented extension (if initial findings support it)
- Independent training seeds and more model checkpoints.
- Controlled opponent-mixing or population-training ablations with comparable training compute.
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
2. van der Ree & Wiering, **Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play**.  
   **Why read:** Prior Othello-specific work comparing self-play and learning against a fixed opponent. Important to investigate novelty before claiming that opponent diversity or self-play comparisons are new. **Bibliographic details and canonical link to be verified before formal citation.**

### Additional related direction
- Czempin & Gleave (2022), **Reducing Exploitability with Population Based Training**. Relevant background on exploitability and opponent populations; assess after the core papers if opponent mixing becomes the main question.

## Scope and claims
- Othello is substantially smaller than Go, but supports inexpensive controlled experiments, strong reference engines, and detailed failure analysis.
- A single model and environment can support an informative undergraduate study; broad RL claims and publication-level generalization would need stronger evidence.
- Do not present the preliminary Greedy/minimax result as evidence of self-play overfitting until alternatives have been tested.
