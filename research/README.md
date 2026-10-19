# Reversi-Zero Research

**Working topic:** Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents.

**Status:** Faculty-supervised independent-study literature review and experiment planning (October 2026).

- **[Experiment design proposal for Dr. Boutell (October 2026)](experiment-design-proposal-2026-10.md)** — research questions, evidence, protocol, stages, and meeting decisions
- [Research plan and reading list](research-plan.md)
- [Literature reviews](lit-reviews/README.md)
- [Existing experimental evidence and hypotheses](../docs/experiments.md)

## Research framing

Existing evaluations suggest that performance against Greedy and depth-4 Minimax can differ unexpectedly even when overall strength rises. The historical results motivate investigation, but **do not yet prove** that self-play distribution shift causes the behavior or that the pattern generalizes.

First establish a reproducible, controlled baseline (paired colors/openings, game counts, MCTS budget, reference opponents), then examine failure modes. An optional second step is to investigate training-opponent diversity and whether any improvement transfers to *unseen* strategies.

This directory is for research plans, reading notes, and future study results; existing benchmark claims and run details remain in [docs/experiments.md](../docs/experiments.md).
