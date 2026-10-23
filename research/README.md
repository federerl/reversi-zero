# Reversi-Zero Research

**Working topic:** Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents.

**Status:** Advisor-supported experiment planning, updated after October 22 discussion on unseen-opponent generalization and Edax training.

- **[Updated experiment design and October 22 advisor feedback](experiment-design-proposal-2026-10.md)** — controlled anomaly replication, held-out opponent generalization, Edax integration and level semantics
- [Research plan and reading list](research-plan.md)
- [Literature reviews](lit-reviews/README.md)
- [Existing experimental evidence and hypotheses](../docs/experiments.md)

## Research framing

Existing evaluations suggest that performance against Greedy and depth-4 Minimax can differ unexpectedly even when overall strength rises. The historical results motivate investigation, but **do not yet prove** that self-play distribution shift causes the behavior or that the pattern generalizes.

First establish a reproducible, controlled baseline (paired colors/openings, game counts, MCTS budget, reference opponents), then examine failure modes. The advisor-requested next direction is a conditional opponent-diversity comparison: train against Greedy/Minimax or Edax, and evaluate on *unseen* strategies while accounting for compute. Edax levels are search-depth/selectivity configurations of the same engine, not independent neural-network versions.

This directory is for research plans, reading notes, and future study results; existing benchmark claims and run details remain in [docs/experiments.md](../docs/experiments.md).
