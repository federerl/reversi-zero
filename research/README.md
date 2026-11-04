# Reversi-Zero Research

**Working topic:** Investigating Opponent-Specific Weaknesses in AlphaZero-Style Reversi Agents.

**Status:** Advisor-supported experiment planning, updated after October 22 discussion on unseen-opponent generalization and Edax training.

- **[Updated experiment design and October 22 advisor feedback](experiment-design-proposal-2026-10.md)** — controlled anomaly replication, held-out opponent generalization, Edax integration and level semantics
- [Held-out opponent roster for the opponent-diversity experiment](held-out-opponents.md) — which opponents count as unseen, the similarity check as run, and the frozen roster
- [Research plan and reading list](research-plan.md)
- [Literature reviews](lit-reviews/README.md)
- [Existing experimental evidence and hypotheses](../docs/experiments.md)
- [Step 0 audit](step0-audit.md) and the [frozen evaluation protocol](eval-protocol-frozen.json) — what the historical numbers rest on, and exactly what Step 1 runs
- [Step 1 pilot results](step1-pilot-results.md) — the pattern reproduces on 3,000 games, and every loss to Greedy is an early wipeout
- **[Step 2 failure analysis](step2-failure-analysis.md)** — the network rates one disc as a near-certain win, because self-play taught it so; which head lets the wipeout through; the same loss against any disc-hungry opponent
- **[Step 3 design](step3-design.md)** — the pre-registered opponent-diversity experiment: groups, mixing rule, scorecard, predictions, how to launch
- **[Step 3 results](step3-results.md)** — playing Greedy in training did not transfer and barely took, because the external games never trained the value head where the search reads it; a random-playout search finds the same hole; the material target cuts it, seen and unseen, at a steep cost in strength
- [Agenda for the late-October advisor meeting](meetings/2026-10-advisor-meeting-agenda.html) — answers to the October 22 questions, the Step 0 to 2 results, and the Step 3 decisions
- [Step 3b design](step3b-design.md) — the follow-up: record the opponent's positions from external games as value-only examples, with the predictions written before the runs
- **[Step 3b results](step3b-results.md)** — the value head was trained where the search reads it and did not move; the training rows show why: the wipeout is a one-move tactic in a state that is usually favourable, outnumbered fifty to one in the data
- **[Step 3c: the terminal check](step3c-terminal-check.md)** — one exact rule in the evaluation search, no training: wipeouts against Greedy 65 → 12 on the same weights, the same on every checkpoint, nothing else moves; the weakness is a search-depth artefact
- [Agenda for the November 2 advisor meeting](meetings/2026-11-02-advisor-meeting-agenda.md) — the Step 3, 3b, 3c chain in one table, the revised explanation, what changes in the report, Phase 2
- [Step 4 design](step4-design.md) — the terminal check switched on in self-play, on top of the opponent mix that makes the low-disc positions: the strongest form of the data-side route, with the predictions written before the runs and what each outcome would mean for the claim
- **[What this study contributes](contribution.md)** — the claim in one sentence, what is known and what is new, tactical versus strategic weaknesses as a horizon of exact calculation, with the Go cyclic attack placed on it, and why the remedies differ, what would strengthen it

## Research framing

Existing evaluations suggest that performance against Greedy and depth-4 Minimax can differ unexpectedly even when overall strength rises. The historical results motivate investigation, but **do not yet prove** that self-play distribution shift causes the behavior or that the pattern generalizes.

First establish a reproducible, controlled baseline (paired colors/openings, game counts, MCTS budget, reference opponents), then examine failure modes. The advisor-requested next direction is a conditional opponent-diversity comparison: train against Greedy/Minimax or Edax, and evaluate on *unseen* strategies while accounting for compute. Edax levels are search-depth/selectivity configurations of the same engine, not independent neural-network versions.

This directory is for research plans, reading notes, and future study results; existing benchmark claims and run details remain in [docs/experiments.md](../docs/experiments.md).
