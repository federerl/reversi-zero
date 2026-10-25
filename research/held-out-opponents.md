# Held-Out Opponent Roster for the Opponent-Diversity Experiment

**Applies to:** Step 3 of the [experiment design proposal](experiment-design-proposal-2026-10.md)
**Prepared:** October 25, 2026
**Status:** Proposal. The roster must be frozen, and written here as final, before any mixed-opponent training starts. It does not change Steps 0–2.

## 1. What the roster is for

Step 3 trains the agent against extra opponents (Greedy and Minimax-d4, and possibly Edax) and asks whether it gets better against opponents it has *not* trained against. That question only has an answer if the test opponents are different from the training opponents in a way that matters. This document says what "different" means, lists the opponents that meet the bar, and fixes the rule for deciding before training which ones are in.

The lesson behind it comes from the Go literature. In Wang et al. (2023) and Tseng et al. (2025), training a Go agent to defend against one attack made it resistant to that attack and still vulnerable to new ones. Czempin and Gleave (2022) attribute this to the training opponents being too alike. Improvement against the opponents used in training is therefore not evidence of improvement in general, and a test set that resembles the training set cannot tell the two apart.

## 2. "Different" means a different source of knowledge

Every Othello opponent judges positions somehow. The judgment has to come from somewhere, and that source is what makes two opponents alike or not. Search depth, randomness added on top, and playing strength do not change the source.

The training opponents in Step 3 all draw on hand-written Othello knowledge:

- **Greedy** scores a move by the number of discs it flips.
- **Minimax-d4** searches four moves ahead and scores the resulting position with a hand-written table ([`src/reversi/agents/minimax.py`](../src/reversi/agents/minimax.py)). The table has three ingredients: corners are valuable, the squares next to corners are traps, and having more legal moves (mobility) is good.

The candidates first considered for the held-out set do not clear the bar, because each is one of those ingredients used on its own:

| Candidate | What it actually is |
|---|---|
| Corner-priority heuristic | The corner term of the Minimax table, alone |
| Mobility heuristic | The mobility term of the Minimax table, alone |
| Noisy Greedy | Greedy with randomness added |
| Minimax at another depth | The same table, searched deeper or shallower |

An agent that has learned to beat Minimax-d4 has learned to beat that table. Winning against the table's parts, or against the same table searched to depth 6, shows the same thing again. These candidates are kept only as *seen-family* controls (Section 5), never as held-out opponents.

## 3. The sources available in Othello

| Source of knowledge | Opponents | Relation to the training opponents |
|---|---|---|
| Hand-written rules | Greedy; Minimax; single-feature heuristics; parity (who gets the last move in each empty region); frontier discs (discs next to empty squares are weak); "evaporation" (keep few discs early) | Same source as training. Parity and frontier are concepts the Minimax table does not contain, so they are the least-alike members of this family, but still the same kind |
| No knowledge, search only | MCTS with random rollouts: no network and no table; a position is judged by playing random moves to the end many times and counting wins | Genuinely different. The repository's MCTS ([`src/reversi/search/mcts.py`](../src/reversi/search/mcts.py)) already has the tree; only the leaf evaluation changes |
| Learned from recorded games | Edax 4.6; Egaroucid (an independent open-source engine, GPL-3, whose author reports about 58% against Edax 4.4); n-tuple networks trained by temporal-difference learning or coevolution on Othello, with published weights | Not seen by Group B. Edax levels are one engine with different search settings; Egaroucid is a second, independently written engine with its own evaluation |
| Learned by self-play | Reversi-Zero checkpoints from other generations or other training runs; other AlphaZero-style Othello reimplementations | Same source as the agent itself. Useful for checking that strength has not dropped, not for testing transfer |
| Human play | The WTHOR database (French Othello federation tournament games, 1977 to date). A small policy network trained to predict the human move gives a human-imitation opponent | Entirely different source, and the kind of real-world strategy distribution Czarnecki et al. (2020) describe. Availability of the raw files is unconfirmed |
| Trained to beat this agent | An adversarial policy trained against a frozen checkpoint, as in Wang et al. (2023) | Measures exploitability rather than transfer. It is the direct analogue of the Go experiments and belongs in Phase 2 |

## 4. Proposed held-out roster

One opponent per source, so that a result cannot be explained by a single family. Listed in priority order; the first three are required, the rest are added if they can be obtained in time.

| # | Opponent | Source | Status | Role |
|---|---|---|---|---|
| 1 | Edax 4.6 at a level not used in training (candidate: L7; L1–L10 search to the end of the game once at most 2×level squares are empty, so L7 also tests a stronger endgame than any training opponent) | Learned from games | In the repository ([`src/reversi/agents/edax.py`](../src/reversi/agents/edax.py)) | Held-out, primary |
| 2 | Random-rollout MCTS at a fixed number of rollouts per move, chosen so that it scores roughly even against Minimax-d4 | None | To implement | Held-out, primary |
| 3 | A checkpoint from a different training run (different seed where available; otherwise a much earlier generation of the same run) | Self-play | In the repository | Strength retention |
| 4 | Egaroucid, or a published n-tuple evaluation searched to a fixed depth | Learned from games, independent of Edax | To build / obtain | Held-out, secondary |
| 5 | Human-imitation policy trained on WTHOR games | Human play | Data availability to confirm | Held-out, secondary |

**Seen set**, reported separately and never counted as held-out: Greedy, Minimax-d4, and for Group C the Edax levels it trained against. **Sanity check:** Random.

**Excluded from training and from tuning:** every opponent in the held-out roster. Edax may still be used in Step 2 as a reference evaluator for annotating failures, because that use does not feed the training signal.

## 5. The similarity check, fixed before training

A candidate is admitted to the held-out roster only after this check, and the check is run before Step 3 training begins.

1. Take a fixed set of positions: 500 mid-game positions (between 20 and 44 empty squares) sampled from the Step 1 game logs, seeded, saved to disk.
2. For each position, record the move chosen by each seen opponent (Greedy, Minimax-d4) and by each candidate. For opponents with random tie-breaking, record the full set of tied moves.
3. *Agreement* between two opponents is the fraction of positions where the candidate's move is among the seen opponent's chosen moves.
4. Measure the agreement between the seen opponents themselves (Greedy vs. Minimax-d4). That number is the baseline for "two opponents everyone agrees are different".
5. A candidate whose agreement with any seen opponent is higher than that baseline is not held out. It may still be reported as a seen-family control.

The baseline is measured rather than chosen so that the threshold is not an arbitrary number. The single-feature heuristics in Section 2 are expected to fail this check against Minimax-d4; the check is what documents it.

## 6. Scorecard

Reported for every training group, with the same colour-swapped opening pairs and the same agent-side MCTS budget as Step 1:

- **Primary:** the score against each held-out opponent, one row per opponent, with a paired uncertainty interval. There is no single aggregate held-out number; an average can hide a regression against one opponent behind a gain against another.
- **Retention:** the score against opponent 3 and against the strongest Edax level the agent can meaningfully be measured on. A group whose held-out scores rise while retention falls has traded general strength for a narrower one, and that is reported as such.
- **Seen:** the scores against Greedy and Minimax-d4. A gain here with no gain on the primary rows is the "improved only against training opponents" outcome the design exists to detect.

## 7. Relation to Steps 0–2

Nothing in this document changes the protocol audit, the paired reproduction pilot, or the failure analysis. Those use Greedy, Minimax-d4, Random, and Edax as a reference evaluator, exactly as the proposal states. The only constraints this document adds are timing ones:

- The roster and the similarity check are finalised before any mixed-opponent training starts.
- New opponents (the random-rollout MCTS, any imported engine) may be implemented at any time, but their results must not be used to choose training settings.

## 8. Open items

- Confirm whether the raw WTHOR files can be obtained; the parser format is documented, the download location is not.
- Confirm that Egaroucid builds in the project's environment and can be driven move by move like Edax.
- Pick the rollout count for opponent 2 by measuring it against Minimax-d4, and record it.
- Decide whether the adversarial-policy test (Section 3, last row) is in scope for Phase 2.

## References

- Wang et al. (2023), *Adversarial Policies Beat Superhuman Go AIs*. https://proceedings.mlr.press/v202/wang23g.html
- Tseng et al. (2025), *Can Go AIs Be Adversarially Robust?* https://arxiv.org/abs/2406.12843
- Czempin and Gleave (2022), *Reducing Exploitability with Population Based Training*. https://arxiv.org/abs/2208.05083
- Czarnecki et al. (2020), *Real World Games Look Like Spinning Tops*. https://arxiv.org/abs/2004.09468
- Egaroucid, an open-source Othello engine. https://www.egaroucid.nyanyan.dev/en/
- Buro, *Logistello* (pattern-based evaluation learned from games). https://web.cecs.pdx.edu/~bart/cs510games-summer2000/papers/buro/logistello.pdf
- WTHOR format, as documented by a parser library. https://docs.rs/crate/wthor/0.4.1
