# Held-Out Opponent Roster for the Opponent-Diversity Experiment

**Applies to:** Step 3 of the [experiment design proposal](experiment-design-proposal-2026-10.md) and the [Step 3 design](step3-design.md)
**Prepared:** October 25, 2026. **Frozen:** October 28, 2026, after the similarity check in section 5 and the playout calibration in section 8 were run. Any later change is recorded here with its reason.
**Status:** Final. It does not change Steps 0–2.

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

An agent that has learned to beat Minimax-d4 has learned to beat that table. Winning against the table's parts, or against the same table searched to depth 6, shows the same thing again. These candidates are kept only as *seen-family* controls, never as held-out opponents. This is a rule about where an opponent's judgement comes from, and it is decided by construction; the measured check in section 5 is a second filter that catches near-copies, not a replacement for it.

## 3. The sources available in Othello

| Source of knowledge | Opponents | Relation to the training opponents |
|---|---|---|
| Hand-written rules | Greedy; Minimax; single-feature heuristics; parity (who gets the last move in each empty region); frontier discs (discs next to empty squares are weak); "evaporation" (keep few discs early) | Same source as training. Parity and frontier are concepts the Minimax table does not contain, so they are the least-alike members of this family, but still the same kind |
| No knowledge, search only | MCTS with random rollouts: no network and no table; a position is judged by playing random moves to the end many times and counting wins | Genuinely different. The repository's MCTS ([`src/reversi/search/mcts.py`](../src/reversi/search/mcts.py)) already has the tree; only the leaf evaluation changes |
| Learned from recorded games | Edax 4.6; Egaroucid (an independent open-source engine, GPL-3, whose author reports about 58% against Edax 4.4); n-tuple networks trained by temporal-difference learning or coevolution on Othello, with published weights | Not seen by Group B. Edax levels are one engine with different search settings; Egaroucid is a second, independently written engine with its own evaluation |
| Learned by self-play | Reversi-Zero checkpoints from other generations or other training runs; other AlphaZero-style Othello reimplementations | Same source as the agent itself. Useful for checking that strength has not dropped, not for testing transfer |
| Human play | The WTHOR database (French Othello federation tournament games, 1977 to date). A small policy network trained to predict the human move gives a human-imitation opponent | Entirely different source, and the kind of real-world strategy distribution Czarnecki et al. (2020) describe. Availability of the raw files is unconfirmed |
| Trained to beat this agent | An adversarial policy trained against a frozen checkpoint, as in Wang et al. (2023) | Measures exploitability rather than transfer. It is the direct analogue of the Go experiments and belongs in Phase 2 |

## 4. The held-out roster

One opponent per source, so that a result cannot be explained by a single family. Listed in priority order; the first three are required, the rest are added if they can be obtained in time.

| # | Opponent | Source | Status | Role |
|---|---|---|---|---|
| 1 | Edax 4.6 at level 7 (L1–L10 search to the end of the game once at most 2×level squares are empty, so L7 also tests a stronger endgame than any training opponent). Levels 3 and 5 are also held out for groups A and B, which never see Edax; only a group C would make them seen | Learned from games | In the repository ([`src/reversi/agents/edax.py`](../src/reversi/agents/edax.py)) | Held-out, primary |
| 2 | Random-playout tree search, `rollout-r3000`: a PUCT search with every prior equal and each leaf judged by one random playout, 3000 playouts per move. The budget was chosen by the calibration in section 8 | None | In the repository ([`src/reversi/agents/rollout.py`](../src/reversi/agents/rollout.py)) | Held-out, primary |
| 3 | Run 1's generation 60 (`models/reversi-8x8-gen60.pt`), a different training run from the `control120` the Step 3 groups start from | Self-play | In the repository | Strength retention |
| 4 | Egaroucid, or a published n-tuple evaluation searched to a fixed depth | Learned from games, independent of Edax | To build / obtain | Held-out, secondary |
| 5 | Human-imitation policy trained on WTHOR games | Human play | Data availability to confirm | Held-out, secondary |

**Seen set**, reported separately and never counted as held-out: Greedy, Minimax-d4, and for a Group C the Edax levels it trained against. **Seen family**, reported as its own row: greedy-corner, greedy-d2, noisy-greedy, corner, mobility, minimax-d2 and minimax-d6, each a training opponent's own ideas in another form (section 5). **Sanity check:** Random.

**Excluded from training and from tuning:** every opponent in the held-out roster. Edax may still be used in Step 2 as a reference evaluator for annotating failures, because that use does not feed the training signal.

## 5. The similarity check, as run

The check asks how often a candidate plays a move the training opponent would have played, on the same positions. Run on October 28, 2026 by [`scripts/similarity_check.py`](../scripts/similarity_check.py); report: [`results/step2/similarity-check.json`](results/step2/similarity-check.json); positions: [`results/step2/similarity-positions.json`](results/step2/similarity-positions.json).

1. 500 mid-game positions (20 to 44 empty squares, at least two legal moves, all distinct) were sampled from the Step 1 game logs with a fixed seed and saved, so every opponent answers the same questions and a later candidate can be checked against the same set.
2. Each opponent's move was recorded in each position. Opponents that break ties at random (Greedy and its variants, the one-idea controls) report their whole tie set; the random-playout search is sampled once with a seed derived from the position; Random's set is every legal move.
3. The *agreement* of candidate A with reference B at a position is the share of A's moves that are in B's set, which is the probability that a random tie-break lands on a move B could have chosen; averaged over the 500 positions. Random's row is therefore the chance level.

**The threshold first proposed does not work, and the measurement says why.** The plan was to use the agreement between the two training opponents themselves as the ceiling, on the grounds that everybody accepts Greedy and Minimax-d4 as different strategies. Measured, Minimax-d4 plays one of Greedy's moves in 17% of positions and Greedy plays Minimax-d4's move in 10%, both *below* chance (23% and 14%). Two different strategies agree less often than a random player agrees with either, so that number cannot be a ceiling: it would exclude Random.

What the data does show is a gap between two groups whose relation to the training opponents is known by construction:

| training opponent | chance level | lowest agreement among its own variants | highest agreement among strong opponents from another source | threshold (midpoint) |
|---|---:|---:|---:|---:|
| Greedy | 22.7% | 74.3% (greedy-d2; greedy-corner 88.5%, noisy-greedy 84.6%) | 18.0% (Edax L5; L3 17.2%, L7 17.0%; the self-play networks 14.6% to 15.2%) | **46.1%** |
| Minimax-d4 | 13.6% | 56.6% (minimax-d2; minimax-d6 68.0%) | 45.2% (Edax L7; L3 42.8%, L5 43.0%; the self-play networks 40.4% to 41.6%) | **50.9%** |

A candidate passes the check when its agreement with each training opponent is at most that opponent's threshold. The threshold sits halfway between the groups rather than at either edge so that a candidate has to be clearly on one side.

| candidate | agrees with Greedy | with Minimax-d4 | check | by source (section 2) | role |
|---|---:|---:|---|---|---|
| Edax L7 | 17.0% | 45.2% | passes | learned from games | **held out** |
| Edax L3, L5 | 17.2%, 18.0% | 42.8%, 43.0% | pass | learned from games | held out for groups without Edax |
| rollout-r250 / r1000 / r3000 | 27.6% / 30.2% / 30.2% | 16.8% / 22.2% / 23.6% | pass | none | **held out** (one budget, section 8) |
| gen60, control120 | 15.2%, 14.6% | 40.4%, 41.6% | pass | self-play | retention |
| greedy-corner | 88.5% | 15.2% | fails: Greedy | hand-written, Greedy's idea | seen family |
| greedy-d2 | 74.3% | 12.5% | fails: Greedy | hand-written, Greedy's idea | seen family |
| noisy-greedy | 84.6% | 10.4% | fails: Greedy | hand-written, Greedy's idea | seen family |
| minimax-d2, minimax-d6 | 17.6%, 17.8% | 56.6%, 68.0% | fail: Minimax-d4 | the same table | seen family |
| corner | 24.3% | 23.4% | passes | one term of the table | seen family, by source |
| mobility | 27.4% | 33.3% | passes | one term of the table | seen family, by source |

Three things the table says beyond the verdicts:

- **Agreement with a strong opponent partly measures strength.** Edax L7, a far stronger player than Minimax-d4 with a different evaluation, agrees with it 45% of the time; the self-play networks, 40%; Minimax searched to depth 6 with the same table, 68%. Good players converge on good moves, so this check cannot certify two opponents unrelated. It can show that a candidate is not a near-copy, which is what it is for.
- **The one-idea controls pass the measured check and are excluded anyway.** Corner and mobility are weak players, and a weak player agrees with Minimax-d4 about as often as a random one plus a little. The proposal's expectation that they would fail the check against Minimax-d4 was wrong; the source rule in section 2 is what keeps them out of the held-out set, and they are reported as seen family on that ground.
- **The random-playout search leans slightly toward Greedy.** 28% to 30% against a chance level of 23%, and 33% to 38% against greedy-corner. A random game ends by counting discs, so playouts reward a disc lead a little. It is well below the threshold and shares no code, table or data with Greedy; the lean is noted so that a gain against it is read with this in mind.

## 6. Scorecard

Reported for every training group, with the same colour-swapped opening pairs and the same agent-side MCTS budget as Step 1:

- **Primary:** the score against each held-out opponent, one row per opponent, with a paired uncertainty interval. There is no single aggregate held-out number; an average can hide a regression against one opponent behind a gain against another.
- **Retention:** the score against opponent 3 and against the strongest Edax level the agent can meaningfully be measured on. A group whose held-out scores rise while retention falls has traded general strength for a narrower one, and that is reported as such.
- **Seen:** the scores against Greedy and Minimax-d4. A gain here with no gain on the primary rows is the "improved only against training opponents" outcome the design exists to detect.

## 7. Relation to Steps 0–2

Nothing in this document changes the protocol audit, the paired reproduction pilot, or the failure analysis. Those use Greedy, Minimax-d4, Random, and Edax as a reference evaluator, exactly as the proposal states. The only constraints this document adds are timing ones:

- The roster and the similarity check are finalised before any mixed-opponent training starts.
- New opponents (the random-rollout MCTS, any imported engine) may be implemented at any time, but their results must not be used to choose training settings.

## 8. The playout budget for opponent 2

The roster asks for a playout budget at which the knowledge-free search "scores roughly even against Minimax-d4", so that it tests the networks at about the strength of a training opponent while judging positions in a way no training opponent does. [`scripts/calibrate_rollout.py`](../scripts/calibrate_rollout.py) played each budget against Minimax-d4 and, for context, against Greedy: 100 games per match, 50 seeded 4-ply openings each played with both colours, seed 20261028, one process per match on the cluster's CPU partition. Report: [`results/step2/rollout-calibration.json`](results/step2/rollout-calibration.json).

| playouts per move | vs Minimax-d4 [95%] | vs Greedy [95%] | seconds per game, one core |
|---:|---|---|---:|
| 250 | 19.5% [12.9%, 28.3%] | 99.0% [94.6%, 99.8%] | 21 |
| 1,000 | 41.0% [31.9%, 50.8%] | 100% [96.3%, 100%] | 58 |
| **3,000** | **47.5% [38.0%, 57.2%]** | 100% [96.3%, 100%] | 134 |

**The budget is 3,000 playouts per move: `rollout-r3000`.** Its interval against Minimax-d4 is centred on even. A thousand playouts is clearly short of it, and a search that already beats Greedy in every game at 250 playouts is not a Greedy-like opponent, whatever its slight lean in the similarity check. The cost is about two and a quarter minutes a game on one cluster core, so a 200-game pairing is one core for seven or eight hours, which the evaluation runs as one job among many. An 8,000-playout budget was submitted in the same job for the shape of the curve; it is not needed for the choice and its row is added here when it finishes.

Two things worth knowing about this opponent. It plays a perfect-information search with a noisy leaf estimate, so unlike Greedy and Minimax it is not deterministic: the same position can get a different move on a different seed, and every match seeds it from the game index, as the arena seeds Greedy's tie-breaks. And its strength comes entirely from search depth over random play, so it is at its weakest early, where playouts from a near-empty board say little, and at its strongest late, where a few thousand playouts cover most of what can happen. That is the opposite profile from Minimax-d4, which plays the opening well and stops four plies short of the end.

## 9. Open items

- Confirm whether the raw WTHOR files can be obtained; the parser format is documented, the download location is not. Until then opponent 5 is not in the roster.
- Confirm that Egaroucid builds in the project's environment and can be driven move by move like Edax. Until then opponent 4 is not in the roster.
- The adversarial-policy test (section 3, last row) is Phase 2.

## References

- Wang et al. (2023), *Adversarial Policies Beat Superhuman Go AIs*. https://proceedings.mlr.press/v202/wang23g.html
- Tseng et al. (2025), *Can Go AIs Be Adversarially Robust?* https://arxiv.org/abs/2406.12843
- Czempin and Gleave (2022), *Reducing Exploitability with Population Based Training*. https://arxiv.org/abs/2208.05083
- Czarnecki et al. (2020), *Real World Games Look Like Spinning Tops*. https://arxiv.org/abs/2004.09468
- Egaroucid, an open-source Othello engine. https://www.egaroucid.nyanyan.dev/en/
- Buro, *Logistello* (pattern-based evaluation learned from games). https://web.cecs.pdx.edu/~bart/cs510games-summer2000/papers/buro/logistello.pdf
- WTHOR format, as documented by a parser library. https://docs.rs/crate/wthor/0.4.1
