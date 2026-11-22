# References for the Phase 1 report

**Purpose:** the bibliography the [Phase 1 report](phase1-report-outline.md) draws on, with what each entry supports. The engine's own sources, the AlphaZero papers, UCT, PUCB and the two statistics papers, are documented with their provenance in [`docs/references.md`](../docs/references.md) and are listed here only by name.

Every entry below was checked on November 22, 2026 against the publisher's record (the Crossref API for every DOI; the PMLR, NeurIPS, ICLR, AAAI or IJCAI proceedings page; the arXiv abstract page for preprints). Where a record disagrees with another, the entry says so. Nothing is cited from memory.

---

## 1. The algorithm and the statistics (see `docs/references.md`)

Silver et al. 2016 (AlphaGo, *Nature* 529), Silver et al. 2017 (AlphaGo Zero, *Nature* 550), Silver et al. 2018 (AlphaZero, *Science* 362) and its 2017 preprint (arXiv:1712.01815), Kocsis & Szepesvári 2006 (UCT), Rosin 2011 (PUCB), Bradley & Terry 1952, Wilson 1927. Eight entries; full citations, DOIs and the traps between the two Silver papers are in `docs/references.md`.

Used in: section 2 (the loop, the search), section 4.1 (the system), section 4.2 (the intervals and ratings).

## 2. Monte-Carlo tree search and exact resolution inside it

**Coulom, R. (2006). "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search." In *Computers and Games* (CG 2006), LNCS 4630, pp. 72–83. Springer.**
DOI [10.1007/978-3-540-75538-8_7](https://doi.org/10.1007/978-3-540-75538-8_7)
Crossref dates the chapter 2007 (the volume's print date); the conference was May 2006 and most bibliographies say 2006. The paper that named the method.
Used in: section 2.

**Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M., Cowling, P. I., Rohlfshagen, P., Tavener, S., Perez, D., Samothrakis, S., & Colton, S. (2012). "A Survey of Monte Carlo Tree Search Methods." *IEEE Transactions on Computational Intelligence and AI in Games* 4(1), 1–43.**
DOI [10.1109/TCIAIG.2012.2186810](https://doi.org/10.1109/TCIAIG.2012.2186810)
The survey; its section on MCTS enhancements is where the solver and first-play-urgency ideas are catalogued.
Used in: section 2, section 8 (the residue of the check is a first-play-urgency effect).

**Winands, M. H. M., Björnsson, Y., & Saito, J.-T. (2008). "Monte-Carlo Tree Search Solver." In *Computers and Games* (CG 2008), LNCS 5131, pp. 25–36. Springer.**
DOI [10.1007/978-3-540-87608-3_3](https://doi.org/10.1007/978-3-540-87608-3_3)
Proves game-theoretic values inside the tree and backs them up exactly instead of averaging them. The Step 3c terminal check is the one-ply, no-backup form of this; the report has to say so before a reader does.
Used in: section 3, section 8.1, section 10 ("what is not new").

## 3. Learning from search, and the targets the heads are given

**Anthony, T., Tian, Z., & Barber, D. (2017). "Thinking Fast and Slow with Deep Learning and Tree Search." In *Advances in Neural Information Processing Systems* 30 (NIPS 2017).**
[arXiv:1705.08439](https://arxiv.org/abs/1705.08439), [NeurIPS page](https://neurips.cc/virtual/2017/poster/9311)
Expert Iteration: the search is the expert, the network the apprentice, and the policy target is what the search found. The Step 5 route, a search run at the opponent's position to give the policy head a target there, is this split applied at a position the loop would otherwise never search.
Used in: section 3, section 9.1.

**Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., & Silver, D. (2020). "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature* 588(7839), 604–609.**
DOI [10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4), preprint [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
Cited for the value target as a learned average over outcomes and for the search's reliance on the learned prior at every depth; the one paper after AlphaZero in which the evaluator sees no exact rule at all.
Used in: section 3, section 10.

**Wu, D. J. (2019). "Accelerating Self-Play Learning in Go."** [arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
KataGo. Cited for the auxiliary ownership target, which is the Step 2 comparison recipe. **Not** cited for first-play urgency: `docs/references.md` records that this attribution was not verified.
Used in: section 4.1, section 6.

## 4. Exploitability of self-play agents

**Lanctot, M., Zambaldi, V., Gruslys, A., Lazaridou, A., Tuyls, K., Pérolat, J., Silver, D., & Graepel, T. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." In *Advances in Neural Information Processing Systems* 30 (NIPS 2017).**
[arXiv:1711.00832](https://arxiv.org/abs/1711.00832), [NeurIPS page](https://neurips.cc/virtual/2017/poster/9199)
Names the problem: a policy trained against itself overfits to itself, measured as joint-policy correlation. The first five authors and the venue were checked on the arXiv and NeurIPS pages; the remaining author names are from the arXiv listing and should be read off the PDF before the report goes out.
Used in: section 3, section 10 ("what is not new").

**Wang, T. T., Gleave, A., Tseng, T., Pelrine, K., Belrose, N., Miller, J., Dennis, M. D., Duan, Y., Pogrebniak, V., Levine, S., & Russell, S. (2023). "Adversarial Policies Beat Superhuman Go AIs." In *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 35655–35739.**
[PMLR page](https://proceedings.mlr.press/v202/wang23g.html), preprint [arXiv:2211.00241](https://arxiv.org/abs/2211.00241)
The cyclic attack on KataGo. The framing for Step 1 and the case the discussion places on the horizon axis beside the wipeout. Review: [paper 1](lit-reviews/paper-1-wang-2023.md).
Used in: sections 1, 3, 10.

**Tseng, T., McLean, E., Pelrine, K., Wang, T. T., & Gleave, A. (2025). "Can Go AIs Be Adversarially Robust?" *Proceedings of the AAAI Conference on Artificial Intelligence* 39(26), 27662–27670.**
DOI [10.1609/aaai.v39i26.34980](https://doi.org/10.1609/aaai.v39i26.34980), preprint [arXiv:2406.12843](https://arxiv.org/abs/2406.12843)
Three defences, none of which survives an adaptive attacker. The "fixed opponents are not adaptive adversaries" limit. Review: [paper 2](lit-reviews/paper-2-tseng-2025.md).
Used in: sections 3, 10.

**Lan, L.-C., Zhang, H., Wu, T.-R., Tsai, M.-Y., Wu, I.-C., & Hsieh, C.-J. (2022). "Are AlphaZero-like Agents Robust to Adversarial Perturbations?" In *Advances in Neural Information Processing Systems* 35 (NeurIPS 2022).**
[NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/48adb34f7ee39177c4c23a8e4253a492-Abstract.html), preprint [arXiv:2211.03769](https://arxiv.org/abs/2211.03769)
Adding meaningless stones makes a Go agent play an obviously inferior move; the perturbation is found by search, not by a trained adversary. The nearest published case of a *tactical* blind spot in an AlphaZero-style agent found without training an opponent, which is what the random-playout row does here.
Used in: sections 3, 10.

**Czempin, P., & Gleave, A. (2022). "Reducing Exploitability with Population Based Training." ICML 2022 Workshop on New Frontiers in Adversarial Machine Learning.**
[arXiv:2208.05083](https://arxiv.org/abs/2208.05083)
The arXiv listing names the New Frontiers workshop; the ICML site also lists the paper under its ML4Cyber workshop. Cite the workshop the paper's own page names. The remedy the literature recommends, and the Phase 2 contrast. Review: [paper 6](lit-reviews/paper-6-czempin-2022.md).
Used in: sections 3, 10, 11.

**Timbers, F., Bard, N., Lockhart, E., Lanctot, M., Schmid, M., Burch, N., Schrittwieser, J., Hubert, T., & Bowling, M. (2022). "Approximate Exploitability: Learning a Best Response." In *Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence* (IJCAI-22), pp. 3487–3493.**
DOI [10.24963/ijcai.2022/484](https://doi.org/10.24963/ijcai.2022/484), preprint [arXiv:2004.09677](https://arxiv.org/abs/2004.09677) (the preprint's title adds "in large games")
Exploitability as worst-case performance, approximated by a learned best response. The held-out roster is a fixed, cheap stand-in for that quantity, and the report should say which of the two it measures.
Used in: section 4.3, section 10.

**Czarnecki, W. M., Gidel, G., Tracey, B., Tuyls, K., Omidshafiei, S., Balduzzi, D., & Jaderberg, M. (2020). "Real World Games Look Like Spinning Tops." In *Advances in Neural Information Processing Systems* 33 (NeurIPS 2020).**
[NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/ca172e964907a97d5ebd876bfdd4adbd-Abstract.html), preprint [arXiv:2004.09468](https://arxiv.org/abs/2004.09468)
Non-transitivity: why Greedy rates below Minimax and is the harder opponent. Review: [paper 3](lit-reviews/paper-3-czarnecki-2020.md).
Used in: sections 3, 5.

## 5. What the network knows

**McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). "Acquisition of chess knowledge in AlphaZero." *Proceedings of the National Academy of Sciences* 119(47), e2206625119.**
DOI [10.1073/pnas.2206625119](https://doi.org/10.1073/pnas.2206625119), preprint [arXiv:2111.09259](https://arxiv.org/abs/2111.09259) (the preprint's author list lacks Wattenberg)
Probing what the network represents. The background for the head probe and the calibration check. Review: [paper 4](lit-reviews/paper-4-mcgrath-2022.md).
Used in: sections 3, 4.5, 6.

**Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task." In *The Eleventh International Conference on Learning Representations* (ICLR 2023).**
[arXiv:2210.13382](https://arxiv.org/abs/2210.13382), [ICLR page](https://iclr.cc/virtual/2023/oral/12641)
Othello-GPT. Cited once, for Othello as a probing testbed; not load-bearing.
Used in: section 2.

## 6. Othello

**van der Ree, M., & Wiering, M. (2013). "Reinforcement learning in the game of Othello: Learning against a fixed opponent and learning from self-play." In *2013 IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning* (ADPRL), pp. 108–115.**
DOI [10.1109/ADPRL.2013.6614996](https://doi.org/10.1109/ADPRL.2013.6614996)
Self-play against a fixed opponent in Othello, no search, no failure analysis; the paper also found that learning from the opponent's moves made results *worse*, which Step 3b reproduces for the value head and Step 5 reverses for the policy head. Review: [paper 5](lit-reviews/paper-5-vanderree-2013.md).
Used in: sections 3, 7, 9.

**Buro, M. (2002). "Improving heuristic mini-max search by supervised learning." *Artificial Intelligence* 134(1–2), 85–99.**
DOI [10.1016/S0004-3702(01)00093-5](https://doi.org/10.1016/S0004-3702(01)00093-5)
Logistello: the pattern-based evaluation, ProbCut and the opening book, from the program's author. The citation for "what a classical Othello evaluator looks like" and for Edax's lineage. The file linked from `held-out-opponents.md` is the 2000 preprint of this article.
Used in: section 2, section 4.3.

**Takizawa, H. (2023). "Othello is Solved."** [arXiv:2310.19387](https://arxiv.org/abs/2310.19387)
Weak solution of 8×8 Othello (a draw), computed with a modified Edax. Not peer-reviewed; cite as a preprint. Establishes that the game is small enough to be read exactly in the endgame, and that Edax is the reference engine.
Used in: section 2, section 4.3.

**Delorme, R. Edax, version 4.6.** Software. [github.com/abulmo/edax-reversi](https://github.com/abulmo/edax-reversi)
The engine behind every Edax row. Cite the version and the commit the evaluation reports record.
Used in: section 4.3.

**Egaroucid** (software, [egaroucid.nyanyan.dev](https://www.egaroucid.nyanyan.dev/en/)) and the **WTHOR** game database format: named in `held-out-opponents.md` as secondary held-out sources that were not obtained. Mention in section 4.3 only if the report explains why they were not used.

---

## Count

| group | entries |
|---|---:|
| algorithm and statistics (`docs/references.md`) | 8 |
| MCTS and exact resolution | 3 |
| learning from search | 3 |
| exploitability | 7 |
| what the network knows | 2 |
| Othello (papers) | 3 |
| software | 1 (+2 optional) |
| **total citable** | **27** |

## Not yet found, and wanted

- A published source for **first-play urgency** as the engines implement it. Leela Zero and Leela Chess Zero document it in their repositories, not in a paper; Browne et al. 2012 is the nearest citable survey. The report should cite the engine documentation by URL and say that no paper defines the 0.25.
- A source for the **"usually favourable state with one refuting reply"** pattern outside this project: a second game with a documented one-move tactical blind spot in a learned evaluator. None has been located; its absence is what makes the Phase 2 second-game item worth running.
