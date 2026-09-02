# Where the ideas came from

This project is described as "AlphaZero-style". That phrase does a lot of work, so
this file says exactly which parts come from published papers, which come from
practice that has no paper behind it, and which are things everyone repeats
without a source.

The three tiers matter more than the bibliography. Anyone can implement a paper.
The interesting question is which numbers in this repository are backed by
research, which are backed by other people's engines, and which are backed by
nothing in particular — and being honest about the third group is the point.

Every citation below was checked against the publisher's record (Crossref, dblp,
PubMed) or the paper's own PDF. Where a check failed or was not done, this file
says so rather than guessing.

---

## Tier 1 — from the papers

### The core algorithm

**Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez,
A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F.,
Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). "Mastering
the game of Go without human knowledge." *Nature* 550(7676), 354-359.**
DOI [10.1038/nature24270](https://doi.org/10.1038/nature24270)

Called **AlphaGo Zero** below. Most of the algorithm in this repository comes from
here: learning from self-play starting at random weights, one network with a
policy head and a value head, the search's visit counts used as the training
target for the policy, and the game's final result used as the target for the
value.

**Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A.,
Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., &
Hassabis, D. (2018). "A general reinforcement learning algorithm that masters
chess, shogi, and Go through self-play." *Science* 362(6419), 1140-1144.**
DOI [10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404) - PMID 30523106

Called **AlphaZero** below. Its earlier preprint has a **different title** and
should be cited separately when the preprint is what you mean:

**Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). "Mastering Chess and
Shogi by Self-Play with a General Reinforcement Learning Algorithm."**
[arXiv:1712.01815](https://arxiv.org/abs/1712.01815) (v1, 5 December 2017; only
v1 was ever posted).

> **Two traps.** The *Science* title uses a serial comma - "chess, shogi, and
> Go". And "Silver et al." plus a year does not disambiguate the two: the author
> lists are not nested. AlphaGo Zero has 17 authors including Huang, Baker,
> Bolton, Chen, Hui and van den Driessche; AlphaZero has 13 including Lanctot and
> Kumaran. Neither list contains the other.

**Silver, D., Huang, A., Maddison, C. J., et al. (2016). "Mastering the game of
Go with deep neural networks and tree search." *Nature* 529(7587), 484-489.**

The original AlphaGo. Cited because its selection rule already used a policy
prior, which is the shape this project's search inherits.

### The search underneath it

**Kocsis, L., & Szepesvari, C. (2006). "Bandit based Monte-Carlo Planning." In
*Machine Learning: ECML 2006*, LNAI 4212, pp. 282-293. Springer.**
DOI [10.1007/11871842_29](https://doi.org/10.1007/11871842_29)

UCT - tree search that treats each choice as a bandit problem. **UCT has no
policy prior**: it is a value estimate plus a visit-count exploration bonus, and
that bonus contains a logarithm.

**Rosin, C. D. (2011). "Multi-armed bandits with episode context." *Annals of
Mathematics and Artificial Intelligence* 61(3), 203-230.**
DOI [10.1007/s10472-011-9258-6](https://doi.org/10.1007/s10472-011-9258-6)
(An earlier preliminary version appeared at ISAIM 2010.)

> **Do not credit the name "PUCT" to Rosin.** He calls his algorithm **PUCB** -
> "Predictor + UCB". The name *PUCT*, and the specific
> `sqrt(sum_b N(s,b)) / (1 + N(s,a))` form this project uses, come from the
> AlphaGo Zero and AlphaZero papers, which describe theirs as a *variant of*
> Rosin's work rather than a transcription of it.

### The statistics

**Bradley, R. A., & Terry, M. E. (1952). "Rank Analysis of Incomplete Block
Designs: I. The Method of Paired Comparisons." *Biometrika* 39(3-4), 324-345.**
DOI [10.1093/biomet/39.3-4.324](https://doi.org/10.1093/biomet/39.3-4.324)

Behind every rating in `docs/experiments.md`. The roman numeral "I." is part of
the title.

**Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
Statistical Inference." *Journal of the American Statistical Association*
22(158), 209-212.**
DOI [10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)

Behind every win-rate interval here. The interval is genuinely derived in this
four-page note, and p. 211 gives the successes-and-failures form that
`arena/stats.py` actually implements - so the code can cite the primary source
rather than a textbook. Wilson writes the multiplier as lambda where modern texts
write z. It is a "NOTES" item, so call it *primary* rather than *peer-reviewed*.

### Which paper states which number

| setting | value | stated in |
|---|---|---|
| Root noise weight epsilon | 0.25 | **AlphaGo Zero** only, Methods, "Self-play" |
| Dirichlet alpha, Go | 0.03 | **AlphaGo Zero** |
| Dirichlet alpha, chess / shogi | 0.3 / 0.15 | **AlphaZero**, Methods, "Configuration" |
| Temperature schedule | tau = 1 for the **first 30 moves of the game**, then tau to 0 | **AlphaGo Zero**, Methods, "Self-play" |
| No gating: always self-play with the newest network | - | **AlphaZero** |

Three details worth keeping straight.

**epsilon = 0.25 is not an AlphaZero number.** The AlphaZero preprint never states
an epsilon at all; it says "unless otherwise specified, the training and search
algorithm and parameters are identical to AlphaGo Zero." Cite AlphaGo Zero for it.

**The 30 is 30 plies in total, not 30 per player**, and it applies to *self-play
data generation only* - competitive play uses tau near 0 throughout. This project
keeps the same split; see contract C4 in `docs/architecture.md`, which holds the
stored policy target and the move actually played on separate temperatures.

**On gating**, AlphaGo Zero has one and AlphaZero removed it. AlphaGo Zero
promoted a new network only after it beat the current best over 400 games at
1,600 simulations with a win rate above 55%. AlphaZero generates self-play "using
the latest parameters for this neural network, omitting the evaluation step and
the selection of best player." **This project follows AlphaZero** - decision D9.

---

## Tier 2 — engine practice, no paper

### First Play Urgency

`search/mcts.py` gives an unvisited child an optimistic-but-reduced value
(`q_parent - 0.25 * sqrt(sum P)`) instead of zero.

**"First play urgency" and "FPU" appear zero times in the AlphaGo Zero paper**,
Methods included. That paper says only that a newly expanded edge starts at
N = 0, W = 0, Q = 0, P = p_a. So **FPU must not be attributed to AlphaGo Zero or
AlphaZero.** It belongs to the computer-Go lineage and the engines that grew out
of it - Leela Zero, Leela Chess Zero, KataGo.

The **0.25** here is a common engine default, not a value derived or measured
anywhere. It has not been tuned or ablated in this project. Asked why 0.25, the
honest answer is "it is what the engines use, and I did not test alternatives."

---

## Tier 3 — folklore, no citable origin

### "Dirichlet alpha is about 10 divided by the branching factor"

`docs/configuration.md` uses this rule to justify alpha = 1.0 for Reversi. **No
paper states it.**

What the AlphaZero preprint actually says is weaker: the noise "was scaled in
inverse proportion to the approximate number of legal moves in a typical
position" - inverse proportionality, with **no constant and no formula**.

Work the published values backwards and the constant is not even stable:

| game | alpha | typical legal moves | implied constant |
|---|---|---|---|
| chess | 0.3 | ~35 | ~10.5 |
| shogi | 0.15 | ~80 | ~12 |
| Go | 0.03 | ~250 | ~7.5 |

So "10 over the branching factor" is a **fit through three points after the
fact**, not a published rule. It is a reasonable way to pick a starting value and
should be described that way. The best citable source for the rule as stated is a
community write-up, not a paper.

---

## Not established — do not cite on this project's authority

**KataGo.** David J. Wu's "Accelerating Self-Play Learning in Go" is widely cited
as documenting FPU, and its arXiv identifier is commonly given as 1902.10565.
**Neither was verified here.** Check it yourself before citing it.

**Whether the AlphaGo Zero paper states a numeric c_puct.** Its Methods describes
c_puct qualitatively as "a constant determining the level of exploration" and says
the search parameters "were selected by Gaussian process optimization" - but
whether either paper's body gives a number is **unsettled here**. Separately,
AlphaZero's *released pseudocode* makes c_puct grow with the visit count:
`log((N(s) + c_base + 1) / c_base) + c_init`, with c_base = 19652 and
c_init = 1.25.

That last point cuts against a tempting simplification. It is accurate to say
**the formula as written in the papers has no logarithm** - the log lives in the
UCT lineage, whose bias term is `2 * C_p * sqrt(ln t / s)`. It is *not* accurate
to say there is no logarithm anywhere in AlphaZero: there is one, in the
coefficient rather than in the visit-count term.

**This project uses a fixed `c_puct = 1.5`**, which is neither a value the papers
state nor the pseudocode's schedule. It has not been tuned. That is a real gap and
an obvious thing to test.

---

## What this project changed, and why

Everything above is what was taken. These are the departures, made for Reversi or
for a laptop rather than for a data centre.

* **Three input planes, no history and no colour plane.** Reversi is Markov - no
  repetition, no ko - so past positions carry nothing. And it is exactly
  colour-symmetric once the board is seen from the mover's point of view, so a
  colour plane would add a feature that means nothing and halve how much each
  symmetry teaches. See ADR-0003.
* **A 6-block, 64-channel network** - about 458k parameters, against AlphaGo
  Zero's 40 blocks. Sized to a 4 GB laptop GPU, and to the fact that self-play
  throughput rather than model capacity is the scarce resource here.
* **Independent worker processes that each write their own shard**, rather than a
  central inference server. Chosen for the constraints in ADR-0004: eight cores,
  Windows, and a job that can be interrupted.
* **A value guardrail on the easy difficulty levels** - after searching, discard
  moves whose value is worse than the best by more than a threshold, then sample
  among what survives. This is what makes the Casual level *weak* rather than
  *stupid*: it plays reasonable but suboptimal moves instead of throwing away
  corners. Not from any source above.
* **Two independent rules engines and a differential test.** Not an algorithmic
  idea at all - it is the reason every number in this repository can be trusted.
