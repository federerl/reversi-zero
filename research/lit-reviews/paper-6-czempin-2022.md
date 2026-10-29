# Paper 6: Reducing Exploitability with Population Based Training

**Czempin & Gleave (2022), ICML Workshop on New Frontiers in Adversarial Machine Learning**

## Summary

Self-play policies can be beaten by *adversarial policies*: opponents trained specifically against one fixed victim, which are weak against everyone else (Gleave et al., 2020). This paper tests a defence. Instead of training against itself, the protagonist trains against a population of n opponents, each a separate self-play agent with its own random seed, alternating between updating the protagonist and the opponents. Robustness is measured by how many training steps an attacker needs before it beats the protagonist, in two small two-player zero-sum games, Laser Tag and Simple Push.

## Key Findings

- Self-play baselines in both games can be exploited, which extends the earlier finding to far lower-dimensional environments than the robotics tasks where adversarial policies were first found.
- In Laser Tag, larger populations (20 to 80 opponents) make attacks take longer; against the 80-opponent protagonist the attacker needs about twice the steps it needs against self-play. In Simple Push, two opponents are already as good as sixteen.
- The hardened protagonist keeps its ordinary strength, and the cost is n+1 times the compute of self-play. An attacker that keeps training still wins in the end.

## Critical Thoughts

The experiments are small, the paper is six workshop pages, and the environments have nothing to do with board games. What I take from it is the framing rather than the numbers. An adversarial policy is an opponent that is weak in general but exploits one blind spot of its victim. Greedy fits that description: it loses to every reference engine and still wipes out my networks in a third of their games. Nobody trained Greedy against them; a one-line rule found the blind spot on its own.

The defence is a useful contrast. Their opponents are learned and expensive, and the paper cannot say what the population teaches. Mine are hand-written, cheap, chosen for a known weakness, and I can say exactly which positions they add to training.

## Connection to Reversi-Zero

This paper gives my Step 2 result a name: the self-play network is exploitable, and the exploit did not need a trained attacker. It also accounts for two parts of the Step 3 design. The retention row of the scorecard, the Step 3 networks against Run 1's generation 60, corresponds to their check that hardening does not cost ordinary strength. The adversarial-policy test deferred to Phase 2 in the held-out roster is their attack applied to Othello.

A population of past checkpoints or other seeds as training opponents would be their defence in my setting. It is a natural group for a later phase, not the question Step 3 asks, which is whether a small share of games against known, fixed opponents is enough to fix a failure that self-play cannot see.
