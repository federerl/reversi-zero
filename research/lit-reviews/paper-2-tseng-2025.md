# Paper 2: Can Go AIs Be Adversarially Robust?

**Tseng et al. (2025), AAAI**

## Summary

This paper extends Wang et al. (2023) by investigating whether the adversarial vulnerabilities found in KataGo can be addressed through additional training or changes to the neural network architecture. Rather than discovering new attacks alone, the authors focus on improving robustness against adversarial opponents. They evaluate three defense approaches: positional adversarial training, iterated adversarial training, and replacing the convolutional neural network (CNN) backbone with a Vision Transformer (ViT).

## Key Findings

- **Positional adversarial training:** Training KataGo on positions associated with known cyclic attacks helps defend against those attacks. However, newly trained adversaries can discover alternative strategies that still exploit the model.
- **Iterated adversarial training:** The authors repeatedly alternate between training an attacker and improving the defender. Although each defender becomes better at handling previous attacks, new attackers continue to find weaknesses. After nine iterations, the final attacker still achieves a 42% win rate against the defended model using 65,536 MCTS visits.
- **Vision Transformer:** Replacing the CNN backbone with a Transformer does not eliminate the vulnerability. The resulting model can still be defeated by cyclic adversarial strategies, suggesting that the problem is not solely caused by the CNN architecture.

Overall, the experiments show that defending against specific attacks does not necessarily lead to robustness against unseen adversarial strategies.

## Critical Thoughts

I found the iterated adversarial training experiment particularly interesting because it demonstrates the difficulty of generalizing from known attacks. Even when the defender improves against previously encountered strategies, relatively small changes in the attack can still cause failures.

However, I would not conclude that adversarial training is ineffective. The experiments show that these defenses can reduce attack success rates and increase the effort required to discover new vulnerabilities. The remaining issue is whether the model learns a generalizable defense or simply adapts to the specific attacks encountered during training.

## Connection to Reversi-Zero

This paper is relevant to my hypothesis that limited opponent diversity during self-play may contribute to opponent-specific weaknesses in Reversi-Zero.

One possible experiment would be to introduce Greedy or other opponent strategies into training and compare the results against a self-play-only baseline. However, improving performance against the opponents included in training would not necessarily demonstrate better generalization.

Following the approach in this paper, I should also evaluate against previously unseen opponents to determine whether increased training diversity improves robustness more broadly. Before testing such interventions, I still need to confirm that the weaknesses observed in my original Reversi-Zero evaluations are reproducible.
