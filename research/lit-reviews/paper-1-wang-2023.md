# Paper 1: Adversarial Policies Beat Superhuman Go AIs

**Wang et al. (2023), ICML**

## Summary

This paper investigates whether superhuman Go agents can be exploited by weaker adversarial policies. The authors train an adversary specifically against KataGo using victim-play instead of traditional self-play. They also introduce Adversarial MCTS (A-MCTS), which models the victim's behavior during search, and curriculum learning to gradually increase the difficulty of the opponent.

## Key Findings

The authors discovered two major attacks:

- **Pass attack:** The adversary exploits KataGo's tendency to pass before securing its territory, achieving a 99.9% win rate against KataGo without search.
- **Cyclic attack:** After defending against the pass attack, the researchers discovered another strategy that causes KataGo to misjudge the safety of large connected groups. The adversary achieves a 97.3% win rate against KataGo with 4,096 visits and still wins 72% of games against KataGo with 10 million visits.

The researchers also found that adversarial training could defend against the original cyclic attack, but further training allowed the adversary to discover successful variations. Their analysis of KataGo's value predictions and neural-network activations suggests that the model systematically misjudges certain unusual positions.

## Critical Thoughts

I initially questioned the pass attack results after examining the games on the authors' website. KataGo appeared to control most of the board, yet the adversary was declared the winner. After learning more about the Tromp–Taylor scoring rules, I understood why these results were valid under the experimental setup. However, this attack depends heavily on the game's termination and scoring rules.

The cyclic attack seems more relevant to my research because it reveals a weakness in KataGo's evaluation of certain board positions. I also found it interesting that increasing MCTS simulations significantly did not completely eliminate the vulnerability.

## Connection to Reversi-Zero

My Reversi-Zero agent previously showed a higher win rate against depth-4 minimax than against Greedy. Although Greedy is not a deliberately trained adversary, this may indicate an opponent-specific weakness.

This paper suggests several possible directions for my research: testing different opponent strategies, examining the effect of MCTS search budgets, and analyzing the network's value predictions or internal activations.

However, I still need to determine whether the observed weakness is reproducible and whether it comes from the network, MCTS, or another factor.
