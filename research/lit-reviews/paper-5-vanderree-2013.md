# Paper 5: Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play

**van der Ree & Wiering (2013), IEEE ADPRL**

## Summary

This paper compares three ways of producing the training games for an Othello agent: self-play, playing against a fixed opponent while learning only from the agent's own moves, and playing against a fixed opponent while also learning from the opponent's moves. Each is tried with Q-learning, Sarsa and TD-learning, all using a small multi-layer perceptron that reads the 64 squares directly. There is no search: the agent plays the move whose resulting position the network rates highest. Three fixed players serve as both training and test opponents: a random player and two positional players, HEUR and BENCH, which score a position by a fixed table of square weights.

## Key Findings

Agents train for 2,000,000 games and are tested on 236 four-ply openings with both colours.

- Q-learning and Sarsa score highest against the opponent they trained against (0.87 against BENCH after training on BENCH, compared with 0.72 from self-play) and lose that advantage against a different one: training on HEUR gives 0.58 against BENCH, below self-play.
- TD-learning, which learns state values rather than action values, learns best from self-play and gains little or nothing from a fixed opponent.
- Learning from the opponent's moves as well as the agent's own is worse in every case, even though it doubles the training data.

The conclusion names as future work fixed opponents with more diverse strategies, and an analysis of how the training opponent's style shapes the learned play.

## Critical Thoughts

This is the closest prior work to my Step 3 question, so I read it mainly to check whether the question had already been answered. It has not. Each run trains on one source of games, so there is no mixture of self-play and fixed-opponent games, and the paper reports scores only, never why a game was lost. Both positional players evaluate the same way, so testing against the other one is a mild transfer test, and none of the three opponents plays for discs the way Greedy does.

The results do carry a warning. For the action-value methods, training against a fixed opponent made the agent better against that opponent and worse against others. That is the overfitting my held-out roster is designed to detect.

## Connection to Reversi-Zero

Step 3 mixes a fixed share of games against Greedy and Minimax-d4 into self-play, a treatment this paper did not try, and asks whether the gain transfers to opponents the network never trained against, which is the test its conclusion asks for. Two of my design choices have support here: external games produce training examples only at the network's own decisions, and the scorecard has retention and held-out rows because training against one opponent can lower the score against another.

The paper also sharpens what a positive result would mean. The TD agent, the one closest to AlphaZero's value learning, gained nothing from a fixed opponent. If Group B improves, the claim is not that fixed opponents help in general, but that they supply the one-disc positions self-play never produces, so that the value head learns the right sign for them.
