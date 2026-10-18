# Paper 3: Real World Games Look Like Spinning Tops

**Czarnecki et al. (2020), NeurIPS**

## Summary

This paper investigates the structure of strategy spaces in competitive games, particularly symmetric zero-sum games. The authors propose the Spinning Top Hypothesis, suggesting that game strategies can be organized by two dimensions: transitive strength and non-transitive interactions.

Unlike a simple ranking where stronger agents consistently defeat weaker ones, non-transitive games can contain cycles in which different strategies counter one another. The authors suggest that these relationships are most diverse at intermediate levels of strength, giving the strategy space a spinning-top-like shape.

## Key Findings

The authors provide theoretical results supporting this structure and examine nine competitive games. They introduce Nash clustering to analyze the relationship between strategy strength and non-transitive interactions.

An important implication is that training against a single opponent may not expose an agent to the full diversity of strategies. Population-based training can help by introducing multiple opponents with different strengths and behaviors. The paper also connects the structure of the strategy space to the population size needed for effective learning.

## Critical Thoughts

I found the Spinning Top Hypothesis interesting because it challenges the idea that playing strength can always be represented by a single ranking. An agent may perform well against stronger opponents while struggling against certain weaker strategies.

However, the hypothesis does not mean that every unexpected matchup result is caused by non-transitivity. Establishing this relationship would require more extensive evaluations across multiple strategies.

## Connection to Reversi-Zero

My Reversi-Zero agent performs much better against depth-4 minimax than against Greedy, even though Greedy is generally weaker. This resembles the opponent-specific performance differences discussed in the paper, although my current results do not demonstrate a non-transitive cycle.

I also wondered whether my agent's approximate strength relative to Edax places it in the intermediate region of the Spinning Top Hypothesis. However, Edax difficulty levels cannot directly determine its position in the overall strategy space.

This provides a possible direction for future research using payoff matrices across different agents and checkpoints. For now, I would keep non-transitivity as a potential explanation rather than making it the main research focus.
