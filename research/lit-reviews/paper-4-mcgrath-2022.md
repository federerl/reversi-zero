# Paper 4: Acquisition of Chess Knowledge in AlphaZero

**McGrath et al. (2022), PNAS**

## Summary

This paper investigates what AlphaZero learns internally through self-play training, particularly whether its neural network develops representations of human-understandable chess concepts without learning from human games. The authors study what concepts are learned, when they emerge during training, and where they are represented within the network.

## Key Findings

The researchers use linear probing to examine AlphaZero's internal representations. By extracting activations from different network layers and training checkpoints, they train simple linear models to predict chess concepts such as material, mobility, and king safety.

Their results show that many human-understandable concepts can be decoded from AlphaZero's neural network, even though it was trained entirely through self-play. They also find that different concepts emerge at different stages of training and are represented differently across network layers.

The authors additionally analyze AlphaZero's value function and opening preferences to investigate how its understanding of chess develops over time.

## Critical Thoughts

I found the idea of examining what a neural network learns during training particularly interesting. Rather than evaluating AlphaZero only through its playing strength, the authors investigate the knowledge represented inside the model.

However, one important limitation is that successfully decoding a concept using linear probing does not necessarily mean the network actually relies on that concept when making decisions. The analysis provides evidence about internal representations, but further experiments would be needed to establish causal relationships.

## Connection to Reversi-Zero

This paper provides a possible method for investigating the internal behavior of my Reversi-Zero agent. For example, I could examine whether its network learns strategic concepts such as mobility, corner control, and disc count, and how these representations change across training generations.

If the opponent-specific performance differences I previously observed are reproducible, analyzing internal activations could help identify whether certain game states are represented differently.

I could initially use t-SNE to explore these representations visually, followed by linear probing if specific concepts appear relevant. However, this would be a possible extension rather than the primary focus of my current research.
