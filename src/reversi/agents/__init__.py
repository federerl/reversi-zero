"""Things that can play Reversi, behind one interface.

The trained agent is one of these; so are the opponents it gets measured against.
Everything that plays a game -- self-play, the arena, the web API -- talks to this
interface rather than to a specific implementation, so a tournament can be a loop
over a list without caring what is inside each entry.

The opponents exist to answer one question the training loss cannot: *is it
actually any good?* A falling loss says the network is getting better at
predicting its own search. It says nothing about whether that search plays well.
Only playing somebody tells you that.
"""

from __future__ import annotations

from reversi.agents.base import Agent
from reversi.agents.greedy import GreedyAgent
from reversi.agents.random_agent import RandomAgent

__all__ = ["Agent", "GreedyAgent", "RandomAgent"]
