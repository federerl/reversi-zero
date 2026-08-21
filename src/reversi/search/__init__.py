"""Looking ahead: the tree search that turns a network's guesses into a move.

The network alone is a fast opinion. The search is what makes that opinion
strong: it spends a few hundred network calls exploring the lines the network
thinks are worth exploring, and comes back with a better answer than the network
gave on its own. That gap -- search result better than raw network -- is the
entire engine of AlphaZero-style learning, because the improved answer becomes
the training target for the next network.

Nothing here imports torch. The search talks to an ``Evaluator``, which is a
protocol -- a description of the two methods it needs -- so the tests can hand it
a stub that returns fixed numbers, and every search test runs in milliseconds on
a CPU with no model in sight.
"""

from __future__ import annotations
