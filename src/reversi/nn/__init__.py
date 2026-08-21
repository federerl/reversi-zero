"""The neural network: turning a position into numbers, and guessing from them.

Two jobs, kept in separate files because only one of them needs torch.

* ``features.py`` turns a board position into the array the network reads. Pure
  numpy, so the engine, the tests, and the web API can all use it cheaply.
* ``model.py`` is the network itself, and ``evaluator.py`` is the adapter that
  lets the tree search call it. Both import torch.

Nothing in ``reversi.game`` or ``reversi.search`` imports this package's torch
half; the search talks to an ``Evaluator`` protocol instead, which is what keeps
the whole test suite runnable without a GPU.
"""

from __future__ import annotations
