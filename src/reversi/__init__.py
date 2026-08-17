"""reversi-zero: an AlphaZero-style Reversi system.

Layering (enforced by ``tests/unit/test_import_boundaries.py``)::

    game/    pure Python, no torch          -- the rules, and only the rules
    search/  numpy; NN behind a Protocol    -- PUCT MCTS
    nn/      torch                          -- policy-value network
    agents/  game + optionally search/nn    -- baselines and the AZ agent
    data/ selfplay/ train/ ckpt/            -- the training pipeline
    arena/ difficulty/                      -- evaluation; never touches replay
    api/                                    -- product; imports NOTHING from
                                               train/ or selfplay/
"""

from __future__ import annotations

__version__ = "0.1.0"
