"""Four opponents of different strength, from one trained network.

Nobody wants to play something that beats them every time, and nobody is
impressed by something that plays randomly. So the same network is wrapped four
ways, and the settings that separate them are the ones that actually change how
well it plays.

The interesting problem is making the easy levels *weak* rather than *stupid*.
See ``levels.py``.
"""

from __future__ import annotations

from reversi.difficulty.levels import (
    LEVELS,
    DifficultyLevel,
    choose_move,
    level_by_name,
)

__all__ = ["LEVELS", "DifficultyLevel", "choose_move", "level_by_name"]
