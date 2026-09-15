"""Solving the last few moves exactly, instead of estimating them.

Everything else in this project answers "who is probably winning" and attaches an
interval to the answer. This package answers "who *is* winning" and attaches
nothing, because near the end of an Othello game the position is small enough to
search to the last square. The result is a proof, not a measurement.

Nothing here touches torch, and nothing here changes how the agent plays. It is
analysis: it powers the endgame puzzles, and it is the tool that could later tell
a player which move actually threw a won game away.
"""

from __future__ import annotations

from reversi.endgame.solver import (
    MAX_EMPTIES,
    NoChoiceError,
    RootSolution,
    TooManyEmptiesError,
    empties,
    solve_exact,
    solve_root,
)

__all__ = [
    "MAX_EMPTIES",
    "NoChoiceError",
    "RootSolution",
    "TooManyEmptiesError",
    "empties",
    "solve_exact",
    "solve_root",
]
