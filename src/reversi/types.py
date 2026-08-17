"""Core value types shared by every layer.

This module is deliberately dependency-free (stdlib only) so that `reversi.game`
can import it without dragging in numpy or torch.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

# A bitboard is a plain Python int whose bit `i` is set iff the square with
# index `i` is occupied. Index convention (contract C1, docs/architecture.md):
#
#     index = row * board_size + col,  row 0 = top, col 0 = left
#
# This exact mapping is used by the engine, the feature encoder, the symmetry
# permutations, the API JSON payloads, and the web UI. It is never re-derived.
Bitboard: TypeAlias = int

# An action is a square index in [0, board_size**2), or `board_size**2` for PASS.
Action: TypeAlias = int


class Player(IntEnum):
    """The side to move. Values are stable and used as array indices."""

    BLACK = 0
    WHITE = 1

    @property
    def opponent(self) -> Player:
        return Player.WHITE if self is Player.BLACK else Player.BLACK

    @property
    def label(self) -> str:
        return "black" if self is Player.BLACK else "white"

    @classmethod
    def from_label(cls, label: str) -> Player:
        match label.lower():
            case "black" | "b":
                return cls.BLACK
            case "white" | "w":
                return cls.WHITE
            case _:
                msg = f"unknown player label {label!r}; expected 'black' or 'white'"
                raise ValueError(msg)


# Game result, always reported from the perspective of a named player:
#   +1 win, 0 draw, -1 loss.
Outcome: TypeAlias = int


def pass_action(board_size: int) -> Action:
    """The PASS action index for a board of the given size.

    PASS is the single action beyond the square indices. It is legal *iff* the
    side to move has no placement and the opponent has at least one (contract
    C3) -- so PASS and placements are never simultaneously legal, and PASS is
    never legal at a terminal state.
    """
    return board_size * board_size


def policy_size(board_size: int) -> int:
    """Length of the policy vector: one entry per square, plus PASS."""
    return board_size * board_size + 1
