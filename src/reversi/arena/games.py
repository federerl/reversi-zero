"""Writing down the games themselves, not just who won.

A match report says "11 wins from 30" and splits that by colour. That is enough
to rate an agent and not enough to find out *why* it lost: which move went
wrong, whether the search saw the problem coming, whether the losses share a
shape. Answering those needs the moves, and for the network agent, what its
search thought at each of them.

So a match can optionally keep a record of every game: the opening it started
from, every move in order, and for each move the search's own numbers when the
mover has any -- the value it gave the position and how it spread its visits
across the legal moves. Fixed opponents like Greedy and Minimax have no such
numbers, and their plies simply carry none.

**Recording changes nothing about the match.** The same seeds draw the same
openings, the same agents pick the same moves, and the report comes out
identical with or without it. It is a tap on the side of the loop, not a branch
in it, which is what makes a recorded result comparable to an unrecorded one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reversi.atomicio import atomic_write_json
from reversi.search.mcts import SearchResult
from reversi.types import Action

__all__ = ["GameRecord", "PlyRecord", "SearchReporting", "write_games"]

GAMES_VERSION = 1


@runtime_checkable
class SearchReporting(Protocol):
    """An agent that remembers what its last search found.

    The arena only needs one thing beyond ``select``: a way to ask, right after
    the move, what the search thought. Agents that search by other means (a fixed
    evaluation, an external engine) do not have this and are not expected to.
    """

    @property
    def last_search(self) -> SearchResult | None: ...


@dataclass(frozen=True, slots=True)
class PlyRecord:
    """One move, and what the mover knew when it chose it."""

    action: Action
    by_a: bool
    """Whether the first agent in the match played this move."""
    net_value: float | None = None
    """The network's opinion of the position before searching, for the mover,
    in [-1, +1]. ``None`` when the mover has no network."""
    search_value: float | None = None
    """The same after searching: the visit-weighted value at the root."""
    visits: tuple[tuple[Action, int], ...] | None = None
    """How the search spread its simulations over the legal moves, as
    ``(action, visits)`` pairs. Only the legal moves, so it stays small."""

    @classmethod
    def from_move(cls, mover: object, action: Action, *, by_a: bool) -> PlyRecord:
        """Capture the mover's search numbers if it has any, else just the move."""
        search = mover.last_search if isinstance(mover, SearchReporting) else None
        if search is None:
            return cls(action=action, by_a=by_a)
        return cls(
            action=action,
            by_a=by_a,
            net_value=float(search.root_value),
            search_value=float(search.value()),
            visits=tuple(zip(search.actions, search.visits, strict=True)),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": int(self.action), "by_a": self.by_a}
        if self.visits is not None:
            payload["net_value"] = self.net_value
            payload["search_value"] = self.search_value
            payload["visits"] = [[int(a), int(n)] for a, n in self.visits]
        return payload


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One finished game: where it started, every move, how it ended."""

    index: int
    """Position in the match. Games ``2k`` and ``2k+1`` share opening ``k``."""
    opening: tuple[Action, ...]
    a_is_black: bool
    plies: tuple[PlyRecord, ...]
    black_discs: int
    white_discs: int
    outcome_for_a: int
    """+1 if the first agent won, -1 if it lost, 0 for a draw."""

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "opening": [int(a) for a in self.opening],
            "a_is_black": self.a_is_black,
            "plies": [ply.to_json() for ply in self.plies],
            "final": {"black": self.black_discs, "white": self.white_discs},
            "outcome_for_a": self.outcome_for_a,
        }


def write_games(
    path: Path,
    records: list[GameRecord],
    *,
    agent_a: str,
    agent_b: str,
    seed: int,
    opening_plies: int,
    board_size: int,
) -> dict[str, Any]:
    """Write a match's games next to its report. Returns what was written.

    The header repeats the match's identity so the file stands on its own: a
    directory of these can be read back without the report that produced them.
    """
    payload = {
        "games_version": GAMES_VERSION,
        "agents": {"a": agent_a, "b": agent_b},
        "board_size": board_size,
        "protocol": {"seed": seed, "opening_plies": opening_plies},
        "games": [record.to_json() for record in records],
    }
    atomic_write_json(path, payload)
    return payload
