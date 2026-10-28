"""Games against a fixed opponent, mixed into a generation of self-play.

Ordinary self-play has the network on both sides of every game. The research
experiments (``research/step3-design.md``) add a second kind of training game:
the network on one side, a fixed opponent such as Greedy or Minimax on the
other. The network searches, records its policy targets and learns from the
result exactly as in self-play; the opponent's moves only advance the game.

**Why it is a fixed share, decided by the game index.** A generation of 600
games is split across workers by index range, and a worker that fails is
re-run on the same range. If which games were external depended on anything
else, a re-run could play a different mix and the generation would not
reproduce. So the schedule is a pure function of the game index: external
games are spread evenly through the index range, each one is assigned its
opponent in turn, and the network's colour alternates per opponent. Every
worker, every re-run and every resume gets the same answer.

**What it does not do.** Nothing here touches the value target: a position
from an external game is labelled with that game's result, as a self-play
position is. Whether that is the right choice is an experimental question and
is argued in the design document, not decided in code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from reversi.agents.base import Agent
from reversi.errors import ConfigError
from reversi.types import Player

__all__ = ["Assignment", "OpponentSchedule", "build_opponents"]


@dataclass(frozen=True, slots=True)
class Assignment:
    """One external game: who the network plays, and as which colour."""

    opponent: str
    network: Player


@dataclass(frozen=True, slots=True)
class OpponentSchedule:
    """Which games of a generation are played against a fixed opponent.

    ``selfplay_games`` ordinary games and ``opponent_games`` external games make
    up a generation of ``total`` games, numbered ``0`` to ``total - 1``. The
    external ones are spread evenly through that range.
    """

    selfplay_games: int
    opponent_games: int
    opponents: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selfplay_games < 0 or self.opponent_games < 0:
            msg = (
                f"game counts cannot be negative: selfplay_games={self.selfplay_games}, "
                f"opponent_games={self.opponent_games}"
            )
            raise ConfigError(msg)
        if self.opponent_games > 0 and not self.opponents:
            msg = f"{self.opponent_games} games against a fixed opponent, but no opponent named"
            raise ConfigError(msg)
        if len(set(self.opponents)) != len(self.opponents):
            msg = f"the same opponent is named twice: {self.opponents}"
            raise ConfigError(msg)

    @property
    def total(self) -> int:
        return self.selfplay_games + self.opponent_games

    def ordinal(self, index: int) -> int | None:
        """Which external game ``index`` is (0-based), or ``None`` for a self-play game.

        The external games are the indices at which ``floor(index * X / T)``
        steps up, for ``X`` external games in ``T``: exactly ``X`` of them, as
        evenly spaced as integers allow.
        """
        if self.opponent_games == 0:
            return None
        if not 0 <= index < self.total:
            msg = f"game index {index} is outside this generation of {self.total} games"
            raise ConfigError(msg)
        before = (index * self.opponent_games) // self.total
        after = ((index + 1) * self.opponent_games) // self.total
        return before if after > before else None

    def assignment(self, index: int) -> Assignment | None:
        """The opponent and the network's colour for game ``index``, if it is external.

        Opponents take turns in the order given; the network's colour flips each
        time the list wraps around, so every opponent is met equally often with
        each colour.
        """
        k = self.ordinal(index)
        if k is None:
            return None
        rounds, which = divmod(k, len(self.opponents))
        network = Player.BLACK if rounds % 2 == 0 else Player.WHITE
        return Assignment(opponent=self.opponents[which], network=network)

    def external_indices(self) -> list[int]:
        return [i for i in range(self.total) if self.ordinal(i) is not None]


def build_opponents(names: Iterable[str]) -> dict[str, Agent]:
    """One agent per opponent name, keyed by the name the schedule uses.

    Names are the arena's entrant names (``greedy``, ``minimax-d4``, ``edax-l5``),
    so a training opponent is exactly the thing the evaluation would call by that
    name. Networks are refused: a game against another network is a different
    experiment, and it would need the worker to hold two models.
    """
    from reversi.arena.entrants import build_agent, parse_entrant

    agents: dict[str, Agent] = {}
    seen: dict[str, str] = {}
    for name in names:
        spec = parse_entrant(name, default_simulations=1)
        if spec.needs_network:
            msg = f"training opponent {name!r} is a network; only fixed opponents are supported"
            raise ConfigError(msg)
        if spec.name in seen:
            msg = f"{name!r} and {seen[spec.name]!r} are the same opponent ({spec.name})"
            raise ConfigError(msg)
        seen[spec.name] = name
        agents[name] = build_agent(spec)
    return agents
