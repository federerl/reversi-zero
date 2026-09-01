"""Four difficulty levels from one network, and the trick that makes the easy
ones pleasant to play against.

**The obvious way to make an agent weaker is to make it random, and it is wrong.**
An agent that plays randomly some fraction of the time does not feel like a
weaker opponent; it feels like a broken one. It will hand you a corner for no
reason, then play three strong moves, then blunder its whole position. Losing to
it is annoying and beating it teaches you nothing.

So weakness here comes from three dials, and one guardrail:

* **How long it thinks.** 16 simulations against 800 is the biggest single lever
  on strength, and the most honest: the agent is genuinely considering less.
* **How much it samples.** At temperature 0 it always plays its best move. Above
  0 it picks in proportion to how much it liked each one, so it varies.
* **How many moves it will consider.** Restricting to the top few visits keeps it
  from picking something it barely looked at.

* **The guardrail** is what stops sampling from producing nonsense. After the
  search, any move whose value is worse than the best by more than ``guard`` is
  discarded *before* sampling. So an easy level plays a move it thinks is
  slightly worse -- never one it knows is terrible. It gives away small
  advantages, not corners.

That is the difference between an opponent that is beatable and one that is
broken, and it is the reason ``guard`` exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reversi.errors import ConfigError
from reversi.search.mcts import SearchResult
from reversi.types import Action

__all__ = ["LEVELS", "DifficultyLevel", "choose_move", "level_by_name"]


@dataclass(frozen=True, slots=True)
class DifficultyLevel:
    """One rung of the ladder."""

    name: str
    label: str
    simulations: int
    temperature: float
    top_k: int | None
    guard: float
    """How far below the best move's value a candidate may be and still be
    played. 0 means "only the best move"; 0.35 means "anything not clearly
    losing"."""

    description: str = ""

    def __post_init__(self) -> None:
        if self.simulations < 1:
            msg = f"{self.name}: simulations must be at least 1"
            raise ConfigError(msg)
        if self.temperature < 0.0:
            msg = f"{self.name}: temperature cannot be negative"
            raise ConfigError(msg)
        if self.top_k is not None and self.top_k < 1:
            msg = f"{self.name}: top_k must be at least 1 when set"
            raise ConfigError(msg)
        if self.guard < 0.0:
            msg = f"{self.name}: guard cannot be negative"
            raise ConfigError(msg)


LEVELS: tuple[DifficultyLevel, ...] = (
    DifficultyLevel(
        name="casual",
        label="Casual",
        simulations=16,
        temperature=0.8,
        top_k=3,
        guard=0.35,
        description="Thinks briefly and varies its play. Gives away small advantages.",
    ),
    DifficultyLevel(
        name="club",
        label="Club",
        simulations=64,
        temperature=0.35,
        top_k=2,
        guard=0.20,
        description="Considers more, and mostly plays what it considers best.",
    ),
    DifficultyLevel(
        name="strong",
        label="Strong",
        simulations=256,
        temperature=0.0,
        top_k=None,
        guard=0.05,
        description="Plays its best move every time.",
    ),
    DifficultyLevel(
        name="max",
        label="Max",
        simulations=800,
        temperature=0.0,
        top_k=None,
        guard=0.0,
        description="Thinks as long as it is allowed to, and never compromises.",
    ),
)


def level_by_name(name: str) -> DifficultyLevel:
    for level in LEVELS:
        if level.name == name:
            return level
    known = ", ".join(level.name for level in LEVELS)
    msg = f"unknown difficulty {name!r}; expected one of {known}"
    raise ConfigError(msg)


def choose_move(
    result: SearchResult,
    level: DifficultyLevel,
    rng: np.random.Generator | None = None,
) -> Action:
    """Pick a move at this difficulty, applying the guardrail first.

    Order matters: **filter, then sample.** Sampling first and filtering after
    would let a bad move be chosen and then vetoed, which changes the
    distribution in a way that is hard to reason about. Filtering first means the
    agent is choosing among moves it considers acceptable, which is exactly what
    a weaker but sane player does.
    """
    if not result.actions:
        msg = "cannot choose a move from a search with no actions"
        raise ConfigError(msg)

    candidates = _acceptable(result, level)

    if level.temperature <= 0.0 or len(candidates) == 1:
        # Best of what survived the guard -- most visited, as always.
        best = max(candidates, key=lambda i: result.visits[i])
        return result.actions[best]

    if rng is None:
        msg = f"difficulty {level.name!r} samples its move, so it needs an rng"
        raise ConfigError(msg)

    weights = np.array([result.visits[i] for i in candidates], dtype=np.float64)
    if weights.sum() <= 0:
        return result.actions[candidates[0]]

    sharpened = weights ** (1.0 / level.temperature)
    total = sharpened.sum()
    if not np.isfinite(total) or total <= 0.0:
        best = max(candidates, key=lambda i: result.visits[i])
        return result.actions[best]

    chosen = int(rng.choice(len(candidates), p=sharpened / total))
    return result.actions[candidates[chosen]]


def _acceptable(result: SearchResult, level: DifficultyLevel) -> list[int]:
    """Indices of the moves this level is willing to play.

    Two filters, both against the *best* move rather than an absolute threshold:
    a position where every move is bad should still produce a move.
    """
    order = sorted(range(len(result.actions)), key=lambda i: -result.visits[i])

    # The guardrail runs first, and top_k narrows what survives it.
    #
    # The other order looks equivalent and is not. Taking the three most-visited
    # moves first makes the guard compare against the best of *those three*
    # rather than the best move available, so a strong move that happened to be
    # searched less becomes invisible and a weaker one passes a guard it should
    # have failed. The criterion is worded against the best move available, and
    # this is the order that delivers that.
    #
    # Found by the calibration: one violation in 500 moves, which is exactly the
    # rate you would expect from a case this narrow.
    if level.guard > 0.0:
        # Only judge moves the search actually looked at -- an unvisited move has
        # no measured value, and treating its 0.0 as an opinion would let a level
        # with a wide guard play something it never considered.
        visited = [i for i in order if result.visits[i] > 0]
        if visited:
            best_value = max(result.q_values[i] for i in visited)
            order = [i for i in visited if result.q_values[i] >= best_value - level.guard]

    if level.top_k is not None:
        order = order[: level.top_k]

    return order or [max(range(len(result.actions)), key=lambda i: result.visits[i])]
