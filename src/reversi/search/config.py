"""The knobs the search runs with, and the one that must be off outside self-play.

This is a separate, plain dataclass rather than the YAML-backed ``MCTSConfig``
because the search gets used in places that have no run configuration at all --
tests, the web API, the arena -- and threading a whole validated config object
through them would be silly.

**The default is the safe one.** ``dirichlet_eps`` and ``temp_moves`` both
default to zero, meaning "no exploration noise, always play the best move". You
have to ask for exploration explicitly, via ``for_selfplay``. That asymmetry is
on purpose, and it follows from which mistake is worse:

* Forgetting to turn noise *on* during self-play costs some variety in the
  training games. It shows up in the metrics as a falling game-diversity number,
  and it is recoverable.
* Forgetting to turn noise *off* during evaluation means every strength number
  you measure is of an agent playing deliberately randomised moves. Nothing
  fails, nothing looks wrong, and every result in the project is quietly
  understated by an unknown amount.

So the dangerous one is opt-in (contract C7).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from reversi.errors import ConfigError

if TYPE_CHECKING:
    from reversi.config import MCTSConfig

__all__ = ["SearchConfig"]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Search parameters. Frozen, so a shared instance cannot be edited by one caller."""

    n_simulations: int = 100
    """How many times to walk the tree before answering. The dominant knob for
    both strength and cost -- time scales linearly with it."""

    c_puct: float = 1.5
    """How much to trust the network's suggestion versus what the search has
    actually found. Higher explores more of what the network likes; lower sticks
    with whatever is already scoring well."""

    fpu_reduction: float = 0.25
    """How pessimistic to be about a move nobody has tried yet.

    A child with no visits has no measured value, so it needs a stand-in. Using
    zero -- the obvious choice -- quietly ruins the search: in a position the
    mover is losing, every real move scores below zero, so an untried move at
    zero always looks better, and the search fans out across every move instead
    of going deeper into any of them. Starting from the parent's own estimate and
    subtracting a little keeps untried moves attractive without making them
    automatically the best thing on the board."""

    dirichlet_alpha: float = 1.0
    """Shape of the random noise added at the root during self-play. The rule of
    thumb is roughly 10 divided by the average number of legal moves; Reversi
    averages 8-10, so 1.0 on the full board and 2.0 on 4x4."""

    dirichlet_eps: float = 0.0
    """How much of that noise to mix in. **Must be 0 outside self-play.**"""

    temp_moves: int = 0
    """How many opening plies to pick a move at random (weighted by how much the
    search liked it) instead of always taking the best. Without this, self-play
    games from the same network are near-identical and the training data is a
    handful of positions repeated thousands of times."""

    temp_init: float = 1.0
    """The randomness level used during those opening plies. 1.0 means "pick in
    proportion to how often the search visited each move"."""

    def __post_init__(self) -> None:
        if self.n_simulations < 1:
            msg = f"n_simulations must be at least 1, got {self.n_simulations}"
            raise ConfigError(msg)
        if self.c_puct <= 0.0:
            msg = f"c_puct must be positive, got {self.c_puct}"
            raise ConfigError(msg)
        if self.fpu_reduction < 0.0:
            msg = f"fpu_reduction cannot be negative, got {self.fpu_reduction}"
            raise ConfigError(msg)
        if self.dirichlet_alpha <= 0.0:
            msg = f"dirichlet_alpha must be positive, got {self.dirichlet_alpha}"
            raise ConfigError(msg)
        if not 0.0 <= self.dirichlet_eps <= 1.0:
            msg = f"dirichlet_eps must be between 0 and 1, got {self.dirichlet_eps}"
            raise ConfigError(msg)
        if self.temp_moves < 0:
            msg = f"temp_moves cannot be negative, got {self.temp_moves}"
            raise ConfigError(msg)
        if self.temp_init < 0.0:
            msg = f"temp_init cannot be negative, got {self.temp_init}"
            raise ConfigError(msg)

    # -----------------------------------------------------------------
    # Constructors for the two situations that exist
    # -----------------------------------------------------------------

    @classmethod
    def for_selfplay(cls, mcts: MCTSConfig) -> SearchConfig:
        """Generating training games: exploration on, exactly as configured."""
        return cls(
            n_simulations=mcts.n_simulations,
            c_puct=mcts.c_puct,
            fpu_reduction=mcts.fpu_reduction,
            dirichlet_alpha=mcts.dirichlet_alpha,
            dirichlet_eps=mcts.dirichlet_eps,
            temp_moves=mcts.temp_moves,
            temp_init=mcts.temp_init,
        )

    @classmethod
    def for_evaluation(cls, mcts: MCTSConfig, *, n_simulations: int | None = None) -> SearchConfig:
        """Measuring strength or answering a web request: play the best move, always.

        Noise and opening randomness are forced off here rather than left to the
        caller, so that an arena run cannot inherit them by accident from the
        training config it was loaded from.
        """
        return cls(
            n_simulations=n_simulations if n_simulations is not None else mcts.n_simulations,
            c_puct=mcts.c_puct,
            fpu_reduction=mcts.fpu_reduction,
            dirichlet_alpha=mcts.dirichlet_alpha,
            dirichlet_eps=0.0,
            temp_moves=0,
            temp_init=0.0,
        )

    def with_simulations(self, n_simulations: int) -> SearchConfig:
        """A copy at a different simulation budget -- how difficulty levels differ."""
        return replace(self, n_simulations=n_simulations)

    # -----------------------------------------------------------------

    @property
    def uses_noise(self) -> bool:
        return self.dirichlet_eps > 0.0

    def assert_no_noise(self, where: str) -> None:
        """Fail loudly if exploration is on where it must not be (contract C7).

        Called by the arena and the web API at construction time. A check that
        runs once at startup is worth more than a comment, because the failure it
        prevents produces no symptom at all -- just quietly wrong numbers.
        """
        if self.dirichlet_eps != 0.0:
            msg = (
                f"{where} must run with dirichlet_eps=0, got {self.dirichlet_eps}. "
                "Exploration noise belongs to self-play only; leaving it on makes "
                "every strength measurement wrong without failing anything "
                "(contract C7). Build this config with SearchConfig.for_evaluation()."
            )
            raise ConfigError(msg)
        if self.temp_moves != 0:
            msg = (
                f"{where} must play its best move every time, but temp_moves="
                f"{self.temp_moves} makes the first {self.temp_moves} plies random. "
                "Build this config with SearchConfig.for_evaluation()."
            )
            raise ConfigError(msg)
