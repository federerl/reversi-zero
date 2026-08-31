"""The one place the model lives, and the only place searches run.

**One model, loaded once, shared by every request.** It is read-only during
inference, so there is nothing to guard -- and loading a 1.9 MB network per
request would dominate the response time.

**Searches run in a small thread pool behind a semaphore.** A tree search is
seconds of CPU, and an async endpoint that ran one inline would block the event
loop and stall every other request, including the health check. Two threads,
because the machine serving this is also the machine that might be training on
it.

**Latency is bounded by construction, not by hope.** Each difficulty has a fixed
simulation budget, so a response takes as long as that budget takes and no
longer. When the pool is full the server says 429 immediately rather than
queueing indefinitely -- a request that will not be served for thirty seconds is
better refused than accepted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from reversi.difficulty import DifficultyLevel, choose_move, level_by_name
from reversi.game import rules, scoring
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.nn.evaluator import TorchEvaluator
from reversi.nn.export import ExportedModel, load_export
from reversi.search.config import SearchConfig
from reversi.search.mcts import MCTS, SearchResult
from reversi.types import Action, Player, pass_action

__all__ = ["EngineService", "SearchOutcome"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    action: Action
    result: SearchResult
    level: DifficultyLevel
    elapsed_ms: float


class EngineService:
    """Loads the model once and answers move requests."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        workers: int = 2,
        seed: int = 0,
    ) -> None:
        self.exported: ExportedModel = load_export(model_path, device=device)
        self.device = device
        self.evaluator = TorchEvaluator(self.exported.model, device=device)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="search")
        self._slots = asyncio.Semaphore(workers)
        self._rng = np.random.default_rng(seed)

        # The API serves; it never trains. Gradient bookkeeping on this path is
        # pure cost, and a model left in training mode would answer differently
        # depending on what it happened to be batched with.
        torch.set_grad_enabled(False)
        self.exported.model.eval()

    @property
    def board_size(self) -> int:
        return self.exported.board_size

    @property
    def model_id(self) -> str:
        return self.exported.label

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -----------------------------------------------------------------

    def search_config(self, level: DifficultyLevel) -> SearchConfig:
        """Build the search settings for a level.

        ``assert_no_noise`` is the point of doing it here rather than inline:
        exploration noise belongs to self-play, and an agent serving a human with
        it switched on would be quietly playing worse than it can (contract C7).
        """
        config = SearchConfig(
            n_simulations=level.simulations,
            dirichlet_eps=0.0,
            temp_moves=0,
        )
        config.assert_no_noise(f"the web API at difficulty {level.name!r}")
        return config

    async def choose(self, state: State, difficulty: str) -> SearchOutcome:
        """Search and return a move, without blocking the event loop.

        Refuses immediately when both search slots are busy. A caller told "try
        again shortly" can do something sensible; a caller left waiting thirty
        seconds cannot.
        """
        level = level_by_name(difficulty)

        if self._slots.locked():
            msg = "all search slots are busy"
            raise BusyError(msg)

        async with self._slots:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._pool, self._search, state, level)

    def _search(self, state: State, level: DifficultyLevel) -> SearchOutcome:
        started = time.perf_counter()
        mcts = MCTS(self.evaluator, self.search_config(level))
        result = mcts.run(state)
        action = choose_move(result, level, self._rng)

        # Contract C5, third layer. The search cannot reach an illegal move by
        # construction, so this firing means something upstream is broken -- and
        # it is far better to fail here than to send a client a move that its own
        # copy of the rules will reject.
        legal = rules.legal_actions(state)
        if action not in legal:
            msg = (
                f"the engine chose action {action}, which is not legal here. "
                f"Legal: {legal}\n{state}"
            )
            raise RuntimeError(msg)

        return SearchOutcome(
            action=action,
            result=result,
            level=level,
            elapsed_ms=1000 * (time.perf_counter() - started),
        )


class BusyError(RuntimeError):
    """Both search slots are occupied. Maps to 429."""


# ---------------------------------------------------------------------------
# Reading a position -- no model involved, so these stay synchronous
# ---------------------------------------------------------------------------


def describe(state: State) -> dict[str, object]:
    """Everything a client needs to render a position, derived from the rules."""
    legal = rules.legal_actions(state)
    terminal = rules.is_terminal(state)
    black, white = scoring.disc_counts(state)

    result: str | None = None
    if terminal:
        if black > white:
            result = "black"
        elif white > black:
            result = "white"
        else:
            result = "draw"

    return {
        "legal": legal,
        "is_terminal": terminal,
        "must_pass": legal == [pass_action(state.size)],
        "score": {"black": black, "white": white},
        "result": result,
    }


def evaluation_for(state: State, result: SearchResult) -> dict[str, object]:
    """The agent's view of the position, with whose view it is stated outright."""
    value = result.value()
    return {
        "value": value,
        "win_probability": (value + 1.0) / 2.0,
        "perspective": state.to_move.label,
    }


def top_moves(result: SearchResult, limit: int = 3) -> list[dict[str, float | int]]:
    order = sorted(range(len(result.actions)), key=lambda i: -result.visits[i])[:limit]
    total = max(1, result.total_visits)
    return [
        {
            "action": int(result.actions[i]),
            "visits": int(result.visits[i]),
            "share": result.visits[i] / total,
            "value": float(result.q_values[i]),
        }
        for i in order
    ]


def starting_state(board_size: int) -> State:
    return rules.initial_state(board_size)


def disc_total(state: State) -> int:
    return popcount(state.occupied)


def player_label(player: Player) -> str:
    return player.label
