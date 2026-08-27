"""Playing many games at once so their network calls arrive together.

**The problem this solves.** Day 5's self-play plays one game at a time and asks
the network about one position at a time. Measured on the target hardware, that
runs at 499 positions per second on the GPU -- barely faster than the 329 the CPU
manages, because the GPU spends nearly all of its time waiting to be handed the
next single position rather than computing anything.

Hand it 48 positions at once and it does 19,296 per second. Same GPU, same
network, **39 times the throughput**, purely from not being starved. That is the
difference between an 8x8 generation taking twenty hours and taking half an hour.

**How.** Hold B games in flight. Advance every one of them to the point where it
needs the network, collect all those positions, ask once, then hand each answer
back to the game it came from. Repeat. Nothing about any individual game changes;
only the order in which the work is interleaved.

**Why no virtual loss.** Parallel MCTS usually needs it: if several simulations
descend the same tree at once they all take the same path, because none of them
has recorded a visit yet. Here each tree contributes exactly *one* leaf per round,
so two simulations in the same tree are never in flight together and the problem
does not arise. That is a deliberate simplification -- virtual loss is a real
source of subtle strength bugs, and batching across games gets the same GPU
efficiency without it.

**Finished positions never take a slot in the batch.** A leaf that ends the game
has an exact value; sending it to the network would be both slower and less
accurate.

The search logic itself is not reimplemented here. ``MCTS`` exposes the pieces of
one simulation -- descend, resolve a terminal, open a node, back up -- and this
module is a different *scheduler* over the same pieces. Two copies would drift,
and the drift would be invisible: both would still produce legal moves.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from reversi.data.schema import GameRecord, Sample
from reversi.errors import WorkerError
from reversi.game import rules
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import Evaluator
from reversi.search.mcts import MCTS, SearchResult, select_move
from reversi.search.node import Node
from reversi.seeding import game_seed
from reversi.seeding import rng as make_rng
from reversi.types import Action, pass_action

__all__ = ["BatchedSelfPlay"]

log = logging.getLogger(__name__)


class _Slot:
    """One game in flight, plus the search currently running on it.

    A small state machine. ``prepare`` runs it forward until it either needs the
    network -- returning the position to ask about -- or the game ends. Everything
    that does not need the network (forced moves, finished positions, choosing a
    move once the simulations are done) happens inside ``prepare`` without
    consuming a slot in the batch.
    """

    __slots__ = (
        "branching",
        "finished",
        "game_index",
        "mcts",
        "pending",
        "ply",
        "record",
        "rng",
        "root",
        "sims_done",
        "simulations",
        "state",
    )

    def __init__(self, mcts: MCTS, board_size: int) -> None:
        self.mcts = mcts
        self.simulations = mcts.config.n_simulations
        self.finished = True
        self.game_index = -1
        self.record = GameRecord(samples=[], board_size=board_size)
        self.state = rules.initial_state(board_size)
        self.rng = make_rng(0)
        self.root: Node | None = None
        self.sims_done = 0
        self.pending: tuple[Node, list[tuple[Node, int]]] | None = None
        self.ply = 0
        self.branching: list[int] = []

    # -- lifecycle ----------------------------------------------------

    def start(self, game_index: int, seed: int, board_size: int) -> None:
        self.game_index = game_index
        self.state = rules.initial_state(board_size)
        self.record = GameRecord(samples=[], board_size=board_size, game_index=game_index)
        self.rng = make_rng(seed)
        self.root = None
        self.sims_done = 0
        self.pending = None
        self.ply = 0
        self.branching = []
        self.finished = False

    def prepare(self, ceiling: int) -> State | None:
        """Run until the network is needed. Returns the position to ask about."""
        while not self.finished:
            if self.root is None:
                self._skip_to_a_real_decision(ceiling)
                if self.finished:
                    return None
                self.root = Node(self.state)
                return self.root.state  # needs expanding

            if self.sims_done >= self.simulations:
                self._play_the_move(ceiling)
                continue

            node, path = self.mcts.descend(self.root)
            exact = self.mcts.terminal_value(node)
            if exact is not None:
                # A finished position: exact answer, no network call, no slot in
                # the batch. Costs one simulation and nothing else.
                self.mcts.backup(path, exact)
                self.sims_done += 1
                continue

            self.pending = (node, path)
            return node.state
        return None

    def receive(self, logits: NDArray[np.float32], value: float, rng_noise: bool) -> None:
        """Hand back one row of the batch to whichever node asked for it."""
        if self.root is not None and not self.root.expanded:
            self.mcts.open_node(self.root, logits, value)
            if rng_noise:
                self.mcts._add_root_noise(self.root, self.rng)
            self.sims_done = 0
            return

        if self.pending is None:  # pragma: no cover - would be a scheduling bug
            msg = "a slot received an evaluation it had not asked for"
            raise WorkerError(msg)

        node, path = self.pending
        self.pending = None
        self.mcts.open_node(node, logits, value)
        self.mcts.backup(path, value)
        self.sims_done += 1

    # -- the parts that need no network -------------------------------

    def _skip_to_a_real_decision(self, ceiling: int) -> None:
        """Apply forced moves and finish the game, until a genuine choice appears."""
        while True:
            if rules.is_terminal(self.state):
                self._finish()
                return
            actions = rules.legal_actions(self.state)
            if len(actions) > 1:
                return
            # Only one legal move: nothing to decide, so nothing to search and
            # nothing to record.
            self.record.skipped_forced_moves += 1
            self._apply(actions[0], ceiling)

    def _play_the_move(self, ceiling: int) -> None:
        root = self.root
        if root is None:  # pragma: no cover - guarded by the caller
            return

        result = SearchResult(
            actions=root.actions,
            visits=tuple(root.visits),
            q_values=tuple(root.q(i) for i in range(root.n_actions)),
            root_value=root.net_value,
            board_size=self.state.size,
        )
        self.branching.append(root.n_actions)
        self.record.samples.append(
            Sample(
                black=self.state.black,
                white=self.state.white,
                to_move=self.state.to_move,
                pi=result.policy_target(),
                move_no=self.ply,
            )
        )

        config = self.mcts.config
        temperature = config.temp_init if self.ply < config.temp_moves else 0.0
        action = select_move(result, temperature=temperature, rng=self.rng)

        self.root = None
        self.sims_done = 0
        self._apply(action, ceiling)

    def _apply(self, action: Action, ceiling: int) -> None:
        if self.ply > ceiling:
            msg = (
                f"a game reached {self.ply} plies on a {self.state.size}x"
                f"{self.state.size} board, past the ceiling of {ceiling}. "
                "The rules engine is misbehaving."
            )
            raise WorkerError(msg)
        if action == pass_action(self.state.size):
            self.record.passes += 1
        self.state = rules.apply(self.state, action)
        self.ply += 1

    def _finish(self) -> None:
        self.record.plies = self.ply
        self.record.finish(self.state)
        self.finished = True


class BatchedSelfPlay:
    """Plays games ``games_in_flight`` at a time, batching their network calls.

    Produces exactly the same games as playing them one at a time -- same seeds,
    same searches, same moves. Only the scheduling differs, which is what makes
    "batched and unbatched agree" a testable property rather than a hope.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        search_config: SearchConfig,
        *,
        board_size: int,
        games_in_flight: int,
        root_seed: int,
        generation: int,
        worker_id: int = 0,
        max_plies: int | None = None,
    ) -> None:
        if games_in_flight < 1:
            msg = f"games_in_flight must be at least 1, got {games_in_flight}"
            raise WorkerError(msg)

        self.evaluator = evaluator
        self.search_config = search_config
        self.board_size = board_size
        self.games_in_flight = games_in_flight
        self.root_seed = root_seed
        self.generation = generation
        self.worker_id = worker_id
        self.ceiling = max_plies if max_plies is not None else 4 * board_size * board_size

        # Counters worth reporting: the average batch size is the number that
        # says whether the batching is actually working.
        self.rounds = 0
        self.positions_evaluated = 0

    @property
    def mean_batch(self) -> float:
        return self.positions_evaluated / self.rounds if self.rounds else 0.0

    def play(self, n_games: int) -> Iterator[tuple[GameRecord, list[int]]]:
        """Play ``n_games``, yielding each as it finishes."""
        if n_games < 1:
            return

        width = min(self.games_in_flight, n_games)
        slots = [
            _Slot(MCTS(self.evaluator, self.search_config), self.board_size) for _ in range(width)
        ]
        wants_noise = self.search_config.uses_noise

        started = 0
        for slot in slots:
            slot.start(started, self._seed(started), self.board_size)
            started += 1

        completed = 0
        while completed < n_games:
            asking: list[_Slot] = []
            states: list[State] = []

            for slot in slots:
                if slot.finished:
                    continue
                position = slot.prepare(self.ceiling)
                if position is not None:
                    asking.append(slot)
                    states.append(position)

            if states:
                logits, values = self.evaluator.evaluate(states)
                self.rounds += 1
                self.positions_evaluated += len(states)
                for index, slot in enumerate(asking):
                    slot.receive(logits[index], float(values[index]), wants_noise)

            # Harvest anything that finished, and refill its slot immediately so
            # the next batch is as full as it can be.
            for slot in slots:
                if not slot.finished:
                    continue
                if slot.game_index >= 0:
                    yield slot.record, slot.branching
                    completed += 1
                    slot.game_index = -1
                    if completed >= n_games:
                        break
                if started < n_games:
                    slot.start(started, self._seed(started), self.board_size)
                    started += 1

            if not states and all(slot.finished for slot in slots) and started >= n_games:
                break

    def _seed(self, game_index: int) -> int:
        return game_seed(self.root_seed, self.generation, self.worker_id, game_index)
