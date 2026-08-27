"""Playing one game against yourself, and writing down what was learned from it.

For every position the agent actually had a decision to make, we keep two things:

* **what the search concluded** -- its visit distribution over the legal moves.
  This becomes the policy target. It is a better answer than the network gave
  before searching, which is the entire mechanism by which the agent improves.
* **how the game turned out** -- but that is not known yet, so it gets filled in
  at the end, for every position at once, from the point of view of whoever was to
  move in each one.

Two details that are easy to get wrong and both matter:

**The move played is not the target stored.** For the first several plies the move
is *sampled* in proportion to the search's visit counts rather than taken as the
best one. That is what stops a deterministic network from playing the identical
game every time and handing the trainer thousands of copies of one position. But
the stored target is always the raw visit distribution -- never the sampled or
sharpened version (contract C4). Two separate things, computed separately, on
purpose.

**Positions with only one legal move are not recorded.** No search is run either.
There was no decision, so there is nothing to learn: a target saying "the only
legal move had probability 1" teaches the network a fact about the rules that the
rules already enforce, while using up a slot in the training batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reversi.data.schema import GameRecord, Sample
from reversi.errors import WorkerError
from reversi.game import rules
from reversi.search.config import SearchConfig
from reversi.search.evaluator import Evaluator
from reversi.search.mcts import MCTS, select_move
from reversi.seeding import game_seed
from reversi.seeding import rng as make_rng
from reversi.types import pass_action

__all__ = ["SelfPlaySummary", "play_game", "play_games"]


@dataclass(slots=True)
class SelfPlaySummary:
    """What a batch of self-play games looked like, for the metrics log."""

    games: int = 0
    positions: int = 0
    plies: list[int] = field(default_factory=list)
    passes: int = 0
    forced_moves: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    branching_total: int = 0
    branching_count: int = 0

    def observe(self, record: GameRecord, branching: list[int]) -> None:
        self.games += 1
        self.positions += len(record.samples)
        self.plies.append(record.plies)
        self.passes += record.passes
        self.forced_moves += record.skipped_forced_moves
        self.branching_total += sum(branching)
        self.branching_count += len(branching)
        if record.result_for_black > 0:
            self.black_wins += 1
        elif record.result_for_black < 0:
            self.white_wins += 1
        else:
            self.draws += 1

    def as_metrics(self) -> dict[str, Any]:
        plies = np.asarray(self.plies, dtype=np.float64) if self.plies else np.zeros(1)
        total_plies = max(1, int(plies.sum()))
        decisive = max(1, self.black_wins + self.white_wins)
        return {
            "games": self.games,
            "positions": self.positions,
            "plies_mean": float(plies.mean()),
            "plies_std": float(plies.std()),
            "pass_rate": self.passes / total_plies,
            "forced_move_rate": self.forced_moves / total_plies,
            "mean_branching": (
                self.branching_total / self.branching_count if self.branching_count else 0.0
            ),
            # Under fair play from a symmetric start this should sit near 0.5.
            # A number far from it points at something structural rather than at
            # the agent having found a strategy.
            "first_player_win_rate": self.black_wins / decisive,
            "draw_rate": self.draws / max(1, self.games),
        }


def play_game(
    evaluator: Evaluator,
    search_config: SearchConfig,
    *,
    board_size: int,
    rng: np.random.Generator,
    max_plies: int | None = None,
    game_index: int = -1,
) -> tuple[GameRecord, list[int]]:
    """Play one game against yourself. Returns the record and the branching seen.

    ``rng`` drives both the exploration noise inside the search and the sampling
    of which move to play, so one seed reproduces the whole game.
    """
    mcts = MCTS(evaluator, search_config)
    record = GameRecord(samples=[], board_size=board_size, game_index=game_index)
    branching: list[int] = []

    state = rules.initial_state(board_size)
    passing = pass_action(board_size)
    # A game cannot exceed one placement per initially-empty square plus a pass
    # between each; the bound exists so a rules bug becomes a loud failure here
    # rather than an infinite loop inside an overnight job.
    ceiling = max_plies if max_plies is not None else 4 * board_size * board_size
    ply = 0

    while not rules.is_terminal(state):
        if ply > ceiling:
            msg = (
                f"a game reached {ply} plies on a {board_size}x{board_size} board, "
                f"past the ceiling of {ceiling}. The rules engine is misbehaving."
            )
            raise WorkerError(msg)

        actions = rules.legal_actions(state)
        if len(actions) == 1:
            # Nothing to decide, so nothing to search and nothing to record.
            action = actions[0]
            record.skipped_forced_moves += 1
        else:
            branching.append(len(actions))
            result = mcts.run(state, rng=rng)
            record.samples.append(
                Sample(
                    black=state.black,
                    white=state.white,
                    to_move=state.to_move,
                    pi=result.policy_target(),
                    move_no=ply,
                )
            )
            temperature = search_config.temp_init if ply < search_config.temp_moves else 0.0
            action = select_move(result, temperature=temperature, rng=rng)

        if action == passing:
            record.passes += 1
        state = rules.apply(state, action)
        ply += 1

    record.plies = ply
    record.finish(state)
    return record, branching


def play_games(
    evaluator: Evaluator,
    search_config: SearchConfig,
    *,
    board_size: int,
    n_games: int,
    root_seed: int,
    generation: int,
    worker_id: int = 0,
    max_plies: int | None = None,
) -> Iterator[tuple[GameRecord, list[int]]]:
    """Play ``n_games``, each with its own derived seed.

    Seeds come from ``(root_seed, generation, worker_id, game_index)`` rather than
    from a running counter, so game 57 of generation 3 is the same game whether it
    was played in one long run or after a resume -- and two workers can never
    accidentally play identical games.
    """
    for index in range(n_games):
        seed = game_seed(root_seed, generation, worker_id, index)
        yield play_game(
            evaluator,
            search_config,
            board_size=board_size,
            rng=make_rng(seed),
            max_plies=max_plies,
            game_index=index,
        )
