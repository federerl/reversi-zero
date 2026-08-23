"""Playing two agents against each other, fairly.

"Fairly" is doing real work in that sentence. Black moves first in Reversi, and
first-move advantage is not zero, so a match where one agent plays black more
often is measuring the colour as much as the agent. Every match here therefore
plays an **exactly equal number of games as each colour**, and refuses to run an
odd number of games rather than silently rounding.

Results are reported split by colour too. That split is a diagnostic worth
looking at: an agent that wins 90% as black and 40% as white has not learned
Reversi, it has learned an opening.

**What is deliberately missing until day 9.** Strong agents are near
deterministic, so without varied starting positions every game between the same
two agents would be identical -- a 200-game match would be one game counted 200
times. Day 9 adds the seeded random opening book (each opening played twice with
colours swapped), and day 10 adds Wilson intervals and Bradley-Terry ratings, so
a win rate becomes a claim with an error bar on it. Until then these numbers are
honest but coarse, and the agents involved are random enough that repetition is
not yet the problem it will be.
"""

from __future__ import annotations

from dataclasses import dataclass

from reversi.agents.base import Agent
from reversi.errors import ArenaError
from reversi.game import rules, scoring
from reversi.seeding import derive_seed
from reversi.seeding import rng as make_rng
from reversi.types import Player

__all__ = ["MatchResult", "play_match"]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of a match, from the first agent's point of view."""

    agent_a: str
    agent_b: str
    games: int
    wins: int
    losses: int
    draws: int
    wins_as_black: int
    wins_as_white: int
    draws_as_black: int
    draws_as_white: int
    mean_plies: float
    seed: int

    @property
    def score(self) -> float:
        """Points per game for the first agent: a win is 1, a draw is half.

        This -- not the win count -- is what gets compared against a threshold,
        because a draw is genuinely half a result and dropping draws entirely
        would flatter whichever agent draws more.
        """
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def score_as_black(self) -> float:
        half = self.games // 2
        return (self.wins_as_black + 0.5 * self.draws_as_black) / half

    @property
    def score_as_white(self) -> float:
        half = self.games // 2
        return (self.wins_as_white + 0.5 * self.draws_as_white) / half

    def summary(self) -> str:
        return (
            f"{self.agent_a} vs {self.agent_b}: "
            f"{self.wins}W {self.losses}L {self.draws}D over {self.games} games "
            f"= {self.score:.1%} "
            f"(as black {self.score_as_black:.1%}, as white {self.score_as_white:.1%})"
        )


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    *,
    games: int,
    board_size: int,
    seed: int,
    max_plies: int | None = None,
) -> MatchResult:
    """Play ``games`` games, half with each agent as black.

    Every game gets its own derived seed, so the whole match replays exactly from
    ``seed`` -- and so two agents that are both deterministic still produce a
    reproducible result rather than depending on call order.
    """
    if games < 2 or games % 2 != 0:
        msg = (
            f"a match needs an even number of games so each agent plays black "
            f"exactly half the time; got {games}"
        )
        raise ArenaError(msg)

    ceiling = max_plies if max_plies is not None else 4 * board_size * board_size

    wins = losses = draws = 0
    wins_as_black = wins_as_white = 0
    draws_as_black = draws_as_white = 0
    total_plies = 0

    for index in range(games):
        a_is_black = index % 2 == 0
        black_agent, white_agent = (agent_a, agent_b) if a_is_black else (agent_b, agent_a)
        rng = make_rng(derive_seed(seed, "match", agent_a.name, agent_b.name, index))

        state = rules.initial_state(board_size)
        plies = 0
        while not rules.is_terminal(state):
            if plies > ceiling:
                msg = f"a game ran past {ceiling} plies; the rules engine is misbehaving"
                raise ArenaError(msg)
            mover = black_agent if state.to_move is Player.BLACK else white_agent
            action = mover.select(state, rng)
            # apply() re-checks legality, so an agent returning a move it is not
            # allowed to play fails here with the position attached rather than
            # corrupting the game (contract C5).
            state = rules.apply(state, action)
            plies += 1

        total_plies += plies
        outcome = scoring.result_for(state, Player.BLACK if a_is_black else Player.WHITE)

        if outcome > 0:
            wins += 1
            if a_is_black:
                wins_as_black += 1
            else:
                wins_as_white += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
            if a_is_black:
                draws_as_black += 1
            else:
                draws_as_white += 1

    return MatchResult(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        wins_as_black=wins_as_black,
        wins_as_white=wins_as_white,
        draws_as_black=draws_as_black,
        draws_as_white=draws_as_white,
        mean_plies=total_plies / games,
        seed=seed,
    )
