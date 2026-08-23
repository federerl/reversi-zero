"""The opponents, and playing a fair match (test matrix T29, T30).

These are the yardstick the whole project is measured against, so a bug here
would not make anything crash -- it would make every strength number wrong. A
Greedy agent that quietly preferred the top-left corner, or a match that gave one
side black more often, would both flatter or punish the trained agent by an
amount nobody could see.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from reversi.agents import Agent, GreedyAgent, RandomAgent
from reversi.agents.az_agent import AZAgent
from reversi.arena import play_match
from reversi.config import MCTSConfig
from reversi.errors import ArenaError, ConfigError
from reversi.game import reference as ref
from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.types import Action, Player, pass_action


def board(text: str, to_move: Player = Player.BLACK) -> State:
    position = ref.from_ascii(text, to_move)
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=to_move, size=position.size)


# ---------------------------------------------------------------------------
# Legality (T30)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent", [RandomAgent(), GreedyAgent()])
@pytest.mark.parametrize("size", [4, 8])
def test_agents_only_ever_play_legal_moves(agent: Agent, size: int) -> None:
    """20 complete games each. ``apply`` would raise on an illegal move anyway,
    which is the point: the engine is the backstop, not the agent's good manners."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = rules.initial_state(size)
        while not rules.is_terminal(state):
            action = agent.select(state, rng)
            assert action in rules.legal_actions(state)
            state = rules.apply(state, action)


@pytest.mark.parametrize("agent", [RandomAgent(), GreedyAgent()])
def test_agents_pass_when_that_is_all_they_can_do(agent: Agent) -> None:
    state = board(
        """
        BBW.
        BBWW
        WBWW
        BBBB
        """,
        Player.WHITE,
    )
    assert rules.must_pass(state)
    assert agent.select(state, np.random.default_rng(0)) == pass_action(4)


@pytest.mark.parametrize("agent", [RandomAgent(), GreedyAgent()])
def test_agents_refuse_a_finished_position(agent: Agent) -> None:
    finished = board("WWWW\nWWWW\nWWWW\nWWWW", Player.BLACK)
    with pytest.raises(ValueError, match="finished position"):
        agent.select(finished, np.random.default_rng(0))


def test_both_agents_satisfy_the_protocol() -> None:
    assert isinstance(RandomAgent(), Agent)
    assert isinstance(GreedyAgent(), Agent)


# ---------------------------------------------------------------------------
# What each one actually does
# ---------------------------------------------------------------------------


def test_random_uses_every_legal_move_eventually() -> None:
    """A "random" agent that always picked index 0 would pass a legality test."""
    state = rules.initial_state(8)
    rng = np.random.default_rng(0)
    chosen = {RandomAgent().select(state, rng) for _ in range(200)}
    assert chosen == set(rules.legal_actions(state))


def test_greedy_takes_the_move_that_flips_the_most() -> None:
    rng = np.random.default_rng(0)
    state = rules.initial_state(8)
    for _ in range(20):
        if rules.is_terminal(state):
            break
        action = GreedyAgent().select(state, rng)
        gains = {a: popcount(rules.flips(state, a)) for a in rules.legal_actions(state)}
        assert gains[action] == max(gains.values())
        state = rules.apply(state, action)


def test_greedy_breaks_ties_randomly_not_by_board_order() -> None:
    """Always taking the lowest-numbered tied square is a positional bias.

    It would make Greedy systematically prefer the top-left of the board, which
    an opponent can learn to exploit -- so the win rate would be measuring that
    quirk rather than the opponent's strength.
    """
    state = rules.initial_state(8)
    gains = {a: popcount(rules.flips(state, a)) for a in rules.legal_actions(state)}
    assert len(set(gains.values())) == 1, "the opening has four equally greedy moves"

    rng = np.random.default_rng(0)
    chosen = {GreedyAgent().select(state, rng) for _ in range(200)}
    assert len(chosen) > 1


def test_greedy_discriminates_when_the_moves_actually_differ() -> None:
    """Sanity that the two opponents are different strategies.

    Many positions offer only equally-greedy moves -- early on, nearly every move
    flips exactly one disc -- so the position is searched for rather than assumed.
    """
    rng = np.random.default_rng(3)
    state = rules.initial_state(8)

    for _ in range(60):
        if rules.is_terminal(state):
            break
        gains = {a: popcount(rules.flips(state, a)) for a in rules.legal_actions(state)}
        if len(set(gains.values())) > 1:
            choice = GreedyAgent().select(state, rng)
            assert gains[choice] == max(gains.values())
            assert min(gains.values()) < max(gains.values())
            return
        state = rules.apply(state, rules.legal_actions(state)[0])

    pytest.fail("no position with differing flip counts was reached")


# ---------------------------------------------------------------------------
# The trained agent (contract C7)
# ---------------------------------------------------------------------------


def test_the_trained_agent_refuses_a_self_play_search_config() -> None:
    """Exploration noise while measuring strength understates every number.

    Nothing would fail and no test would go red -- the results would just be
    quietly wrong -- so this is checked when the agent is built.
    """
    selfplay = SearchConfig.for_selfplay(MCTSConfig(dirichlet_eps=0.25, temp_moves=8))

    with pytest.raises(ConfigError, match="dirichlet_eps=0"):
        AZAgent(StubEvaluator(), selfplay)


def test_the_trained_agent_plays_legal_moves_and_is_deterministic() -> None:
    agent = AZAgent(StubEvaluator(), SearchConfig(n_simulations=8))
    state = rules.initial_state(8)

    first = agent.select(state, np.random.default_rng(0))
    assert first in rules.legal_actions(state)
    # A different rng must change nothing: with no noise and no temperature the
    # agent is deterministic, which is what makes a tournament reproducible.
    assert agent.select(state, np.random.default_rng(12345)) == first


# ---------------------------------------------------------------------------
# Fair matches (T29)
# ---------------------------------------------------------------------------


class ColourRecorder:
    """Plays randomly, and counts how many games it started as black.

    Being asked to move in the *opening* position is exactly what it means to be
    black: black moves first, so white is never consulted there.
    """

    def __init__(self) -> None:
        self.games_as_black = 0

    @property
    def name(self) -> str:
        return "recorder"

    def select(self, state: State, rng: np.random.Generator) -> Action:
        if state == rules.initial_state(state.size):
            self.games_as_black += 1
        actions = rules.legal_actions(state)
        return actions[int(rng.integers(0, len(actions)))]


def test_a_match_gives_each_agent_black_exactly_half_the_time() -> None:
    """Black moves first, and that is worth something.

    An unequal split measures the colour as much as it measures the agent -- and
    it would do so invisibly, since the totals still add up.
    """
    recorder = ColourRecorder()
    result = play_match(recorder, RandomAgent(), games=20, board_size=8, seed=1)

    assert recorder.games_as_black == 10, "exactly half the games, as black"
    assert result.games == 20
    assert result.wins + result.losses + result.draws == 20
    assert result.wins_as_black + result.wins_as_white == result.wins
    assert result.draws_as_black + result.draws_as_white == result.draws


def test_the_colour_split_is_reported_separately() -> None:
    """An agent winning 90% as black and 40% as white has learned an opening,
    not Reversi. The split is what makes that visible."""
    result = play_match(GreedyAgent(), RandomAgent(), games=40, board_size=8, seed=3)

    assert 0.0 <= result.score_as_black <= 1.0
    assert 0.0 <= result.score_as_white <= 1.0
    assert result.score == pytest.approx(
        (result.score_as_black + result.score_as_white) / 2, abs=1e-9
    )


def test_a_match_refuses_an_odd_number_of_games() -> None:
    """Rounding silently would give one agent an extra game as black."""
    with pytest.raises(ArenaError, match="even number of games"):
        play_match(RandomAgent("a"), RandomAgent("b"), games=7, board_size=8, seed=1)


def test_a_match_replays_exactly_from_its_seed() -> None:
    first = play_match(GreedyAgent(), RandomAgent(), games=20, board_size=8, seed=5)
    again = play_match(GreedyAgent(), RandomAgent(), games=20, board_size=8, seed=5)
    different = play_match(GreedyAgent(), RandomAgent(), games=20, board_size=8, seed=6)

    assert (first.wins, first.losses, first.draws) == (again.wins, again.losses, again.draws)
    assert first.mean_plies == again.mean_plies
    assert (first.wins, first.losses, first.draws) != (
        different.wins,
        different.losses,
        different.draws,
    )


def test_the_score_counts_a_draw_as_half() -> None:
    result = play_match(RandomAgent("a"), RandomAgent("b"), games=200, board_size=8, seed=2)

    expected = (result.wins + 0.5 * result.draws) / result.games
    assert result.score == pytest.approx(expected)
    assert 0.0 <= result.score <= 1.0
    # Two identical strategies should end up near even.
    assert 0.35 < result.score < 0.65


def test_greedy_beats_random_on_the_full_board() -> None:
    """A weak strategy, but a real one -- and this is the ordering the 8x8 gate
    assumes. See the 4x4 note in the day-6 integration test: on the small board
    the ordering reverses, which is a property of the fixture, not a bug."""
    result = play_match(GreedyAgent(), RandomAgent(), games=200, board_size=8, seed=1)
    assert result.score > 0.55, result.summary()


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_the_arena_cannot_import_the_training_data_pipeline() -> None:
    """Evaluation games must never be able to reach the replay buffer.

    If a single arena game leaked into training data, the agent would be training
    on its own test set and every number afterwards would be inflated by an
    unknowable amount. Making the import impossible is stronger than remembering
    not to do it.
    """
    code = (
        "import reversi.arena, sys; "
        "print(any(m.startswith(('reversi.train', 'reversi.data', 'reversi.selfplay')) "
        "for m in sys.modules))"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert finished.stdout.strip() == "False", (
        "reversi.arena must not import train/, data/ or selfplay/"
    )
