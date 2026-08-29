"""The depth-4 baseline (test matrix T26, T30).

This agent is the yardstick every later strength claim is measured against, so
the tests care about two things above all: that it is *correct* (alpha-beta
returns what a full search would), and that it is *stable* (the same position
always gives the same move, and the weights cannot drift without noticing).

A yardstick that quietly changes between generation 5 and generation 40 makes the
curve between them meaningless. That is why the fingerprint test exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.agents import GreedyAgent, MinimaxAgent, RandomAgent
from reversi.agents.minimax import (
    CORNER,
    STAGE_WEIGHTS,
    X_SQUARE,
    evaluate,
    square_values,
    weights_fingerprint,
)
from reversi.arena import play_match
from reversi.game import reference as ref
from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.types import Action, Player

RNG = np.random.default_rng(0)


def board(text: str, to_move: Player = Player.BLACK) -> State:
    position = ref.from_ascii(text, to_move)
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=to_move, size=position.size)


def midgame(plies: int = 14, seed: int = 5, size: int = 8) -> State:
    import random

    rng = random.Random(seed)
    state = rules.initial_state(size)
    for _ in range(plies):
        if rules.is_terminal(state):
            break
        state = rules.apply(state, rng.choice(rules.legal_actions(state)))
    return state


# ===========================================================================
# The evaluation
# ===========================================================================


def test_corners_are_worth_far_more_than_anything_else() -> None:
    """A corner can never be flipped. Everything else is priced against that."""
    values = square_values(8)
    assert values[0] == CORNER
    assert values[7] == CORNER
    assert values[56] == CORNER
    assert values[63] == CORNER
    assert max(values) == CORNER


def test_the_squares_next_to_corners_are_penalised() -> None:
    """Playing an X-square is usually how you hand over the corner beside it.

    This is the piece of Othello knowledge the trained agent has to discover for
    itself -- nothing in the training signal mentions corners.
    """
    values = square_values(8)
    assert values[9] == X_SQUARE  # diagonal neighbour of the top-left corner
    assert values[9] < 0
    assert values[1] < 0  # C-square, along the edge
    assert values[9] < values[1], "the diagonal square is the worse trap"


def test_the_middle_is_priced_as_a_liability_early() -> None:
    """Counter-intuitive and deliberate: early discs in the centre cost mobility."""
    values = square_values(8)
    assert values[27] < 0  # a central square
    assert values[0] > values[27]


@pytest.mark.parametrize("size", [4, 6, 8])
def test_the_table_generates_for_every_board_size(size: int) -> None:
    values = square_values(size)
    assert len(values) == size * size
    assert values[0] == CORNER
    assert values[size - 1] == CORNER


def test_the_evaluation_is_from_the_movers_point_of_view() -> None:
    """Contract C2 again. The same board is worth the opposite to the other side.

    This is what lets the evaluation sit inside a negamax search with no sign
    flips anywhere -- and a violation would make the search prefer losing moves
    without failing anything.
    """
    state = midgame()
    flipped = State(
        black=state.black, white=state.white, to_move=state.to_move.opponent, size=state.size
    )
    assert evaluate(state) == pytest.approx(-evaluate(flipped))


def test_a_won_position_outranks_any_heuristic_opinion() -> None:
    won = board("BBBB\nBBBB\nBBBB\nBBBB", Player.BLACK)
    assert rules.is_terminal(won)
    assert scoring.result(won) == 1
    assert evaluate(won) > 1000


def test_the_weights_fingerprint_is_stable_and_covers_the_weights() -> None:
    """A baseline that drifts invalidates every rating measured against it.

    The fingerprint goes into every match report so a result can prove which
    yardstick produced it.
    """
    first = weights_fingerprint(8)
    assert first == weights_fingerprint(8)
    assert len(first) == 16

    original = STAGE_WEIGHTS["midgame"]
    try:
        STAGE_WEIGHTS["midgame"] = (9.0, 9.0, 9.0, 9.0)
        assert weights_fingerprint(8) != first, "the fingerprint must notice a weight change"
    finally:
        STAGE_WEIGHTS["midgame"] = original
    assert weights_fingerprint(8) == first


# ===========================================================================
# The search
# ===========================================================================


def plain_minimax(state: State, depth: int) -> float:
    """A deliberately naive full search, for comparison. No pruning, no ordering."""
    if depth <= 0 or rules.is_terminal(state):
        return evaluate(state)
    return max(
        -plain_minimax(rules.apply(state, action), depth - 1)
        for action in rules.legal_actions(state)
    )


def best_by_plain_minimax(state: State, depth: int) -> set[Action]:
    scored = {
        action: -plain_minimax(rules.apply(state, action), depth - 1)
        for action in rules.legal_actions(state)
    }
    best = max(scored.values())
    return {a for a, v in scored.items() if v == pytest.approx(best)}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_alpha_beta_finds_what_a_full_search_would(seed: int) -> None:
    """Pruning must only skip branches that cannot change the answer.

    This is the test that would catch a broken window -- an alpha-beta bug does
    not crash, it just quietly returns a worse move, and the agent would still
    look like a working baseline while being a weaker yardstick than it claims.
    """
    state = midgame(plies=16, seed=seed)
    depth = 3

    chosen = MinimaxAgent(depth).select(state, RNG)
    acceptable = best_by_plain_minimax(state, depth)

    assert chosen in acceptable, (
        f"alpha-beta chose {chosen}, but a full search rates {sorted(acceptable)} best"
    )


def test_the_same_position_always_gives_the_same_move() -> None:
    """Determinism is what makes a tournament reproducible.

    Any difference between two runs must be the *other* agent's doing.
    """
    state = midgame()
    agent = MinimaxAgent(3)
    first = agent.select(state, np.random.default_rng(0))
    for seed in (1, 2, 99):
        assert agent.select(state, np.random.default_rng(seed)) == first


def test_it_only_plays_legal_moves() -> None:
    agent = MinimaxAgent(2)
    state = rules.initial_state(8)
    while not rules.is_terminal(state):
        action = agent.select(state, RNG)
        assert action in rules.legal_actions(state)
        state = rules.apply(state, action)


def test_it_passes_when_it_must() -> None:
    state = board("BBW.\nBBWW\nWBWW\nBBBB", Player.WHITE)
    assert rules.must_pass(state)
    assert MinimaxAgent(3).select(state, RNG) == rules.legal_actions(state)[0]


def test_it_refuses_a_finished_position() -> None:
    with pytest.raises(ValueError, match="finished position"):
        MinimaxAgent(2).select(board("WWWW\nWWWW\nWWWW\nWWWW", Player.BLACK), RNG)


def test_a_nonsense_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        MinimaxAgent(0)


def count_plain_minimax_nodes(state: State, depth: int) -> int:
    """Nodes an unpruned search would visit. The honest thing to compare against.

    Not ``branching ** depth``: the branching factor grows as the board opens up,
    so that estimate is wrong in the direction that flatters the pruning.
    """
    if depth <= 0 or rules.is_terminal(state):
        return 1
    return 1 + sum(
        count_plain_minimax_nodes(rules.apply(state, action), depth - 1)
        for action in rules.legal_actions(state)
    )


def test_pruning_actually_prunes() -> None:
    """Alpha-beta returns the same answer as a full search -- the point of the
    ordering is that it gets there having looked at far less of the tree.

    Without this the agent would still be *correct*, just too slow to be usable
    as a baseline over hundreds of games.
    """
    state = midgame()
    depth = 4

    agent = MinimaxAgent(depth)
    agent.select(state, RNG)
    unpruned = sum(
        count_plain_minimax_nodes(rules.apply(state, action), depth - 1)
        for action in rules.legal_actions(state)
    )

    assert agent.nodes < unpruned / 3, (
        f"alpha-beta visited {agent.nodes} nodes; an unpruned search visits {unpruned}"
    )


# ===========================================================================
# Strength (T26): the acceptance criterion
# ===========================================================================


def test_looking_further_ahead_plays_better() -> None:
    """The most basic sanity check on a search: depth must buy something.

    If depth 1 matched depth 3, the search would be doing nothing and the
    "depth-4" in the baseline's name would be decoration.
    """
    result = play_match(
        MinimaxAgent(3, name="d3"),
        MinimaxAgent(1, name="d1"),
        games=10,
        board_size=8,
        seed=4,
        opening_plies=4,
    )
    assert result.score > 0.6, result.summary()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_baseline_beats_greedy_convincingly() -> None:
    """Criterion T26: >= 80% over 200 colour-balanced games.

    Measured at 98.3% over 60 games while it was being written; 200 games is the
    number the plan asks for, and it takes several minutes.
    """
    result = play_match(
        MinimaxAgent(4),
        GreedyAgent(),
        games=200,
        board_size=8,
        seed=20260827,
        opening_plies=4,
    )
    assert result.score >= 0.80, result.summary()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_baseline_crushes_random() -> None:
    result = play_match(
        MinimaxAgent(4), RandomAgent(), games=100, board_size=8, seed=7, opening_plies=4
    )
    assert result.score >= 0.95, result.summary()
