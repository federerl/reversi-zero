"""The opponents added for the research experiments: controls, variants, and the playout search.

Each is a fixed opponent the trained agent gets measured against, so the
tests pin what each one claims to do: a corner-first Greedy really takes
corners, a disc-counting search at depth 1 really is Greedy, and the
random-playout search is reproducible from a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.agents import (
    Agent,
    CornerAgent,
    GreedyAgent,
    MaterialAgent,
    MobilityAgent,
    NoisyGreedyAgent,
    RolloutAgent,
)
from reversi.agents.greedy import corners
from reversi.agents.rollout import RolloutEvaluator
from reversi.arena.entrants import build_agent, describe_entrant, parse_entrant
from reversi.game import rules
from reversi.game.bitboard import popcount

CONTROLS = [
    GreedyAgent(corner_first=True),
    MaterialAgent(2),
    NoisyGreedyAgent(),
    CornerAgent(),
    MobilityAgent(),
    RolloutAgent(12),
]


@pytest.mark.parametrize("agent", CONTROLS, ids=lambda a: a.name)
def test_every_control_plays_legal_moves_to_the_end(agent: Agent) -> None:
    assert isinstance(agent, Agent)
    rng = np.random.default_rng(1)
    for _ in range(3):
        state = rules.initial_state(8)
        while not rules.is_terminal(state):
            action = agent.select(state, rng)
            assert action in rules.legal_actions(state)
            state = rules.apply(state, action)


@pytest.mark.parametrize("agent", CONTROLS, ids=lambda a: a.name)
def test_every_control_refuses_a_finished_position(agent: Agent) -> None:
    full = rules.initial_state(4)
    rng = np.random.default_rng(0)
    while not rules.is_terminal(full):
        full = rules.apply(full, rules.legal_actions(full)[0])
    with pytest.raises(ValueError, match="finished position"):
        agent.select(full, rng)


def test_a_depth_one_disc_search_is_exactly_greedy() -> None:
    """A move flipping f discs changes the difference by 2f + 1, so the two
    agents must tie on exactly the same moves in every position."""
    greedy, material = GreedyAgent(), MaterialAgent(1)
    rng = np.random.default_rng(2)
    state = rules.initial_state(8)
    checked = 0
    while not rules.is_terminal(state):
        assert set(greedy.choices(state)) == set(material.choices(state))
        checked += 1
        state = rules.apply(state, greedy.select(state, rng))
    assert checked > 50


def test_corner_first_greedy_takes_a_corner_when_it_can() -> None:
    """Walk random games until a corner is legal; it must be chosen every time."""
    agent = GreedyAgent(corner_first=True)
    rng = np.random.default_rng(3)
    seen = 0
    for _ in range(20):
        state = rules.initial_state(8)
        while not rules.is_terminal(state):
            legal = rules.legal_actions(state)
            available = [a for a in legal if a in corners(8)]
            if available:
                assert set(agent.choices(state)) == set(available)
                seen += 1
            state = rules.apply(state, legal[int(rng.integers(0, len(legal)))])
    assert seen > 0, "no position with a legal corner was reached"


def test_corner_first_greedy_is_plain_greedy_otherwise() -> None:
    state = rules.initial_state(8)
    assert set(GreedyAgent(corner_first=True).choices(state)) == set(GreedyAgent().choices(state))


def test_mobility_leaves_the_opponent_the_fewest_replies() -> None:
    rng = np.random.default_rng(4)
    state = rules.initial_state(8)
    for _ in range(30):
        if rules.is_terminal(state):
            break
        action = MobilityAgent().select(state, rng)
        replies = {
            a: popcount(rules.legal_placements(rules.apply(state, a)))
            for a in rules.legal_actions(state)
        }
        assert replies[action] == min(replies.values())
        state = rules.apply(state, action)


def test_noisy_greedy_sometimes_plays_a_non_greedy_move() -> None:
    """With epsilon 1 it is Random; with epsilon 0 it is Greedy. In between, both."""
    state = rules.initial_state(8)
    # Reach a position where Greedy's tie set is smaller than the legal set.
    rng = np.random.default_rng(5)
    while set(GreedyAgent().choices(state)) == set(rules.legal_actions(state)):
        state = rules.apply(state, GreedyAgent().select(state, rng))
    greedy = set(GreedyAgent().choices(state))

    always = {NoisyGreedyAgent(epsilon=0.0).select(state, rng) for _ in range(50)}
    assert always <= greedy
    sometimes = {NoisyGreedyAgent(epsilon=0.5).select(state, rng) for _ in range(200)}
    assert sometimes - greedy, "epsilon 0.5 never left Greedy's tie set in 200 draws"


def test_a_playout_scores_the_result_for_the_leaf_mover() -> None:
    """Contract C2 for the knowledge-free evaluator: the value is for the player to move."""
    evaluator = RolloutEvaluator(np.random.default_rng(0))
    state = rules.initial_state(4)
    logits, values = evaluator.evaluate([state, state])
    assert logits.shape == (2, 17)
    assert not logits.any(), "no move is preferred before it is tried"
    assert set(values.tolist()) <= {-1.0, 0.0, 1.0}


def test_rollout_search_replays_exactly_from_its_seed() -> None:
    state = rules.initial_state(8)
    state = rules.apply(state, rules.legal_actions(state)[0])
    agent = RolloutAgent(40)
    first = [agent.select(state, np.random.default_rng(7)) for _ in range(3)]
    again = [agent.select(state, np.random.default_rng(7)) for _ in range(3)]
    assert first == again
    assert all(a in rules.legal_actions(state) for a in first)


def test_rollout_search_beats_random_play() -> None:
    """Forty playouts a move is a weak searcher, but a search all the same."""
    from reversi.agents import RandomAgent
    from reversi.arena import play_match

    result = play_match(RolloutAgent(40), RandomAgent(), games=10, board_size=6, seed=11)
    assert result.score >= 0.8, result.summary()


@pytest.mark.parametrize(
    ("text", "name", "kind"),
    [
        ("greedy-corner", "greedy-corner", "heuristic"),
        ("greedy-d2", "greedy-d2", "heuristic"),
        ("noisy-greedy", "noisy-greedy", "heuristic"),
        ("corner", "corner", "heuristic"),
        ("mobility", "mobility", "heuristic"),
        ("rollout-r500", "rollout-r500", "rollout"),
        ("rollout500", "rollout-r500", "rollout"),
    ],
)
def test_the_controls_enter_a_tournament_by_name(text: str, name: str, kind: str) -> None:
    spec = parse_entrant(text, default_simulations=50)
    assert (spec.name, spec.kind) == (name, kind)
    assert not spec.needs_network
    agent = build_agent(spec)
    assert agent.name == name
    described = describe_entrant(spec)
    assert described["kind"] == kind
    if kind == "rollout":
        assert described["playouts_per_move"] == 500
    else:
        assert described["heuristic"] == name
