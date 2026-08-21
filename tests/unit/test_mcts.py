"""The tree search (test matrix T15-T18; contracts C2, C4, C5, C7).

These are the most important tests in the project so far, because the bugs they
catch are the ones that produce no symptom. A search with the value sign
inverted does not crash, does not slow down, and does not make the training loss
look unusual -- it just prefers moves that lose, and the agent dutifully learns
to be bad at Reversi. There is no later test that would notice.

Every test here runs against ``StubEvaluator``, a stand-in whose answers we
choose. That is the point: when a search test fails, the network cannot be at
fault, so the search is.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from reversi.errors import ConfigError, SearchError
from reversi.game import reference as ref
from reversi.game import rules
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.search.mcts import MCTS, select_move
from reversi.types import Player, pass_action, policy_size

# ---------------------------------------------------------------------------
# Positions. Every one of these was found by searching real games for the
# property it demonstrates, not drawn by hand and hoped over.
# ---------------------------------------------------------------------------


def board(text: str, to_move: Player = Player.BLACK) -> State:
    position = ref.from_ascii(text, to_move)
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=to_move, size=position.size)


# Black to play. Playing 13 ends the game immediately, and Black wins it.
# Playing 14 does not end the game. Black is ahead 10-4.
WIN_IN_ONE = board(
    """
    WBBB
    BWBB
    BBWB
    B..W
    """,
    Player.BLACK,
)
WINNING_MOVE = 13

# White to play, and White has exactly one legal move -- which ends the game
# with White losing 4-12. There is no escape, so the root value must be -1.
EVERY_MOVE_LOSES = board(
    """
    WWWB
    BBBB
    BBB.
    BBBB
    """,
    Player.WHITE,
)

# White has no legal square, so White's only action is PASS. After the pass,
# Black plays the last square and wins. The root value must therefore be
# negative -- which is only true if the sign flips correctly across a pass.
MUST_PASS_THEN_LOSE = board(
    """
    BBW.
    BBWW
    WBWW
    BBBB
    """,
    Player.WHITE,
)


def midgame(plies: int = 12, seed: int = 3) -> State:
    """A position a dozen random moves into a game, with plenty of legal moves."""
    import random

    rng = random.Random(seed)
    state = rules.initial_state(8)
    for _ in range(plies):
        state = rules.apply(state, rng.choice(rules.legal_actions(state)))
    return state


def search(
    state: State,
    *,
    simulations: int = 64,
    evaluator: StubEvaluator | None = None,
    **config: float,
) -> tuple:
    stub = evaluator if evaluator is not None else StubEvaluator()
    mcts = MCTS(stub, SearchConfig(n_simulations=simulations, **config))  # type: ignore[arg-type]
    return mcts.run(state), stub


# ===========================================================================
# Contract C2 -- the sign of the value coming back up the tree
# ===========================================================================


def test_a_move_that_wins_the_game_is_valued_at_plus_one() -> None:
    """A win for the player to move is +1 *for that player*."""
    result, _ = search(WIN_IN_ONE)

    assert result.best_action() == WINNING_MOVE
    assert max(result.q_values) >= 0.95
    winning_index = result.actions.index(WINNING_MOVE)
    assert result.q_values[winning_index] == pytest.approx(1.0)


def test_a_position_with_no_escape_is_valued_at_minus_one() -> None:
    """The mirror image. If the sign were flipped, this would read +1."""
    result, _ = search(EVERY_MOVE_LOSES)

    assert result.actions == (11,), "this position has exactly one legal move"
    assert result.q_values[0] == pytest.approx(-1.0)
    assert result.value() <= -0.95


def test_the_sign_flips_correctly_across_a_pass() -> None:
    """Passing changes whose turn it is, so it flips the sign like any other move.

    This is the case a "pass is not a real move" shortcut gets wrong: skip the
    negation on a pass and the value arrives at the root belonging to the wrong
    player, so the search happily passes into losing positions.
    """
    result, _ = search(MUST_PASS_THEN_LOSE)

    assert result.actions == (pass_action(4),), "White's only legal action is PASS"
    assert result.q_values[0] < 0.0
    assert result.value() <= -0.95


def test_a_network_with_no_opinion_produces_exactly_zero_values() -> None:
    """With every leaf valued 0, every Q must be exactly 0 -- no drift, no bias.

    A sign error would not show here, but an accumulation bug would: any
    asymmetry in how values are added up shows as a non-zero average of zeros.
    """
    state = rules.initial_state(8)
    result, stub = search(state, simulations=32)

    # Every leaf was evaluated by the network, so none of them was a finished
    # game -- which is what makes "all values are 0" the expected answer.
    assert stub.positions == 33, "expansion plus one evaluation per simulation"

    assert result.root_value == 0.0
    assert all(q == 0.0 for q in result.q_values)
    assert result.value() == 0.0


# ===========================================================================
# Structure: visits, legality, terminal handling
# ===========================================================================


@pytest.mark.parametrize("simulations", [1, 7, 64, 200])
def test_every_simulation_is_counted_exactly_once(simulations: int) -> None:
    result, _ = search(midgame(), simulations=simulations)
    assert result.total_visits == simulations


def test_illegal_actions_are_never_visited() -> None:
    """Contract C5, layer 2. There is no edge for an illegal move to be visited by."""
    state = midgame()
    result, _ = search(state, simulations=2000)

    counts = result.visit_counts()
    legal = set(rules.legal_actions(state))

    assert counts.shape == (policy_size(8),)
    assert set(result.actions) == legal
    for action in range(policy_size(8)):
        if action not in legal:
            assert counts[action] == 0, f"action {action} is illegal but was visited"
    assert counts.sum() == 2000


def test_finished_positions_are_never_sent_to_the_network() -> None:
    """We know the result exactly; a guess could only be worse -- and slower."""
    result, stub = search(EVERY_MOVE_LOSES, simulations=50)

    # The root is expanded once. Its only child is a finished game, so it is
    # scored exactly and never evaluated, however many times we visit it.
    assert stub.positions == 1
    assert result.total_visits == 50


def test_searching_a_finished_game_is_an_error() -> None:
    finished = board(
        """
        WWWW
        WWWW
        WWWW
        WWWW
        """,
        Player.BLACK,
    )
    assert rules.is_terminal(finished)

    with pytest.raises(SearchError, match="finished position"):
        MCTS(StubEvaluator()).run(finished)


# ===========================================================================
# T15 -- the selection rule behaves as advertised
# ===========================================================================


def test_the_search_follows_a_confident_prior() -> None:
    """If the network is sure, the search spends its budget where the network points."""
    state = midgame()
    preferred = rules.legal_actions(state)[0]

    def policy(position: State) -> np.ndarray:
        logits = np.zeros(policy_size(position.size), dtype=np.float32)
        logits[rules.legal_actions(position)[0]] = 8.0
        return logits

    result, _ = search(state, simulations=50, evaluator=StubEvaluator(policy=policy))

    assert result.best_action() == preferred
    assert result.visits[result.actions.index(preferred)] == 50


def test_the_search_finds_the_move_the_network_values_highly() -> None:
    """With flat priors, the value estimates alone have to steer the search."""
    state = midgame()
    actions = rules.legal_actions(state)
    good_child = rules.apply(state, actions[2])
    key = (good_child.black, good_child.white, good_child.to_move)

    def value(position: State) -> float:
        # -1 for the child's mover means +1 for us, the parent.
        return -1.0 if (position.black, position.white, position.to_move) == key else 0.0

    result, _ = search(state, simulations=200, evaluator=StubEvaluator(value=value))

    assert result.best_action() == actions[2]
    best = result.visits[result.actions.index(actions[2])]
    assert best > max(
        v for a, v in zip(result.actions, result.visits, strict=True) if a != actions[2]
    )


def test_first_play_urgency_controls_how_widely_the_search_spreads() -> None:
    """An untried move is valued from its parent's estimate, minus a penalty.

    With no penalty, untried moves stay attractive and the search fans out across
    all of them. With a large penalty it commits early and goes deep. The penalty
    existing at all is what stops a losing position from looking like a reason to
    try every alternative once instead of reading any of them properly.
    """
    state = midgame()
    n_actions = len(rules.legal_actions(state))

    # Every position looks good for whoever moves there, so every child looks
    # bad from the root -- the situation where the stand-in value matters.
    spread, _ = search(
        state,
        simulations=3 * n_actions,
        evaluator=StubEvaluator(value=lambda _s: 0.8),
        c_puct=1.0,
        fpu_reduction=0.0,
    )
    focused, _ = search(
        state,
        simulations=3 * n_actions,
        evaluator=StubEvaluator(value=lambda _s: 0.8),
        c_puct=1.0,
        fpu_reduction=1.5,
    )

    tried_when_spread = sum(1 for v in spread.visits if v > 0)
    tried_when_focused = sum(1 for v in focused.visits if v > 0)

    assert tried_when_spread == n_actions
    assert tried_when_focused < tried_when_spread
    assert max(focused.visits) > max(spread.visits)


# ===========================================================================
# T17 -- determinism
# ===========================================================================


def test_the_same_search_twice_gives_the_same_answer() -> None:
    """No noise, a fixed stub, no RNG anywhere: the result must be reproducible.

    Evaluation depends on this. A tournament where the same agent plays the same
    position differently from run to run cannot be compared against anything.
    """
    state = midgame()
    first, _ = search(state, simulations=120)
    for _ in range(9):
        again, _ = search(state, simulations=120)
        assert again.visits == first.visits
        assert again.q_values == first.q_values
        assert again.best_action() == first.best_action()


# ===========================================================================
# T18 / contract C7 -- exploration noise, and where it must not be
# ===========================================================================


def test_without_noise_an_rng_changes_nothing() -> None:
    """With eps=0 the search must be bit-identical whether or not an RNG is handed in.

    This is the property the arena relies on: it passes no RNG, and it needs to
    know that nothing random is happening behind its back.
    """
    state = midgame()
    config = SearchConfig(n_simulations=64, dirichlet_eps=0.0)

    without = MCTS(StubEvaluator(), config).run(state)
    with_rng = MCTS(StubEvaluator(), config).run(state, rng=np.random.default_rng(0))
    other_rng = MCTS(StubEvaluator(), config).run(state, rng=np.random.default_rng(99))

    assert without.visits == with_rng.visits == other_rng.visits


def test_asking_for_noise_without_an_rng_fails_loudly() -> None:
    """Silently skipping the noise would cost self-play its variety, invisibly."""
    config = SearchConfig(n_simulations=8, dirichlet_eps=0.25)

    with pytest.raises(SearchError, match="no rng was given"):
        MCTS(StubEvaluator(), config).run(midgame())


def test_noise_changes_the_search_and_is_reproducible_from_its_seed() -> None:
    state = midgame()
    config = SearchConfig(n_simulations=64, dirichlet_eps=0.25, dirichlet_alpha=1.0)

    one = MCTS(StubEvaluator(), config).run(state, rng=np.random.default_rng(1))
    one_again = MCTS(StubEvaluator(), config).run(state, rng=np.random.default_rng(1))
    two = MCTS(StubEvaluator(), config).run(state, rng=np.random.default_rng(2))

    assert one.visits == one_again.visits, "same seed, same search"
    assert one.visits != two.visits, "different seeds explore differently"


def test_noise_is_mixed_in_without_breaking_the_priors() -> None:
    """The mixed priors must still be a distribution over the legal moves only.

    Reaching into the search here on purpose: this is a contract, and checking it
    through visit counts would only show it indirectly.
    """
    state = midgame()
    config = SearchConfig(n_simulations=1, dirichlet_eps=0.25, dirichlet_alpha=1.0)
    mcts = MCTS(StubEvaluator(), config)

    from reversi.search.node import Node

    node = Node(state)
    mcts._expand(node)
    before = list(node.prior)
    mcts._add_root_noise(node, np.random.default_rng(4))

    assert len(node.prior) == len(before) == len(rules.legal_actions(state))
    assert sum(node.prior) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in node.prior)
    assert node.prior != before, "0.25 of the prior mass should have moved"


def test_evaluation_configs_have_exploration_forced_off() -> None:
    from reversi.config import MCTSConfig

    training = MCTSConfig(n_simulations=300, dirichlet_eps=0.25, temp_moves=15)

    selfplay = SearchConfig.for_selfplay(training)
    evaluation = SearchConfig.for_evaluation(training)

    assert selfplay.dirichlet_eps == 0.25
    assert selfplay.temp_moves == 15

    assert evaluation.dirichlet_eps == 0.0
    assert evaluation.temp_moves == 0
    assert evaluation.n_simulations == 300, "only the randomness is dropped"


def test_assert_no_noise_rejects_a_self_play_config() -> None:
    """What the arena and the web API call at construction, so a mistake fails at
    startup rather than quietly understating every strength number."""
    from reversi.config import MCTSConfig

    selfplay = SearchConfig.for_selfplay(MCTSConfig(dirichlet_eps=0.25, temp_moves=12))

    with pytest.raises(ConfigError, match="dirichlet_eps=0"):
        selfplay.assert_no_noise("arena")

    with pytest.raises(ConfigError, match="temp_moves"):
        SearchConfig(dirichlet_eps=0.0, temp_moves=12).assert_no_noise("arena")

    SearchConfig.for_evaluation(MCTSConfig()).assert_no_noise("arena")  # must not raise


def test_search_config_rejects_nonsense() -> None:
    with pytest.raises(ConfigError, match="n_simulations"):
        SearchConfig(n_simulations=0)
    with pytest.raises(ConfigError, match="c_puct"):
        SearchConfig(c_puct=0.0)
    with pytest.raises(ConfigError, match="dirichlet_eps"):
        SearchConfig(dirichlet_eps=1.5)
    with pytest.raises(ConfigError, match="fpu_reduction"):
        SearchConfig(fpu_reduction=-0.1)


# ===========================================================================
# Contract C4 -- the training target and the move played are different things
# ===========================================================================


def test_the_policy_target_is_the_raw_visit_share() -> None:
    state = midgame()
    result, _ = search(state, simulations=100)

    target = result.policy_target()
    legal = set(rules.legal_actions(state))

    assert target.shape == (policy_size(8),)
    assert target.sum() == pytest.approx(1.0, abs=1e-6)
    for action, visits in zip(result.actions, result.visits, strict=True):
        assert target[action] == pytest.approx(visits / 100)
    for action in range(policy_size(8)):
        if action not in legal:
            assert target[action] == 0.0


def test_the_policy_target_ignores_the_playing_temperature() -> None:
    """Contract C4, the mistake this guards against.

    The temperature makes self-play *play* a varied opening. It must not touch
    what gets stored: the network has to be trained on what the search actually
    concluded, not on a sharpened or flattened rewrite of it. These are two
    separate code paths on purpose, and this test is what keeps them separate.
    """
    result, _ = search(midgame(), simulations=100)
    target = result.policy_target()

    rng = np.random.default_rng(0)
    for temperature in (0.0, 0.5, 1.0, 2.0):
        select_move(result, temperature=temperature, rng=rng)
        np.testing.assert_array_equal(result.policy_target(), target)


def test_a_pass_only_position_puts_all_the_target_on_pass() -> None:
    result, _ = search(MUST_PASS_THEN_LOSE, simulations=16)
    target = result.policy_target()

    assert target[pass_action(4)] == pytest.approx(1.0)
    assert target.sum() == pytest.approx(1.0, abs=1e-6)


def test_zero_temperature_plays_the_most_visited_move() -> None:
    result, _ = search(midgame(), simulations=100)
    for _ in range(5):
        assert select_move(result, temperature=0.0) == result.best_action()


def test_temperature_samples_in_proportion_to_visits() -> None:
    """Temperature 1.0 means "pick each move as often as the search visited it".

    Needs a network with opinions: against a stub with none, every move gets the
    same number of visits and any sampling rule looks correct.
    """

    def policy(position: State) -> np.ndarray:
        logits = np.zeros(policy_size(position.size), dtype=np.float32)
        for rank, action in enumerate(rules.legal_actions(position)):
            logits[action] = 0.4 * rank
        return logits

    result, _ = search(midgame(), simulations=200, evaluator=StubEvaluator(policy=policy))
    assert len(set(result.visits)) > 1, "this test needs an uneven visit distribution"

    rng = np.random.default_rng(7)
    draws = 5000
    drawn = Counter(select_move(result, temperature=1.0, rng=rng) for _ in range(draws))

    assert set(drawn) == {a for a, v in zip(result.actions, result.visits, strict=True) if v > 0}
    assert drawn.most_common(1)[0][0] == result.best_action()

    for action, visits in zip(result.actions, result.visits, strict=True):
        assert drawn[action] / draws == pytest.approx(visits / result.total_visits, abs=0.03)


def test_sampling_without_an_rng_is_refused() -> None:
    result, _ = search(midgame(), simulations=16)
    with pytest.raises(SearchError, match="needs an rng"):
        select_move(result, temperature=1.0)


# ===========================================================================
# Layering
# ===========================================================================


def test_importing_the_search_does_not_pull_in_torch() -> None:
    """The search talks to a protocol, not to a model.

    This is what keeps the whole test suite runnable on a machine with no GPU and
    no CUDA build, and it is easy to break by adding one convenient import.
    """
    code = "import reversi.search.mcts, sys; print('torch' in sys.modules)"
    finished = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert finished.stdout.strip() == "False", "reversi.search must not import torch"
