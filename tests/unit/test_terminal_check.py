"""The one-move look-ahead for game-ending replies (``SearchConfig.terminal_check``).

The Step 2 weakness, restated as a search problem: the network rates a position
where it holds one disc as a near-certain win and gives the opponent's capturing
reply a prior of a few percent, so a 50-simulation search never expands that
reply and never learns that the game is over one move down. The check reads
that exact answer when a node is opened, without a network call and without
waiting for the search to get there on its own, and a node whose mover can win
at once is then treated as finished.

These tests pin three things: that the check reports the exact result where
one exists and keeps it, that it changes nothing where none does (so every
published number of the plain search stays valid), and that the batched
self-play scheduler applies it the same way the one-at-a-time search does
(``test_game_batch.py`` carries that case).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reversi.arena.entrants import describe_entrant, parse_entrant
from reversi.errors import ConfigError
from reversi.game import reference as ref
from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.search.mcts import MCTS
from reversi.search.node import Node
from reversi.types import Player, policy_size


def board(text: str, to_move: Player) -> State:
    position = ref.from_ascii(text, to_move)
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=to_move, size=position.size)


# Black to play; playing 13 ends the game with Black winning (the same position
# the sign tests use).
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

# The Step 2 shape on a full board. White to play with two discs and nine legal
# moves. After 44, Black has six replies, and one of them, 52, takes every white
# disc and ends the game. Every other white move leaves no game-ending reply.
# Found by searching random games for exactly this: a move that looks ordinary
# one ply deep and is lost one ply deeper.
TRAP = board(
    """
    ........
    .....B..
    ...BB...
    ...BW...
    ...BBBBB
    .....W..
    ........
    ........
    """,
    Player.WHITE,
)
TRAP_MOVE = 44
WIPING_REPLY = 52
SAFE_MOVES = (10, 12, 18, 26, 29, 31, 42, 46)


def confident(root_logit: float = 1.0, reply_logit: float = -4.0) -> StubEvaluator:
    """A stand-in network with the Step 2 opinions.

    It likes the trap move (a prior of about a quarter, the median the Step 2
    probe measured on the move the network played), gives the wiping reply a
    prior under one percent, and rates every position as good for White.
    """

    def policy(state: State) -> np.ndarray:
        logits = np.zeros(policy_size(8), dtype=np.float32)
        legal = rules.legal_actions(state)
        if state.to_move is Player.WHITE and TRAP_MOVE in legal:
            logits[TRAP_MOVE] = root_logit
        if state.to_move is Player.BLACK and WIPING_REPLY in legal:
            logits[WIPING_REPLY] = reply_logit
        return logits

    def value(state: State) -> float:
        return 0.9 if state.to_move is Player.WHITE else -0.9

    return StubEvaluator(policy=policy, value=value)


def test_the_fixtures_have_the_shape_the_tests_assume() -> None:
    assert rules.legal_actions(TRAP) == sorted([TRAP_MOVE, *SAFE_MOVES])
    after_trap = rules.apply(TRAP, TRAP_MOVE)
    assert len(rules.legal_actions(after_trap)) == 6
    ended = rules.apply(after_trap, WIPING_REPLY)
    assert rules.is_terminal(ended)
    assert scoring.result_for(ended, Player.BLACK) == 1
    assert ended.white == 0, "a wipeout: White has no disc left"
    for move in SAFE_MOVES:
        child = rules.apply(TRAP, move)
        assert not any(rules.is_terminal(rules.apply(child, r)) for r in rules.legal_actions(child))


# ===========================================================================
# What the check reports
# ===========================================================================


def test_a_position_with_a_winning_move_is_worth_plus_one_when_opened() -> None:
    """The network's number is kept on the node; the exact one is what gets backed up."""
    stub = StubEvaluator(value=lambda state: 0.1)
    mcts = MCTS(stub, SearchConfig(n_simulations=1, terminal_check=True))
    node = Node(WIN_IN_ONE)
    logits, values = stub.evaluate([WIN_IN_ONE])

    backed_up = mcts.open_node(node, logits[0], float(values[0]))

    assert backed_up == pytest.approx(1.0)
    assert node.proven_value == pytest.approx(1.0)
    assert node.net_value == pytest.approx(0.1), "the raw network value is not overwritten"
    winning = node.children[node.actions.index(WINNING_MOVE)]
    assert winning is not None and winning.is_terminal, "the ending move's child is kept"
    assert stub.positions == 1, "the look-ahead makes no network calls"


def test_the_check_reads_the_loss_one_ply_down_in_a_single_simulation() -> None:
    """Off, one simulation down the trap backs up the network's cheerful value.
    On, it backs up -1, because the reply that ends the game was looked at."""
    on = MCTS(confident(), SearchConfig(n_simulations=1, terminal_check=True)).run(TRAP)
    off = MCTS(confident(), SearchConfig(n_simulations=1)).run(TRAP)

    trap = on.actions.index(TRAP_MOVE)
    assert on.visits[trap] == 1 and off.visits[trap] == 1, "both went down the trap first"
    assert on.q_values[trap] == pytest.approx(-1.0)
    assert off.q_values[trap] == pytest.approx(0.9)


def test_a_proven_loss_stays_lost_and_the_search_moves_on() -> None:
    """With the check, the trap's value is exactly -1 however often the search
    returns to it, and the most-visited move is a safe one. Without it, the
    plain search prefers the trap at every budget, because the reply that ends
    the game has too small a prior ever to be tried."""
    checked = MCTS(confident(), SearchConfig(n_simulations=16, terminal_check=True)).run(TRAP)
    plain = MCTS(confident(), SearchConfig(n_simulations=256)).run(TRAP)

    trap = checked.actions.index(TRAP_MOVE)
    assert checked.q_values[trap] == pytest.approx(-1.0)
    assert checked.best_action() in SAFE_MOVES
    assert plain.best_action() == TRAP_MOVE
    assert plain.q_values[trap] > 0.8


def test_a_proven_node_is_not_descended_into() -> None:
    """A node whose mover can win at once is as good as finished: nothing below
    it is expanded, so its exact value cannot be diluted by network opinions
    of the other replies."""
    mcts = MCTS(confident(), SearchConfig(n_simulations=32, terminal_check=True))
    root = Node(TRAP)
    mcts._expand(root)
    for _ in range(32):
        mcts._simulate(root)

    child = root.children[root.actions.index(TRAP_MOVE)]
    assert child is not None and child.proven_value == pytest.approx(1.0)
    assert root.visits[root.actions.index(TRAP_MOVE)] > 1, "the trap was visited again"
    assert all(v == 0 for v in child.visits), "but never searched below"


def test_the_root_is_searched_even_when_it_is_proven() -> None:
    """``root_value`` is documented as the network's raw estimate, and the search
    must still visit the root's moves to choose one, win in hand or not."""
    stub = StubEvaluator(value=lambda state: 0.3)
    result = MCTS(stub, SearchConfig(n_simulations=4, terminal_check=True)).run(WIN_IN_ONE)

    assert result.root_value == pytest.approx(0.3)
    assert result.total_visits == 4
    assert result.best_action() == WINNING_MOVE
    assert result.q_values[result.actions.index(WINNING_MOVE)] == pytest.approx(1.0)


# ===========================================================================
# Where nothing ends, nothing changes
# ===========================================================================


def midgame(plies: int = 12, seed: int = 3) -> State:
    import random

    rng = random.Random(seed)
    state = rules.initial_state(8)
    for _ in range(plies):
        state = rules.apply(state, rng.choice(rules.legal_actions(state)))
    return state


def test_the_check_is_off_by_default_and_in_every_derived_config() -> None:
    from reversi.config import MCTSConfig

    assert SearchConfig().terminal_check is False
    assert SearchConfig.for_selfplay(MCTSConfig()).terminal_check is False
    assert SearchConfig.for_evaluation(MCTSConfig(), n_simulations=50).terminal_check is False


def test_without_a_game_ending_move_in_reach_the_search_is_identical() -> None:
    state = midgame()
    plain = MCTS(StubEvaluator(), SearchConfig(n_simulations=48)).run(state)
    checked = MCTS(StubEvaluator(), SearchConfig(n_simulations=48, terminal_check=True)).run(state)

    assert checked.actions == plain.actions
    assert checked.visits == plain.visits
    assert checked.q_values == pytest.approx(plain.q_values)


# ===========================================================================
# The entrant switch
# ===========================================================================


def test_an_entrant_switches_the_check_on_after_a_semicolon(tmp_path: Path) -> None:
    weights = tmp_path / "gen_00120.pt"
    weights.write_bytes(b"x")

    spec = parse_entrant(f"control-tc={weights}@50;terminal_check", default_simulations=1)
    assert spec.terminal_check is True
    assert spec.c_puct == 1.5
    assert describe_entrant(spec)["terminal_check"] is True

    spelled = parse_entrant(f"x={weights};terminal_check=1;c_puct=2.0", default_simulations=1)
    assert spelled.terminal_check is True and spelled.c_puct == 2.0

    off = parse_entrant(f"x={weights};terminal_check=0", default_simulations=1)
    assert off.terminal_check is False
    assert "terminal_check" not in describe_entrant(off)

    with pytest.raises(ConfigError, match="terminal_check takes 0 or 1"):
        parse_entrant(f"x={weights};terminal_check=maybe", default_simulations=1)
