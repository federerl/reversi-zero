"""The eight board turns behave like turns (test matrix T10, contract C6).

If these properties hold, showing the network a rotated position with a rotated
answer is guaranteed to be teaching it something true rather than teaching it
noise. Since symmetry augmentation multiplies our training data by eight, a bug
here would corrupt seven eighths of it.

The five properties, in plain terms:

1. The eight turns are genuinely different from each other.
2. Turning and then un-turning gets you back where you started.
3. The legal moves of a turned board are the turned legal moves.
4. Playing a turned move on a turned board gives the turned result.
5. PASS never moves, because passing is not a square.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reversi.game import rules, scoring
from reversi.game.symmetry import (
    symmetries,
    transform_bits,
    transform_policy,
    transform_state,
)
from reversi.types import pass_action

SIZES = [4, 6, 8]


def random_position(seed: int, size: int) -> rules.State:
    """A position some way into a random game, so tests see real boards."""
    rng = random.Random(seed)
    state = rules.initial_state(size)
    for _ in range(rng.randrange(0, 3 * size)):
        if rules.is_terminal(state):
            break
        state = rules.apply(state, rng.choice(rules.legal_actions(state)))
    return state


# ---------------------------------------------------------------------------
# 1. There really are eight distinct turns
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_there_are_eight_distinct_turns(size: int) -> None:
    syms = symmetries(size)
    assert len(syms) == 8
    assert len({sym.perm for sym in syms}) == 8, "two turns do the same thing"


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_every_turn_is_a_rearrangement_of_the_squares(size: int) -> None:
    """A turn must move every square somewhere, with nothing lost or doubled."""
    n_squares = size * size
    for sym in symmetries(size):
        assert sorted(sym.perm) == list(range(n_squares)), f"{sym.name} is not a permutation"


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_turning_twice_stays_within_the_eight(size: int) -> None:
    """Any turn followed by any other turn is itself one of the eight.

    This is what makes them a closed set rather than eight arbitrary shuffles.
    """
    syms = symmetries(size)
    known = {sym.perm for sym in syms}
    for a in syms:
        for b in syms:
            combined = tuple(b.perm[a.perm[i]] for i in range(size * size))
            assert combined in known, f"{a.name} then {b.name} is not one of the eight"


# ---------------------------------------------------------------------------
# 2. Turning and un-turning is a round trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_inverse_undoes_the_turn(size: int) -> None:
    for sym in symmetries(size):
        for index in range(size * size):
            assert sym.inverse[sym.perm[index]] == index
            assert sym.perm[sym.inverse[index]] == index


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_state_round_trips(size: int) -> None:
    for seed in range(20):
        state = random_position(seed, size)
        for sym in symmetries(size):
            turned = transform_state(state, sym)
            back = rules.State(
                black=transform_bits(turned.black, _inverted(sym)),
                white=transform_bits(turned.white, _inverted(sym)),
                to_move=turned.to_move,
                size=size,
            )
            assert back == state, f"{sym.name} did not round trip"


def _inverted(sym):  # type: ignore[no-untyped-def]
    """The turn that undoes ``sym``."""
    from reversi.game.symmetry import Symmetry

    return Symmetry(name=f"inverse_{sym.name}", size=sym.size, perm=sym.inverse, inverse=sym.perm)


# ---------------------------------------------------------------------------
# 3 and 4. The rules survive being turned
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_legal_moves_of_a_turned_board_are_the_turned_legal_moves(size: int) -> None:
    for seed in range(30):
        state = random_position(seed, size)
        expected_actions = rules.legal_actions(state)

        for sym in symmetries(size):
            turned = transform_state(state, sym)
            assert sorted(rules.legal_actions(turned)) == sorted(
                sym.action(a) for a in expected_actions
            ), f"{sym.name} changed which moves are legal"

            assert rules.is_terminal(turned) == rules.is_terminal(state)
            assert scoring.disc_counts(turned) == scoring.disc_counts(state)


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_playing_a_turned_move_on_a_turned_board_gives_the_turned_result(size: int) -> None:
    """The property that makes augmentation valid.

    Turn the board, turn the move, play it — you get the same thing as playing
    the move first and turning afterwards.
    """
    for seed in range(30):
        state = random_position(seed, size)
        if rules.is_terminal(state):
            continue

        for action in rules.legal_actions(state):
            played_then_turned = rules.apply(state, action)
            for sym in symmetries(size):
                turned_then_played = rules.apply(transform_state(state, sym), sym.action(action))
                assert turned_then_played == transform_state(played_then_turned, sym), (
                    f"{sym.name} broke on action {action}"
                )


# ---------------------------------------------------------------------------
# 5. PASS stays put
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_pass_is_never_moved_by_a_turn(size: int) -> None:
    """Rotating the PASS entry would scramble every pass probability we train on."""
    for sym in symmetries(size):
        assert sym.action(pass_action(size)) == pass_action(size)
        assert sym.inverse_action(pass_action(size)) == pass_action(size)


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_policy_turns_with_the_board_and_keeps_pass_in_place(size: int) -> None:
    n_squares = size * size
    policy = [float(i) for i in range(n_squares)] + [0.5]

    for sym in symmetries(size):
        turned = transform_policy(policy, sym)
        assert len(turned) == n_squares + 1
        assert turned[n_squares] == 0.5, f"{sym.name} moved the PASS entry"
        for index in range(n_squares):
            assert turned[sym.perm[index]] == policy[index]


@pytest.mark.property
@pytest.mark.parametrize("size", SIZES)
def test_turning_a_policy_preserves_its_total(size: int) -> None:
    """A probability distribution stays a probability distribution."""
    n_squares = size * size
    policy = [1.0 / (n_squares + 1)] * (n_squares + 1)
    for sym in symmetries(size):
        assert sum(transform_policy(policy, sym)) == pytest.approx(1.0)


@pytest.mark.property
def test_policy_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="65 entries"):
        transform_policy([0.0] * 64, symmetries(8)[0])


# ---------------------------------------------------------------------------
# The same properties, on positions Hypothesis chooses rather than we do
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=10_000), turn=st.integers(0, 7))
@settings(max_examples=1000, deadline=None)
def test_symmetry_properties_on_arbitrary_positions(seed: int, turn: int) -> None:
    state = random_position(seed, 8)
    sym = symmetries(8)[turn]
    turned = transform_state(state, sym)

    assert rules.is_terminal(turned) == rules.is_terminal(state)
    assert sorted(rules.legal_actions(turned)) == sorted(
        sym.action(a) for a in rules.legal_actions(state)
    )
    assert scoring.disc_counts(turned) == scoring.disc_counts(state)
    if rules.is_terminal(state):
        assert scoring.result(turned) == scoring.result(state)
