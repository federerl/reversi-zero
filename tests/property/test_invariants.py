"""Things that must be true in every Reversi position, ever (test matrix T11).

The hand-written tests in ``test_rules.py`` check positions someone thought of.
These check positions nobody thought of: thousands of random games are played to
completion, and after every single move a set of statements that can never be
false is verified.

This catches a different kind of bug. A hand-written test finds the case you
anticipated; an invariant finds the case you did not — the position 40 moves
into a game where two rules interact in a way nobody considered.

The fast selection plays 60 games per board size, which takes about 8 seconds
against this deliberately slow engine. The nightly job plays the full 5,000.
"""

from __future__ import annotations

import random

import pytest

from reversi.errors import IllegalMoveError
from reversi.game import reference as ref
from reversi.types import Player, pass_action

# The reference engine is slow by design, so the fast selection keeps these
# small. Day 3's bitboard engine is far quicker; these counts get raised then,
# and the 50,000-game cross-check becomes the main workout.
FAST_GAMES = 60
FULL_GAMES = 5000


def check_position(state: ref.RefState) -> None:
    """Every statement here must hold in any legal Reversi position."""
    size = state.size
    n_squares = size * size
    black_bb, white_bb = state.bitboards()
    black, white = ref.disc_counts(state)

    # No square can be held by both players at once. This is the invariant a
    # broken flip would violate first.
    assert black_bb & white_bb == 0, "a square is owned by both players"

    assert black + white <= n_squares
    assert bin(black_bb).count("1") == black
    assert bin(white_bb).count("1") == white

    actions = ref.legal_actions(state)

    # Contract C3: being over and having nothing to do are the same statement.
    assert ref.is_terminal(state) == (actions == [])

    if pass_action(size) in actions:
        assert actions == [pass_action(size)], "PASS is never offered alongside a real move"
        assert ref.legal_placements(state) == []
        assert ref.legal_placements_for(state, state.to_move.opponent) != []

    for action in actions:
        if action == pass_action(size):
            continue
        row, col = ref.to_coords(action, size)
        assert state.cell(row, col) is None, "a legal move must land on an empty square"
        assert ref.flips(state, action), "a legal move must flip at least one disc"


def play_random_game(seed: int, size: int = 8) -> tuple[ref.RefState, int, int]:
    """Play one game with random legal moves, checking invariants throughout."""
    rng = random.Random(seed)
    state = ref.initial_state(size)
    check_position(state)

    plies = 0
    passes = 0
    last_was_pass = False
    previous_discs = sum(ref.disc_counts(state))

    while not ref.is_terminal(state):
        actions = ref.legal_actions(state)
        action = rng.choice(actions)
        mover = state.to_move

        state = ref.apply(state, action)
        plies += 1
        check_position(state)

        # Whose turn it is always changes, including on a pass.
        assert state.to_move is mover.opponent

        discs = sum(ref.disc_counts(state))
        if action == pass_action(size):
            passes += 1
            # Contract C3 in action: two passes in a row would mean neither side
            # could move, which is the game being over, so it can never be
            # played as a move.
            assert not last_was_pass, "two passes in a row should have ended the game"
            last_was_pass = True
            assert discs == previous_discs, "passing must not change the board"
        else:
            last_was_pass = False
            row, col = ref.to_coords(action, size)
            assert state.cell(row, col) is mover, "the played square must be the mover's"
            assert discs > previous_discs, "a placement always adds at least the new disc"
        previous_discs = discs

        assert plies <= 2 * size * size, "a game cannot run this long"

    return state, plies, passes


@pytest.mark.property
@pytest.mark.parametrize("size", [4, 6, 8])
def test_invariants_hold_across_random_games(size: int) -> None:
    for seed in range(FAST_GAMES // 3):
        play_random_game(seed, size)


@pytest.mark.property
@pytest.mark.slow
def test_invariants_hold_across_many_random_games() -> None:
    for seed in range(FULL_GAMES):
        play_random_game(seed, 8)


@pytest.mark.property
def test_games_finish_and_score_consistently() -> None:
    for seed in range(FAST_GAMES):
        state, plies, passes = play_random_game(seed, 8)

        assert ref.is_terminal(state)
        assert ref.legal_actions(state) == []

        black, white = ref.disc_counts(state)
        assert 0 < black + white <= 64

        # These bounds are derived from the rules, not guessed:
        #   - the board starts with 4 discs and every placement adds exactly one
        #     newly occupied square, so placements == discs - 4
        #   - a full 8x8 board has 60 squares to fill, so placements <= 60
        #   - passes can never be consecutive, so there is at most one more pass
        #     than there are placements
        placements = plies - passes
        assert placements == black + white - 4
        assert placements <= 60
        assert passes <= placements + 1

        outcome = ref.result(state)
        mine, theirs = (black, white) if state.to_move is Player.BLACK else (white, black)
        expected = (mine > theirs) - (mine < theirs)
        assert outcome == expected


@pytest.mark.property
def test_result_is_opposite_for_the_two_players() -> None:
    """A finished game is a win for exactly one side, or a draw for both."""
    for seed in range(25):
        state, _, _ = play_random_game(seed, 8)
        flipped = ref.RefState(grid=state.grid, to_move=state.to_move.opponent, size=state.size)
        assert ref.result(state) == -ref.result(flipped)


@pytest.mark.property
def test_illegal_actions_are_always_rejected() -> None:
    """Anything not offered by legal_actions must be refused (contract C5)."""
    rng = random.Random(1234)

    for _ in range(25):
        state = ref.initial_state(8)
        for _ in range(rng.randrange(0, 40)):
            if ref.is_terminal(state):
                break
            state = ref.apply(state, rng.choice(ref.legal_actions(state)))

        allowed = set(ref.legal_actions(state))
        for action in range(65):
            if action in allowed:
                continue
            with pytest.raises(IllegalMoveError):
                ref.apply(state, action)


@pytest.mark.property
@pytest.mark.slow
def test_first_player_advantage_is_not_extreme() -> None:
    """A sanity check on the whole engine rather than on one rule.

    Under random play the two sides should win about equally often. A lopsided
    result would point at something structural - a colour handled asymmetrically,
    or the wrong player moving first.
    """
    black_wins = white_wins = draws = 0
    for seed in range(400):
        state, _, _ = play_random_game(seed, 8)
        black, white = ref.disc_counts(state)
        if black > white:
            black_wins += 1
        elif white > black:
            white_wins += 1
        else:
            draws += 1

    assert draws < 100, "draws should be uncommon under random play"
    decisive = black_wins + white_wins
    share = black_wins / decisive
    assert 0.35 < share < 0.65, (
        f"black won {share:.0%} of decisive games under random play; "
        "a large imbalance suggests the colours are handled asymmetrically"
    )
