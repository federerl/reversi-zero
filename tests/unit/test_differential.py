"""Making the two engines prove each other right (test matrix T12).

There are two implementations of the Reversi rules in this project:

* ``reference.py`` — a grid of squares, walking outward in eight directions.
  Slow, and obviously correct from reading it.
* ``rules.py`` — two integers and bit shifts. Fast, and *not* obviously correct
  from reading it.

Neither was written from the other. Both were written from the rules of
Othello. So if they agree on thousands of random games, checked at every single
move, the odds that they are both wrong in exactly the same way are negligible.
That agreement is the entire correctness argument for the engine.

This matters more than it might seem. A rules bug does not crash: the AI just
learns a slightly different game, and every measurement afterwards is
meaningless. There is no loss curve and no other test in this project that
would catch it.

What is compared, after every move of every game:

* the exact set of legal actions
* whether the game is over
* which discs each legal move would flip
* the resulting position, bit for bit
* the disc counts and the final result
"""

from __future__ import annotations

import random

import pytest

from reversi.errors import IllegalMoveError
from reversi.game import reference as ref
from reversi.game import rules, scoring
from reversi.game.bitboard import indices
from reversi.game.state import State
from reversi.types import Player, pass_action

# Every push runs the quick version; the nightly job runs the long one.
#
# These numbers are set by measurement, not preference. Paired games run at
# about 22 per second (both engines plus every comparison), and the reference
# engine is the bottleneck. So 20,000 games takes ~15 minutes, which fits the
# nightly budget alongside the other slow tests; 50,000 would take ~38 minutes
# and would not. 20,000 games is still roughly 1.2 million positions compared,
# with every legal move's flip set checked at each one.
FAST_GAMES = 400
NIGHTLY_GAMES = 20_000


def as_state(position: ref.RefState) -> State:
    """The same position, in the fast engine's representation."""
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=position.to_move, size=position.size)


def compare(position: ref.RefState, state: State, context: str) -> None:
    """Assert the two engines agree about everything in this position."""
    # The positions themselves must match before anything else is meaningful.
    assert position.bitboards() == (state.black, state.white), f"{context}: boards diverged"
    assert position.to_move is state.to_move, f"{context}: turn diverged"

    ref_actions = sorted(ref.legal_actions(position))
    fast_actions = sorted(rules.legal_actions(state))
    assert ref_actions == fast_actions, (
        f"{context}: legal actions disagree\n"
        f"reference: {ref_actions}\nfast:      {fast_actions}\n{state}"
    )

    assert ref.is_terminal(position) == rules.is_terminal(state), f"{context}: terminal disagrees"
    assert ref.disc_counts(position) == scoring.disc_counts(state), f"{context}: counts disagree"

    # Flip sets have to match for every legal move, not just the one played.
    # Checking only the played move would let a bug hide in the moves the random
    # player happened not to choose.
    for action in fast_actions:
        if action == pass_action(state.size):
            continue
        expected = sorted(ref.flips(position, action))
        actual = sorted(indices(rules.flips(state, action)))
        assert expected == actual, (
            f"{context}: flips disagree for action {action}\n"
            f"reference: {expected}\nfast:      {actual}\n{state}"
        )

    if ref.is_terminal(position):
        assert ref.result(position) == scoring.result(state), f"{context}: result disagrees"


def play_paired_game(seed: int, size: int) -> tuple[int, int]:
    """Play one random game through both engines at once, comparing each step.

    Both engines are driven by the same move choices, so any divergence must
    come from the rules rather than from the two games drifting apart.
    """
    rng = random.Random(seed)
    position = ref.initial_state(size)
    state = rules.initial_state(size)

    compare(position, state, f"seed={seed} ply=0")

    plies = 0
    while not rules.is_terminal(state):
        actions = rules.legal_actions(state)
        action = rng.choice(actions)

        position = ref.apply(position, action)
        state = rules.apply(state, action)
        plies += 1

        compare(position, state, f"seed={seed} ply={plies} action={action}")
        assert plies <= 2 * size * size, "game ran impossibly long"

    return plies, scoring.result(state)


# ---------------------------------------------------------------------------
# The cross-check itself
# ---------------------------------------------------------------------------


@pytest.mark.differential
@pytest.mark.parametrize("size", [4, 6, 8])
def test_engines_agree_across_random_games(size: int) -> None:
    for seed in range(FAST_GAMES // 3):
        play_paired_game(seed, size)


@pytest.mark.differential
@pytest.mark.slow
@pytest.mark.timeout(2400)  # ~15 min expected; the default 300s is for fast tests
def test_engines_agree_across_many_random_games() -> None:
    """The long version. Runs nightly, not on every push."""
    for seed in range(NIGHTLY_GAMES):
        play_paired_game(seed, 8)


# ---------------------------------------------------------------------------
# Agreement on the awkward cases specifically
#
# Random play reaches passes and wipeouts only occasionally, so those get their
# own checks rather than being left to chance.
# ---------------------------------------------------------------------------


@pytest.mark.differential
def test_engines_agree_on_forced_passes() -> None:
    """Search for games containing a pass, and check both engines pass there."""
    passes_seen = 0

    for seed in range(600):
        rng = random.Random(seed)
        position = ref.initial_state(8)
        state = rules.initial_state(8)

        while not rules.is_terminal(state):
            actions = rules.legal_actions(state)
            if actions == [pass_action(8)]:
                passes_seen += 1
                assert ref.legal_actions(position) == [pass_action(8)]
                assert rules.must_pass(state)
                assert not rules.is_terminal(state)
                assert not ref.is_terminal(position)

                after_ref = ref.apply(position, pass_action(8))
                after_fast = rules.apply(state, pass_action(8))
                # A pass moves no disc and hands over the turn.
                assert after_fast.black == state.black
                assert after_fast.white == state.white
                assert after_fast.to_move is state.to_move.opponent
                compare(after_ref, after_fast, f"after pass, seed={seed}")

            action = rng.choice(actions)
            position = ref.apply(position, action)
            state = rules.apply(state, action)

    assert passes_seen > 20, f"only {passes_seen} passes found; the sample is too thin"


@pytest.mark.differential
def test_engines_agree_on_illegal_actions() -> None:
    """Both engines must refuse exactly the same actions (contract C5)."""
    rng = random.Random(99)

    for _seed in range(40):
        position = ref.initial_state(8)
        state = rules.initial_state(8)
        for _ in range(rng.randrange(0, 40)):
            if rules.is_terminal(state):
                break
            action = rng.choice(rules.legal_actions(state))
            position = ref.apply(position, action)
            state = rules.apply(state, action)

        allowed = set(rules.legal_actions(state))
        for action in [*range(65), 65, 100, -1]:
            if action in allowed:
                continue
            with pytest.raises(IllegalMoveError):
                rules.apply(state, action)
            with pytest.raises(IllegalMoveError):
                ref.apply(position, action)


@pytest.mark.differential
def test_engines_agree_on_the_opening() -> None:
    for size in (4, 6, 8):
        position = ref.initial_state(size)
        state = rules.initial_state(size)
        compare(position, state, f"opening size={size}")
        assert sorted(rules.legal_actions(state)) == sorted(ref.legal_actions(position))


@pytest.mark.differential
@pytest.mark.parametrize(("d_row", "d_col"), ref.DIRECTIONS)
def test_engines_agree_on_every_flip_direction(d_row: int, d_col: int) -> None:
    """One direction at a time, checked against both engines.

    Random games exercise all eight directions many times over, but building
    them explicitly means a direction cannot be missed by chance — and if one
    ever breaks, the failure names which one instead of pointing at ply 37 of
    game 4,912.

    The board is 6x6: black plays the centre square, a white disc sits one step
    away in the direction being tested, and a black disc one step beyond closes
    the trap. Exactly one disc may flip.
    """
    size = 6
    play_row, play_col = 2, 2
    grid = [["." for _ in range(size)] for _ in range(size)]
    grid[play_row + d_row][play_col + d_col] = "W"
    grid[play_row + 2 * d_row][play_col + 2 * d_col] = "B"

    position = ref.from_ascii("\n".join("".join(row) for row in grid), to_move=Player.BLACK)
    state = as_state(position)

    play = ref.to_index(play_row, play_col, size)
    trapped = ref.to_index(play_row + d_row, play_col + d_col, size)

    assert ref.flips(position, play) == [trapped]
    assert indices(rules.flips(state, play)) == [trapped]
    compare(position, state, f"direction ({d_row},{d_col})")
    compare(
        ref.apply(position, play),
        rules.apply(state, play),
        f"after playing direction ({d_row},{d_col})",
    )


@pytest.mark.differential
def test_engines_agree_when_a_player_is_wiped_out() -> None:
    """Losing every disc ends the game: a legal move must trap an opponent
    disc, and there are none left to trap."""
    board = """
        BBB.
        BBB.
        BBB.
        ....
        """
    position = ref.from_ascii(board, to_move=Player.WHITE)
    state = as_state(position)

    assert ref.is_terminal(position) and rules.is_terminal(state)
    assert scoring.disc_counts(state) == (9, 0)
    assert ref.result(position) == scoring.result(state) == -1
    compare(position, state, "wipeout")
