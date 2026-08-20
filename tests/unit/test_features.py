"""What the network is shown (test matrix T14, contract C1).

The encoding is small enough that these tests can check it square by square,
which is worth doing: if the planes are wrong, every downstream number is
meaningless and nothing anywhere would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.game import reference as ref
from reversi.game import rules, symmetry
from reversi.game.state import State
from reversi.nn import features
from reversi.types import Player, pass_action, policy_size

# ---------------------------------------------------------------------------


def board(text: str, to_move: Player = Player.BLACK) -> State:
    """A position drawn as a picture, in the fast engine's representation."""
    position = ref.from_ascii(text, to_move)
    black, white = position.bitboards()
    return State(black=black, white=white, to_move=to_move, size=position.size)


MIDGAME = """
........
...BW...
..BBBW..
...BWW..
...WBB..
....W...
........
........
"""


# ---------------------------------------------------------------------------
# Shape and contents
# ---------------------------------------------------------------------------


def test_encoding_shape_and_dtype() -> None:
    planes = features.encode(rules.initial_state(8))
    assert planes.shape == (features.IN_PLANES, 8, 8)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)) <= {0.0, 1.0}


@pytest.mark.parametrize("size", [4, 6, 8])
def test_encoding_shape_for_every_board_size(size: int) -> None:
    planes = features.encode(rules.initial_state(size))
    assert planes.shape == (features.IN_PLANES, size, size)


def test_planes_hold_mover_opponent_and_legal_moves() -> None:
    """Plane 0 is mine, plane 1 is theirs, plane 2 is where I may play."""
    state = board(MIDGAME, Player.BLACK)
    mine, theirs, legal = features.encode(state)

    assert features.planes_to_bits(mine, 8) == state.black
    assert features.planes_to_bits(theirs, 8) == state.white
    assert features.planes_to_bits(legal, 8) == rules.legal_placements(state)

    # No square is in both disc planes, and no legal move sits on an occupied one.
    assert not np.any((mine > 0) & (theirs > 0))
    assert not np.any((legal > 0) & ((mine > 0) | (theirs > 0)))


def test_index_convention_is_row_times_size_plus_col() -> None:
    """Contract C1: bit i lands at grid[i // size, i % size], nowhere else."""
    # A single black disc at row 1, col 2 -> index 1 * 4 + 2 = 6.
    state = State(black=1 << 6, white=0, to_move=Player.BLACK, size=4)
    mine = features.encode(state)[0]

    assert mine[1, 2] == 1.0
    assert mine.sum() == 1.0


# ---------------------------------------------------------------------------
# Contract C1: colours are never shown to the network
# ---------------------------------------------------------------------------


def test_colour_swapped_position_encodes_identically() -> None:
    """The heart of contract C1.

    Take any position, swap every black disc for a white one and vice versa, and
    swap whose turn it is. That is the *same problem* -- the player to move faces
    an identical arrangement of their own and their opponent's discs. The network
    must therefore see exactly the same input, so that everything it learns about
    one applies to the other.

    Without this, the network has to learn Reversi twice, once per colour, from
    half the data each time.
    """
    state = board(MIDGAME, Player.BLACK)
    mirrored = State(
        black=state.white,
        white=state.black,
        to_move=state.to_move.opponent,
        size=state.size,
    )

    assert not np.array_equal(state.black, mirrored.black), "the two positions differ"
    np.testing.assert_array_equal(features.encode(state), features.encode(mirrored))


def test_same_board_different_mover_encodes_differently() -> None:
    """The flip side: the colour swap only cancels out when *both* things swap."""
    state = board(MIDGAME, Player.BLACK)
    other_turn = State(black=state.black, white=state.white, to_move=Player.WHITE, size=state.size)
    assert not np.array_equal(features.encode(state), features.encode(other_turn))


@pytest.mark.parametrize("name_index", range(8))
def test_encoding_turns_with_the_board(name_index: int) -> None:
    """Rotating the board rotates its encoding, plane by plane (C1 meets C6).

    This is what makes the eightfold data augmentation valid: a turned position
    encodes to the turned encoding, so a training sample can be turned without
    re-deriving anything.
    """
    state = board(MIDGAME, Player.BLACK)
    sym = symmetry.symmetries(8)[name_index]

    turned_then_encoded = features.encode(symmetry.transform_state(state, sym))

    encoded_then_turned = np.zeros_like(turned_then_encoded)
    original = features.encode(state)
    for index in range(64):
        source_row, source_col = divmod(index, 8)
        target_row, target_col = divmod(sym.perm[index], 8)
        encoded_then_turned[:, target_row, target_col] = original[:, source_row, source_col]

    np.testing.assert_array_equal(turned_then_encoded, encoded_then_turned)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_encode_batch_matches_encoding_each_position() -> None:
    states = [rules.initial_state(8), board(MIDGAME, Player.WHITE)]
    batch = features.encode_batch(states)

    assert batch.shape == (2, features.IN_PLANES, 8, 8)
    for row, state in enumerate(states):
        np.testing.assert_array_equal(batch[row], features.encode(state))


def test_encode_batch_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        features.encode_batch([])


def test_encode_batch_rejects_mixed_board_sizes() -> None:
    with pytest.raises(ValueError, match="same board size"):
        features.encode_batch([rules.initial_state(8), rules.initial_state(4)])


# ---------------------------------------------------------------------------
# The legal-action mask
# ---------------------------------------------------------------------------


def test_legal_mask_matches_the_engine() -> None:
    state = board(MIDGAME, Player.BLACK)
    mask = features.legal_mask(state)

    assert mask.shape == (policy_size(8),)
    assert set(np.flatnonzero(mask)) == set(rules.legal_actions(state))
    assert not mask[pass_action(8)], "PASS is not legal when placements exist"


def test_legal_mask_sets_pass_only_when_passing_is_forced() -> None:
    # White has no legal square here but Black does, so PASS is White's only move.
    state = board(
        """
        BBW.
        BBWW
        WBWW
        BBBB
        """,
        Player.WHITE,
    )
    mask = features.legal_mask(state)

    assert rules.must_pass(state)
    assert mask[pass_action(4)]
    assert np.count_nonzero(mask) == 1


def test_legal_mask_is_empty_exactly_when_the_game_is_over() -> None:
    """Contract C3, stated in mask form."""
    state = board(
        """
        WWWW
        WWWW
        WWWW
        WWWW
        """,
        Player.BLACK,
    )
    assert rules.is_terminal(state)
    assert not features.legal_mask(state).any()


# ---------------------------------------------------------------------------
# Masked softmax (T14)
# ---------------------------------------------------------------------------


def test_masked_policy_is_zero_on_illegal_moves_and_sums_to_one() -> None:
    state = board(MIDGAME, Player.BLACK)
    mask = features.legal_mask(state)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=policy_size(8)).astype(np.float32)

    probs = features.masked_policy(logits, mask)

    assert probs.shape == mask.shape
    assert probs[~mask].sum() == 0.0
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(probs >= 0.0)


def test_masked_policy_ranks_legal_moves_by_their_logits() -> None:
    state = board(MIDGAME, Player.BLACK)
    mask = features.legal_mask(state)
    legal = np.flatnonzero(mask)

    logits = np.zeros(policy_size(8), dtype=np.float32)
    logits[legal[0]] = 5.0  # strongly preferred
    probs = features.masked_policy(logits, mask)

    assert probs[legal[0]] == probs.max()
    assert probs[legal[0]] > 0.5


def test_masked_policy_with_one_legal_move_gives_it_everything() -> None:
    mask = np.zeros(policy_size(4), dtype=np.bool_)
    mask[7] = True
    probs = features.masked_policy(np.zeros(policy_size(4), dtype=np.float32), mask)

    assert probs[7] == pytest.approx(1.0)
    assert not np.isnan(probs).any()


def test_masked_policy_survives_extreme_logits() -> None:
    """Large positive logits overflow float32 if the max is not subtracted first."""
    mask = np.ones(policy_size(4), dtype=np.bool_)
    logits = np.full(policy_size(4), 90.0, dtype=np.float32)
    logits[3] = 120.0

    probs = features.masked_policy(logits, mask)

    assert not np.isnan(probs).any()
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert probs[3] == probs.max()


def test_masked_policy_with_all_infinite_logits_falls_back_to_uniform() -> None:
    mask = np.zeros(policy_size(4), dtype=np.bool_)
    mask[[1, 2, 3]] = True
    logits = np.full(policy_size(4), -np.inf, dtype=np.float32)

    probs = features.masked_policy(logits, mask)

    assert not np.isnan(probs).any()
    assert probs[mask] == pytest.approx(1 / 3)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_masked_policy_refuses_an_empty_mask() -> None:
    mask = np.zeros(policy_size(4), dtype=np.bool_)
    with pytest.raises(ValueError, match="no legal actions"):
        features.masked_policy(np.zeros(policy_size(4), dtype=np.float32), mask)


def test_masked_policy_rejects_a_shape_mismatch() -> None:
    mask = np.ones(policy_size(4), dtype=np.bool_)
    with pytest.raises(ValueError, match="does not match"):
        features.masked_policy(np.zeros(3, dtype=np.float32), mask)
