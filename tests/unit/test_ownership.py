"""The ownership target: who ends up owning each square, and everything it threads through.

The target exists to give the shared trunk sixty-four small answers about "who is
winning where" instead of the one number ``z`` carries. Four things have to hold
for it to be safe to add:

* the target is *right* -- signs follow the mover, and its sum is the disc margin;
* shards without it still load, and a window can mix old and new shards;
* augmentation turns it with the board, exactly as the policy is turned;
* a network with the head is a drop-in for one without: ``forward`` unchanged,
  old checkpoints still resume, the term absent from the loss unless asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from reversi.config import Config
from reversi.data import replay as replay_module
from reversi.data.replay import ReplayBuffer
from reversi.data.schema import (
    GameRecord,
    Sample,
    arrays_to_samples,
    ownership_for,
    samples_to_arrays,
    validate_arrays,
)
from reversi.data.shards import read_shard, write_shard
from reversi.errors import ReplayError
from reversi.game import symmetry
from reversi.game.state import State, initial_state
from reversi.nn.model import PolicyValueNet, normalise_arch
from reversi.train.loss import policy_value_loss
from reversi.types import Player, policy_size

BOARD = 4
N = BOARD * BOARD
WIDTH = policy_size(BOARD)


def uniform_policy() -> np.ndarray:
    return np.full(WIDTH, 1.0 / WIDTH, dtype=np.float32)


def sample_at(state: State, move_no: int) -> Sample:
    return Sample(
        black=state.black,
        white=state.white,
        to_move=state.to_move,
        pi=uniform_policy(),
        move_no=move_no,
    )


# ===========================================================================
# The target itself
# ===========================================================================


def test_ownership_is_signed_from_the_movers_side_and_sums_to_the_margin() -> None:
    """Black owns squares 0 and 1, white owns square 2, the rest are empty."""
    terminal = State(black=0b011, white=0b100, to_move=Player.BLACK, size=BOARD)

    as_black = ownership_for(terminal, Player.BLACK)
    as_white = ownership_for(terminal, Player.WHITE)

    assert as_black.dtype == np.int8
    assert as_black.tolist()[:3] == [1, 1, -1]
    assert not as_black[3:].any()
    np.testing.assert_array_equal(as_white, -as_black)
    assert int(as_black.sum()) == 2 - 1  # black discs minus white discs


def test_ownership_reads_the_top_bit_of_a_full_board() -> None:
    """Bit 63 on an 8x8 board must not be lost to a signed shift."""
    top = 1 << 63
    terminal = State(black=top, white=0, to_move=Player.BLACK, size=8)
    own = ownership_for(terminal, Player.BLACK)
    assert own.shape == (64,)
    assert own[63] == 1
    assert int(own.sum()) == 1


def test_finish_fills_ownership_for_every_position_from_its_own_movers_view() -> None:
    start = initial_state(BOARD)
    record = GameRecord(samples=[sample_at(start, 0)], board_size=BOARD)
    second = State(black=start.black, white=start.white, to_move=Player.WHITE, size=BOARD)
    record.samples.append(sample_at(second, 1))

    terminal = State(black=0b1111, white=0b110000, to_move=Player.BLACK, size=BOARD)
    record.finish(terminal)

    first, then = record.samples
    assert first.own is not None and then.own is not None
    np.testing.assert_array_equal(first.own, ownership_for(terminal, Player.BLACK))
    np.testing.assert_array_equal(then.own, -first.own)
    # z and the ownership sum agree in sign: both are the result from the mover's view.
    assert np.sign(first.z) == np.sign(first.own.sum())


def test_ownership_is_absent_until_the_game_finishes() -> None:
    assert sample_at(initial_state(BOARD), 0).own is None


# ===========================================================================
# Shards: optional on disk, all or nothing
# ===========================================================================


def finished_samples(n: int) -> list[Sample]:
    record = GameRecord(
        samples=[sample_at(initial_state(BOARD), i) for i in range(n)], board_size=BOARD
    )
    record.finish(State(black=0b1111, white=0b110000, to_move=Player.BLACK, size=BOARD))
    return record.samples


def test_a_shard_carries_ownership_when_every_sample_has_it(tmp_path: Path) -> None:
    arrays = samples_to_arrays(finished_samples(3), generation=1, board_size=BOARD)
    assert arrays["own"].shape == (3, N)
    assert arrays["own"].dtype == np.int8

    info = write_shard(tmp_path / "gen_00001.npz", arrays, board_size=BOARD)
    back = read_shard(tmp_path / "gen_00001.npz", board_size=BOARD, expect_sha256=info.sha256)
    np.testing.assert_array_equal(back["own"], arrays["own"])
    assert all(s.own is not None for s in arrays_to_samples(back, board_size=BOARD))


def test_a_shard_without_ownership_is_still_valid() -> None:
    """Every shard written before the target existed looks like this."""
    unfinished = [sample_at(initial_state(BOARD), i) for i in range(2)]
    arrays = samples_to_arrays(unfinished, generation=1, board_size=BOARD)
    assert "own" not in arrays
    validate_arrays(arrays, board_size=BOARD)
    assert all(s.own is None for s in arrays_to_samples(arrays, board_size=BOARD))


def test_mixing_finished_and_unfinished_samples_in_one_shard_is_refused() -> None:
    mixed = [*finished_samples(1), sample_at(initial_state(BOARD), 5)]
    with pytest.raises(ReplayError, match="all or nothing"):
        samples_to_arrays(mixed, generation=1, board_size=BOARD)


def test_malformed_ownership_is_refused() -> None:
    arrays = samples_to_arrays(finished_samples(2), generation=1, board_size=BOARD)
    bad_value = dict(arrays, own=arrays["own"].copy())
    bad_value["own"][0, 0] = 2
    with pytest.raises(ReplayError, match="ownership targets must be"):
        validate_arrays(bad_value, board_size=BOARD)
    bad_shape = dict(arrays, own=arrays["own"][:, :-1])
    with pytest.raises(ReplayError, match="ownership targets have shape"):
        validate_arrays(bad_shape, board_size=BOARD)


# ===========================================================================
# The replay window
# ===========================================================================


def test_the_window_mixes_shards_with_and_without_ownership() -> None:
    buffer = ReplayBuffer(board_size=BOARD, window=1000, symmetry_aug=False)
    buffer.add(
        samples_to_arrays([sample_at(initial_state(BOARD), 0)], generation=1, board_size=BOARD)
    )
    buffer.add(samples_to_arrays(finished_samples(1), generation=2, board_size=BOARD))

    rng = np.random.default_rng(0)
    seen_valid: set[bool] = set()
    for _ in range(50):
        batch = buffer.sample(1, rng)
        assert batch.own is not None and batch.own_valid is not None
        assert batch.own.shape == (1, N)
        assert batch.own_valid.shape == (1,)
        seen_valid.add(bool(batch.own_valid[0]))
        if not batch.own_valid[0]:
            assert not batch.own.any(), "a row without a target must carry zeros, not garbage"
    assert seen_valid == {True, False}


def test_augmentation_turns_the_ownership_with_the_board() -> None:
    """Turning the board but not the owners would teach that the corner belongs to
    whoever owned a different corner. Same failure the policy test guards against.

    The recorded position is asymmetric on purpose: the opening position is fixed
    by four of the eight symmetries, which would make "which turn was applied"
    ambiguous and the check meaningless.
    """
    position = State(black=0b1011, white=0b0100_0000_0000, to_move=Player.BLACK, size=BOARD)
    terminal = State(black=0b1111, white=0b110000, to_move=Player.BLACK, size=BOARD)
    record = GameRecord(samples=[sample_at(position, 3)], board_size=BOARD)
    record.finish(terminal)

    buffer = ReplayBuffer(board_size=BOARD, window=1000, symmetry_aug=True)
    buffer.add(samples_to_arrays(record.samples, generation=1, board_size=BOARD))

    expected = [
        (
            replay_module.features.encode(symmetry.transform_state(position, sym)),
            ownership_for(symmetry.transform_state(terminal, sym), Player.BLACK).astype(np.float32),
        )
        for sym in symmetry.symmetries(BOARD)
    ]

    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(100):
        batch = buffer.sample(1, rng)
        assert batch.own is not None
        matches = [
            index
            for index, (planes, owners) in enumerate(expected)
            if np.array_equal(batch.planes[0], planes) and np.array_equal(batch.own[0], owners)
        ]
        assert matches, "the sampled board and its owners were not turned together"
        seen.update(matches)
    assert len(seen) > 1, "augmentation should actually vary the orientation"


# ===========================================================================
# The network and the loss
# ===========================================================================


def test_the_ownership_head_is_shaped_like_the_board_and_bounded() -> None:
    model = PolicyValueNet(BOARD, n_blocks=1, channels=8, value_hidden=16, ownership=True).eval()
    x = torch.randn(3, 3, BOARD, BOARD)
    policy, value, own = model.forward_all(x)
    assert own is not None
    assert own.shape == (3, N)
    assert policy.shape == (3, WIDTH) and value.shape == (3, 1)
    assert torch.all(own.abs() <= 1.0)

    # forward stays two outputs, identical to forward_all's first two.
    p2, v2 = model(x)
    torch.testing.assert_close(p2, policy)
    torch.testing.assert_close(v2, value)


def test_a_network_without_the_head_reports_none() -> None:
    model = PolicyValueNet(BOARD, n_blocks=1, channels=8, value_hidden=16)
    _, _, own = model.forward_all(torch.randn(1, 3, BOARD, BOARD))
    assert own is None
    assert model.arch()["ownership"] is False


def test_old_architecture_records_still_describe_the_same_network() -> None:
    """A checkpoint written before the head existed has no ``ownership`` key."""
    model = PolicyValueNet(BOARD, n_blocks=1, channels=8, value_hidden=16)
    old_record = {k: v for k, v in model.arch().items() if k != "ownership"}
    assert normalise_arch(old_record) == normalise_arch(model.arch())
    with_head = PolicyValueNet(BOARD, n_blocks=1, channels=8, value_hidden=16, ownership=True)
    assert normalise_arch(old_record) != normalise_arch(with_head.arch())


def test_the_ownership_term_is_masked_and_absent_when_not_asked_for() -> None:
    logits = torch.zeros(4, WIDTH)
    pi = torch.full((4, WIDTH), 1.0 / WIDTH)
    value = torch.zeros(4, 1)
    z = torch.zeros(4)
    own_pred = torch.zeros(4, N)
    own_target = torch.ones(4, N)  # every valid row is off by exactly 1 on every square
    valid = torch.tensor([True, True, False, False])

    without = policy_value_loss(logits, value, pi, z)
    assert without.ownership is None
    assert "ownership_loss" not in without.as_metrics()

    off = policy_value_loss(
        logits, value, pi, z, ownership_pred=own_pred, own_target=own_target, own_valid=valid
    )
    assert off.ownership is None, "weight 0 means no term, even with a head"
    torch.testing.assert_close(off.total, without.total)

    on = policy_value_loss(
        logits,
        value,
        pi,
        z,
        ownership_pred=own_pred,
        own_target=own_target,
        own_valid=valid,
        ownership_weight=0.5,
    )
    assert on.ownership is not None
    torch.testing.assert_close(on.ownership, torch.tensor(1.0))  # invalid rows did not dilute it
    torch.testing.assert_close(on.total, without.total + 0.5 * 1.0)
    assert on.as_metrics()["ownership_loss"] == pytest.approx(1.0)

    none_valid = policy_value_loss(
        logits,
        value,
        pi,
        z,
        ownership_pred=own_pred,
        own_target=own_target,
        own_valid=torch.zeros(4, dtype=torch.bool),
        ownership_weight=0.5,
    )
    assert none_valid.ownership is not None
    assert float(none_valid.ownership) == 0.0


def test_gradients_reach_the_ownership_head_and_the_trunk_through_it() -> None:
    model = PolicyValueNet(BOARD, n_blocks=1, channels=8, value_hidden=16, ownership=True)
    x = torch.randn(2, 3, BOARD, BOARD)
    policy, value, own = model.forward_all(x)
    assert own is not None
    parts = policy_value_loss(
        policy,
        value,
        torch.full((2, WIDTH), 1.0 / WIDTH),
        torch.zeros(2),
        ownership_pred=own,
        own_target=torch.ones(2, N),
        own_valid=torch.ones(2, dtype=torch.bool),
        ownership_weight=1.0,
    )
    parts.total.backward()
    assert model.ownership_head is not None
    head_grads = [p.grad for p in model.ownership_head.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in head_grads)


# ===========================================================================
# Config
# ===========================================================================


def test_a_weight_without_a_head_is_a_config_error(smoke_config: Config) -> None:
    payload = smoke_config.model_dump()
    payload["train"]["ownership_loss_weight"] = 1.0
    with pytest.raises(ValueError, match="no head to train"):
        Config.model_validate(payload)

    payload["net"]["ownership"] = True
    config = Config.model_validate(payload)
    assert config.net.ownership and config.train.ownership_loss_weight == 1.0
