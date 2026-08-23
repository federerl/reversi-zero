"""Storing games and sampling them back (test matrix T20, T21).

A shard is the only artifact in this project that outlives the process that made
it, so the tests here lean on the two things that make that safe: everything is
validated on the way in and on the way out, and everything is checksummed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reversi.data.replay import ReplayBuffer
from reversi.data.schema import Sample, arrays_to_samples, samples_to_arrays, validate_arrays
from reversi.data.shards import Manifest, read_shard, shard_filename, write_shard
from reversi.errors import ReplayError
from reversi.game import rules, symmetry
from reversi.nn import features
from reversi.types import Player, policy_size

BOARD = 4
WIDTH = policy_size(BOARD)


def make_samples(count: int, *, seed: int = 0, z: float = 1.0) -> list[Sample]:
    """Synthetic but *valid* samples: real positions, real distributions."""
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    state = rules.initial_state(BOARD)

    for index in range(count):
        if rules.is_terminal(state):
            state = rules.initial_state(BOARD)
        actions = rules.legal_actions(state)

        pi = np.zeros(WIDTH, dtype=np.float32)
        weights = rng.random(len(actions)).astype(np.float32) + 0.1
        weights /= weights.sum()
        for action, weight in zip(actions, weights, strict=True):
            pi[action] = weight

        out.append(
            Sample(
                black=state.black,
                white=state.white,
                to_move=state.to_move,
                pi=pi,
                move_no=index % 12,
                z=z if index % 2 == 0 else -z,
            )
        )
        state = rules.apply(state, actions[int(rng.integers(0, len(actions)))])

    return out


def arrays_for(count: int, *, generation: int = 1, seed: int = 0) -> dict:
    return samples_to_arrays(
        make_samples(count, seed=seed), generation=generation, board_size=BOARD
    )


# ===========================================================================
# The record format
# ===========================================================================


def test_samples_survive_a_round_trip_through_arrays() -> None:
    original = make_samples(50, seed=1)
    arrays = samples_to_arrays(original, generation=4, board_size=BOARD)
    restored = arrays_to_samples(arrays, board_size=BOARD)

    assert len(restored) == len(original)
    for before, after in zip(original, restored, strict=True):
        assert (before.black, before.white, before.to_move) == (
            after.black,
            after.white,
            after.to_move,
        )
        assert before.move_no == after.move_no
        assert before.z == after.z
        np.testing.assert_allclose(before.pi, after.pi, atol=1e-6)
    assert set(arrays["generation"]) == {4}


def test_bitboards_survive_the_full_64_bit_range() -> None:
    """uint64 is exact only if nothing quietly routes through a float on the way."""
    samples = [
        Sample(
            black=(1 << 63) | 1,
            white=(1 << 62),
            to_move=Player.WHITE,
            pi=np.eye(1, WIDTH, 0, dtype=np.float32)[0],
            move_no=3,
            z=-1.0,
        )
    ]
    arrays = samples_to_arrays(samples, generation=1, board_size=BOARD)
    restored = arrays_to_samples(arrays, board_size=BOARD)

    assert restored[0].black == (1 << 63) | 1
    assert restored[0].white == 1 << 62


def test_a_policy_target_that_is_not_a_distribution_is_refused() -> None:
    arrays = arrays_for(10)
    arrays["pi"][3] *= 0.5  # now sums to 0.5

    with pytest.raises(ReplayError, match="sums to"):
        validate_arrays(arrays, board_size=BOARD)


def test_overlapping_bitboards_are_refused() -> None:
    arrays = arrays_for(10)
    arrays["white"] = arrays["black"].copy()

    with pytest.raises(ReplayError, match="same square for both"):
        validate_arrays(arrays, board_size=BOARD)


def test_an_impossible_result_is_refused() -> None:
    arrays = arrays_for(10)
    arrays["z"][0] = 0.5

    with pytest.raises(ReplayError, match="z must be"):
        validate_arrays(arrays, board_size=BOARD)


def test_a_missing_field_is_refused() -> None:
    arrays = arrays_for(10)
    del arrays["move_no"]

    with pytest.raises(ReplayError, match="missing"):
        validate_arrays(arrays, board_size=BOARD)


def test_the_wrong_policy_width_is_refused() -> None:
    samples = make_samples(3)
    bad = Sample(
        black=samples[0].black,
        white=samples[0].white,
        to_move=samples[0].to_move,
        pi=np.full(WIDTH + 1, 1.0 / (WIDTH + 1), dtype=np.float32),
        move_no=0,
        z=1.0,
    )
    with pytest.raises(ReplayError, match="expected"):
        samples_to_arrays([bad], generation=1, board_size=BOARD)


def test_an_empty_shard_is_refused() -> None:
    with pytest.raises(ReplayError, match="zero samples"):
        samples_to_arrays([], generation=1, board_size=BOARD)


# ===========================================================================
# Shards on disk (T20)
# ===========================================================================


def test_a_shard_round_trips_exactly(tmp_path: Path) -> None:
    arrays = arrays_for(2000, seed=2)
    path = tmp_path / shard_filename(3)

    info = write_shard(path, arrays, board_size=BOARD)
    reloaded = read_shard(path, board_size=BOARD, expect_sha256=info.sha256)

    assert info.n_positions == 2000
    assert info.generation == 1
    assert info.filename == "gen_00003_w00.npz"
    for field, values in arrays.items():
        np.testing.assert_array_equal(reloaded[field], values)


def test_a_corrupted_shard_is_caught_by_its_checksum(tmp_path: Path) -> None:
    """The check that matters. numpy will happily load a damaged file.

    Without the checksum, corruption arrives as slightly wrong training data
    rather than as an error -- and nothing downstream could tell the difference.
    """
    path = tmp_path / shard_filename(1)
    info = write_shard(path, arrays_for(100), board_size=BOARD)

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(ReplayError, match="does not match its manifest checksum"):
        read_shard(path, board_size=BOARD, expect_sha256=info.sha256)


def test_a_truncated_shard_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / shard_filename(1)
    write_shard(path, arrays_for(100), board_size=BOARD)
    path.write_bytes(path.read_bytes()[:200])

    with pytest.raises(ReplayError):
        read_shard(path, board_size=BOARD)


def test_a_missing_shard_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="is missing"):
        read_shard(tmp_path / "nope.npz", board_size=BOARD)


def test_writing_a_shard_leaves_no_temporary_files(tmp_path: Path) -> None:
    """The atomic write must not leave the real data under a temporary name.

    numpy's savez appends '.npz' to any path lacking it, which would have written
    the data beside the temporary file and then renamed an empty one into place.
    """
    path = tmp_path / shard_filename(1)
    write_shard(path, arrays_for(20), board_size=BOARD)

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    assert path.stat().st_size > 0


# ===========================================================================
# The manifest
# ===========================================================================


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path)
    assert manifest.shards == []

    for generation in (1, 2, 3):
        path = tmp_path / shard_filename(generation)
        manifest.add(write_shard(path, arrays_for(30, generation=generation), board_size=BOARD))

    reloaded = Manifest.load(tmp_path)
    assert [s.filename for s in reloaded.shards] == [shard_filename(g) for g in (1, 2, 3)]
    assert reloaded.total_positions == 90


def test_the_manifest_drops_shards_that_no_longer_match(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path)
    for generation in (1, 2):
        path = tmp_path / shard_filename(generation)
        manifest.add(write_shard(path, arrays_for(30, generation=generation), board_size=BOARD))

    (tmp_path / shard_filename(1)).write_bytes(b"not a shard any more")

    dropped = manifest.verify()

    assert [s.filename for s in dropped] == [shard_filename(1)]
    assert [s.filename for s in manifest.shards] == [shard_filename(2)]
    assert Manifest.load(tmp_path).total_positions == 30


def test_pruning_deletes_the_oldest_files(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path)
    for generation in range(1, 6):
        path = tmp_path / shard_filename(generation)
        manifest.add(write_shard(path, arrays_for(10, generation=generation), board_size=BOARD))

    removed = manifest.prune(keep=2)

    assert len(removed) == 3
    assert not (tmp_path / shard_filename(1)).exists()
    assert (tmp_path / shard_filename(5)).exists()
    assert [s.generation for s in manifest.shards] == [4, 5]


def test_an_unreadable_manifest_is_an_error_not_a_fresh_start(tmp_path: Path) -> None:
    """Silently starting over would throw away a run's whole history."""
    (tmp_path / "manifest.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ReplayError, match="unreadable"):
        Manifest.load(tmp_path)


# ===========================================================================
# The sliding window (T21)
# ===========================================================================


def test_the_window_keeps_the_newest_positions() -> None:
    buffer = ReplayBuffer(board_size=BOARD, window=100, symmetry_aug=False)

    for generation in (1, 2, 3):
        buffer.add(arrays_for(60, generation=generation, seed=generation))

    assert len(buffer) == 100
    # 180 positions arrived, the window holds 100, so generation 1 is entirely
    # gone and generation 2 is partly gone.
    assert buffer.generations == [2, 3]


def test_the_window_rebuilds_from_disk(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path)
    for generation in (1, 2, 3):
        manifest.add(
            write_shard(
                tmp_path / shard_filename(generation),
                arrays_for(40, generation=generation, seed=generation),
                board_size=BOARD,
            )
        )

    buffer = ReplayBuffer(board_size=BOARD, window=100, symmetry_aug=False)
    recovered = buffer.load_from(manifest)

    assert recovered == 100
    # 120 positions exist and the window holds 100, so the newest 100 are kept:
    # all of generations 2 and 3, plus the tail of generation 1.
    assert buffer.generations == [1, 2, 3]


def test_rebuilding_only_reads_the_shards_it_needs(tmp_path: Path) -> None:
    """A window of one generation must not read thirty files to fill itself."""
    manifest = Manifest.load(tmp_path)
    for generation in range(1, 11):
        manifest.add(
            write_shard(
                tmp_path / shard_filename(generation),
                arrays_for(20, generation=generation, seed=generation),
                board_size=BOARD,
            )
        )
    # Delete an old shard's *contents* without telling the manifest: if loading
    # touched it, the checksum check would fire.
    (tmp_path / shard_filename(1)).write_bytes(b"corrupt")

    buffer = ReplayBuffer(board_size=BOARD, window=40, symmetry_aug=False)
    assert buffer.load_from(manifest) == 40
    assert buffer.generations == [9, 10]


def test_sampling_draws_from_generations_in_proportion() -> None:
    """Equal-sized generations should each supply about a fair share of a batch."""
    buffer = ReplayBuffer(board_size=BOARD, window=10_000, symmetry_aug=False)
    for generation in (1, 2, 3, 4):
        buffer.add(arrays_for(250, generation=generation, seed=generation))

    rng = np.random.default_rng(0)
    counts = dict.fromkeys(buffer.generations, 0)
    draws = 4000
    for _ in range(draws // 100):
        indices = buffer._choose_indices(100, rng)
        for index in indices:
            counts[int(buffer._columns["generation"][index])] += 1

    expected = draws / 4
    # Three standard deviations of a binomial with p=1/4.
    tolerance = 3 * (draws * 0.25 * 0.75) ** 0.5
    for generation, count in counts.items():
        assert abs(count - expected) < tolerance, f"generation {generation} drew {count}"


def test_one_oversized_generation_cannot_dominate_a_batch() -> None:
    """The cap: no generation may exceed its fair share by more than the factor.

    Without it, a generation that happened to produce four times as many
    positions as the others would supply four times as much of every batch --
    turning the window into a snapshot of one lucky afternoon.
    """
    buffer = ReplayBuffer(
        board_size=BOARD, window=10_000, per_gen_cap_factor=1.5, symmetry_aug=False
    )
    buffer.add(arrays_for(100, generation=1, seed=1))
    buffer.add(arrays_for(100, generation=2, seed=2))
    buffer.add(arrays_for(2000, generation=3, seed=3))  # ten times the others

    rng = np.random.default_rng(1)
    draws = 4000
    counts = dict.fromkeys(buffer.generations, 0)
    for _ in range(draws // 200):
        for index in buffer._choose_indices(200, rng):
            counts[int(buffer._columns["generation"][index])] += 1

    share = counts[3] / draws
    fair = 1 / 3
    assert share <= 1.5 * fair + 0.03, f"generation 3 took {share:.2%} of the batches"
    assert share > fair * 0.9, "the cap should not push it below a fair share either"


def test_a_batch_is_ready_for_the_network() -> None:
    buffer = ReplayBuffer(board_size=BOARD, window=1000)
    buffer.add(arrays_for(200, generation=1))

    batch = buffer.sample(32, np.random.default_rng(0))

    assert len(batch) == 32
    assert batch.planes.shape == (32, features.IN_PLANES, BOARD, BOARD)
    assert batch.planes.dtype == np.float32
    assert batch.pi.shape == (32, WIDTH)
    assert batch.z.shape == (32,)
    np.testing.assert_allclose(batch.pi.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(batch.z)) <= {-1.0, 0.0, 1.0}


def test_sampling_an_empty_buffer_is_an_error() -> None:
    with pytest.raises(ReplayError, match="empty"):
        ReplayBuffer(board_size=BOARD, window=10).sample(4, np.random.default_rng(0))


# ===========================================================================
# Sample-time augmentation
# ===========================================================================


def test_without_augmentation_a_sample_matches_what_was_stored() -> None:
    buffer = ReplayBuffer(board_size=BOARD, window=1000, symmetry_aug=False)
    samples = make_samples(1, seed=4)
    buffer.add(samples_to_arrays(samples, generation=1, board_size=BOARD))

    batch = buffer.sample(1, np.random.default_rng(0))

    np.testing.assert_allclose(batch.pi[0], samples[0].pi, atol=1e-6)
    np.testing.assert_array_equal(batch.planes[0], features.encode(samples[0].state(BOARD)))


def test_augmentation_turns_the_board_and_the_target_together() -> None:
    """The whole point: an orientation is only free if the answer turns with it.

    Turning the board but not the policy would teach the network that the good
    move is wherever the *un-turned* good move used to be -- which is worse than
    no augmentation at all.
    """
    buffer = ReplayBuffer(board_size=BOARD, window=1000, symmetry_aug=True)
    samples = make_samples(1, seed=6)
    buffer.add(samples_to_arrays(samples, generation=1, board_size=BOARD))

    original = samples[0]
    candidates = [
        (
            features.encode(symmetry.transform_state(original.state(BOARD), sym)),
            np.asarray(symmetry.transform_policy(original.pi.tolist(), sym), dtype=np.float32),
        )
        for sym in symmetry.symmetries(BOARD)
    ]

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        batch = buffer.sample(1, rng)
        matches = [
            index
            for index, (planes, pi) in enumerate(candidates)
            if np.array_equal(batch.planes[0], planes) and np.allclose(batch.pi[0], pi, atol=1e-6)
        ]
        assert matches, "a sampled batch was not any of the eight orientations"
        seen.update(matches)

    assert len(seen) > 1, "augmentation should actually vary the orientation"


def test_the_fast_policy_permutation_matches_the_reference_one() -> None:
    """The buffer reorders policies with one array index for speed.

    ``symmetry.transform_policy`` is the readable version. They must agree, or
    every augmented target in the project is subtly scrambled.
    """
    buffer = ReplayBuffer(board_size=BOARD, window=10)
    rng = np.random.default_rng(0)
    policy = rng.random(WIDTH).astype(np.float32)
    policy /= policy.sum()

    for index, sym in enumerate(symmetry.symmetries(BOARD)):
        fast = policy[buffer._policy_gather[index]]
        readable = np.asarray(symmetry.transform_policy(policy.tolist(), sym), dtype=np.float32)
        np.testing.assert_allclose(fast, readable, atol=1e-6)


def test_augmentation_never_moves_the_pass_entry() -> None:
    """PASS is not a square, so no rotation can move it (contract C6)."""
    buffer = ReplayBuffer(board_size=BOARD, window=10)
    passing = BOARD * BOARD
    for gather in buffer._policy_gather:
        assert gather[passing] == passing


# ===========================================================================
# Metrics
# ===========================================================================


def test_buffer_stats_describe_staleness_and_variety() -> None:
    buffer = ReplayBuffer(board_size=BOARD, window=10_000, symmetry_aug=False)
    buffer.add(arrays_for(100, generation=1, seed=1))
    buffer.add(arrays_for(100, generation=3, seed=2))

    stats = buffer.stats(current_generation=3)

    assert stats["buffer_size"] == 200
    assert stats["buffer_generations"] == 2
    assert stats["buffer_age_mean"] == pytest.approx(1.0)
    assert 0.0 < stats["unique_positions_fraction"] <= 1.0


def test_stats_on_an_empty_buffer_do_not_explode() -> None:
    assert ReplayBuffer(board_size=BOARD, window=10).stats(0) == {"buffer_size": 0}
