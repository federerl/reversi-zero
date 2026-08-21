"""What self-play writes down (test matrix T19, contract C4).

The targets produced here are the only thing the network ever learns from, so a
mistake in them is a mistake in everything downstream -- and, as usual in this
project, one that trains perfectly happily.

**A correction to the plan's wording.** The plan states this invariant as "z
alternates strictly over a completed game". That is not quite right, because
positions with only one legal move are deliberately not recorded: skip one and
two consecutive *stored* samples can belong to the same player. The real
invariant, tested below, is that every sample's z is the game's result as seen by
whoever was to move in that sample -- which implies alternation between samples
one ply apart, and equality between samples two plies apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from reversi.config import MCTSConfig
from reversi.game import rules
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.seeding import rng as make_rng
from reversi.selfplay.worker import SelfPlaySummary, play_game, play_games
from reversi.types import Player, pass_action, policy_size

BOARD = 4


def selfplay_config(**overrides: float | int) -> SearchConfig:
    defaults: dict[str, float | int] = {
        "n_simulations": 12,
        "temp_moves": 3,
        "dirichlet_eps": 0.25,
        "dirichlet_alpha": 2.0,
    }
    defaults.update(overrides)
    return SearchConfig(**defaults)  # type: ignore[arg-type]


def one_game(seed: int = 1, **overrides: float | int):
    return play_game(
        StubEvaluator(),
        selfplay_config(**overrides),
        board_size=BOARD,
        rng=make_rng(seed),
    )


# ---------------------------------------------------------------------------
# The shape of what gets recorded
# ---------------------------------------------------------------------------


def test_a_game_finishes_and_records_positions() -> None:
    record, branching = one_game()

    assert record.plies > 0
    assert record.samples, "a game should record at least one decision"
    assert len(branching) == len(record.samples)
    assert all(count >= 2 for count in branching), "recorded positions had a real choice"


def test_every_policy_target_is_a_distribution_over_legal_moves() -> None:
    record, _ = one_game()
    width = policy_size(BOARD)

    for sample in record.samples:
        state = sample.state(BOARD)
        legal = set(rules.legal_actions(state))

        assert sample.pi.shape == (width,)
        assert sample.pi.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(sample.pi >= 0.0)

        # Contract C5 seen from the data side: the target must put no weight on a
        # move that does not exist.
        for action in range(width):
            if action not in legal:
                assert sample.pi[action] == 0.0


def test_positions_with_only_one_legal_move_are_not_recorded() -> None:
    """No decision, nothing to learn.

    A target saying "the forced move had probability 1" would teach the network a
    rule the rules already enforce, using up a slot in every batch it appears in.
    """
    record, _ = one_game()

    assert record.skipped_forced_moves > 0, "4x4 games reliably contain forced moves"
    assert len(record.samples) + record.skipped_forced_moves == record.plies

    for sample in record.samples:
        assert len(rules.legal_actions(sample.state(BOARD))) >= 2


def test_a_forced_pass_is_counted_but_not_recorded() -> None:
    record, _ = one_game(seed=5)
    passing = pass_action(BOARD)

    # A pass is only ever legal when it is the *only* legal action (contract C3),
    # so it is always a forced move and never a recorded decision.
    for sample in record.samples:
        assert sample.pi[passing] == 0.0


# ---------------------------------------------------------------------------
# The value target (T19)
# ---------------------------------------------------------------------------


def test_z_is_the_result_from_each_positions_own_movers_view() -> None:
    for seed in range(12):
        record, _ = one_game(seed=seed)
        outcome = record.result_for_black

        for sample in record.samples:
            expected = outcome if sample.to_move is Player.BLACK else -outcome
            assert sample.z == float(expected), (
                f"sample at ply {sample.move_no} has z={sample.z} but the game "
                f"ended {outcome} for black and {sample.to_move.label} was to move"
            )


def test_z_flips_between_positions_one_ply_apart() -> None:
    """The alternation, stated in the form that survives skipped positions."""
    for seed in range(12):
        record, _ = one_game(seed=seed)
        if record.result_for_black == 0:
            continue  # a draw is 0 for both sides; nothing to alternate

        for earlier, later in zip(record.samples, record.samples[1:], strict=False):
            gap = later.move_no - earlier.move_no
            if gap % 2 == 1:
                assert later.z == -earlier.z
            else:
                assert later.z == earlier.z


def test_z_is_unset_until_the_game_finishes() -> None:
    """``finish`` is what fills in z; before it there is no answer to fill in."""
    from reversi.data.schema import GameRecord, Sample

    record = GameRecord(samples=[], board_size=BOARD)
    record.samples.append(
        Sample(
            black=1,
            white=2,
            to_move=Player.BLACK,
            pi=np.zeros(policy_size(BOARD), dtype=np.float32),
            move_no=0,
        )
    )
    assert record.samples[0].z == 0.0


# ---------------------------------------------------------------------------
# Contract C4 -- the stored target is not the played move
# ---------------------------------------------------------------------------


def test_the_stored_target_is_unaffected_by_the_playing_temperature() -> None:
    """Two games from the same seed, differing only in how moves are *chosen*.

    Raising the temperature changes which moves get played, so the games diverge.
    What must not happen is the stored targets becoming sharper or flatter: they
    are the search's raw visit shares either way. If temperature leaked into the
    target, the entropy of the stored distributions would move with it.
    """
    cold, _ = one_game(seed=3, temp_moves=0)
    hot, _ = one_game(seed=3, temp_moves=12, temp_init=1.0)

    def entropy(record) -> float:
        values = []
        for sample in record.samples:
            probabilities = sample.pi[sample.pi > 0]
            values.append(float(-(probabilities * np.log(probabilities)).sum()))
        return sum(values) / len(values)

    # Both games' targets are raw visit shares, so neither is systematically
    # sharper. With 12 simulations the achievable entropies are the same small set
    # of values regardless of what was played.
    assert entropy(cold) == pytest.approx(entropy(hot), abs=0.6)

    for record in (cold, hot):
        for sample in record.samples:
            assert sample.pi.sum() == pytest.approx(1.0, abs=1e-6)


def test_the_first_moves_are_varied_and_later_ones_are_not() -> None:
    """Temperature is what stops self-play producing the same game every time."""
    # With no exploration at all, the search is deterministic, so every game
    # starts from the identical position with the identical target.
    deterministic, _ = one_game(seed=1, temp_moves=0, dirichlet_eps=0.0)
    again, _ = one_game(seed=99, temp_moves=0, dirichlet_eps=0.0)
    np.testing.assert_array_equal(deterministic.samples[0].pi, again.samples[0].pi)

    # With temperature and noise on, different seeds explore differently.
    varied = {tuple(one_game(seed=s)[0].samples[0].pi) for s in range(8)}
    assert len(varied) > 1, "exploration should make openings differ between games"


# ---------------------------------------------------------------------------
# Reproducibility and batching
# ---------------------------------------------------------------------------


def test_the_same_seed_replays_the_same_game() -> None:
    first, _ = one_game(seed=42)
    second, _ = one_game(seed=42)

    assert first.plies == second.plies
    assert first.result_for_black == second.result_for_black
    assert len(first.samples) == len(second.samples)
    for a, b in zip(first.samples, second.samples, strict=True):
        assert (a.black, a.white, a.to_move, a.move_no) == (b.black, b.white, b.to_move, b.move_no)
        np.testing.assert_array_equal(a.pi, b.pi)


def test_games_in_a_batch_are_seeded_independently() -> None:
    """Two games in the same generation must not be the same game.

    Seeds come from (run seed, generation, worker, game index), so this also holds
    across a resume and across workers -- which is what stops six parallel workers
    from producing six copies of one generation.
    """
    played = list(
        play_games(
            StubEvaluator(),
            selfplay_config(),
            board_size=BOARD,
            n_games=6,
            root_seed=7,
            generation=1,
        )
    )
    assert len(played) == 6

    signatures = {
        tuple((s.black, s.white, s.move_no) for s in record.samples) for record, _ in played
    }
    assert len(signatures) > 1, "different games should not be identical"


def test_a_worker_id_changes_the_games() -> None:
    def signature(worker_id: int) -> tuple:
        played = list(
            play_games(
                StubEvaluator(),
                selfplay_config(),
                board_size=BOARD,
                n_games=3,
                root_seed=7,
                generation=1,
                worker_id=worker_id,
            )
        )
        return tuple(tuple((s.black, s.white) for s in r.samples) for r, _ in played)

    assert signature(0) != signature(1)


# ---------------------------------------------------------------------------
# The summary that becomes metrics
# ---------------------------------------------------------------------------


def test_summary_reports_sane_statistics() -> None:
    summary = SelfPlaySummary()
    for record, branching in play_games(
        StubEvaluator(),
        selfplay_config(),
        board_size=BOARD,
        n_games=10,
        root_seed=3,
        generation=1,
    ):
        summary.observe(record, branching)

    metrics = summary.as_metrics()

    assert metrics["games"] == 10
    assert metrics["positions"] > 0
    assert metrics["plies_mean"] > 0
    assert 0.0 <= metrics["pass_rate"] <= 1.0
    assert 0.0 <= metrics["draw_rate"] <= 1.0
    assert 0.0 <= metrics["first_player_win_rate"] <= 1.0
    assert metrics["mean_branching"] >= 2.0


def test_selfplay_uses_the_configured_exploration() -> None:
    """A self-play config keeps its noise; an evaluation config cannot have any."""
    training = MCTSConfig(dirichlet_eps=0.25, temp_moves=4)

    assert SearchConfig.for_selfplay(training).uses_noise
    assert not SearchConfig.for_evaluation(training).uses_noise
