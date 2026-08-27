"""Batched self-play (test matrix T22).

The claim this file exists to defend: **batching changes only the order the work
happens in, never the games.** Everything else here is secondary.

That claim is worth testing rather than reasoning about, because the failure mode
is quiet. A batched scheduler that subtly diverged -- an off-by-one in the
simulation count, exploration noise drawn at the wrong moment, a stale root -- would
still produce legal moves and plausible-looking training data. The agent would
just learn from something slightly different from what the search actually
concluded, and nothing would say so.
"""

from __future__ import annotations

import pytest

from reversi.errors import WorkerError
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.selfplay.game_batch import BatchedSelfPlay
from reversi.selfplay.worker import play_games

BOARD = 4


def fingerprint(records) -> dict:
    """Everything about a game that could possibly differ, keyed by game index.

    Keyed rather than ordered on purpose: batched games finish when they finish,
    so the *order* legitimately differs. The games themselves must not.
    """
    out = {}
    for record, branching in records:
        out[record.game_index] = (
            record.plies,
            record.passes,
            record.skipped_forced_moves,
            record.result_for_black,
            tuple(branching),
            tuple(
                (s.black, s.white, int(s.to_move), s.move_no, s.z, tuple(s.pi))
                for s in record.samples
            ),
        )
    return out


def solo(config: SearchConfig, games: int = 8):
    return play_games(
        StubEvaluator(), config, board_size=BOARD, n_games=games, root_seed=7, generation=1
    )


def batched(config: SearchConfig, games: int = 8, in_flight: int = 5) -> BatchedSelfPlay:
    return BatchedSelfPlay(
        StubEvaluator(),
        config,
        board_size=BOARD,
        games_in_flight=in_flight,
        root_seed=7,
        generation=1,
    )


# ===========================================================================
# The property that matters
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("deterministic", SearchConfig(n_simulations=12)),
        ("with noise", SearchConfig(n_simulations=12, dirichlet_eps=0.25, dirichlet_alpha=2.0)),
        ("with temperature", SearchConfig(n_simulations=12, temp_moves=3)),
        (
            "with both",
            SearchConfig(n_simulations=12, temp_moves=3, dirichlet_eps=0.25, dirichlet_alpha=2.0),
        ),
    ],
)
def test_batching_produces_exactly_the_same_games(name: str, config: SearchConfig) -> None:
    """Position for position, target for target, in every configuration.

    The noise and temperature cases are the ones with teeth: both draw from the
    per-game random generator, so a scheduler that interleaved the draws
    differently would diverge here and nowhere else.
    """
    one_at_a_time = fingerprint(solo(config))
    together = fingerprint(batched(config).play(8))

    assert set(one_at_a_time) == set(together) == set(range(8))
    assert one_at_a_time == together, f"batched play diverged ({name})"


@pytest.mark.parametrize("in_flight", [1, 2, 3, 8, 16])
def test_the_batch_width_does_not_change_the_games(in_flight: int) -> None:
    """Including widths larger than the number of games, and a width of one."""
    config = SearchConfig(n_simulations=10, temp_moves=2, dirichlet_eps=0.25, dirichlet_alpha=2.0)
    expected = fingerprint(solo(config, games=6))

    actual = fingerprint(batched(config, in_flight=in_flight).play(6))

    assert actual == expected


def test_games_may_finish_out_of_order() -> None:
    """Documenting the one thing batching *does* change.

    Short games finish first, so records arrive in completion order. Anything
    that needs to know which game it is reads ``game_index``.
    """
    config = SearchConfig(n_simulations=8, temp_moves=3, dirichlet_eps=0.25, dirichlet_alpha=2.0)
    order = [record.game_index for record, _ in batched(config, in_flight=8).play(8)]

    assert sorted(order) == list(range(8))


# ===========================================================================
# That the batching is actually happening
# ===========================================================================


def test_the_batches_are_nearly_full() -> None:
    """A scheduler that quietly submitted one position at a time would pass every
    correctness test above while being no faster than not batching at all."""
    config = SearchConfig(n_simulations=16)
    play = batched(config, in_flight=8)
    list(play.play(24))

    assert play.rounds > 0
    assert play.mean_batch > 6.0, f"mean batch was only {play.mean_batch:.1f} of 8"


def test_finished_positions_never_enter_the_batch() -> None:
    """We know their value exactly; asking the network would be slower and worse.

    On 4x4 games end quickly, so terminal leaves are common -- if they were being
    sent to the network, the evaluated count would exceed the rounds by more than
    the batch width allows.
    """
    config = SearchConfig(n_simulations=16)
    stub = StubEvaluator()
    play = BatchedSelfPlay(
        stub, config, board_size=BOARD, games_in_flight=8, root_seed=3, generation=1
    )
    list(play.play(16))

    assert stub.positions == play.positions_evaluated
    assert stub.calls == play.rounds


def test_a_width_of_one_still_works() -> None:
    config = SearchConfig(n_simulations=8)
    play = batched(config, in_flight=1)
    records = list(play.play(3))

    assert len(records) == 3
    assert play.mean_batch == pytest.approx(1.0)


def test_zero_games_produces_nothing() -> None:
    play = batched(SearchConfig(n_simulations=4))
    assert list(play.play(0)) == []


def test_a_nonsense_width_is_refused() -> None:
    with pytest.raises(WorkerError, match="at least 1"):
        BatchedSelfPlay(
            StubEvaluator(),
            SearchConfig(n_simulations=4),
            board_size=BOARD,
            games_in_flight=0,
            root_seed=1,
            generation=1,
        )


def test_every_game_asked_for_is_produced() -> None:
    play = batched(SearchConfig(n_simulations=6), in_flight=4)
    records = list(play.play(17))

    assert len(records) == 17
    assert sorted(r.game_index for r, _ in records) == list(range(17))
