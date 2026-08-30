"""Fair matches and honest reports (test matrix T27, T29; contract S13).

Nothing here is about who wins. It is about whether a result deserves to be
believed: were the colours even, did the games actually differ from each other,
and is enough written down to reconstruct the claim later.

Each of these would bias a number rather than break it, which is the reason they
are checked in code rather than trusted to a checklist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reversi.agents import GreedyAgent, MinimaxAgent, RandomAgent
from reversi.arena import MatchReport, check_fairness, play_match, write_report
from reversi.arena.openings import apply_opening, random_openings
from reversi.errors import ArenaError
from reversi.game import rules
from reversi.game.bitboard import popcount

BOARD = 8


# ===========================================================================
# The opening book
# ===========================================================================


def test_openings_are_reproducible_from_their_seed() -> None:
    """Two runs of the same tournament must use the same openings, or their
    results cannot be compared at all."""
    first = random_openings(count=10, board_size=BOARD, seed=42, plies=4)
    again = random_openings(count=10, board_size=BOARD, seed=42, plies=4)
    other = random_openings(count=10, board_size=BOARD, seed=43, plies=4)

    assert first == again
    assert first != other


def test_openings_are_distinct() -> None:
    """A book with the same line twice is a book with fewer lines than it claims."""
    book = random_openings(count=30, board_size=BOARD, seed=1, plies=4)
    assert len(set(book)) == 30


def test_openings_have_the_length_asked_for() -> None:
    for plies in (2, 4, 6):
        book = random_openings(count=5, board_size=BOARD, seed=2, plies=plies)
        assert all(len(line) == plies for line in book)


def test_both_sides_still_have_choices_after_the_opening() -> None:
    """A position where one side is already forced is not a fair place to start.

    Colour swapping cancels a first-move advantage; it cannot rescue a position
    that is simply lopsided.
    """
    for line in random_openings(count=20, board_size=BOARD, seed=3, plies=4):
        state = apply_opening(BOARD, line)

        assert not rules.is_terminal(state)
        assert len(rules.legal_actions(state)) >= 2
        theirs = rules.legal_placements_for(state, state.to_move.opponent)
        assert popcount(theirs) >= 2


def test_every_opening_is_reachable_by_legal_play() -> None:
    """They are built by playing legal moves, so replaying one must never raise."""
    for line in random_openings(count=10, board_size=BOARD, seed=4, plies=4):
        state = rules.initial_state(BOARD)
        for action in line:
            assert action in rules.legal_actions(state)
            state = rules.apply(state, action)


def test_asking_for_impossible_openings_says_so() -> None:
    """A 4x4 board has few enough distinct lines that this is reachable.

    Failing loudly beats looping forever, and beats silently returning fewer
    openings than were asked for.

    Only 12 usable two-ply openings exist on 4x4, so 60 is already impossible.
    Asking for more does not make the test stronger, only slower: the generator
    tries `count * 100` times before giving up, so a large `count` spends
    hundreds of thousands of attempts re-proving what 6,000 already showed.
    """
    with pytest.raises(ArenaError, match="only found"):
        random_openings(count=60, board_size=4, seed=1, plies=2)


def test_nonsense_requests_are_refused() -> None:
    with pytest.raises(ArenaError, match="at least one opening"):
        random_openings(count=0, board_size=BOARD, seed=1)
    with pytest.raises(ArenaError, match="cannot be negative"):
        random_openings(count=2, board_size=BOARD, seed=1, plies=-1)


# ===========================================================================
# Fair matches (T29, S13)
# ===========================================================================


def test_a_match_uses_each_opening_exactly_twice() -> None:
    """Once with each agent as black -- that is what cancels the first-move edge."""
    result = play_match(
        RandomAgent("a"), RandomAgent("b"), games=20, board_size=BOARD, seed=1, opening_plies=4
    )

    assert result.openings_used == 10
    assert result.openings_used * 2 == result.games
    assert result.opening_plies == 4


class Recorder:
    """Wraps an agent and remembers every position it was asked about."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: set[tuple[int, int, int]] = set()

    @property
    def name(self) -> str:
        return f"rec-{self.inner.name}"

    def select(self, state, rng):
        self.seen.add((state.black, state.white, int(state.to_move)))
        return self.inner.select(state, rng)


def test_the_book_makes_deterministic_agents_play_different_games() -> None:
    """The whole reason the book exists.

    Two agents with no randomness play the *identical* game from the standard
    start, so an 8-game match without a book is one game counted eight times.

    Note what is measured: how many distinct *positions* occur, not how the games
    came out. Varied openings need not produce varied results -- the stronger
    agent may well win all of them -- so counting wins would test the wrong thing.
    """

    def distinct_positions(*, opening_plies: int) -> int:
        recorder = Recorder(MinimaxAgent(2, name="d2"))
        play_match(
            recorder,
            MinimaxAgent(1, name="d1"),
            games=8,
            board_size=BOARD,
            seed=5,
            opening_plies=opening_plies,
        )
        return len(recorder.seen)

    without = distinct_positions(opening_plies=0)
    with_book = distinct_positions(opening_plies=4)

    assert with_book > without * 2, (
        f"the book produced {with_book} distinct positions against {without} "
        "without it -- it is not adding variety"
    )


def test_a_match_is_reproducible_including_its_book() -> None:
    args = {"games": 10, "board_size": BOARD, "seed": 9, "opening_plies": 4}
    first = play_match(GreedyAgent(), RandomAgent(), **args)
    again = play_match(GreedyAgent(), RandomAgent(), **args)

    assert (first.wins, first.losses, first.draws) == (again.wins, again.losses, again.draws)
    assert first.mean_plies == again.mean_plies


# ===========================================================================
# The fairness checks
# ===========================================================================


def fair_result():
    return play_match(
        RandomAgent("a"), RandomAgent("b"), games=10, board_size=BOARD, seed=2, opening_plies=4
    )


def test_a_fair_match_passes() -> None:
    check_fairness(fair_result())


def test_a_match_without_a_book_is_refused_by_default() -> None:
    """Because against deterministic agents it is one game repeated."""
    result = play_match(RandomAgent("a"), RandomAgent("b"), games=10, board_size=BOARD, seed=2)

    with pytest.raises(ArenaError, match="standard position"):
        check_fairness(result)

    # ...but two genuinely random agents may opt out and say why.
    check_fairness(result, require_openings=False)


def test_a_miscounted_match_is_refused() -> None:
    """Totals that do not add up mean games were lost somewhere in the counting."""
    from dataclasses import replace

    broken = replace(fair_result(), wins=99)
    with pytest.raises(ArenaError, match="sums to"):
        check_fairness(broken)


def test_a_colour_imbalance_is_refused() -> None:
    from dataclasses import replace

    result = fair_result()
    broken = replace(result, wins_as_black=result.wins + 1, wins_as_white=0)
    with pytest.raises(ArenaError, match="by colour do not add up"):
        check_fairness(broken)


def test_a_book_that_does_not_cover_the_games_is_refused() -> None:
    from dataclasses import replace

    broken = replace(fair_result(), openings_used=3)
    with pytest.raises(ArenaError, match="exactly twice"):
        check_fairness(broken)


# ===========================================================================
# Reports
# ===========================================================================


def test_a_report_records_everything_needed_to_judge_it(tmp_path: Path) -> None:
    """The question asked later is not "what was the number" but "can I trust it"."""
    result = fair_result()
    report = MatchReport(
        result=result,
        board_size=BOARD,
        agent_specs={"a": {"kind": "random"}, "b": {"kind": "random"}},
        notes="a test",
    )

    path = tmp_path / "report.json"
    payload = write_report(path, report)
    on_disk = json.loads(path.read_text(encoding="utf-8"))

    assert on_disk == payload
    assert on_disk["games"] == 10
    assert on_disk["protocol"]["seed"] == 2
    assert on_disk["protocol"]["opening_plies"] == 4
    assert on_disk["protocol"]["openings_used"] == 5
    assert on_disk["by_colour"]["a_as_black"]["games"] == 5
    assert on_disk["by_colour"]["a_as_white"]["games"] == 5
    assert "git" in on_disk["provenance"]
    assert on_disk["agent_specs"]["a"]["kind"] == "random"


def test_an_unfair_match_is_never_written(tmp_path: Path) -> None:
    """The check runs before the file is created, so a bad result leaves no
    artifact that could later be mistaken for a real one."""
    result = play_match(RandomAgent("a"), RandomAgent("b"), games=10, board_size=BOARD, seed=2)
    path = tmp_path / "report.json"

    with pytest.raises(ArenaError):
        write_report(path, MatchReport(result=result, board_size=BOARD))

    assert not path.exists()


def test_the_report_names_which_baseline_weights_were_used(tmp_path: Path) -> None:
    """A rating is relative to its yardstick. Two reports disagreeing about a
    baseline's strength usually turn out to have used different weights."""
    from reversi.agents.minimax import weights_fingerprint

    result = play_match(
        MinimaxAgent(1, name="d1"),
        RandomAgent(),
        games=4,
        board_size=BOARD,
        seed=1,
        opening_plies=4,
    )
    report = MatchReport(
        result=result,
        board_size=BOARD,
        agent_specs={
            "a": {"kind": "minimax", "depth": 1, "weights": weights_fingerprint(BOARD)},
            "b": {"kind": "random"},
        },
    )
    payload = write_report(tmp_path / "r.json", report)

    assert payload["agent_specs"]["a"]["weights"] == weights_fingerprint(BOARD)
