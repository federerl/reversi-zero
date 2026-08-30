"""The fixtures a TypeScript port will be judged against must be right first.

These tests are a layer below the browser tests. Those check that TypeScript
reproduces the fixtures; these check that the fixtures describe the frozen
engine. A fixture generated from a bug would make the port agree with the bug.

The other thing checked here is *portability*: the stub evaluator's hash has to
be reproducible in JavaScript, which constrains it to 32-bit integer arithmetic.
The recorded test vectors are what the TypeScript side asserts against, so a
change to the hash breaks loudly on both sides rather than quietly on one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reversi.game import rules, scoring
from reversi.types import pass_action
from reversi.web.fixtures import (
    FIXTURE_VERSION,
    _mix32,
    encoding_fixture,
    games_fixture,
    rules_fixture,
    sample_positions,
    search_fixture,
    state_of,
    stub_logit,
    stub_seed,
    stub_value,
    write_fixtures,
)

BOARD = 8
SEED = 12345


# ===========================================================================
# The fixtures describe the engine that generated them
# ===========================================================================


def test_the_rules_fixture_matches_the_engine() -> None:
    """Every recorded legal set, flip mask, terminal flag and score, re-derived."""
    fixture = rules_fixture(count=120, board_size=BOARD, seed=SEED)

    assert fixture["version"] == FIXTURE_VERSION
    assert fixture["pass_action"] == pass_action(BOARD)

    for black, white, to_move, legal, flips, terminal, black_discs, white_discs in fixture["cases"]:
        state = state_of(black, white, to_move, BOARD)

        assert list(rules.legal_actions(state)) == legal
        assert int(rules.is_terminal(state)) == terminal
        assert list(scoring.disc_counts(state)) == [black_discs, white_discs]
        for action, mask in flips:
            assert rules.flips(state, action) == int(mask, 16)


def test_the_rules_fixture_records_a_flip_mask_for_every_placement() -> None:
    """A missing entry is a case the port never gets checked on.

    PASS is the one action with no flips, and it is excluded rather than recorded
    as zero -- so "no entry" means PASS, not "we forgot".
    """
    fixture = rules_fixture(count=200, board_size=BOARD, seed=SEED)
    passing = pass_action(BOARD)

    for _, _, _, legal, flips, _, _, _ in fixture["cases"]:
        placements = [a for a in legal if a != passing]
        assert [a for a, _ in flips] == placements


def test_the_rules_fixture_reaches_the_awkward_positions() -> None:
    """Coverage claim, asserted rather than assumed.

    A fixture of only comfortable mid-game positions would pass a port that gets
    passing or termination wrong, which are two of the three things it is for.
    """
    fixture = rules_fixture(count=400, board_size=BOARD, seed=SEED)
    passing = pass_action(BOARD)

    terminal = sum(1 for case in fixture["cases"] if case[5])
    forced_pass = sum(1 for case in fixture["cases"] if case[3] == [passing])

    assert terminal >= 1, "no terminal position in the fixture"
    assert forced_pass >= 1, "no forced-pass position in the fixture"


def test_the_encoding_fixture_puts_each_disc_on_the_right_square() -> None:
    """Contract C1: bit i is row i // size, column i % size, from the mover's view.

    Stored as bitmasks so the comparison is exact per square. A transposed board
    is the failure being guarded against, and it survives any check that only
    compares totals.
    """
    fixture = encoding_fixture(count=60, board_size=BOARD, seed=SEED)

    for black, white, to_move, planes in fixture["cases"]:
        state = state_of(black, white, to_move, BOARD)
        mine, theirs, legal = (int(p, 16) for p in planes)

        assert mine == state.mine
        assert theirs == state.theirs
        assert legal == rules.legal_placements(state)


def test_the_search_fixture_never_visits_an_illegal_action() -> None:
    """Contract C5, restated where a port could break it.

    The search creates edges only for legal actions, so this cannot fail in
    Python. Recording it means the TypeScript port has to earn the same
    guarantee rather than inherit it.
    """
    fixture = search_fixture(count=12, board_size=BOARD, seed=SEED, simulations=32)
    assert fixture["cases"], "the search fixture is empty"

    for black, white, to_move, actions, visits, _, best in fixture["cases"]:
        state = state_of(black, white, to_move, BOARD)
        legal = set(rules.legal_actions(state))

        assert set(actions) <= legal
        assert sum(visits) == fixture["simulations"]
        assert best in legal


def test_the_search_fixture_is_deterministic() -> None:
    """Same position, same evaluator, same tree -- or an exact comparison is a lie.

    Exploration noise is what would break this, which is why the generator
    asserts it is off rather than trusting the default.
    """
    first = search_fixture(count=8, board_size=BOARD, seed=SEED, simulations=32)
    again = search_fixture(count=8, board_size=BOARD, seed=SEED, simulations=32)

    assert first["cases"] == again["cases"]


def test_the_games_fixture_replays_move_for_move() -> None:
    """Each rule is checked in isolation elsewhere; this checks them in sequence.

    Two engines can agree on every rule and still drift apart, which is the
    failure a whole-game fixture exists to catch.
    """
    fixture = games_fixture(count=6, board_size=BOARD, seed=SEED)

    for game in fixture["games"]:
        state = rules.initial_state(BOARD)
        for index, action in enumerate(game["moves"]):
            assert action in rules.legal_actions(state)
            state = rules.apply(state, action)

            black, white, to_move = game["positions"][index + 1]
            assert state.black == int(black, 16)
            assert state.white == int(white, 16)
            assert int(state.to_move) == to_move

        assert rules.is_terminal(state)
        assert list(scoring.disc_counts(state)) == game["final_score"]


# ===========================================================================
# The stub evaluator has to survive the trip to JavaScript
# ===========================================================================


def test_the_hash_stays_inside_32_bits() -> None:
    """JavaScript's numbers stop being exact above 2^53.

    The hash therefore has to be expressible with `Math.imul` and unsigned
    shifts. Anything that leaves the 32-bit range here would be reproducible in
    Python and not in the browser.
    """
    for value in (0, 1, 0x9E3779B9, 0xFFFFFFFF, 0x7FFFFFFF):
        assert 0 <= _mix32(value) <= 0xFFFFFFFF


def test_the_hash_test_vectors_are_frozen() -> None:
    """The TypeScript port asserts these same numbers.

    Recording them in both places means a change to the hash fails on both sides
    at once, instead of leaving the fixtures and the port quietly disagreeing.
    """
    assert _mix32(0x00000000) == 0x00000000
    assert _mix32(0x00000001) == 0x688990C0
    assert _mix32(0x9E3779B9) == 0x01FCE552
    assert _mix32(0xFFFFFFFF) == 0x6768824A


def test_the_stub_has_an_opinion() -> None:
    """A uniform stand-in would hide a prior applied to the wrong action.

    With every move rated equally the tree explores symmetrically, so visit
    counts look plausible whether or not the priors line up with the actions
    they belong to.
    """
    states = sample_positions(count=20, board_size=BOARD, seed=SEED)
    seeds = {stub_seed(s) for s in states}
    assert len(seeds) == len(states), "the stub gives different positions the same seed"

    seed = stub_seed(states[0])
    logits = [stub_logit(seed, a) for a in range(BOARD * BOARD + 1)]
    assert max(logits) - min(logits) > 1.0, "the stub's policy is nearly flat"
    assert all(-2.0 <= value < 2.0 for value in logits)
    assert -1.0 <= stub_value(seed) < 1.0


def test_the_stub_reads_the_whole_board() -> None:
    """Squares above bit 31 have to reach the hash.

    Feeding a 64-bit board in as two halves is what makes the port possible, and
    dropping the high half is the way that goes wrong -- positions differing only
    in the bottom rows would hash the same and the fixture would still look fine.
    """
    from reversi.game.state import State
    from reversi.types import Player

    low = State(black=0x0000000018000000, white=0x0000000000000000, to_move=Player.BLACK, size=8)
    high = State(black=0x1800000000000000, white=0x0000000000000000, to_move=Player.BLACK, size=8)

    assert stub_seed(low) != stub_seed(high)


# ===========================================================================
# What gets written to disk
# ===========================================================================


def test_every_fixture_file_stays_small_enough_to_commit(tmp_path: Path) -> None:
    """These files live in git, so their size is a real constraint.

    500 KB is the threshold the pre-commit large-file hook uses. The coverage
    that does not fit is not lost -- it is regenerated at a much larger count in
    the nightly job.
    """
    sizes = write_fixtures(tmp_path, seed=SEED)

    assert set(sizes) == {"rules", "encoding", "search", "games"}
    for name, size in sizes.items():
        assert size < 500_000, f"{name}.json is {size:,} bytes, too large to commit"


def test_the_fixtures_are_reproducible(tmp_path: Path) -> None:
    """CI regenerates them and fails on a difference.

    That check only means something if generating twice from the same seed gives
    the same bytes -- otherwise it would fail on every run and get switched off.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_fixtures(first, seed=SEED)
    write_fixtures(second, seed=SEED)

    for name in ("rules", "encoding", "search", "games"):
        assert (first / f"{name}.json").read_bytes() == (second / f"{name}.json").read_bytes()


def test_the_fixtures_can_be_built_without_a_trained_model(tmp_path: Path) -> None:
    """CI checks fixture freshness on every push, where no checkpoint exists.

    If that check needed a model it could not run, and the fixtures would drift
    away from the engine without anything noticing.
    """
    sizes = write_fixtures(tmp_path, seed=SEED, onnx_path=None)
    assert "network" not in sizes
    assert not (tmp_path / "network.json").exists()


def test_a_written_fixture_is_valid_json_a_browser_can_read(tmp_path: Path) -> None:
    write_fixtures(tmp_path, seed=SEED, rules_count=50)

    payload = json.loads((tmp_path / "rules.json").read_text(encoding="utf-8"))
    assert payload["fixture"] == "rules"
    assert payload["board_size"] == BOARD
    assert len(payload["schema"]) == len(payload["cases"][0])


@pytest.mark.slow
def test_the_committed_fixtures_still_match_the_engine() -> None:
    """The freshness check, as a test rather than only as a CI step.

    Skips rather than fails when the fixtures have not been generated yet, so it
    does not block the first run of a fresh clone.
    """
    directory = Path("web/src/engine/__fixtures__")
    if not (directory / "rules.json").exists():
        pytest.skip("fixtures have not been generated yet")

    payload = json.loads((directory / "rules.json").read_text(encoding="utf-8"))
    regenerated = rules_fixture(
        count=len(payload["cases"]),
        board_size=payload["board_size"],
        seed=payload["seed"],
    )
    assert payload["cases"] == regenerated["cases"], (
        "the committed fixtures no longer match the engine. Regenerate them with "
        "`reversi export-fixtures web/src/engine/__fixtures__`."
    )
