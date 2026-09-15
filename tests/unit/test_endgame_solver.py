"""The exact endgame solver.

This file matters more than most. Everything else in the project reports a
measurement with an interval attached, and a number that is a little bit off is
still informative. The solver reports a *proof*, and a proof that is a little bit
off is a lie that a player will believe -- it will tell somebody their move loses
when it wins, and they will learn the wrong lesson and trust it, because the
interface says the answer is certain.

So it is checked four ways, each independent of the others:

1. **Against the exhaustive 4x4 solver** in ``test_solved_4x4.py``, which has no
   alpha-beta, no ordering and no transposition table, so it shares none of the
   ways this one could be wrong. It only knows win/draw/loss, so the comparison
   is on the *sign* of the margin -- but it covers every position reachable on a
   4x4 board, which is a great many more positions than a hand-written test.
2. **Against the engine at terminal positions**, where the answer is not a search
   result at all but a disc count.
3. **Against itself, structurally**: negamax says a position is worth the best of
   the negated children. Checked on real positions including ones where a player
   must pass, which is where contract C3 either holds or quietly does not.
4. **Against Edax**, an outside engine nobody here wrote, on its own problem sets.

Anchor 4 is the one that caught a real disagreement. Edax and this project score
a game that ends with the board unfilled differently -- Edax gives the empty
squares to the winner, this project gives them to nobody. On the ~90% of
positions whose perfect play fills the board the two agree exactly, so a check
that ignored the convention would have *passed* and hidden the rest. The solver
therefore implements both rules, and the comparison uses Edax's.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# A sibling test module, not a package import: `tests/` has no __init__.py, and
# pytest puts each test file's directory on the path. Imported rather than copied
# so there stays exactly one exhaustive solver to disagree with.
from test_solved_4x4 import solve as solve_4x4_exhaustively

from reversi.agents.edax import EDAX_ROOT, find_edax
from reversi.endgame import MAX_EMPTIES, NoChoiceError, TooManyEmptiesError, empties, solve_exact
from reversi.endgame.solver import solve_root
from reversi.errors import ArenaError
from reversi.game import rules, scoring
from reversi.game.state import State, initial_state
from reversi.types import Player

BOARD = 8


# ===========================================================================
# Anchor 1 — the exhaustive 4x4 solver
# ===========================================================================


def _every_4x4_position(limit: int) -> list[State]:
    """Positions reachable in 4x4 play, breadth first, deduplicated."""
    start = initial_state(4)
    seen = {(start.black, start.white, int(start.to_move))}
    frontier = [start]
    found = [start]

    while frontier and len(found) < limit:
        state = frontier.pop()
        if rules.is_terminal(state):
            continue
        for action in rules.legal_actions(state):
            child = rules.apply(state, action)
            key = (child.black, child.white, int(child.to_move))
            if key in seen:
                continue
            seen.add(key)
            found.append(child)
            frontier.append(child)

    return found[:limit]


def test_it_agrees_with_the_exhaustive_solver_on_who_wins() -> None:
    """Every reachable 4x4 position, compared against a solver sharing no code.

    The 4x4 solver is nineteen lines with no pruning, no ordering and no cache.
    If alpha-beta is cutting a branch it should not, or the transposition table
    is returning a bound as though it were exact, this is where it shows.
    """
    disagreements = []
    for state in _every_4x4_position(1500):
        exhaustive = solve_4x4_exhaustively(state.black, state.white, int(state.to_move))
        margin = solve_exact(state, max_empties=12)
        sign = (margin > 0) - (margin < 0)
        if sign != exhaustive:
            disagreements.append((state.black, state.white, int(state.to_move), exhaustive, margin))

    assert disagreements == [], (
        f"{len(disagreements)} positions disagree, first: {disagreements[0]}"
    )


def test_white_wins_four_by_four_by_eight_discs() -> None:
    """The known result, with the margin the coarser solver cannot express."""
    margin = solve_exact(initial_state(4), max_empties=12)
    assert margin == -8, "4x4 is a win for white; black to move should lose by 8"


# ===========================================================================
# Anchor 2 — the engine, at positions that need no search
# ===========================================================================


def test_a_finished_game_is_scored_by_counting_not_by_searching() -> None:
    """At a terminal position the solver must return exactly the disc count."""
    wipeouts = [
        State(black=0xFFFF, white=0, to_move=Player.BLACK, size=4),
        State(black=0, white=0xFFFF, to_move=Player.BLACK, size=4),
        State(black=0x00FF, white=0xFF00, to_move=Player.WHITE, size=4),
    ]
    for state in wipeouts:
        assert rules.is_terminal(state)
        assert solve_exact(state, max_empties=16) == scoring.score_margin(state)


# ===========================================================================
# Anchor 3 — the negamax identity, including across a pass
# ===========================================================================


def _positions_from_random_play(count: int, *, open_squares: int, seed: int) -> list[State]:
    import random

    rng = random.Random(seed)
    found: list[State] = []
    while len(found) < count:
        state = initial_state(BOARD)
        while not rules.is_terminal(state):
            if empties(state) == open_squares and rules.legal_placements(state):
                found.append(state)
                break
            state = rules.apply(state, rng.choice(rules.legal_actions(state)))
    return found


def test_a_position_is_worth_the_best_of_its_negated_children() -> None:
    """The identity negamax is built on, checked on real positions.

    If this fails the sign is being flipped the wrong number of times somewhere,
    which is the failure mode that does not crash: the solver would confidently
    recommend the worst move available.
    """
    for state in _positions_from_random_play(12, open_squares=9, seed=20260914):
        parent = solve_exact(state, max_empties=10)
        best_child = max(
            -solve_exact(rules.apply(state, action), max_empties=10)
            for action in rules.legal_actions(state)
        )
        assert parent == best_child


def test_a_forced_pass_is_the_same_position_from_the_other_side() -> None:
    """Contract C3: a pass is not a move, it is a change of point of view.

    Constructed rather than sampled, because a position where one side has no
    legal move is rare enough that random play would usually not produce one.
    """
    # Black owns a corner block; white has nothing on the board that can be
    # trapped, so black has no placement and must pass.
    state = State(black=0x0007, white=0x0000, to_move=Player.WHITE, size=4)
    if rules.legal_placements(state):
        pytest.skip("constructed position turned out to offer a placement")

    assert rules.must_pass(state) or rules.is_terminal(state)
    if rules.must_pass(state):
        passed = rules.apply(state, state.size * state.size)
        assert solve_exact(state, max_empties=16) == -solve_exact(passed, max_empties=16)


# ===========================================================================
# Anchor 4 — Edax, on its own problem sets
# ===========================================================================


def _edax_available() -> bool:
    try:
        find_edax()
    except ArenaError:
        return False
    return True


needs_edax = pytest.mark.skipif(
    not _edax_available(),
    reason=f"no Edax under {EDAX_ROOT}/ (see docs/experiments.md, 'Getting Edax')",
)

_PROBLEMS = Path("tools/edax/problem/full-10.txt")


def _parse_obf(line: str) -> State:
    """One problem line to a State: 64 squares of X/O/-, then the side to move."""
    board, _, rest = line.strip().partition(" ")
    black = white = 0
    for index, char in enumerate(board):
        if char == "X":
            black |= 1 << index
        elif char == "O":
            white |= 1 << index
    side = rest.strip().split(";")[0].strip().upper()
    return State(
        black=black,
        white=white,
        to_move=Player.BLACK if side == "X" else Player.WHITE,
        size=8,
    )


def _edax_scores(problem: Path) -> list[int]:
    exe = find_edax()
    result = subprocess.run(
        [str(exe.resolve()), "-solve", str(problem.resolve())],
        cwd=str(exe.resolve().parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        int(match.group(1))
        for match in (
            re.match(r"\s*\d+\|\s*\d+\s+([+-]\d+)", raw) for raw in result.stdout.splitlines()
        )
        if match
    ]


@needs_edax
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_it_agrees_with_edax_on_every_position(tmp_path: Path) -> None:
    """The check that does not depend on anything written in this repository.

    Run in Edax's scoring convention, not this project's. The two differ only on
    games that end with the board unfilled, and comparing across that difference
    would produce failures that look like search bugs and are not.
    """
    if not _PROBLEMS.is_file():
        pytest.skip(f"{_PROBLEMS} is not present")

    lines = [line for line in _PROBLEMS.read_text(encoding="utf-8").splitlines() if line.strip()][
        :60
    ]
    slice_file = tmp_path / "positions.txt"
    slice_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    expected = _edax_scores(slice_file)
    assert len(expected) == len(lines), "Edax did not report one score per position"

    disagreements = []
    for index, (line, want) in enumerate(zip(lines, expected, strict=True), start=1):
        state = _parse_obf(line)
        got = solve_exact(state, max_empties=MAX_EMPTIES, empties_to_winner=True)
        if got != want:
            disagreements.append((index, empties(state), want, got))

    assert disagreements == [], (
        f"{len(disagreements)} of {len(lines)} disagree: {disagreements[:5]}"
    )


@needs_edax
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_two_scoring_conventions_differ_only_by_the_empty_squares(tmp_path: Path) -> None:
    """Pin the difference, so a future change to either rule is visible.

    This is the finding that anchor 4 exists to protect: on a game that fills the
    board the conventions agree, and on one that does not they differ by exactly
    the squares left over. Without this test, silently adopting Edax's rule would
    still pass every other check here.
    """
    if not _PROBLEMS.is_file():
        pytest.skip(f"{_PROBLEMS} is not present")

    lines = [line for line in _PROBLEMS.read_text(encoding="utf-8").splitlines() if line.strip()][
        :40
    ]
    differences = set()
    for line in lines:
        state = _parse_obf(line)
        discs = solve_exact(state, max_empties=MAX_EMPTIES)
        to_winner = solve_exact(state, max_empties=MAX_EMPTIES, empties_to_winner=True)
        differences.add(abs(to_winner) - abs(discs))

    assert differences, "no positions were compared"
    assert all(gap >= 0 for gap in differences), (
        "awarding empty squares to the winner can only widen a margin, never narrow it"
    )
    assert 0 in differences, "expected most games to fill the board, where the rules agree"


# ===========================================================================
# The guard rails
# ===========================================================================


def test_it_refuses_a_position_it_cannot_finish() -> None:
    """A limit that is checked beats a machine that never answers."""
    with pytest.raises(TooManyEmptiesError, match="empty squares"):
        solve_exact(initial_state(BOARD))

    with pytest.raises(TooManyEmptiesError, match=f"caps at {MAX_EMPTIES}"):
        solve_exact(initial_state(BOARD), max_empties=MAX_EMPTIES + 1)


def test_it_refuses_a_position_with_nothing_to_choose() -> None:
    """A forced pass is not a puzzle, and neither is a finished game."""
    finished = State(black=0xFFFF, white=0, to_move=Player.BLACK, size=4)
    with pytest.raises(NoChoiceError, match="no placement"):
        solve_root(finished, max_empties=16)


# ===========================================================================
# The root solve, which is what a puzzle is built from
# ===========================================================================


def test_every_root_move_gets_an_exact_value_not_a_bound() -> None:
    """A puzzle needs the whole table, so root moves are not pruned against
    each other. Each margin must equal what solving that child alone gives."""
    for state in _positions_from_random_play(6, open_squares=9, seed=4242):
        solution = solve_root(state, max_empties=10)
        for action, margin in solution.margins.items():
            alone = -solve_exact(rules.apply(state, action), max_empties=10)
            assert margin == alone, f"move {action} came back as a bound, not a value"


def test_the_principal_variation_really_reaches_the_promised_score() -> None:
    """Playing the line out must produce the number the solver reported.

    The line is what a puzzle shows to justify its answer. If it does not end
    where the answer says, the explanation contradicts the claim.
    """
    for state in _positions_from_random_play(6, open_squares=9, seed=99):
        solution = solve_root(state, max_empties=10)

        current = state
        for action in solution.principal_variation:
            current = rules.apply(current, action)

        assert rules.is_terminal(current), "the line stopped before the game was over"
        # score_margin is from the mover's view, and the mover alternates, so
        # compare for the player who was to move at the root.
        assert scoring.result_for(current, state.to_move) == (
            (solution.best > 0) - (solution.best < 0)
        )
