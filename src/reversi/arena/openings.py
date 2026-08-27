"""Varied starting positions, so a match measures more than one game.

**The problem.** Two strong agents are near-deterministic: no exploration noise,
no temperature, same position in, same move out. Play them against each other 200
times from the standard start and you get **one game, counted 200 times**. The
win rate is then 0% or 100%, its confidence interval is meaningless, and the
whole tournament measures a single line of play.

**The fix.** Start each pair of games from a different position, reached by
playing a few random moves. Now the agents are compared across 100 genuinely
different openings, and the result says something about how they play Reversi
rather than how they play one line of it.

**Each opening is played twice, with the colours swapped.** Black moves first and
that is worth something, so an opening that happens to favour the first player
would otherwise flatter whoever drew it. Playing it both ways cancels that out
exactly -- both agents face the identical position from the identical side.

**The book is seeded, so it is reproducible.** Two runs of the same tournament
use the same openings, which is what makes their results comparable at all. The
seed goes in the report.
"""

from __future__ import annotations

from reversi.errors import ArenaError
from reversi.game import rules
from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.seeding import derive_seed
from reversi.seeding import rng as make_rng
from reversi.types import Action

__all__ = ["Opening", "apply_opening", "random_openings"]

Opening = tuple[Action, ...]


def random_openings(
    *,
    count: int,
    board_size: int,
    seed: int,
    plies: int = 4,
    min_branching: int = 2,
    label: str = "openings",
) -> tuple[Opening, ...]:
    """Build ``count`` distinct opening lines of ``plies`` random moves.

    An opening is rejected unless **both** sides still have at least
    ``min_branching`` legal moves at the end of it. A position where one side is
    already forced is not a fair starting point -- it hands part of the game to
    whichever agent happens to be on the good side of it, and no amount of colour
    swapping fixes a position that is simply lopsided.

    Duplicates are rejected too: a book with the same line twice is a book with
    fewer lines than it claims.
    """
    if count < 1:
        msg = f"need at least one opening, got {count}"
        raise ArenaError(msg)
    if plies < 0:
        msg = f"plies cannot be negative, got {plies}"
        raise ArenaError(msg)

    found: list[Opening] = []
    seen: set[Opening] = set()

    # Bounded rather than a while-true: on a small board there may simply not be
    # `count` distinct usable openings, and looping forever is a worse failure
    # than saying so.
    attempts = 0
    ceiling = max(200, count * 100)

    while len(found) < count and attempts < ceiling:
        rng = make_rng(derive_seed(seed, label, board_size, plies, attempts))
        attempts += 1

        state = rules.initial_state(board_size)
        line: list[Action] = []
        usable = True

        for _ in range(plies):
            actions = rules.legal_actions(state)
            if not actions:
                usable = False
                break
            action = actions[int(rng.integers(0, len(actions)))]
            line.append(action)
            state = rules.apply(state, action)

        if not usable or rules.is_terminal(state):
            continue
        # Both sides, not just the one to move: a position where the *opponent*
        # is about to be forced is just as lopsided a place to start from.
        if len(rules.legal_actions(state)) < min_branching:
            continue
        theirs = rules.legal_placements_for(state, state.to_move.opponent)
        if popcount(theirs) < min_branching:
            continue

        opening = tuple(line)
        if opening in seen:
            continue
        seen.add(opening)
        found.append(opening)

    if len(found) < count:
        msg = (
            f"only found {len(found)} usable openings of {count} asked for on a "
            f"{board_size}x{board_size} board after {attempts} attempts. Try fewer "
            "openings, or fewer opening plies."
        )
        raise ArenaError(msg)

    return tuple(found)


def apply_opening(board_size: int, opening: Opening) -> State:
    """Replay an opening line and return the position it reaches."""
    state = rules.initial_state(board_size)
    for action in opening:
        state = rules.apply(state, action)
    return state
