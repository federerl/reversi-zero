"""Counting discs and deciding who won.

The perspective rule here is small but load-bearing. ``result`` always answers
"how did this go for the player to move", never "how did this go for black".
One convention, used by the engine, the search, the training targets, the
evaluation, and the web API alike (contract C2).

The alternative — sometimes black's view, sometimes the mover's — is how sign
errors get into a tree search, and a sign error there does not crash. The AI
just confidently learns to lose.
"""

from __future__ import annotations

from reversi.game.bitboard import popcount
from reversi.game.state import State
from reversi.types import Outcome, Player


def disc_counts(state: State) -> tuple[int, int]:
    """How many discs each player has, as (black, white)."""
    return popcount(state.black), popcount(state.white)


def result(state: State) -> Outcome:
    """+1 / 0 / -1 for a win / draw / loss, from the point of view of ``to_move``.

    Meaningful only once the game is over. "The player to move" still means
    something at a finished game even though they have no move left: it is
    whoever would have played next.

    Empty squares count for nobody. Some Othello variants award them to the
    winner; we use the plain disc count and say so in the docs.
    """
    mine = popcount(state.mine)
    theirs = popcount(state.theirs)
    if mine > theirs:
        return 1
    if mine < theirs:
        return -1
    return 0


def result_for(state: State, player: Player) -> Outcome:
    """The same result, expressed for a specific colour rather than the mover.

    Used when recording a finished game: each stored position needs the outcome
    from the point of view of whoever was to move *in that position*, and the
    two players alternate.
    """
    outcome = result(state)
    return outcome if player is state.to_move else -outcome


def score_margin(state: State) -> int:
    """Disc difference from the mover's point of view.

    Not used for training targets, which are win/draw/loss only. Reported in
    the web UI and the evaluation logs, where "won by 2" and "won by 40" are
    worth telling apart.
    """
    return popcount(state.mine) - popcount(state.theirs)
