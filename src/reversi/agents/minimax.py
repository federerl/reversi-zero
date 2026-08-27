"""A depth-4 alpha-beta searcher with a hand-written evaluation.

This is the yardstick. "Our agent beats a depth-4 alpha-beta searcher with a
positional evaluation" is a claim a reader can calibrate against; "our loss went
down" is not.

**It knows things the agent has to discover.** The evaluation encodes what human
Othello players worked out over decades: corners are permanent and worth far more
than discs, the squares diagonally adjacent to corners are traps because playing
one hands the corner over, mobility matters more than material until the very
end, and a disc lead in the opening is usually a liability. None of that is
anywhere in the training signal. If the trained agent beats this, it found those
ideas on its own, from games.

**It is completely deterministic.** Fixed depth, no iterative deepening, no time
budget, no randomness anywhere. The same position always produces the same move,
so a tournament result is exactly reproducible and any difference between two
runs is the *other* agent's doing.

**The weights are frozen.** They are constants in this file rather than a
configurable, and the reason is that a baseline which drifts invalidates every
rating measured against it. Elo is relative: if the yardstick changes between
generation 5 and generation 40, the curve between them means nothing. Their hash
goes into every match report so a result can prove which yardstick it used.
"""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

from reversi.game import rules, scoring
from reversi.game.bitboard import indices, popcount
from reversi.game.state import State
from reversi.types import Action, pass_action

__all__ = ["MinimaxAgent", "evaluate", "square_values", "weights_fingerprint"]

# ---------------------------------------------------------------------------
# The evaluation's constants. FROZEN -- see the module docstring.
# ---------------------------------------------------------------------------

CORNER = 120.0
"""A corner can never be flipped. Everything else in the table is relative to
this fact."""

X_SQUARE = -40.0
"""Diagonally adjacent to a corner. Playing one is usually how you lose the
corner, which is why it scores worse than an empty square would."""

C_SQUARE = -20.0
"""Orthogonally adjacent to a corner, along the edge. Same trap, milder."""

EDGE = 15.0
"""Edges can only be flipped along one line, so they are more stable than the
middle."""

INTERIOR = -3.0
"""Slightly negative on purpose: early discs in the middle are a liability, not
an asset. This is the single most counter-intuitive number here."""

NEAR_EDGE = -5.0
"""The ring just inside the edge. Occupying it tends to hand the edge away."""

# How the three terms are weighted at each stage of the game. Mobility matters
# most early, when running out of moves is how you lose; material is worthless
# until the end, when it is the only thing that counts.
STAGE_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    #            positional  mobility  frontier  material
    "opening": (1.0, 12.0, -2.5, 0.0),
    "midgame": (1.0, 8.0, -2.0, 0.5),
    "endgame": (0.6, 2.0, -0.5, 12.0),
}


@cache
def square_values(board_size: int) -> tuple[float, ...]:
    """The disc-square table, generated rather than typed out.

    The classic table is written for 8x8; generating it keeps 4x4 and 6x6 working
    without a second set of magic numbers to keep in step.
    """
    if board_size < 4:
        msg = f"board_size must be at least 4, got {board_size}"
        raise ValueError(msg)

    values = []
    last = board_size - 1
    for row in range(board_size):
        for col in range(board_size):
            on_edge_row = row in (0, last)
            on_edge_col = col in (0, last)
            near_row = row in (1, last - 1)
            near_col = col in (1, last - 1)

            if on_edge_row and on_edge_col:
                value = CORNER
            elif near_row and near_col:
                value = X_SQUARE  # diagonal neighbour of a corner
            elif (on_edge_row and near_col) or (on_edge_col and near_row):
                value = C_SQUARE  # edge neighbour of a corner
            elif on_edge_row or on_edge_col:
                value = EDGE
            elif near_row or near_col:
                value = NEAR_EDGE
            else:
                value = INTERIOR
            values.append(value)
    return tuple(values)


def weights_fingerprint(board_size: int = 8) -> str:
    """A short hash of everything the evaluation depends on.

    Recorded in every match report. If two reports disagree about a baseline's
    strength, this says immediately whether they were even measuring the same
    opponent.
    """
    payload = repr((square_values(board_size), sorted(STAGE_WEIGHTS.items())))
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def _stage(state: State) -> str:
    filled = popcount(state.occupied)
    total = state.size * state.size
    share = filled / total
    if share < 0.35:
        return "opening"
    if share < 0.80:
        return "midgame"
    return "endgame"


def _frontier(discs: int, empty: int, board_size: int) -> int:
    """How many of these discs touch an empty square.

    Frontier discs are the ones the opponent can attack. Having fewer than your
    opponent means they run out of safe moves first, which is the whole basis of
    mobility play.
    """
    count = 0
    for index in indices(discs):
        row, col = divmod(index, board_size)
        touching = False
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                r, c = row + d_row, col + d_col
                inside = 0 <= r < board_size and 0 <= c < board_size
                if inside and empty & (1 << (r * board_size + c)):
                    touching = True
                    break
            if touching:
                break
        count += touching
    return count


def evaluate(state: State) -> float:
    """Score a position from the point of view of the player to move.

    Same convention as everything else in the project (contract C2), so this can
    sit inside a negamax search without a single sign flip anywhere.
    """
    if rules.is_terminal(state):
        # A decided game is worth more than any heuristic opinion about it.
        return 10_000.0 * scoring.result(state)

    size = state.size
    table = square_values(size)
    mine, theirs = state.mine, state.theirs
    empty = state.empty

    positional = sum(table[i] for i in indices(mine)) - sum(table[i] for i in indices(theirs))

    my_moves = popcount(rules.legal_placements(state))
    their_moves = popcount(rules._placements(theirs, mine, size))
    mobility = _ratio(my_moves, their_moves)

    frontier = _ratio(_frontier(mine, empty, size), _frontier(theirs, empty, size))
    material = _ratio(popcount(mine), popcount(theirs))

    w_pos, w_mob, w_front, w_mat = STAGE_WEIGHTS[_stage(state)]
    return w_pos * positional + w_mob * mobility + w_front * frontier + w_mat * material


def _ratio(mine: int, theirs: int) -> float:
    """A difference normalised to [-100, 100], so terms stay comparable.

    A raw difference would let whichever term happens to have the largest range
    dominate the others regardless of the weights.
    """
    if mine + theirs == 0:
        return 0.0
    return 100.0 * (mine - theirs) / (mine + theirs)


class MinimaxAgent:
    """Alpha-beta to a fixed depth, with the evaluation above.

    Alpha-beta returns *exactly* what a full minimax search would; the pruning
    only skips branches that provably cannot change the answer. So "depth 4" is a
    complete statement of how far it looks, and the result does not depend on how
    fast the machine is -- which is what makes it usable as a stable yardstick.
    """

    def __init__(self, depth: int = 4, name: str | None = None) -> None:
        if depth < 1:
            msg = f"depth must be at least 1, got {depth}"
            raise ValueError(msg)
        self.depth = depth
        self._name = name if name is not None else f"minimax-d{depth}"
        self.nodes = 0

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: State, rng: np.random.Generator) -> Action:
        _ = rng  # deterministic on purpose
        actions = rules.legal_actions(state)
        if not actions:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ValueError(msg)
        if len(actions) == 1:
            return actions[0]

        best_action = actions[0]
        best_score = -float("inf")
        alpha = -float("inf")

        for action in self._ordered(state, actions):
            score = -self._search(rules.apply(state, action), self.depth - 1, -float("inf"), -alpha)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
        return best_action

    def _ordered(self, state: State, actions: list[Action]) -> list[Action]:
        """Look at promising moves first, so alpha-beta can prune more.

        Ordering changes only the speed, never the answer -- but it changes the
        speed a lot, because a cutoff found early skips an entire subtree.
        """
        if len(actions) < 3:
            return actions
        table = square_values(state.size)
        passing = pass_action(state.size)
        return sorted(actions, key=lambda a: -(table[a] if a != passing else 0.0))

    def _search(self, state: State, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1

        if depth <= 0 or rules.is_terminal(state):
            return evaluate(state)

        actions = rules.legal_actions(state)
        best = -float("inf")
        for action in self._ordered(state, actions):
            score = -self._search(rules.apply(state, action), depth - 1, -beta, -alpha)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # this branch cannot change the answer above
        return best
