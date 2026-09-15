"""An exact endgame solver: the final disc difference, under perfect play.

The search everywhere else in this project stops early and asks the network to
guess the rest. This one does not stop. It plays every line to the last legal
move and returns the score that actually results, so its answer is not an
estimate that happens to be confident -- it is the answer.

That is only affordable because the end of an Othello game is small, and it stops
being affordable quickly. Measured by `bench/endgame_bench.py`, solving every move
at a position drawn from real play costs about 0.07 s at eight empty squares, 1.4 s
at eleven, 10.6 s at thirteen and 24 s at fourteen -- roughly two and a half times
more per extra square. At thirty squares there is no number, only a machine that
never answers. So the solver takes an explicit limit and refuses to start above it,
rather than appearing to hang.

**What it returns is a disc difference, not a win or a loss.** The 4x4 solver in
``tests/unit/test_solved_4x4.py`` returns +1/0/-1, which is the right answer to
"who wins" and the wrong one for teaching: "this move still wins, by 2" and "this
move wins by 20" are different lessons, and a puzzle that cannot tell them apart
cannot explain why one move is better than another that also wins.

Three implementation notes that are the difference between seconds and minutes:

**It works on raw integers, not on ``State``.** ``rules.apply`` re-validates the
move and builds an error string on the failure path; ``rules.legal_actions``
allocates a list at every node; ``rules.is_terminal`` runs move generation
*twice*. None of that is wrong -- it is what you want at the edges of the system,
where a bad move should be caught loudly -- and all of it is waste in a loop that
runs millions of times. Here the position is two integers, and legality is
guaranteed by construction because every move came out of the generator.

**Terminal is detected with one extra generation, not two.** Contract C3 says a
position is over exactly when neither side can place. So generate my moves; only
if I have none is it worth asking whether the opponent has any; and only if they
have none is the game over.

**The transposition table never expires.** A normal search caches "this position
is worth X *if you search it 6 more plies*", so an entry is only reusable at the
same remaining depth. This search always goes to the end, so the value of a
position is a property of the position alone. One table serves the whole solve,
including across different first moves, which is most of why solving every root
move costs so much less than solving one of them nine times over.
"""

from __future__ import annotations

from dataclasses import dataclass

from reversi.errors import ReversiError
from reversi.game import rules
from reversi.game.bitboard import geometry, indices, popcount
from reversi.game.state import State
from reversi.types import Action, Bitboard, pass_action

__all__ = [
    "MAX_EMPTIES",
    "NoChoiceError",
    "RootSolution",
    "TooManyEmptiesError",
    "empties",
    "solve_exact",
    "solve_root",
]


MAX_EMPTIES = 16
"""The most empty squares this solver will accept.

Not a property of the algorithm -- the search would happily run at thirty and
never come back. It is where Python stops being fast enough to be useful: the
measured cost roughly triples every two squares, reaching 24 s a position at
fourteen and a worst case of 66 s. Sixteen is therefore a guard rail rather than
a capability claim, so that a mistake in a mining script fails immediately
instead of looking like a hang. Raise it only with a fresh benchmark in hand.
"""

_MOBILITY_ORDERING_FROM = 9
"""Empty-square count above which move ordering pays for itself.

Ordering moves by how few replies they leave the opponent means running the move
generator once per child before searching any of them. Deep in the tree that buys
far more in cut-offs than it costs; near the leaves the tree is so small that the
extra generations dominate. Measured, not guessed -- see `bench/endgame_bench.py`.
"""

_EXACT = 0
_LOWER = 1
_UPPER = 2


def _terminal_margin(mine: Bitboard, theirs: Bitboard, size: int, empties_to_winner: bool) -> int:
    """The final disc difference, under one of the two scoring conventions.

    This project counts discs and gives empty squares to nobody -- contract C3,
    and what `scoring.result` does. A game that ends with the board unfilled is
    scored exactly as the board looks, which is also what a player sees when the
    last move is played.

    Othello tooling more often awards the empty squares to the winner, and Edax
    does. On a game that fills the board the two agree; on one that ends early
    they differ by the number of empty squares, and they optimise *different*
    objectives, so they can prefer different moves.

    Both live here so the solver can be checked against Edax in Edax's own terms
    and still ship in this project's. Comparing across a convention mismatch
    would produce disagreements that look like search bugs and are not -- and,
    worse, agreement on the 90% of positions that fill the board, which reads as
    a passing check.
    """
    mine_count = popcount(mine)
    theirs_count = popcount(theirs)
    if empties_to_winner:
        open_squares = size * size - mine_count - theirs_count
        if mine_count > theirs_count:
            mine_count += open_squares
        elif theirs_count > mine_count:
            theirs_count += open_squares
    return mine_count - theirs_count


class TooManyEmptiesError(ReversiError):
    """Raised when a position has more empty squares than the solver will accept."""


class NoChoiceError(ReversiError):
    """Raised when a position offers no placement, so there is no move to rank.

    Separate from ``TooManyEmptiesError`` because it means the caller handed over a
    position that cannot be a puzzle, not one that is too expensive.
    """


def empties(state: State) -> int:
    """How many squares are still empty."""
    geo = geometry(state.size)
    return popcount(geo.full & ~(state.black | state.white))


def _square_order(size: int) -> tuple[int, ...]:
    """Squares from most to least promising, by shape alone.

    A corner can never be flipped once taken, so it is the one square whose value
    does not depend on the rest of the position. The squares beside a corner are
    the worst for the mirror-image reason: playing one hands the corner over. In
    between, edges beat the interior.

    This is a *tiebreak*, not an evaluation -- the search still computes the true
    value of every move. Its only job is to try likely-good moves first so that
    alpha-beta can discard the rest without looking at them.
    """
    last = size - 1
    order: list[tuple[int, int]] = []
    for row in range(size):
        for col in range(size):
            near_row = min(row, last - row)
            near_col = min(col, last - col)
            if near_row == 0 and near_col == 0:
                rank = 0  # corner
            elif near_row <= 1 and near_col <= 1:
                rank = 4  # the squares that give a corner away
            elif near_row == 0 or near_col == 0:
                rank = 1  # edge
            elif near_row == 1 or near_col == 1:
                rank = 3  # one in from an edge
            else:
                rank = 2  # interior
            order.append((rank, row * size + col))
    order.sort()
    return tuple(square for _, square in order)


_ORDER_CACHE: dict[int, tuple[int, ...]] = {}


def _order_for(size: int) -> tuple[int, ...]:
    cached = _ORDER_CACHE.get(size)
    if cached is None:
        cached = _square_order(size)
        _ORDER_CACHE[size] = cached
    return cached


def _ordered_moves(moves: Bitboard, mine: Bitboard, theirs: Bitboard, size: int) -> list[Action]:
    """The legal moves, most promising first.

    Deep in the tree, order by how few replies each move leaves the opponent --
    the classic Othello heuristic, and the reason a solver finishes at all. Near
    the leaves, fall back to the static shape ordering, because there the
    generator calls cost more than the cut-offs save.
    """
    squares = indices(moves)
    if len(squares) < 2:
        return squares

    remaining = popcount(geometry(size).full & ~(mine | theirs))
    if remaining < _MOBILITY_ORDERING_FROM:
        rank = _order_for(size).index
        return sorted(squares, key=rank)

    scored: list[tuple[int, int]] = []
    for square in squares:
        bit = 1 << square
        flipped = rules._flips_from(bit, mine, theirs, size)
        child_mine = theirs & ~flipped
        child_theirs = mine | flipped | bit
        replies = popcount(rules._placements(child_mine, child_theirs, size))
        scored.append((replies, square))
    scored.sort()
    return [square for _, square in scored]


def _search(
    mine: Bitboard,
    theirs: Bitboard,
    size: int,
    alpha: int,
    beta: int,
    table: dict[tuple[int, int], tuple[int, int]],
    counter: list[int],
    empties_to_winner: bool,
) -> int:
    """Negamax with alpha-beta. Returns the final disc difference for ``mine``.

    ``alpha`` and ``beta`` bracket the values worth knowing about: anything at or
    below ``alpha`` is already beaten by a line the caller has, and anything at or
    above ``beta`` is good enough that the caller will avoid this branch entirely.
    Inside that window the value returned is exact; outside it the value is only
    known to be a bound, which is why the table records which of the two it has.
    """
    counter[0] += 1

    key = (mine, theirs)
    cached = table.get(key)
    if cached is not None:
        value, flag = cached
        if flag == _EXACT:
            return value
        if flag == _LOWER and value >= beta:
            return value
        if flag == _UPPER and value <= alpha:
            return value

    my_moves = rules._placements(mine, theirs, size)
    if not my_moves:
        their_moves = rules._placements(theirs, mine, size)
        if not their_moves:
            # Contract C3: neither side can place, so the game is over.
            return _terminal_margin(mine, theirs, size, empties_to_winner)
        # A pass is not a move and is never stored: it has no position of its
        # own, it is the same position seen from the other side.
        return -_search(theirs, mine, size, -beta, -alpha, table, counter, empties_to_winner)

    original_alpha = alpha
    best = -(size * size + 1)

    for square in _ordered_moves(my_moves, mine, theirs, size):
        bit = 1 << square
        flipped = rules._flips_from(bit, mine, theirs, size)
        value = -_search(
            theirs & ~flipped,
            mine | flipped | bit,
            size,
            -beta,
            -alpha,
            table,
            counter,
            empties_to_winner,
        )
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= original_alpha:
        flag = _UPPER
    elif best >= beta:
        flag = _LOWER
    else:
        flag = _EXACT
    table[key] = (best, flag)
    return best


def solve_exact(
    state: State, *, max_empties: int = MAX_EMPTIES, empties_to_winner: bool = False
) -> int:
    """The final disc difference from the point of view of the player to move.

    Positive means the player to move wins by that many discs with perfect play
    from both sides; negative means they lose by that many.

    ``empties_to_winner`` switches to the convention Edax and most Othello
    tooling use, where a game that ends with the board unfilled awards the
    remaining squares to the winner. It exists so the solver can be validated
    against an outside engine; everything this project publishes uses the
    default, which is the plain disc count.
    """
    _check_size(state, max_empties)
    counter = [0]
    limit = state.size * state.size + 1
    return _search(
        state.mine, state.theirs, state.size, -limit, limit, {}, counter, empties_to_winner
    )


@dataclass(frozen=True, slots=True)
class RootSolution:
    """Every legal move at one position, with the score it really leads to."""

    margins: dict[Action, int]
    """Exact final disc difference after each legal move, for the mover."""
    best: int
    """The best difference available."""
    best_moves: tuple[Action, ...]
    """Every move achieving it -- more than one is common, and it is what makes a
    position easy: a puzzle with six winning moves is not a puzzle."""
    principal_variation: tuple[Action, ...]
    """One line of perfect play from here, starting with a best move. Enough to
    show *why* the answer is the answer instead of asserting it."""
    nodes: int
    """Positions visited. Reported so the cost of a claim is visible."""

    @property
    def winning_moves(self) -> tuple[Action, ...]:
        """Moves that leave the mover ahead. May be empty in a lost position."""
        return tuple(sorted(move for move, margin in self.margins.items() if margin > 0))

    def describe(self) -> str:
        best = ", ".join(str(move) for move in self.best_moves)
        return (
            f"{len(self.margins)} moves, best {self.best:+d} via [{best}], "
            f"{len(self.winning_moves)} winning, {self.nodes} nodes"
        )


def solve_root(
    state: State, *, max_empties: int = MAX_EMPTIES, empties_to_winner: bool = False
) -> RootSolution:
    """Solve every legal move at ``state``, not just the best one.

    A puzzle needs the whole table. Telling somebody their move was wrong is
    worth little; telling them it loses by 2 where the best move wins by 6 is a
    lesson. Alpha-beta is therefore *not* applied between root moves -- each is
    searched with a full window so its value comes back exact rather than as a
    bound saying only "no better than the one before".

    The cost of that is far less than it looks, because the transposition table
    is shared across root moves and this search never expires an entry.
    """
    _check_size(state, max_empties)

    my_moves = rules._placements(state.mine, state.theirs, state.size)
    if not my_moves:
        over = rules.is_terminal(state)
        msg = (
            "this position offers no placement, so there is nothing to choose "
            f"between: {'the game is over' if over else 'the only legal action is a pass'}"
        )
        raise NoChoiceError(msg)

    table: dict[tuple[int, int], tuple[int, int]] = {}
    counter = [0]
    limit = state.size * state.size + 1

    margins: dict[Action, int] = {}
    for square in indices(my_moves):
        bit = 1 << square
        flipped = rules._flips_from(bit, state.mine, state.theirs, state.size)
        margins[square] = -_search(
            state.theirs & ~flipped,
            state.mine | flipped | bit,
            state.size,
            -limit,
            limit,
            table,
            counter,
            empties_to_winner,
        )

    best = max(margins.values())
    best_moves = tuple(sorted(move for move, margin in margins.items() if margin == best))

    return RootSolution(
        margins=margins,
        best=best,
        best_moves=best_moves,
        principal_variation=_principal_variation(state, table, counter, empties_to_winner),
        nodes=counter[0],
    )


def _check_size(state: State, max_empties: int) -> None:
    if max_empties > MAX_EMPTIES:
        msg = (
            f"asked to solve up to {max_empties} empty squares, but this solver caps "
            f"at {MAX_EMPTIES}. Above that it stops being fast enough to be useful; "
            "raise MAX_EMPTIES deliberately if the benchmark says otherwise"
        )
        raise TooManyEmptiesError(msg)
    open_squares = empties(state)
    if open_squares > max_empties:
        msg = (
            f"this position has {open_squares} empty squares and the limit is "
            f"{max_empties}. Solving it exactly would take far longer than it looks; "
            "pass a larger max_empties only if you have measured the cost"
        )
        raise TooManyEmptiesError(msg)


def _principal_variation(
    state: State,
    table: dict[tuple[int, int], tuple[int, int]],
    counter: list[int],
    empties_to_winner: bool,
) -> tuple[Action, ...]:
    """One line of perfect play from here, walked out to the end of the game.

    Re-solving at each step looks wasteful and costs almost nothing: every
    position along the line is already in the table from the root solve, and the
    table never expires because this search always runs to the end.

    Passes are recorded, not skipped. The line is meant to be replayable on a
    board, and a replay that silently drops a pass puts the wrong player on move
    for the rest of it.

    Ties are broken by the shape ordering rather than arbitrarily, so the line
    shown for a position is the same every time it is generated -- a puzzle whose
    answer moved between two runs of the generator would be worse than no puzzle.
    """
    line: list[Action] = []
    current = state
    limit = state.size * state.size + 1
    # A game cannot last longer than one move per square plus a pass apiece.
    guard = state.size * state.size + 2

    while len(line) < guard and not rules.is_terminal(current):
        skip = pass_action(current.size)
        if rules.must_pass(current):
            line.append(skip)
            current = rules.apply(current, skip)
            continue

        my_moves = rules._placements(current.mine, current.theirs, current.size)
        best_value: int | None = None
        best_action: Action | None = None
        for square in _ordered_moves(my_moves, current.mine, current.theirs, current.size):
            bit = 1 << square
            flipped = rules._flips_from(bit, current.mine, current.theirs, current.size)
            value = -_search(
                current.theirs & ~flipped,
                current.mine | flipped | bit,
                current.size,
                -limit,
                limit,
                table,
                counter,
                empties_to_winner,
            )
            if best_value is None or value > best_value:
                best_value, best_action = value, square

        if best_action is None:  # pragma: no cover - guarded by must_pass above
            break
        line.append(best_action)
        current = rules.apply(current, best_action)

    return tuple(line)
