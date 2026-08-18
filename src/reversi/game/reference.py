"""The obvious Reversi implementation. Slow on purpose.

This file is the **specification**. Every rule is written the most direct way
possible: the board is a grid of squares, and to find which discs a move flips
we literally walk outward in each of the eight directions and look.

Nothing here is clever and nothing here is fast. That is the point. The real
engine (``rules.py``, using two 64-bit integers) is far quicker but its
correctness is not obvious from reading it. So we keep this version forever and
make the two play 50,000 random games against each other, checking they agree on
**every single move**. If they ever disagree, one of them is wrong, and we find
out in seconds instead of a week later.

Why that matters so much here: a subtly wrong rules engine does not crash. The
AI simply learns a slightly different game, and every measurement afterwards is
meaningless. No loss curve and no other test would catch it.

This module must not import from ``rules.py`` or ``bitboard.py``. Both engines
are written independently from the rules of Othello, and that independence is
what makes agreement between them evidence of anything.

The rules implemented (standard Othello):

1. Black moves first.
2. A move is legal only if it traps at least one unbroken line of opponent
   discs between the new disc and another of your own discs.
3. Every trapped disc flips to your colour.
4. If you have no legal move but your opponent does, you pass.
5. If neither player has a legal move, the game ends.
6. Whoever has more discs wins. Equal counts is a draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from reversi.errors import IllegalMoveError
from reversi.types import Action, Bitboard, Outcome, Player, pass_action

# A square is owned by a player, or empty.
Cell: TypeAlias = Player | None

# The eight directions, as (row step, column step). Reversi only ever flips in
# straight lines, so this list is the whole of the game's geometry.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class RefState:
    """A board position. Immutable: applying a move returns a new one.

    ``grid[row][col]`` is the owner of that square, or ``None`` if empty.
    ``to_move`` is whose turn it is.
    """

    grid: tuple[tuple[Cell, ...], ...]
    to_move: Player
    size: int

    def cell(self, row: int, col: int) -> Cell:
        return self.grid[row][col]

    def bitboards(self) -> tuple[Bitboard, Bitboard]:
        """Convert to the two-integer form the fast engine uses.

        Only the cross-check test needs this, so the two engines can be compared
        without either knowing the other's internals. Bit ``i`` is set when
        square ``i`` is occupied, with ``i = row * size + col`` (contract C1).
        """
        black = 0
        white = 0
        for row in range(self.size):
            for col in range(self.size):
                owner = self.grid[row][col]
                if owner is Player.BLACK:
                    black |= 1 << (row * self.size + col)
                elif owner is Player.WHITE:
                    white |= 1 << (row * self.size + col)
        return black, white


# ===========================================================================
# Reading and writing boards as text
#
# Used by the hand-written tests, so each one shows the position it is checking
# and a human can verify it by eye. Also the fastest way to see what went wrong
# when something fails.
# ===========================================================================

_CHARS: dict[str, Cell] = {".": None, "B": Player.BLACK, "W": Player.WHITE}


def from_ascii(text: str, to_move: Player = Player.BLACK) -> RefState:
    """Build a position from a picture of the board.

    ``B`` is a black disc, ``W`` white, ``.`` empty. Row 0 is the top line.

    >>> from_ascii(".B..\\n..W.\\n....\\n....").size
    4
    """
    rows = [line.strip() for line in text.strip().splitlines()]
    rows = [row for row in rows if row]
    size = len(rows)
    if any(len(row) != size for row in rows):
        msg = f"board must be square; got {size} rows with widths {[len(r) for r in rows]}"
        raise ValueError(msg)

    grid: list[tuple[Cell, ...]] = []
    for row in rows:
        if any(char not in _CHARS for char in row):
            msg = f"board may only contain '.', 'B', 'W'; got {row!r}"
            raise ValueError(msg)
        grid.append(tuple(_CHARS[char] for char in row))

    return RefState(grid=tuple(grid), to_move=to_move, size=size)


def to_ascii(state: RefState) -> str:
    """Render a position as text. The inverse of :func:`from_ascii`."""
    symbol = {None: ".", Player.BLACK: "B", Player.WHITE: "W"}
    return "\n".join("".join(symbol[cell] for cell in row) for row in state.grid)


# ===========================================================================
# Coordinates
#
# Contract C1: square index = row * size + col, row 0 at the top, column 0 on
# the left. This exact mapping is shared by the engine, the neural network's
# input encoding, the board symmetries, the web API, and the UI. Defined here
# once, never re-derived.
# ===========================================================================


def to_index(row: int, col: int, size: int) -> Action:
    return row * size + col


def to_coords(index: Action, size: int) -> tuple[int, int]:
    return divmod(index, size)


def on_board(row: int, col: int, size: int) -> bool:
    return 0 <= row < size and 0 <= col < size


# ===========================================================================
# Setting up
# ===========================================================================


def initial_state(size: int = 8) -> RefState:
    """The standard opening: four discs in the centre, arranged crosswise.

    On a full-size board that is White on d4 and e5, Black on d5 and e4, Black
    to move. The same crossed pattern is used for the smaller boards.
    """
    if size < 4 or size % 2 != 0:
        msg = f"board size must be even and at least 4, got {size}"
        raise ValueError(msg)

    mid = size // 2
    grid: list[list[Cell]] = [[None] * size for _ in range(size)]
    grid[mid - 1][mid - 1] = Player.WHITE
    grid[mid][mid] = Player.WHITE
    grid[mid - 1][mid] = Player.BLACK
    grid[mid][mid - 1] = Player.BLACK

    return RefState(
        grid=tuple(tuple(row) for row in grid),
        to_move=Player.BLACK,
        size=size,
    )


# ===========================================================================
# Which moves are legal, and what they flip
# ===========================================================================


def flips_in_direction(
    state: RefState, row: int, col: int, d_row: int, d_col: int, player: Player
) -> list[Action]:
    """Discs flipped by playing at (row, col), looking one direction only.

    Step outward collecting opponent discs. If we then run into one of our own
    discs, having collected at least one, that whole run is trapped and flips.
    If we run off the board or hit an empty square first, nothing flips this way.
    """
    captured: list[Action] = []
    r, c = row + d_row, col + d_col

    while on_board(r, c, state.size) and state.grid[r][c] is player.opponent:
        captured.append(to_index(r, c, state.size))
        r, c = r + d_row, c + d_col

    # The run only counts if it is closed off by one of our own discs.
    if captured and on_board(r, c, state.size) and state.grid[r][c] is player:
        return captured
    return []


def flips_for(state: RefState, index: Action, player: Player) -> list[Action]:
    """Every disc playing ``index`` would flip, across all eight directions.

    Empty if the square is occupied or the move traps nothing — which is exactly
    the condition for that move being illegal.
    """
    row, col = to_coords(index, state.size)
    if not on_board(row, col, state.size) or state.grid[row][col] is not None:
        return []

    captured: list[Action] = []
    for d_row, d_col in DIRECTIONS:
        captured.extend(flips_in_direction(state, row, col, d_row, d_col, player))
    return captured


def flips(state: RefState, index: Action) -> list[Action]:
    """Discs flipped if the player to move plays ``index``."""
    return flips_for(state, index, state.to_move)


def legal_placements_for(state: RefState, player: Player) -> list[Action]:
    """Every square ``player`` could legally play, in ascending index order."""
    return [index for index in range(state.size * state.size) if flips_for(state, index, player)]


def legal_placements(state: RefState) -> list[Action]:
    """Every square the player to move could legally play."""
    return legal_placements_for(state, state.to_move)


# ===========================================================================
# Passing and ending — contract C3
#
#   if I have a placement:      play it;      not over
#   elif my opponent has one:   I must pass;  not over
#   else:                       game over
#
# Note what this does NOT do: it never counts consecutive passes. "Two passes in
# a row" is not a state this code can be in, because a second pass could only
# arise when neither player can move — and that is caught here as the game being
# over. A full board is terminal by this same rule rather than by a special
# case. Fewer branches, fewer places to get it wrong.
# ===========================================================================


def legal_actions(state: RefState) -> list[Action]:
    """Everything the player to move may legally do.

    Squares they can play, or exactly ``[PASS]`` when they have no move but the
    opponent does, or an empty list when the game is over.
    """
    placements = legal_placements(state)
    if placements:
        return placements
    if legal_placements_for(state, state.to_move.opponent):
        return [pass_action(state.size)]
    return []


def is_terminal(state: RefState) -> bool:
    """True when neither player has a legal move."""
    return not legal_placements(state) and not legal_placements_for(state, state.to_move.opponent)


# ===========================================================================
# Playing a move
# ===========================================================================


def apply(state: RefState, action: Action) -> RefState:
    """Play ``action`` and return the resulting position.

    The original is never modified. Copying the whole board is wasteful, and
    that is fine — this file is not on any hot path, and immutability removes a
    whole class of bug where something holds a board that later changes
    underneath it.
    """
    allowed = legal_actions(state)
    if action not in allowed:
        raise IllegalMoveError(_explain_illegal(state, action, allowed))

    if action == pass_action(state.size):
        return RefState(grid=state.grid, to_move=state.to_move.opponent, size=state.size)

    row, col = to_coords(action, state.size)
    grid = [list(r) for r in state.grid]
    grid[row][col] = state.to_move
    for flipped in flips(state, action):
        f_row, f_col = to_coords(flipped, state.size)
        grid[f_row][f_col] = state.to_move

    return RefState(
        grid=tuple(tuple(r) for r in grid),
        to_move=state.to_move.opponent,
        size=state.size,
    )


def _explain_illegal(state: RefState, action: Action, allowed: list[Action]) -> str:
    """Error text carrying enough detail to reconstruct the position.

    An illegal move reaching this point outside a test means some other layer
    failed to filter it (contract C5), so the message has to work as a bug
    report on its own.
    """
    black, white = state.bitboards()
    if action == pass_action(state.size):
        reason = "cannot pass while a placement is available"
    elif not 0 <= action < state.size * state.size:
        reason = f"action out of range for a {state.size}x{state.size} board"
    elif not allowed:
        reason = "the game is already over"
    else:
        row, col = to_coords(action, state.size)
        occupied = state.grid[row][col] is not None
        reason = "square is occupied" if occupied else "move traps no opposing discs"
    return (
        f"illegal action {action}: {reason}. "
        f"to_move={state.to_move.label} size={state.size} "
        f"black=0x{black:016x} white=0x{white:016x} legal={allowed}"
    )


# ===========================================================================
# Scoring
# ===========================================================================


def disc_counts(state: RefState) -> tuple[int, int]:
    """How many discs each player has, as (black, white)."""
    black = sum(1 for row in state.grid for cell in row if cell is Player.BLACK)
    white = sum(1 for row in state.grid for cell in row if cell is Player.WHITE)
    return black, white


def result(state: RefState) -> Outcome:
    """+1 / 0 / -1 for win / draw / loss, **from the mover's point of view**.

    "The player to move" still means something at a finished game even though
    they have no move left: it is whoever would have played next. Keeping the
    perspective rule identical everywhere — one convention, no exceptions — is
    what stops sign errors creeping into the search later (contract C2).

    Empty squares count for nobody. Some Othello variants award them to the
    winner; we use the plain disc count.
    """
    black, white = disc_counts(state)
    mine, theirs = (black, white) if state.to_move is Player.BLACK else (white, black)
    if mine > theirs:
        return 1
    if mine < theirs:
        return -1
    return 0
