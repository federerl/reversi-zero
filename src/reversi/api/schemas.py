"""What goes over the wire.

**Bitboards travel as 16-hex-digit strings, never as numbers.** A board is a
64-bit integer, and JavaScript's `Number` is a double: it represents integers
exactly only up to 2^53. Send a full 8x8 board as JSON and the browser silently
rounds it, so a position with discs in the high squares comes back subtly wrong.
Nothing errors -- the game just develops differently on each side.

**The server is stateless.** Every request carries the whole position, and the
server holds no game. That means no session store, no expiry, no cleanup, and a
restart mid-game costs nothing: the next request rebuilds everything it needs.
It also makes the API trivially testable -- any position can be posted directly,
with no setup.

**The server is authoritative about the rules.** It re-derives the legal moves
for every request rather than trusting the client's view, because a client is
something anyone can edit.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reversi.game.bitboard import geometry, popcount
from reversi.game.state import State
from reversi.types import Player

HEX64 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{16}$", description="a 64-bit board, in hex")]


class Position(BaseModel):
    """A board and whose turn it is. Everything the server needs."""

    model_config = ConfigDict(extra="forbid")

    black: HEX64
    white: HEX64
    to_move: Literal["black", "white"]
    board_size: int = Field(default=8, ge=4, le=8)

    @field_validator("board_size")
    @classmethod
    def _known_size(cls, value: int) -> int:
        if value not in (4, 6, 8):
            msg = f"board_size must be 4, 6 or 8; got {value}"
            raise ValueError(msg)
        return value

    def to_state(self) -> State:
        """Convert and validate. Raises ValueError on anything impossible."""
        black = int(self.black, 16)
        white = int(self.white, 16)
        geo = geometry(self.board_size)

        if black & white:
            msg = "a square cannot hold both a black and a white disc"
            raise ValueError(msg)
        if (black | white) & ~geo.full:
            msg = f"a disc is outside the {self.board_size}x{self.board_size} board"
            raise ValueError(msg)
        if popcount(black | white) < 4:
            msg = "a Reversi position has at least the four starting discs"
            raise ValueError(msg)

        return State(
            black=black,
            white=white,
            to_move=Player.from_label(self.to_move),
            size=self.board_size,
        )

    @classmethod
    def from_state(cls, state: State) -> Position:
        # `label` returns a plain str; the field is a two-value Literal, so the
        # narrowing is spelled out rather than left to the checker to infer.
        to_move: Literal["black", "white"] = "black" if state.to_move is Player.BLACK else "white"
        return cls(
            black=f"{state.black:016x}",
            white=f"{state.white:016x}",
            to_move=to_move,
            board_size=state.size,
        )


class GameState(BaseModel):
    """A position plus everything a client needs to render it."""

    model_config = ConfigDict(extra="forbid")

    position: Position
    legal: list[int]
    is_terminal: bool
    must_pass: bool
    score: dict[str, int]
    result: str | None = Field(
        default=None, description="'black', 'white' or 'draw' once the game is over"
    )


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board_size: int = Field(default=8, ge=4, le=8)


class MoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Position
    action: int = Field(ge=0, description="square index, or board_size**2 for PASS")


class LegalMovesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Position


class AiMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Position
    difficulty: str = "strong"
    want_analysis: bool = False


class Evaluation(BaseModel):
    """How the agent sees the position it is about to move in.

    ``perspective`` is stated explicitly rather than left to a convention. Every
    value in this project is from the point of view of the player to move, and a
    client that assumed "always black" would display the win probability
    backwards on every other turn -- while looking entirely plausible.
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=-1.0, le=1.0)
    win_probability: float = Field(ge=0.0, le=1.0)
    perspective: Literal["black", "white"]


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visits: list[int]
    top_moves: list[dict[str, float | int]]
    simulations: int
    elapsed_ms: float


class AiMoveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: int
    state: GameState
    evaluation: Evaluation
    difficulty: str
    analysis: Analysis | None = None


class DifficultyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    description: str
    simulations: int


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_id: str
    board_size: int
    generation: int | None
    git_commit: str | None
    difficulties: list[DifficultyInfo]


class ErrorBody(BaseModel):
    """What a 4xx carries. The legal moves are included on an illegal move so a
    client can correct itself rather than guess."""

    model_config = ConfigDict(extra="forbid")

    error: str
    legal: list[int] | None = None
