"""The web API: five endpoints, no session state, the server owns the rules.

Every endpoint takes the whole position and returns the whole position. The
server never remembers a game, which means it can be restarted mid-match without
anyone noticing and any position can be posted directly in a test.

**The server re-derives the legal moves on every request.** A client is something
anyone can edit; if the rules lived only in the browser, a modified client could
play illegal moves and the server would record them as real. An illegal move gets
422 *and the legal move list*, so an honest client can correct itself rather than
guess.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from reversi.api import service
from reversi.api.schemas import (
    AiMoveRequest,
    AiMoveResponse,
    Analysis,
    DifficultyInfo,
    Evaluation,
    GameState,
    Health,
    LegalMovesRequest,
    MoveRequest,
    NewGameRequest,
    Position,
)
from reversi.api.service import BusyError, EngineService
from reversi.difficulty import LEVELS
from reversi.errors import ConfigError, IllegalMoveError
from reversi.game import rules
from reversi.obs.runmeta import git_info

log = logging.getLogger(__name__)

MODEL_ENV = "RZ_MODEL_PATH"
DEFAULT_MODEL = Path("models/reversi-8x8-gen60.pt")


def build_app(model_path: Path | None = None, *, device: str = "cpu") -> FastAPI:
    """Create the application. The model is loaded once, at startup.

    Startup fails loudly if the model is missing or does not match the code.
    A server that starts without a working model would answer health checks
    happily and fail on the first real request.
    """
    resolved = model_path or Path(os.environ.get(MODEL_ENV, DEFAULT_MODEL))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = EngineService(resolved, device=device)
        log.info("serving %s on %s", app.state.engine.model_id, device)
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(
        title="reversi-zero",
        summary="Play against an agent that learned Reversi from self-play.",
        lifespan=lifespan,
    )

    def engine(request: Request) -> EngineService:
        return request.app.state.engine

    # -----------------------------------------------------------------

    @app.exception_handler(BusyError)
    async def _busy(request: Request, exc: BusyError) -> JSONResponse:
        _ = request, exc
        return JSONResponse(
            status_code=429,
            content={"error": "the engine is busy; try again in a moment"},
            headers={"Retry-After": "1"},
        )

    @app.get("/api/health", response_model=Health)
    async def health(request: Request) -> Health:
        service_ = engine(request)
        return Health(
            status="ok",
            model_id=service_.model_id,
            board_size=service_.board_size,
            generation=service_.exported.meta.get("generation"),
            git_commit=git_info().get("commit_short"),
            difficulties=[
                DifficultyInfo(
                    name=level.name,
                    label=level.label,
                    description=level.description,
                    simulations=level.simulations,
                )
                for level in LEVELS
            ],
        )

    @app.post("/api/new-game", response_model=GameState)
    async def new_game(body: NewGameRequest) -> GameState:
        state = service.starting_state(body.board_size)
        return _game_state(state)

    @app.post("/api/legal-moves", response_model=GameState)
    async def legal_moves(body: LegalMovesRequest) -> GameState:
        return _game_state(_state_of(body.position))

    @app.post("/api/move", response_model=GameState)
    async def move(body: MoveRequest) -> GameState:
        state = _state_of(body.position)
        legal = rules.legal_actions(state)

        if body.action not in legal:
            # 422 with the legal set: an honest client can correct itself, and a
            # dishonest one gets nowhere.
            raise HTTPException(
                status_code=422,
                detail={"error": f"action {body.action} is not legal here", "legal": legal},
            )

        try:
            return _game_state(rules.apply(state, body.action))
        except IllegalMoveError as error:  # pragma: no cover - guarded above
            raise HTTPException(
                status_code=422, detail={"error": str(error), "legal": legal}
            ) from error

    @app.post("/api/ai-move", response_model=AiMoveResponse)
    async def ai_move(body: AiMoveRequest, request: Request) -> AiMoveResponse:
        state = _state_of(body.position)

        if rules.is_terminal(state):
            raise HTTPException(
                status_code=422,
                detail={"error": "the game is over; there is no move to make", "legal": []},
            )

        try:
            outcome = await engine(request).choose(state, body.difficulty)
        except ConfigError as error:
            raise HTTPException(status_code=422, detail={"error": str(error)}) from error

        after = rules.apply(state, outcome.action)
        evaluation = service.evaluation_for(state, outcome.result)

        analysis = None
        if body.want_analysis:
            analysis = Analysis(
                visits=[int(v) for v in outcome.result.visit_counts()],
                top_moves=service.top_moves(outcome.result),
                simulations=outcome.level.simulations,
                elapsed_ms=outcome.elapsed_ms,
            )

        return AiMoveResponse(
            action=outcome.action,
            state=_game_state(after),
            evaluation=Evaluation(**evaluation),  # type: ignore[arg-type]
            difficulty=outcome.level.name,
            analysis=analysis,
        )

    return app


# ---------------------------------------------------------------------------


def _state_of(position: Position):  # noqa: ANN202
    """Validate a client-supplied position, or refuse it with a reason."""
    try:
        return position.to_state()
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"error": str(error)}) from error


def _game_state(state) -> GameState:  # noqa: ANN001
    described = service.describe(state)
    return GameState(position=Position.from_state(state), **described)  # type: ignore[arg-type]


app = None
"""Created lazily by ``build_app``; uvicorn is pointed at a factory instead.

A module-level app would load the model at import time, which makes importing
this module for a test require a trained network on disk.
"""
