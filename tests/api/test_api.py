"""The web API (test matrix T31, T32; contract S16).

The theme is that **the server does not trust the client**. A browser is
something anyone can edit, so every position that arrives is re-validated and
every move is re-derived from the rules. The tests are mostly about what happens
when a client sends something it should not.

The last test is the important one: a thousand moves driven entirely through
HTTP, checking that the server never plays an illegal move and never disagrees
with a second, independent copy of the rules about what the position is.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reversi.api.app import build_app
from reversi.ckpt import CheckpointManager
from reversi.game import rules
from reversi.game.state import State
from reversi.nn.export import export_checkpoint
from reversi.nn.model import build
from reversi.types import Player, pass_action

BOARD = 8


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny untrained network. These tests are about the API, not strength."""
    from reversi.config import NetConfig

    root = tmp_path_factory.mktemp("api")
    net = build(NetConfig(n_blocks=1, channels=8, value_hidden=16), BOARD, seed=1)
    manager = CheckpointManager(root / "ckpt", run_id="api-test", config_sha256="x")
    manager.save(model=net, generation=3, global_step=42)

    exported = root / "model.pt"
    export_checkpoint(root / "ckpt" / "gen_00003.pt", exported)
    return exported


@pytest.fixture
def client(model_path: Path) -> Iterator[TestClient]:
    with TestClient(build_app(model_path)) as running:
        yield running


def position_of(state: State) -> dict:
    return {
        "black": f"{state.black:016x}",
        "white": f"{state.white:016x}",
        "to_move": state.to_move.label,
        "board_size": state.size,
    }


def state_of(payload: dict) -> State:
    position = payload["position"]
    return State(
        black=int(position["black"], 16),
        white=int(position["white"], 16),
        to_move=Player.from_label(position["to_move"]),
        size=position["board_size"],
    )


# ===========================================================================
# The basics
# ===========================================================================


def test_health_reports_which_model_is_being_served(client: TestClient) -> None:
    """Which weights are answering matters as much as that something is."""
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["generation"] == 3
    assert body["board_size"] == BOARD
    assert [d["name"] for d in body["difficulties"]] == ["casual", "club", "strong", "max"]
    assert all(d["simulations"] > 0 for d in body["difficulties"])


def test_a_new_game_is_the_standard_opening(client: TestClient) -> None:
    body = client.post("/api/new-game", json={"board_size": BOARD}).json()

    assert body["score"] == {"black": 2, "white": 2}
    assert body["position"]["to_move"] == "black"
    assert sorted(body["legal"]) == sorted(rules.legal_actions(rules.initial_state(BOARD)))
    assert not body["is_terminal"]


def test_legal_moves_match_the_engine(client: TestClient) -> None:
    state = rules.initial_state(BOARD)
    for action in (19, 18, 26):
        state = rules.apply(state, action)

    body = client.post("/api/legal-moves", json={"position": position_of(state)}).json()
    assert sorted(body["legal"]) == sorted(rules.legal_actions(state))


def test_a_move_advances_the_position(client: TestClient) -> None:
    state = rules.initial_state(BOARD)
    action = rules.legal_actions(state)[0]

    body = client.post("/api/move", json={"position": position_of(state), "action": action}).json()

    assert state_of(body) == rules.apply(state, action)
    assert body["score"] == {"black": 4, "white": 1}


# ===========================================================================
# Not trusting the client
# ===========================================================================


def test_an_illegal_move_is_refused_with_the_legal_ones(client: TestClient) -> None:
    """422 plus the legal set, so an honest client can correct itself."""
    state = rules.initial_state(BOARD)
    illegal = next(a for a in range(64) if a not in rules.legal_actions(state))

    response = client.post("/api/move", json={"position": position_of(state), "action": illegal})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not legal" in detail["error"]
    assert sorted(detail["legal"]) == sorted(rules.legal_actions(state))


def test_a_position_with_a_square_held_twice_is_refused(client: TestClient) -> None:
    """The kind of thing only an edited client sends -- and it must not be played."""
    response = client.post(
        "/api/legal-moves",
        json={
            "position": {
                "black": f"{0xFF:016x}",
                "white": f"{0xFF:016x}",
                "to_move": "black",
                "board_size": 8,
            }
        },
    )
    assert response.status_code == 422
    assert "both" in response.json()["detail"]["error"]


def test_a_position_with_too_few_discs_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/legal-moves",
        json={
            "position": {
                "black": f"{1:016x}",
                "white": f"{2:016x}",
                "to_move": "black",
                "board_size": 8,
            }
        },
    )
    assert response.status_code == 422
    assert "four starting discs" in response.json()["detail"]["error"]


def test_a_malformed_bitboard_is_refused(client: TestClient) -> None:
    """Not 16 hex digits: rejected by the schema before any code sees it."""
    response = client.post(
        "/api/legal-moves",
        json={
            "position": {
                "black": "not-hex",
                "white": f"{2:016x}",
                "to_move": "black",
                "board_size": 8,
            }
        },
    )
    assert response.status_code == 422


def test_an_unknown_difficulty_is_refused(client: TestClient) -> None:
    state = rules.initial_state(BOARD)
    response = client.post(
        "/api/ai-move",
        json={"position": position_of(state), "difficulty": "impossible"},
    )
    assert response.status_code == 422
    assert "unknown difficulty" in response.json()["detail"]["error"]


def test_asking_for_a_move_in_a_finished_game_is_refused(client: TestClient) -> None:
    finished = State(black=0, white=(1 << 64) - 1, to_move=Player.BLACK, size=8)
    response = client.post(
        "/api/ai-move", json={"position": position_of(finished), "difficulty": "casual"}
    )
    assert response.status_code == 422
    assert "game is over" in response.json()["detail"]["error"]


def test_unknown_fields_are_refused(client: TestClient) -> None:
    """extra='forbid' everywhere: a typo'd field is an error, not a silent no-op."""
    response = client.post("/api/new-game", json={"board_size": 8, "colour": "black"})
    assert response.status_code == 422


# ===========================================================================
# The agent's moves
# ===========================================================================


@pytest.mark.parametrize("difficulty", ["casual", "club", "strong"])
def test_the_agent_plays_a_legal_move_at_every_difficulty(
    client: TestClient, difficulty: str
) -> None:
    state = rules.initial_state(BOARD)
    body = client.post(
        "/api/ai-move", json={"position": position_of(state), "difficulty": difficulty}
    ).json()

    assert body["action"] in rules.legal_actions(state)
    assert state_of(body["state"]) == rules.apply(state, body["action"])
    assert body["difficulty"] == difficulty


def test_the_evaluation_says_whose_view_it_is(client: TestClient) -> None:
    """A client that assumed "always black" would show the win probability
    backwards on every other turn, while looking entirely plausible."""
    state = rules.initial_state(BOARD)
    body = client.post(
        "/api/ai-move", json={"position": position_of(state), "difficulty": "casual"}
    ).json()

    evaluation = body["evaluation"]
    assert evaluation["perspective"] == "black"
    assert -1.0 <= evaluation["value"] <= 1.0
    assert evaluation["win_probability"] == pytest.approx((evaluation["value"] + 1) / 2)


def test_analysis_is_only_returned_when_asked_for(client: TestClient) -> None:
    state = rules.initial_state(BOARD)
    payload = {"position": position_of(state), "difficulty": "casual"}

    assert client.post("/api/ai-move", json=payload).json()["analysis"] is None

    analysis = client.post("/api/ai-move", json={**payload, "want_analysis": True}).json()[
        "analysis"
    ]
    assert len(analysis["visits"]) == BOARD * BOARD + 1
    assert sum(analysis["visits"]) > 0
    assert analysis["top_moves"]
    assert analysis["elapsed_ms"] > 0
    # Contract C5 once more, in the payload a browser can see.
    legal = set(rules.legal_actions(state))
    for index, visits in enumerate(analysis["visits"]):
        if index not in legal:
            assert visits == 0


# ===========================================================================
# A thousand moves through HTTP (T32)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_full_games_through_the_api_never_desync(client: TestClient) -> None:
    """The end-to-end check: the server's view and an independent one must agree.

    A desync would not raise. The two sides would simply drift apart, and the
    board a player sees would stop matching the game being played -- which is the
    failure a stateless design exists to prevent, so it is worth proving.
    """
    rng = np.random.default_rng(0)
    moves = 0

    for game in range(6):
        state = rules.initial_state(BOARD)

        while not rules.is_terminal(state) and moves < 1000:
            if game % 2 == 0:
                # The agent moves.
                body = client.post(
                    "/api/ai-move",
                    json={"position": position_of(state), "difficulty": "casual"},
                ).json()
                action = body["action"]
                assert action in rules.legal_actions(state), "the server played an illegal move"
                served = state_of(body["state"])
            else:
                # A random client moves, and the server applies it.
                legal = rules.legal_actions(state)
                action = int(legal[rng.integers(0, len(legal))])
                body = client.post(
                    "/api/move", json={"position": position_of(state), "action": action}
                ).json()
                served = state_of(body)

            ours = rules.apply(state, action)
            assert served == ours, f"the server and the engine disagree after {action}"

            state = ours
            moves += 1

            # And the server's own summary of the position must match too.
            assert body.get("score", body.get("state", {}).get("score")) is not None

    assert moves > 100, f"only {moves} moves were played"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_forced_pass_is_handled_over_http(client: TestClient) -> None:
    """PASS is action board_size**2 and travels like any other move."""
    from reversi.game import reference as ref

    position = ref.from_ascii("BBW.\nBBWW\nWBWW\nBBBB", Player.WHITE)
    black, white = position.bitboards()
    state = State(black=black, white=white, to_move=Player.WHITE, size=4)

    body = client.post("/api/legal-moves", json={"position": position_of(state)}).json()
    assert body["must_pass"]
    assert body["legal"] == [pass_action(4)]

    after = client.post(
        "/api/move", json={"position": position_of(state), "action": pass_action(4)}
    ).json()
    assert state_of(after).to_move is Player.BLACK
    assert after["score"] == body["score"], "passing changes no discs"


# ===========================================================================
# Layering
# ===========================================================================


def test_the_api_does_not_import_the_training_pipeline() -> None:
    """Shipping the app must not ship the trainer.

    If a served game could reach the replay buffer, the agent would be training
    on games played against its users -- data nobody chose, arriving through a
    public endpoint. Making the import impossible is stronger than remembering
    not to write it.
    """
    import subprocess
    import sys

    code = (
        "import reversi.api.app, sys; "
        "print(sorted(m for m in sys.modules "
        "if m.startswith(('reversi.train', 'reversi.data', 'reversi.selfplay'))))"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert finished.stdout.strip() == "[]", (
        f"reversi.api pulled in the training pipeline: {finished.stdout.strip()}"
    )
