"""The data a TypeScript port of this engine has to reproduce.

The strongest thing about this project is that the rules were *proved* rather
than assumed: two independent implementations, compared move by move over 50,000
random games, zero mismatches, then frozen. Every strength number since rests on
that.

Running the agent in a browser means writing the rules a third time, in
TypeScript. A subtly wrong port would fail in the worst available way -- it would
not crash. The browser would play a slightly different game from the one the
agent was trained on, the agent would look weaker than it is, and nothing would
report a problem.

So the port is held to the same standard, and the expectations are **generated
rather than hand-written**. A hand-written test encodes what somebody believed
the rules were; a generated one encodes what the frozen engine actually does,
which is what the agent was trained against.

Five fixtures, each aimed at a specific way a port goes wrong:

``rules``
    Legal actions, the exact set of discs each move flips, terminal, score. Ports
    fail here on edge wrap-around -- a shift that walks off column H and reappears
    on column A -- and on the pass rule.
``encoding``
    Positions to the three input planes. Catches a transposed board or the wrong
    player's point of view, which would leave the network reading a position that
    does not exist.
``network``
    Positions to the exported network's own outputs. Catches a broken ONNX export
    or wrong input scaling in the browser.
``search``
    Visit counts from a deterministic evaluator with no exploration noise. This
    is the one that catches the value-sign inversion -- the classic silent killer
    -- because a search with a flipped sign still returns a legal move.
``games``
    Whole games played out move by move. Catches anything the per-position
    fixtures let through.

**On the stub evaluator.** The search fixture cannot use the real network,
because two runtimes doing float arithmetic in different orders will not always
break an exact tie the same way, and one different choice early in a tree changes
every count after it. Instead it uses a stand-in whose answers are a plain
integer hash of the position -- reproducible exactly in any language with a
32-bit multiply. That isolates what this fixture is for: the search arithmetic,
not the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reversi.game import rules, scoring
from reversi.game.state import State
from reversi.nn.features import encode
from reversi.search.config import SearchConfig
from reversi.search.evaluator import StubEvaluator
from reversi.search.mcts import MCTS
from reversi.types import Player, pass_action, policy_size

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "FIXTURE_VERSION",
    "STUB_SEED_SALT",
    "encoding_fixture",
    "games_fixture",
    "network_fixture",
    "rules_fixture",
    "search_fixture",
    "stub_evaluator",
    "write_fixtures",
]

FIXTURE_VERSION = 1

# Mixed into the position hash so the stub's answers are specific to this
# purpose and cannot be confused with any other seeding in the project.
STUB_SEED_SALT = 0x9E3779B9

_U32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# The stub evaluator: a network stand-in that any language can reproduce
# ---------------------------------------------------------------------------


def _mix32(x: int) -> int:
    """A 32-bit integer hash (the well-known "lowbias32" mixer).

    Chosen because it ports exactly. Every step is a 32-bit xor, shift or
    multiply, and JavaScript has all three -- the multiply being ``Math.imul``,
    which is defined to be exactly this operation. A hash built from ordinary
    JavaScript arithmetic would overflow into floating point and silently stop
    matching.
    """
    x &= _U32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & _U32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & _U32
    return (x ^ (x >> 16)) & _U32


def stub_seed(state: State) -> int:
    """A 32-bit fingerprint of a position.

    The board is fed in as four 32-bit halves rather than two 64-bit integers,
    because that is how a JavaScript port has to hold it anyway -- ordinary
    JavaScript numbers stop being exact above 2^53, so a 64-bit board cannot
    survive as one of them.
    """
    seed = STUB_SEED_SALT
    for part in (
        state.black & _U32,
        (state.black >> 32) & _U32,
        state.white & _U32,
        (state.white >> 32) & _U32,
        int(state.to_move),
    ):
        seed = _mix32(seed ^ part)
    return seed


def stub_logit(seed: int, action: int) -> float:
    """One policy logit, in [-2, 2). Division by 2^32 is exact in both languages."""
    return (_mix32(seed ^ (action + 1)) / 4294967296.0) * 4.0 - 2.0


def stub_value(seed: int) -> float:
    """The position's score, in [-1, 1)."""
    return (_mix32(seed ^ 0x5BF03635) / 4294967296.0) * 2.0 - 1.0


def stub_evaluator() -> StubEvaluator:
    """The evaluator the search fixture is generated with.

    Deliberately *opinionated* rather than uniform. A stub that rates every move
    equally makes the search explore in a flat, symmetric way, which would hide a
    prior that was being applied to the wrong action -- the visit counts would
    look plausible either way.
    """

    def policy(state: State) -> NDArray[np.float32]:
        seed = stub_seed(state)
        width = policy_size(state.size)
        return np.array([stub_logit(seed, a) for a in range(width)], dtype=np.float32)

    def value(state: State) -> float:
        return stub_value(stub_seed(state))

    return StubEvaluator(policy=policy, value=value)


# ---------------------------------------------------------------------------
# Sampling positions
# ---------------------------------------------------------------------------


def sample_positions(
    *,
    count: int,
    board_size: int,
    seed: int,
) -> list[State]:
    """Positions drawn from random games, including terminal ones.

    Random play rather than agent play, deliberately. A trained agent visits a
    narrow, sensible slice of the game; the awkward positions where a port breaks
    -- long forced sequences, near-wipeouts, boards where one side must pass
    repeatedly -- are exactly the ones it avoids.
    """
    rng = np.random.default_rng(seed)
    found: list[State] = []
    seen: set[tuple[int, int, int]] = set()

    while len(found) < count:
        state = rules.initial_state(board_size)
        while True:
            key = (state.black, state.white, int(state.to_move))
            if key not in seen:
                seen.add(key)
                found.append(state)
                if len(found) >= count:
                    break
            if rules.is_terminal(state):
                break
            legal = rules.legal_actions(state)
            state = rules.apply(state, int(legal[rng.integers(0, len(legal))]))

    return found[:count]


def _position(state: State) -> list[Any]:
    """A position as the compact triple every fixture keys on."""
    width = (state.size * state.size + 3) // 4
    return [f"{state.black:0{width}x}", f"{state.white:0{width}x}", int(state.to_move)]


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------


def rules_fixture(*, count: int, board_size: int, seed: int) -> dict[str, Any]:
    """Legal actions, flip masks, terminal flag and score for many positions.

    The flip masks are the valuable part. Legal-move generation and flip
    generation share the same eight-direction walk, so a direction that is
    subtly wrong often still produces a plausible-looking legal set while
    flipping the wrong discs.
    """
    cases = []
    for state in sample_positions(count=count, board_size=board_size, seed=seed):
        legal = rules.legal_actions(state)
        black, white = scoring.disc_counts(state)
        cases.append(
            [
                *_position(state),
                list(legal),
                [
                    [int(a), f"{rules.flips(state, a):x}"]
                    for a in legal
                    if a != pass_action(board_size)
                ],
                int(rules.is_terminal(state)),
                black,
                white,
            ]
        )

    return {
        "fixture": "rules",
        "version": FIXTURE_VERSION,
        "board_size": board_size,
        "seed": seed,
        "pass_action": pass_action(board_size),
        "schema": [
            "black_hex",
            "white_hex",
            "to_move",
            "legal_actions",
            "flips_by_action",
            "is_terminal",
            "black_discs",
            "white_discs",
        ],
        "cases": cases,
    }


def encoding_fixture(*, count: int, board_size: int, seed: int) -> dict[str, Any]:
    """Positions to input planes, as bitmasks rather than arrays of floats.

    The planes only ever hold 0.0 or 1.0, so storing them as bitmasks keeps the
    file small *and* keeps the check sharp: it compares the exact square each
    value landed on, which is what contract C1 is about. A transposed board is
    the failure this catches, and it is invisible in a summary statistic.
    """
    cases = []
    width = (board_size * board_size + 3) // 4
    for state in sample_positions(count=count, board_size=board_size, seed=seed):
        planes = encode(state)
        packed = []
        for plane in planes:
            bits = 0
            flat = plane.reshape(-1)
            for index in range(board_size * board_size):
                if flat[index] > 0.5:
                    bits |= 1 << index
            packed.append(f"{bits:0{width}x}")
        cases.append([*_position(state), packed])

    return {
        "fixture": "encoding",
        "version": FIXTURE_VERSION,
        "board_size": board_size,
        "seed": seed,
        "planes": ["mine", "theirs", "my_legal_placements"],
        "note": "plane bit i is row i // size, column i % size (contract C1)",
        "schema": ["black_hex", "white_hex", "to_move", "planes_hex"],
        "cases": cases,
    }


def network_fixture(
    *, count: int, board_size: int, seed: int, onnx_path: Path, decimals: int = 5
) -> dict[str, Any]:
    """What the exported network answers for a set of positions.

    Run through ONNX Runtime rather than PyTorch on purpose: this fixture exists
    to prove the browser and this machine get the same answers from the same
    file, so it should be generated by the same kind of runtime the browser uses.
    """
    import onnxruntime as ort

    from reversi.nn.onnx import run_onnx

    states = sample_positions(count=count, board_size=board_size, seed=seed)
    batch = np.stack([encode(s) for s in states]).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    policy, value = run_onnx(session, batch)

    cases = [
        [
            *_position(state),
            [round(float(v), decimals) for v in policy[row]],
            round(float(value[row][0]), decimals),
        ]
        for row, state in enumerate(states)
    ]

    return {
        "fixture": "network",
        "version": FIXTURE_VERSION,
        "board_size": board_size,
        "seed": seed,
        "onnx": onnx_path.name,
        "decimals": decimals,
        "tolerance": 10.0**-decimals * 5,
        "schema": ["black_hex", "white_hex", "to_move", "policy_logits", "value"],
        "cases": cases,
    }


def search_fixture(
    *,
    count: int,
    board_size: int,
    seed: int,
    simulations: int = 64,
) -> dict[str, Any]:
    """Visit counts from a search with a reproducible stand-in for the network.

    Exploration noise is off, so the search is fully deterministic: the same
    position and the same evaluator must produce the same tree every time. That
    is what makes an exact comparison meaningful.

    Terminal positions are skipped -- there is no search to run -- and so are
    positions with a single legal action, where the real code skips the search
    entirely anyway.
    """
    evaluator = stub_evaluator()
    config = SearchConfig(n_simulations=simulations, dirichlet_eps=0.0, temp_moves=0)
    config.assert_no_noise("the search fixture")

    cases = []
    for state in sample_positions(count=count * 3, board_size=board_size, seed=seed):
        if rules.is_terminal(state) or len(rules.legal_actions(state)) < 2:
            continue

        result = MCTS(evaluator, config).run(state)
        cases.append(
            [
                *_position(state),
                list(result.actions),
                list(result.visits),
                round(float(result.root_value), 6),
                int(result.best_action()),
            ]
        )
        if len(cases) >= count:
            break

    return {
        "fixture": "search",
        "version": FIXTURE_VERSION,
        "board_size": board_size,
        "seed": seed,
        "simulations": simulations,
        "evaluator": {
            "kind": "hash-stub",
            "salt": STUB_SEED_SALT,
            "note": (
                "logit(a) = mix32(seed ^ (a+1)) / 2^32 * 4 - 2; "
                "value = mix32(seed ^ 0x5BF03635) / 2^32 * 2 - 1; "
                "seed = fold of mix32 over black_lo, black_hi, white_lo, white_hi, to_move"
            ),
        },
        "note": (
            "Exact visit-count agreement requires the port to round its PUCT "
            "arithmetic to float32 the way numpy does (Math.fround in JavaScript). "
            "Without that, compare the chosen move and allow a stated tolerance on "
            "the counts -- and say which was done."
        ),
        "schema": ["black_hex", "white_hex", "to_move", "actions", "visits", "root_value", "best"],
        "cases": cases,
    }


def games_fixture(*, count: int, board_size: int, seed: int) -> dict[str, Any]:
    """Whole games as move lists, with the position after every move.

    Per-position fixtures check each rule in isolation. This one checks that
    applying them in sequence stays in step -- the failure where two engines each
    look right and drift apart anyway.
    """
    rng = np.random.default_rng(seed)
    games = []

    for _ in range(count):
        state = rules.initial_state(board_size)
        moves: list[int] = []
        positions: list[list[Any]] = [_position(state)]

        while not rules.is_terminal(state):
            legal = rules.legal_actions(state)
            action = int(legal[rng.integers(0, len(legal))])
            state = rules.apply(state, action)
            moves.append(action)
            positions.append(_position(state))

        black, white = scoring.disc_counts(state)
        games.append(
            {
                "moves": moves,
                "positions": positions,
                "final_score": [black, white],
                "winner": "black" if black > white else "white" if white > black else "draw",
            }
        )

    return {
        "fixture": "games",
        "version": FIXTURE_VERSION,
        "board_size": board_size,
        "seed": seed,
        "games": games,
    }


# ---------------------------------------------------------------------------


def write_fixtures(
    destination: Path,
    *,
    board_size: int = 8,
    seed: int = 20260830,
    rules_count: int = 1000,
    encoding_count: int = 200,
    network_count: int = 64,
    search_count: int = 40,
    games_count: int = 20,
    simulations: int = 64,
    onnx_path: Path | None = None,
) -> dict[str, int]:
    """Write every fixture into ``destination``, returning file sizes in bytes.

    The network fixture is skipped when no ONNX file is given, so the rules,
    encoding, search and game fixtures can be regenerated without a trained
    model present -- which is what CI does when it checks they are still current.

    The committed sizes are deliberately modest, following the split this repo
    already uses for the differential test: enough coverage to catch a regression
    within minutes of a push, with a far larger run (``rules_count=20000``)
    regenerated nightly. Every file stays under the 500 KB limit that keeps large
    artifacts out of git history.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    payloads: dict[str, dict[str, Any]] = {
        "rules": rules_fixture(count=rules_count, board_size=board_size, seed=seed),
        "encoding": encoding_fixture(count=encoding_count, board_size=board_size, seed=seed + 1),
        "search": search_fixture(
            count=search_count, board_size=board_size, seed=seed + 2, simulations=simulations
        ),
        "games": games_fixture(count=games_count, board_size=board_size, seed=seed + 3),
    }
    if onnx_path is not None:
        payloads["network"] = network_fixture(
            count=network_count, board_size=board_size, seed=seed + 4, onnx_path=onnx_path
        )

    for name, payload in payloads.items():
        path = destination / f"{name}.json"
        # Separators without spaces: these files are committed, and a fixture is
        # read by a test rather than by a person.
        path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n", encoding="utf-8"
        )
        written[name] = path.stat().st_size

    return written


def player_of(code: int) -> Player:
    """Turn the fixture's ``to_move`` code back into a player."""
    return Player.BLACK if code == int(Player.BLACK) else Player.WHITE


def state_of(black_hex: str, white_hex: str, to_move: int, board_size: int) -> State:
    """Rebuild a state from a fixture row, so tests can round-trip one."""
    return State(
        black=int(black_hex, 16),
        white=int(white_hex, 16),
        to_move=player_of(to_move),
        size=board_size,
    )
