"""Edax, as one more entrant in the tournament.

Every strength number in this project so far is the agent measured against
baselines written in this repository. That is internally consistent and entirely
self-referential: the scale is anchored at a random player we wrote, shaped by a
minimax we wrote, and it says nothing about how the agent compares to anything
outside.

`Edax <https://github.com/abulmo/edax-reversi>`_ is the reference Othello engine
-- open source, widely used, and far stronger than anything here at full
strength. Running it as one more agent in the same round robin puts our ratings
next to something a reader can recognise and, if they doubt it, download and
check.

**It is a reference point, not a target.** Edax at its default level is
superhuman; a 458k-parameter network trained for nine hours is not going to beat
it, and a comparison that only reported "we lost" would be worth little. The
useful question is *which level* it takes to hold our agent even, because that
converts an unanchored rating into a statement other people can place.

**How it is driven.** Edax's own protocol has `setboard`, which its GTP mode does
not, and that difference decides the design: with `setboard` each move is a fresh
question about a position, so this needs no game history and drops straight into
the `Agent` protocol. Through GTP the moves would have to be replayed from the
start of the game, which would mean threading a move list through the arena for
one agent's benefit.

**On the opening book.** Turned off. Edax ships a book of memorised lines, and
with it the early moves would come from a lookup rather than from the search --
so "Edax at level 4" would not mean what it says, and the arena's own opening
book (which is what keeps the games varied and fair) would be fighting it.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from reversi.errors import ArenaError
from reversi.game import rules
from reversi.game.bitboard import indices
from reversi.types import Action, Player, pass_action

if TYPE_CHECKING:
    import numpy as np

    from reversi.game.state import State

__all__ = ["EDAX_ROOT", "EdaxAgent", "find_edax"]

# Where `make edax` (and the runbook) put it. Not in git: it is a 14 MB binary
# plus a 14 MB evaluation table, and the same rule that keeps checkpoints out
# applies.
EDAX_ROOT = Path("tools/edax")

# The build to prefer, most specific first. `v3` wants AVX2 and will not start
# on a CPU without it, so the plain build is the one that always works.
_BUILDS = ("wEdax-x86-64-v3.exe", "wEdax-x86-64-v2.exe", "wEdax-x86-64.exe", "lEdax-x86-64", "edax")

_FILES = "abcdefgh"


def find_edax(root: Path = EDAX_ROOT) -> Path:
    """Locate an Edax executable, or explain how to get one."""
    for name in _BUILDS:
        candidate = root / name
        if candidate.is_file():
            return candidate

    msg = (
        f"no Edax executable under {root}/. It is not in this repository -- it is a "
        "binary and an evaluation table, and binaries do not go in git. Fetch it with:\n"
        "  gh release download v4.6 --repo abulmo/edax-reversi "
        f"--pattern '*MS-windows*' --dir {root}\n"
        f"then unzip it there, so that {root}/data/eval.dat exists."
    )
    raise ArenaError(msg)


def square_name(action: Action) -> str:
    """An action as Edax writes it: ``19`` becomes ``D3``.

    The two agree already -- index ``row * 8 + col`` with row 0 at the top reads
    the board in the same order Edax does, so this is a relabelling rather than a
    transformation. Worth stating explicitly all the same: a silent disagreement
    here would show up as an opponent that plays legally and badly.
    """
    return f"{_FILES[action % 8].upper()}{action // 8 + 1}"


def square_index(name: str) -> Action:
    """The inverse. Raises on anything that is not a square."""
    text = name.strip().lower()
    if len(text) != 2 or text[0] not in _FILES or not text[1].isdigit():
        msg = f"{name!r} is not a square Edax should have produced"
        raise ArenaError(msg)
    row = int(text[1]) - 1
    if not 0 <= row < 8:
        msg = f"{name!r} is off the board"
        raise ArenaError(msg)
    return row * 8 + _FILES.index(text[0])


def board_string(state: State) -> str:
    """A position in Edax's notation: 64 squares, then whose turn it is.

    ``*`` is black and ``O`` is white, which is what Edax prints and what it
    reads back.
    """
    if state.size != 8:
        msg = f"Edax plays 8x8 Othello; asked for a {state.size}x{state.size} board"
        raise ArenaError(msg)

    squares = ["-"] * 64
    for index in indices(state.black):
        squares[index] = "*"
    for index in indices(state.white):
        squares[index] = "O"

    to_move = "*" if state.to_move is Player.BLACK else "O"
    return "".join(squares) + " " + to_move


class EdaxAgent:
    """One Edax process, answering one position at a time.

    The process is kept alive between moves. Starting it costs about a second --
    it loads a 14 MB evaluation table -- which would dominate a match at the
    lower levels where a move takes milliseconds.
    """

    __slots__ = ("_lines", "_name", "_process", "_reader", "level", "path")

    def __init__(
        self,
        level: int = 4,
        *,
        path: Path | None = None,
        name: str | None = None,
        hash_bits: int = 18,
    ) -> None:
        if level < 0:
            msg = f"Edax level cannot be negative, got {level}"
            raise ArenaError(msg)

        self.level = level
        self.path = path if path is not None else find_edax()
        self._name = name if name is not None else f"edax-l{level}"

        self._process = subprocess.Popen(
            [
                str(self.path.resolve()),
                "-edax",
                "-level",
                str(level),
                # One thread: the arena runs many of these at once, and letting
                # each grab every core turns parallel matches into contention --
                # the same mistake the calibration workers made.
                "-n-tasks",
                "1",
                # See the module docstring: with the book on, "level N" would
                # describe a lookup rather than a search.
                "-book-usage",
                "off",
                "-verbose",
                "0",
                "-auto-start",
                "off",
                "-hash-table-size",
                str(hash_bits),
            ],
            cwd=str(self.path.resolve().parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read on a thread. Edax writes prompts and warnings as well as answers,
        # and a blocking readline against a process that has nothing more to say
        # would hang the match rather than fail it.
        self._lines: Queue[str] = Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    @property
    def name(self) -> str:
        return self._name

    def _pump(self) -> None:
        stdout = self._process.stdout
        if stdout is None:  # pragma: no cover - set by Popen above
            return
        for line in stdout:
            self._lines.put(line.rstrip("\r\n"))

    def _send(self, command: str) -> None:
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            msg = f"{self._name} is no longer running"
            raise ArenaError(msg)
        stdin.write(command + "\n")
        stdin.flush()

    def _await_move(self, timeout: float) -> Action:
        """Read until Edax announces a move, or give up and say so."""
        seen: list[str] = []
        deadline = timeout
        while deadline > 0:
            try:
                line = self._lines.get(timeout=0.05)
            except Empty:
                deadline -= 0.05
                continue

            stripped = line.strip()
            if not stripped or stripped == ">":
                continue
            seen.append(stripped)

            if stripped.lower().startswith("edax plays"):
                return square_index(stripped.split()[-1])
            if "illegal" in stripped.lower() or "unknown command" in stripped.lower():
                msg = f"{self._name} refused the position: {stripped}"
                raise ArenaError(msg)

        msg = (
            f"{self._name} did not answer within {timeout:.0f}s at level {self.level}. "
            f"It said: {seen if seen else '(nothing)'}"
        )
        raise ArenaError(msg)

    def select(self, state: State, rng: np.random.Generator) -> Action:
        """Set the position, ask for a move, and check what comes back."""
        _ = rng  # Edax at a fixed level is deterministic; nothing to sample.

        legal = rules.legal_actions(state)
        if not legal:
            msg = f"asked for a move in a finished position:\n{state}"
            raise ArenaError(msg)

        # A forced pass never reaches Edax. There is nothing to decide, and it
        # spares us depending on how it spells one.
        if legal == [pass_action(state.size)]:
            return legal[0]

        self._send(f"setboard {board_string(state)}")
        self._send("go")
        action = self._await_move(timeout=30.0 + 5.0 * self.level)

        if action not in legal:
            # Would mean the board was transcribed wrongly rather than that Edax
            # played badly. Failing here beats recording the game as if it were
            # real.
            msg = (
                f"{self._name} returned {square_name(action)}, which is not legal here. "
                f"Legal: {[square_name(a) for a in legal if a != pass_action(8)]}\n{state}"
            )
            raise ArenaError(msg)
        return action

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._send("quit")
                self._process.wait(timeout=5)
            except (ArenaError, subprocess.TimeoutExpired):  # pragma: no cover
                self._process.kill()

    def __del__(self) -> None:  # pragma: no cover - best effort
        with contextlib.suppress(Exception):
            self.close()
