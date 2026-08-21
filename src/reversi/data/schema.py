"""What one recorded position is, and what makes it valid.

Each position self-play keeps produces one training example with three parts:

* **the position itself**, stored as the two bitboards plus whose turn it is;
* **``pi``**, what the search concluded -- the share of its simulations that went
  to each move. This is what the policy head is trained to predict;
* **``z``**, how the game actually ended, from the point of view of whoever was
  to move *in this position*. This is what the value head is trained to predict.

**We store the position, not the encoded planes.** Re-deriving the three input
planes at sampling time costs microseconds. Storing them instead would mean that
the day we change the encoding -- add a plane, drop one, change their order --
every game collected up to that point becomes unusable. Positions are the durable
thing; the encoding is a detail of the current network.

It also happens to be far smaller: two 8-byte integers instead of 192 floats.

**Why ``z`` is the final result and not a discounted return.** There are no
intermediate rewards in Reversi. Nothing happens until the game ends and you
count discs. So the honest target for every position in a game is the same
number, flipped for whichever side was to move. No discounting, no bootstrapping,
no TD error -- the game is short enough to just wait for the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from reversi.errors import ReplayError
from reversi.game.state import State
from reversi.types import Outcome, Player, policy_size

__all__ = [
    "FIELDS",
    "Arrays",
    "GameRecord",
    "Sample",
    "arrays_to_samples",
    "samples_to_arrays",
    "validate_arrays",
]

# The columns of a shard, keyed by field name. Deliberately not narrowed to a
# single dtype: the fields genuinely differ (uint64 bitboards, int8 colours,
# float32 targets), and a narrower alias only forces casts at every use.
Arrays: TypeAlias = dict[str, NDArray[Any]]

# The arrays that make up a shard. Everything is fixed-width, so a shard is a
# handful of contiguous blocks rather than a pickle of Python objects -- which
# means it can be memory-mapped later if the window outgrows RAM.
FIELDS: tuple[str, ...] = ("black", "white", "to_move", "pi", "z", "move_no", "generation")


@dataclass(frozen=True, slots=True)
class Sample:
    """One position, with the two things the network is trained to predict."""

    black: int
    white: int
    to_move: Player
    pi: NDArray[np.float32]
    """The search's visit distribution over all actions, at temperature 1.

    Raw visit shares -- never sharpened, never flattened (contract C4). The
    temperature that makes self-play vary its openings changes which move gets
    *played*, and must not touch this.
    """
    move_no: int
    z: float = 0.0
    """The final result, from this position's mover's point of view.

    Zero until the game finishes and ``GameRecord.finish`` fills it in -- the
    answer does not exist while the game is still being played.
    """

    def state(self, board_size: int) -> State:
        return State(black=self.black, white=self.white, to_move=self.to_move, size=board_size)


@dataclass(slots=True)
class GameRecord:
    """One finished self-play game: its positions, its result, and some statistics."""

    samples: list[Sample]
    board_size: int
    plies: int = 0
    passes: int = 0
    result_for_black: Outcome = 0
    skipped_forced_moves: int = 0
    """Plies where only one move was legal. No search was run and no sample was
    kept: there was no decision to record, and training on "the only legal move
    had probability 1" teaches the network nothing it can use."""

    def finish(self, terminal: State) -> None:
        """Fill in ``z`` for every stored position, now that the result is known.

        Each position gets the outcome as seen by whoever was to move *there*, so
        consecutive positions in a game alternate sign (contract C2 again, this
        time on the training target rather than in the tree).
        """
        from reversi.game.scoring import result_for

        self.result_for_black = result_for(terminal, Player.BLACK)
        self.samples = [
            Sample(
                black=sample.black,
                white=sample.white,
                to_move=sample.to_move,
                pi=sample.pi,
                move_no=sample.move_no,
                z=float(result_for(terminal, sample.to_move)),
            )
            for sample in self.samples
        ]


def samples_to_arrays(
    samples: Sequence[Sample],
    *,
    generation: int,
    board_size: int,
) -> Arrays:
    """Pack samples into the arrays a shard is made of."""
    if not samples:
        msg = "cannot build a shard from zero samples"
        raise ReplayError(msg)

    width = policy_size(board_size)
    count = len(samples)

    pi = np.zeros((count, width), dtype=np.float32)
    for row, sample in enumerate(samples):
        if sample.pi.shape != (width,):
            msg = (
                f"sample {row} has a policy of shape {sample.pi.shape}, "
                f"expected ({width},) for a {board_size}x{board_size} board"
            )
            raise ReplayError(msg)
        pi[row] = sample.pi

    arrays = {
        "black": np.fromiter((s.black for s in samples), dtype=np.uint64, count=count),
        "white": np.fromiter((s.white for s in samples), dtype=np.uint64, count=count),
        "to_move": np.fromiter((int(s.to_move) for s in samples), dtype=np.int8, count=count),
        "pi": pi,
        "z": np.fromiter((s.z for s in samples), dtype=np.float32, count=count),
        "move_no": np.fromiter((s.move_no for s in samples), dtype=np.int16, count=count),
        "generation": np.full(count, generation, dtype=np.int32),
    }
    validate_arrays(arrays, board_size=board_size)
    return arrays


def arrays_to_samples(
    arrays: Arrays,
    *,
    board_size: int,
) -> list[Sample]:
    """Unpack a shard back into samples. Used by tests and by inspection tools."""
    validate_arrays(arrays, board_size=board_size)
    return [
        Sample(
            black=int(arrays["black"][row]),
            white=int(arrays["white"][row]),
            to_move=Player(int(arrays["to_move"][row])),
            pi=np.asarray(arrays["pi"][row], dtype=np.float32),
            move_no=int(arrays["move_no"][row]),
            z=float(arrays["z"][row]),
        )
        for row in range(len(arrays["black"]))
    ]


def validate_arrays(arrays: Arrays, *, board_size: int) -> None:
    """Refuse anything malformed before it reaches disk, or after it comes back.

    These checks are cheap and they run on every write and every read. That is
    deliberate: a shard is the one artifact in this project that outlives the
    process which made it, so a bad one poisons every generation that samples it,
    and it would do so silently -- a policy target that does not sum to 1 is still
    a perfectly trainable number.
    """
    missing = [field for field in FIELDS if field not in arrays]
    if missing:
        msg = f"shard is missing {missing}; expected all of {list(FIELDS)}"
        raise ReplayError(msg)

    count = len(arrays["black"])
    if count == 0:
        msg = "shard contains no positions"
        raise ReplayError(msg)
    for field in FIELDS:
        if len(arrays[field]) != count:
            msg = f"shard field {field!r} has {len(arrays[field])} rows but 'black' has {count}"
            raise ReplayError(msg)

    width = policy_size(board_size)
    if arrays["pi"].shape != (count, width):
        msg = f"policy targets have shape {arrays['pi'].shape}, expected ({count}, {width})"
        raise ReplayError(msg)

    black = arrays["black"].astype(np.uint64)
    white = arrays["white"].astype(np.uint64)
    if np.any(black & white):
        msg = "a position claims the same square for both players"
        raise ReplayError(msg)

    to_move = arrays["to_move"]
    if np.any((to_move != int(Player.BLACK)) & (to_move != int(Player.WHITE))):
        msg = "to_move must be 0 (black) or 1 (white)"
        raise ReplayError(msg)

    sums = arrays["pi"].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        worst = int(np.argmax(np.abs(sums - 1.0)))
        msg = (
            f"policy target at row {worst} sums to {sums[worst]:.6f}, not 1. "
            "A target that is not a distribution still trains, which is why this "
            "is checked rather than assumed."
        )
        raise ReplayError(msg)
    if np.any(arrays["pi"] < 0.0):
        msg = "policy targets contain a negative probability"
        raise ReplayError(msg)

    z = arrays["z"]
    if not np.all(np.isin(z, (-1.0, 0.0, 1.0))):
        msg = f"z must be -1, 0 or +1; found values like {z[~np.isin(z, (-1.0, 0.0, 1.0))][:3]}"
        raise ReplayError(msg)
