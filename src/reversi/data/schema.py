"""What one recorded position is, and what makes it valid.

Each position self-play keeps produces one training example with three parts:

* **the position itself**, stored as the two bitboards plus whose turn it is;
* **``pi``**, what the search concluded -- the share of its simulations that went
  to each move. This is what the policy head is trained to predict;
* **``z``**, how the game actually ended, from the point of view of whoever was
  to move *in this position*. This is what the value head is trained to predict;
* **``own``**, optionally, who ended up owning each square: +1 the mover, -1 the
  opponent, 0 empty. Sixty-four small answers to "who is winning where", against
  ``z``'s one. A network with an ownership head is trained to predict it; a
  network without one ignores it. Shards written before this field existed have
  no ``own`` column and are still valid.

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
    "OPTIONAL_FIELDS",
    "Arrays",
    "GameRecord",
    "Sample",
    "arrays_to_samples",
    "ownership_for",
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

# Columns a shard may carry. A reader fills in what is missing; a writer includes
# what it has. This is how a training target can be added without invalidating
# every game collected before it existed.
OPTIONAL_FIELDS: tuple[str, ...] = ("own",)


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
    own: NDArray[np.int8] | None = None
    """Who owns each square when the game ends, from this position's mover's view.

    +1 the mover's disc, -1 the opponent's, 0 empty; one entry per square in
    ``row * size + col`` order (contract C1). ``None`` until ``finish``, and
    ``None`` for positions read from a shard written before this field existed.
    """

    def state(self, board_size: int) -> State:
        return State(black=self.black, white=self.white, to_move=self.to_move, size=board_size)


@dataclass(slots=True)
class GameRecord:
    """One finished self-play game: its positions, its result, and some statistics."""

    samples: list[Sample]
    board_size: int
    game_index: int = -1
    """Which game of the generation this was.

    Batched self-play finishes games in whatever order they end, not the order
    they started, so without this a record cannot be matched back to the seed
    that produced it -- which is what makes "batched and unbatched agree" a
    checkable claim rather than a hope.
    """
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
        ownership = {
            Player.BLACK: ownership_for(terminal, Player.BLACK),
            Player.WHITE: ownership_for(terminal, Player.WHITE),
        }
        self.samples = [
            Sample(
                black=sample.black,
                white=sample.white,
                to_move=sample.to_move,
                pi=sample.pi,
                move_no=sample.move_no,
                z=float(result_for(terminal, sample.to_move)),
                own=ownership[sample.to_move],
            )
            for sample in self.samples
        ]


def ownership_for(terminal: State, perspective: Player) -> NDArray[np.int8]:
    """Who owns each square of a finished board, as seen by ``perspective``.

    The ownership target. It is the value target ``z`` broken into its parts: the
    final disc count that decides ``z`` is exactly the sum of this array, so a
    network that predicts ownership well has predicted the result and also
    *where* it comes from. That is far more to learn from per position than one
    number in [-1, 1], which is the whole reason the field exists.
    """
    n = terminal.size * terminal.size
    squares = np.arange(n, dtype=np.uint64)
    mine = terminal.black if perspective == Player.BLACK else terminal.white
    theirs = terminal.white if perspective == Player.BLACK else terminal.black
    mine_bits = (np.uint64(mine) >> squares) & np.uint64(1)
    theirs_bits = (np.uint64(theirs) >> squares) & np.uint64(1)
    return (mine_bits.astype(np.int8) - theirs_bits.astype(np.int8)).astype(np.int8)


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

    with_ownership = sum(1 for s in samples if s.own is not None)
    if with_ownership == count:
        n_squares = board_size * board_size
        own = np.zeros((count, n_squares), dtype=np.int8)
        for row, sample in enumerate(samples):
            assert sample.own is not None  # for the type checker; counted above
            if sample.own.shape != (n_squares,):
                msg = (
                    f"sample {row} has an ownership target of shape {sample.own.shape}, "
                    f"expected ({n_squares},)"
                )
                raise ReplayError(msg)
            own[row] = sample.own
        arrays["own"] = own
    elif with_ownership:
        msg = (
            f"{with_ownership} of {count} samples carry an ownership target; a shard is "
            "all or nothing, so a game that was not finished cannot be mixed in"
        )
        raise ReplayError(msg)

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
            own=np.asarray(arrays["own"][row], dtype=np.int8) if "own" in arrays else None,
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

    if "own" in arrays:
        own = arrays["own"]
        n_squares = board_size * board_size
        if own.shape != (count, n_squares):
            msg = f"ownership targets have shape {own.shape}, expected ({count}, {n_squares})"
            raise ReplayError(msg)
        if not np.isin(own, (-1, 0, 1)).all():
            msg = "ownership targets must be -1 (theirs), 0 (empty) or +1 (mine)"
            raise ReplayError(msg)

    z = arrays["z"]
    if not np.all(np.isin(z, (-1.0, 0.0, 1.0))):
        msg = f"z must be -1, 0 or +1; found values like {z[~np.isin(z, (-1.0, 0.0, 1.0))][:3]}"
        raise ReplayError(msg)
