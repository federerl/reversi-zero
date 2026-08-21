"""Turning a board position into the numbers the network reads.

The network never sees colours. It sees three square-shaped grids, always from
the point of view of whoever is about to move:

    plane 0   my discs
    plane 1   my opponent's discs
    plane 2   the squares I may legally play

That choice is contract C1, and it is the reason the network only has to learn
Reversi once instead of twice. Black to move with a given arrangement and White
to move with the mirrored arrangement are the *same problem*, so they get the
same input and share every bit of training signal. Adding a "which colour am I"
plane would split that in half for no gain.

**Why the third plane, when it is derivable from the first two.** It is
redundant in the strict sense: a network could work out the legal moves from the
disc positions. But working them out means tracing runs of discs outward in
eight directions, which is a long chain of reasoning for a small convolutional
network to synthesise, and it would have to spend capacity doing it in every
position. We already know the answer for free -- the engine just computed it --
so we hand it over. Cheap for us, expensive for the network.

**What is deliberately absent.** No history planes. Chess and Go need them
because of repetition and ko rules; Reversi does not, because the position alone
tells you everything about what may happen next. Feeding the last eight
positions would triple the input size to encode nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from reversi.game import rules
from reversi.game.state import State
from reversi.types import Bitboard, policy_size

__all__ = [
    "IN_PLANES",
    "encode",
    "encode_batch",
    "legal_mask",
    "masked_policy",
    "planes_to_bits",
]

# Planes per position: mine, theirs, my legal placements.
IN_PLANES = 3


def _bits_to_plane(bits: Bitboard, size: int) -> NDArray[np.float32]:
    """Unpack a bitboard into a ``size x size`` grid of 0.0 / 1.0.

    Bit ``i`` becomes ``grid[i // size, i % size]``, which is contract C1's
    ``index = row * size + col`` and nothing else. Every layer that converts
    between squares and array positions goes through this line, so there is one
    place to get it wrong rather than five.
    """
    n_squares = size * size
    packed = np.frombuffer(bits.to_bytes((n_squares + 7) // 8, "little"), dtype=np.uint8)
    flat = np.unpackbits(packed, bitorder="little")[:n_squares]
    return flat.astype(np.float32).reshape(size, size)


def encode(state: State) -> NDArray[np.float32]:
    """One position as a ``(3, size, size)`` float32 array, from the mover's view."""
    size = state.size
    return np.stack(
        (
            _bits_to_plane(state.mine, size),
            _bits_to_plane(state.theirs, size),
            _bits_to_plane(rules.legal_placements(state), size),
        )
    )


def encode_batch(states: Sequence[State]) -> NDArray[np.float32]:
    """Several positions as a ``(batch, 3, size, size)`` array.

    The batch is the unit that matters for speed: one forward pass over 48
    positions costs barely more than one over a single position, because the GPU
    spends most of its time waiting to be handed work rather than doing it.
    """
    if not states:
        msg = "encode_batch requires at least one state"
        raise ValueError(msg)
    size = states[0].size
    if any(s.size != size for s in states):
        msg = "every state in a batch must have the same board size"
        raise ValueError(msg)
    return np.stack([encode(s) for s in states])


def planes_to_bits(plane: NDArray[np.float32], size: int) -> Bitboard:
    """The inverse of ``_bits_to_plane``, for tests and debugging.

    Only used to prove the encoding round-trips; nothing in the training path
    needs to go this direction.
    """
    flat = np.asarray(plane, dtype=np.float32).reshape(-1)
    bits = 0
    for index in range(size * size):
        if flat[index] > 0.5:
            bits |= 1 << index
    return bits


def legal_mask(state: State) -> NDArray[np.bool_]:
    """Which of the ``size**2 + 1`` actions are legal here, PASS included.

    The last entry is PASS. It is set only when the mover has no square to play
    and the opponent still does -- so this mask is all-False exactly when the
    game is over (contract C3).
    """
    mask = np.zeros(policy_size(state.size), dtype=np.bool_)
    for action in rules.legal_actions(state):
        mask[action] = True
    return mask


def masked_policy(
    logits: NDArray[np.float32] | Sequence[float],
    mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Turn raw network output into probabilities that are zero on illegal moves.

    The network emits one number per action with no idea which are legal -- that
    is deliberate (contract C5, layer 1): it keeps the model a plain function of
    its input, which is what makes it exportable and testable. Masking happens
    here, at the point of use.

    Returns a full-width vector summing to 1. Used for the analysis payload the
    web UI draws as a heatmap; the tree search uses its own narrower version over
    just the legal actions, because it never needs the illegal entries at all.
    """
    values = np.asarray(logits, dtype=np.float32)
    if values.shape != mask.shape:
        msg = f"logits shape {values.shape} does not match mask shape {mask.shape}"
        raise ValueError(msg)
    if not mask.any():
        msg = "no legal actions to distribute probability over (is the game over?)"
        raise ValueError(msg)

    out = np.zeros_like(values)
    legal_values = values[mask]
    largest = legal_values.max()

    if not np.isfinite(largest):
        # Every legal logit was -inf, or the network produced garbage. There is
        # no basis for preferring any move, so say so rather than returning NaN
        # and poisoning whatever consumes this.
        out[mask] = 1.0 / np.count_nonzero(mask)
        return out

    # Subtract the largest legal logit before exponentiating. Without this a
    # logit of +90 overflows float32 and the whole vector becomes NaN.
    exp = np.exp(legal_values - largest)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        out[mask] = 1.0 / np.count_nonzero(mask)
        return out

    out[mask] = exp / total
    return out
