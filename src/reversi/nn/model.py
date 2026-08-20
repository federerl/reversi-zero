"""The network: one board in, a move preference and a score prediction out.

It has a shared body and two heads, because the two questions it answers need
most of the same understanding:

* **Policy head** -- "which moves look worth considering?" One number per action.
  During the search these numbers decide what gets explored first, which is what
  lets a few hundred simulations go further than a few hundred random guesses.
* **Value head** -- "who is winning here?" One number in [-1, +1], always from
  the point of view of the player to move. This replaces the random playouts
  older programs used to estimate a position: instead of playing the game out to
  the end thousands of times, the network guesses the answer directly.

Both heads share the trunk because "what matters on this board" is nearly the
same question either way, and training them together makes each one better --
the value signal teaches the trunk things the policy signal alone would not.

**On size.** Six blocks of 64 channels is about 400k numbers, which is small as
networks go. That is deliberate. Self-play calls this network tens of millions
of times per generation and trains on it for a few seconds, so the thing that
actually costs us time is how long one forward pass takes, not how many
parameters it has. Doubling the depth roughly doubles the time to produce a
generation of games, in exchange for an accuracy gain that is usually not worth
it. Six blocks also gives each output square a view of roughly +/-12 squares in
every direction, which covers an 8x8 board with room to spare -- past that, extra
depth is not even buying more context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from reversi.nn.features import IN_PLANES
from reversi.types import policy_size

if TYPE_CHECKING:
    from reversi.config import NetConfig

__all__ = ["PolicyValueNet", "ResidualBlock"]


class ResidualBlock(nn.Module):
    """Two convolutions plus a shortcut that carries the input straight through.

    The shortcut is the whole point. Without it, a deep stack has to relearn how
    to pass information along at every layer, and training gets harder the deeper
    you go. With it, each block only has to learn a *correction* to what it was
    given, so adding blocks stays cheap in training difficulty.

    Normalisation and activation come *before* each convolution here rather than
    after ("pre-activation"). That keeps the shortcut path a clean sum with
    nothing applied to it, which trains slightly more stably at depth.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        return out + x


class PolicyValueNet(nn.Module):
    """The policy-value network, sized for a board of ``board_size``.

    ``forward`` returns ``(policy_logits, value)`` with shapes
    ``(batch, board_size**2 + 1)`` and ``(batch, 1)``.

    The policy output is **raw logits with no masking applied**. That is contract
    C5 layer 1 and it is deliberate: the model stays a plain function of its
    input with no knowledge of the rules baked in, which is what makes it
    straightforward to export and to test. Illegal moves are excluded by the
    search, which only ever creates edges for legal actions -- and nothing
    anywhere relies on the network having learned to avoid them.
    """

    def __init__(
        self,
        board_size: int,
        *,
        n_blocks: int = 4,
        channels: int = 48,
        value_hidden: int = 64,
        in_planes: int = IN_PLANES,
    ) -> None:
        super().__init__()
        if board_size < 2:
            msg = f"board_size must be at least 2, got {board_size}"
            raise ValueError(msg)

        self.board_size = board_size
        self.in_planes = in_planes
        self.n_blocks = n_blocks
        self.channels = channels
        self.value_hidden = value_hidden

        n_squares = board_size * board_size
        self.policy_size = policy_size(board_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(n_blocks)))
        self.trunk_out = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

        # Both heads squeeze the trunk down to one or two planes first. That
        # 1x1 convolution is a cheap way to cut the width by ~30x before the
        # linear layer, which is where the parameters would otherwise pile up.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * n_squares, self.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_squares, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),  # bounds the answer to [-1, +1]: a loss, a draw, or a win
        )

    @classmethod
    def from_config(cls, net: NetConfig, board_size: int) -> PolicyValueNet:
        """Build the network described by a validated config block."""
        return cls(
            board_size,
            n_blocks=net.n_blocks,
            channels=net.channels,
            value_hidden=net.value_hidden,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4:
            msg = f"expected a (batch, planes, size, size) tensor, got shape {tuple(x.shape)}"
            raise ValueError(msg)
        trunk = self.trunk_out(self.blocks(self.stem(x)))
        return self.policy_head(trunk), self.value_head(trunk)

    def arch(self) -> dict[str, Any]:
        """The shape of this network, for the checkpoint sidecar.

        Saved next to the weights so that loading can *check* the architecture
        matches instead of failing with a shape error forty lines deep in torch,
        and so the lineage of a checkpoint is readable without importing torch
        at all.
        """
        return {
            "board_size": self.board_size,
            "in_planes": self.in_planes,
            "n_blocks": self.n_blocks,
            "channels": self.channels,
            "value_hidden": self.value_hidden,
            "policy_size": self.policy_size,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _init_weights(module: nn.Module) -> None:
    """Kaiming initialisation for convolutions, zeros for the final biases."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)
        nn.init.normal_(module.weight, std=0.01)


def build(net_config: NetConfig, board_size: int, *, seed: int | None = None) -> PolicyValueNet:
    """Create a freshly initialised network.

    Pass ``seed`` to make the starting weights reproducible -- generation 0 of a
    training run is otherwise the one thing in it that cannot be replayed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = PolicyValueNet.from_config(net_config, board_size)
    model.apply(_init_weights)
    return model
