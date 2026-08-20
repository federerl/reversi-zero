"""The policy-value network and its adapter (test matrix T13).

These run on CPU with tiny networks. The point is not that the network is any
good -- it is randomly initialised and knows nothing -- but that its shapes,
ranges and determinism are what every layer above assumes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reversi.config import NetConfig
from reversi.game import rules
from reversi.nn import features
from reversi.nn.evaluator import TorchEvaluator
from reversi.nn.model import PolicyValueNet, build
from reversi.search.evaluator import Evaluator
from reversi.types import policy_size


def tiny(board_size: int = 8) -> PolicyValueNet:
    model = PolicyValueNet(board_size, n_blocks=2, channels=8, value_hidden=16)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shapes and ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("board_size", [4, 6, 8])
def test_output_shapes(board_size: int) -> None:
    model = tiny(board_size)
    batch = torch.zeros(5, features.IN_PLANES, board_size, board_size)

    logits, value = model(batch)

    assert logits.shape == (5, policy_size(board_size))
    assert value.shape == (5, 1)


def test_value_is_bounded_to_the_outcome_range() -> None:
    """tanh keeps the value head inside [-1, +1]: a loss, a draw, or a win.

    Nothing downstream clips it, and MCTS averages these directly, so a value
    that could drift outside the range would quietly distort every Q in the tree.
    """
    model = tiny(8)
    rng = torch.Generator().manual_seed(0)
    batch = torch.randn(32, features.IN_PLANES, 8, 8, generator=rng) * 10.0

    _, value = model(batch)

    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_output_is_unmasked_logits() -> None:
    """Contract C5, layer 1: the model knows nothing about legality.

    Every action gets a finite number, including illegal ones and PASS. Masking
    is the search's job -- keeping it out of the model is what makes the model a
    plain function that can be exported and tested on its own.
    """
    model = tiny(8)
    state = rules.initial_state(8)
    batch = torch.from_numpy(features.encode_batch([state]))

    logits, _ = model(batch)

    assert torch.all(torch.isfinite(logits))
    assert logits.shape[1] == policy_size(8)


def test_forward_rejects_wrongly_shaped_input() -> None:
    with pytest.raises(ValueError, match="batch, planes, size, size"):
        tiny(8)(torch.zeros(features.IN_PLANES, 8, 8))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_one_position_gets_the_same_answer_whatever_it_is_batched_with() -> None:
    """In eval mode the answer for a position must not depend on its neighbours.

    This is the test that catches a model left in training mode. Batch norm in
    training mode normalises using the current batch's statistics, so the same
    position batched with 63 others gets a different answer than it does alone --
    and self-play would then be searching on numbers that wobble with whatever
    else happened to be in flight.
    """
    model = tiny(8)
    rng = torch.Generator().manual_seed(1)
    batch = torch.randn(64, features.IN_PLANES, 8, 8, generator=rng)

    batched_logits, batched_value = model(batch)
    for row in (0, 17, 63):
        single_logits, single_value = model(batch[row : row + 1])
        torch.testing.assert_close(single_logits[0], batched_logits[row], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(single_value[0], batched_value[row], atol=1e-5, rtol=1e-5)


def test_build_with_a_seed_is_reproducible() -> None:
    """Generation 0 is otherwise the one checkpoint in a run that cannot be replayed."""
    config = NetConfig(n_blocks=1, channels=8, value_hidden=16)
    first = build(config, 8, seed=123)
    second = build(config, 8, seed=123)
    different = build(config, 8, seed=124)

    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b)

    assert any(
        not torch.equal(a, b)
        for a, b in zip(first.parameters(), different.parameters(), strict=True)
    )


# ---------------------------------------------------------------------------
# Architecture metadata and training readiness
# ---------------------------------------------------------------------------


def test_arch_describes_everything_needed_to_rebuild_the_model() -> None:
    model = PolicyValueNet(8, n_blocks=6, channels=64, value_hidden=64)
    arch = model.arch()

    rebuilt = PolicyValueNet(
        arch["board_size"],
        n_blocks=arch["n_blocks"],
        channels=arch["channels"],
        value_hidden=arch["value_hidden"],
        in_planes=arch["in_planes"],
    )

    assert rebuilt.arch() == arch
    assert arch["policy_size"] == policy_size(8)
    # Loading is a state_dict copy, so the two must agree key for key.
    assert rebuilt.state_dict().keys() == model.state_dict().keys()


def test_the_full_size_network_stays_small() -> None:
    """Roughly 400k parameters at the headline size.

    Size is a deliberate constraint, not an accident: self-play calls this
    network tens of millions of times per generation, so forward-pass time -- not
    parameter count -- is what bounds how much training data we can produce.
    """
    model = PolicyValueNet(8, n_blocks=6, channels=64, value_hidden=64)
    assert 200_000 < model.parameter_count() < 700_000


def test_gradients_reach_both_heads() -> None:
    """A shared trunk with a dead head would train silently and pointlessly."""
    model = PolicyValueNet(4, n_blocks=1, channels=8, value_hidden=16)
    model.train()

    # Real positions, not zeros: the gradient of a convolution's weights is
    # proportional to its input, so an all-zero input gives an all-zero gradient
    # for a perfectly healthy network.
    rng = torch.Generator().manual_seed(3)
    logits, value = model(torch.randn(4, features.IN_PLANES, 4, 4, generator=rng))
    (logits.sum() + value.sum()).backward()

    grads = {name: param.grad for name, param in model.named_parameters()}
    assert all(g is not None for g in grads.values()), "every parameter should get a gradient"

    def carries_signal(prefix: str) -> bool:
        return any(
            g is not None and g.abs().sum().item() > 0.0
            for name, g in grads.items()
            if name.startswith(prefix)
        )

    assert carries_signal("stem"), "the shared trunk is not learning"
    assert carries_signal("policy_head")
    assert carries_signal("value_head")


# ---------------------------------------------------------------------------
# The evaluator adapter
# ---------------------------------------------------------------------------


def test_torch_evaluator_satisfies_the_search_protocol() -> None:
    """The search never imports this class; it only requires the shape of it."""
    assert isinstance(TorchEvaluator(tiny(8)), Evaluator)


def test_torch_evaluator_returns_numpy_in_the_shapes_the_search_expects() -> None:
    evaluator = TorchEvaluator(tiny(8))
    states = [rules.initial_state(8), rules.apply(rules.initial_state(8), 19)]

    logits, values = evaluator.evaluate(states)

    assert isinstance(logits, np.ndarray)
    assert logits.shape == (2, policy_size(8))
    assert values.shape == (2,)
    assert logits.dtype == np.float32
    assert values.dtype == np.float32
    assert np.all(np.abs(values) <= 1.0)
    assert evaluator.calls == 1
    assert evaluator.positions == 2


def test_torch_evaluator_refuses_to_run_a_model_left_in_training_mode() -> None:
    model = tiny(8)
    evaluator = TorchEvaluator(model)
    model.train()

    with pytest.raises(RuntimeError, match="training mode"):
        evaluator.evaluate([rules.initial_state(8)])


def test_torch_evaluator_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        TorchEvaluator(tiny(8)).evaluate([])
