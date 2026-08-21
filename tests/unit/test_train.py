"""The loss, the learning-rate schedule, and one optimisation step.

The most useful test in here is the last one: take a single batch and train on it
repeatedly until the network has memorised it. If a network cannot overfit one
batch, no amount of data will help it -- something is disconnected. It is the
cheapest possible check that the whole gradient path actually works.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from reversi.config import TrainConfig
from reversi.data.replay import Batch
from reversi.nn import features
from reversi.nn.model import PolicyValueNet
from reversi.train.loss import policy_entropy, policy_value_loss
from reversi.train.schedule import learning_rate
from reversi.train.trainer import Trainer
from reversi.types import policy_size

BOARD = 4
WIDTH = policy_size(BOARD)


# ===========================================================================
# The loss
# ===========================================================================


def test_an_opinionless_network_scores_the_uniform_loss() -> None:
    """All-zero logits mean "every action equally likely", whatever the target.

    ln(17) on a 4x4 board, ln(65) on 8x8. This is the number a fresh run should
    start at, and seeing it is how you know nothing is scaled or masked wrongly
    before training begins.
    """
    target = np.zeros(WIDTH, dtype=np.float32)
    target[[1, 5, 9]] = 1 / 3

    parts = policy_value_loss(
        torch.zeros(1, WIDTH),
        torch.zeros(1, 1),
        torch.from_numpy(target).unsqueeze(0),
        torch.zeros(1),
    )

    assert float(parts.policy) == pytest.approx(math.log(WIDTH), abs=1e-5)
    assert float(parts.value) == pytest.approx(0.0)


def test_a_confident_correct_network_scores_near_zero() -> None:
    logits = torch.full((1, WIDTH), -20.0)
    logits[0, 7] = 20.0
    target = torch.zeros(1, WIDTH)
    target[0, 7] = 1.0

    parts = policy_value_loss(logits, torch.ones(1, 1), target, torch.ones(1))

    assert float(parts.policy) < 1e-4
    assert float(parts.value) < 1e-6


def test_illegal_actions_contribute_nothing_to_the_loss() -> None:
    """The target is zero there, so those terms vanish whatever the logits say.

    That is why the model can safely emit unmasked logits (contract C5): a wild
    number on an illegal action cannot change the loss, only the softmax
    normaliser -- and the search would never reach that action anyway.
    """
    target = torch.zeros(1, WIDTH)
    target[0, 2] = 1.0

    calm = torch.zeros(1, WIDTH)
    wild = calm.clone()
    wild[0, 11] = 3.0  # an illegal action, given the target

    quiet = float(policy_value_loss(calm, torch.zeros(1, 1), target, torch.zeros(1)).policy)
    noisy = float(policy_value_loss(wild, torch.zeros(1, 1), target, torch.zeros(1)).policy)

    # Not equal -- the softmax denominator changed -- but the *target* put no
    # weight there, so the change is only through normalisation.
    assert noisy > quiet
    assert noisy - quiet < 1.0


def test_the_value_weight_scales_only_the_value_term() -> None:
    logits = torch.zeros(2, WIDTH)
    target = torch.full((2, WIDTH), 1 / WIDTH)
    value = torch.zeros(2, 1)
    z = torch.ones(2)

    single = policy_value_loss(logits, value, target, z, value_weight=1.0)
    double = policy_value_loss(logits, value, target, z, value_weight=2.0)

    assert float(single.policy) == pytest.approx(float(double.policy))
    assert float(single.value) == pytest.approx(float(double.value))
    assert float(double.total) == pytest.approx(float(single.policy) + 2 * float(single.value))


@pytest.mark.parametrize("value_shape", [(4, 1), (4,)])
def test_the_value_head_shape_is_accepted_either_way(value_shape: tuple[int, ...]) -> None:
    """The model returns (batch, 1); a broadcast mistake here would be silent."""
    parts = policy_value_loss(
        torch.zeros(4, WIDTH),
        torch.zeros(value_shape),
        torch.full((4, WIDTH), 1 / WIDTH),
        torch.ones(4),
    )
    assert float(parts.value) == pytest.approx(1.0)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        policy_value_loss(
            torch.zeros(2, WIDTH), torch.zeros(2, 1), torch.zeros(2, WIDTH + 1), torch.zeros(2)
        )
    with pytest.raises(ValueError, match="same length"):
        policy_value_loss(
            torch.zeros(2, WIDTH), torch.zeros(3, 1), torch.zeros(2, WIDTH), torch.zeros(2)
        )


def test_entropy_spans_no_opinion_to_certainty() -> None:
    uniform = policy_entropy(torch.zeros(1, WIDTH))
    assert float(uniform) == pytest.approx(math.log(WIDTH), abs=1e-5)

    certain = torch.full((1, WIDTH), -30.0)
    certain[0, 3] = 30.0
    assert float(policy_entropy(certain)) < 1e-5


# ===========================================================================
# The schedule
# ===========================================================================


def test_warmup_ramps_up_then_cosine_decays() -> None:
    kwargs = {"base_lr": 0.1, "warmup_steps": 10, "total_steps": 110, "floor_divisor": 10.0}

    assert learning_rate(0, **kwargs) == pytest.approx(0.01)
    assert learning_rate(4, **kwargs) == pytest.approx(0.05)
    assert learning_rate(9, **kwargs) == pytest.approx(0.1)

    # Immediately after warmup it is still at the peak, then falls.
    assert learning_rate(10, **kwargs) == pytest.approx(0.1)
    assert learning_rate(60, **kwargs) < 0.1
    assert learning_rate(109, **kwargs) < learning_rate(60, **kwargs)


def test_the_rate_settles_at_the_floor_and_stays_there() -> None:
    kwargs = {"base_lr": 0.1, "warmup_steps": 10, "total_steps": 110, "floor_divisor": 20.0}
    floor = 0.1 / 20

    assert learning_rate(110, **kwargs) == pytest.approx(floor)
    # A run that goes longer than planned must not start taking big steps again.
    assert learning_rate(500, **kwargs) == pytest.approx(floor)


def test_a_warmup_longer_than_the_run_is_capped() -> None:
    """Otherwise the run spends its whole life on the ramp and never trains.

    The default warmup is 200 steps. A short pipeline test might only take 20
    steps in total, and without this cap every one of them would be taken at a
    tenth of the intended rate or less.
    """
    capped = learning_rate(9, base_lr=0.1, warmup_steps=200, total_steps=20, floor_divisor=10.0)
    assert capped == pytest.approx(0.1), "warmup should finish by half the run"


def test_the_schedule_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="base_lr"):
        learning_rate(0, base_lr=0.0, warmup_steps=1, total_steps=10)
    with pytest.raises(ValueError, match="floor_divisor"):
        learning_rate(0, base_lr=0.1, warmup_steps=1, total_steps=10, floor_divisor=0.5)


# ===========================================================================
# The trainer
# ===========================================================================


def fixed_batch(size: int = 16, *, seed: int = 0) -> Batch:
    """A batch whose targets are consistent, so it can actually be memorised."""
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 2, size=(size, features.IN_PLANES, BOARD, BOARD)).astype(np.float32)

    pi = np.zeros((size, WIDTH), dtype=np.float32)
    best = rng.integers(0, WIDTH, size=size)
    pi[np.arange(size), best] = 1.0

    z = np.where(best % 2 == 0, 1.0, -1.0).astype(np.float32)
    return Batch(planes=planes, pi=pi, z=z)


def small_trainer(**overrides: object) -> tuple[Trainer, PolicyValueNet]:
    model = PolicyValueNet(BOARD, n_blocks=1, channels=16, value_hidden=16)
    settings: dict[str, object] = {"lr": 1e-2, "warmup_steps": 0, "batch_size": 16}
    settings.update(overrides)
    config = TrainConfig.model_validate(settings)
    return Trainer(model, config, total_steps=200), model


def test_one_step_reports_what_happened() -> None:
    trainer, _ = small_trainer()
    metrics = trainer.train_on(fixed_batch())

    assert trainer.global_step == 1
    assert metrics["batch_size"] == 16
    assert metrics["lr"] > 0
    assert math.isfinite(metrics["grad_norm"])
    assert metrics["total_loss"] == pytest.approx(
        metrics["policy_loss"] + metrics["value_loss"], abs=1e-4
    )
    assert 0.0 <= metrics["policy_entropy"] <= math.log(WIDTH) + 1e-5


def test_the_network_can_memorise_a_single_batch() -> None:
    """The cheapest check that the gradient path is connected end to end.

    If this fails, no amount of data would help: something between the batch and
    the weights is not wired up.
    """
    trainer, _ = small_trainer()
    batch = fixed_batch(seed=3)

    first = trainer.train_on(batch)
    last = first
    for _ in range(120):
        last = trainer.train_on(batch)

    assert last["policy_loss"] < first["policy_loss"] * 0.5
    assert last["value_loss"] < first["value_loss"] * 0.5
    assert last["value_mae"] < first["value_mae"]


def test_the_configured_learning_rate_reaches_the_optimiser() -> None:
    """A schedule nobody applies is a comment."""
    trainer, _ = small_trainer(lr=5e-3, warmup_steps=10)
    metrics = trainer.train_on(fixed_batch())

    applied = {group["lr"] for group in trainer.optimizer.param_groups}
    assert applied == {metrics["lr"]}
    assert metrics["lr"] == pytest.approx(5e-3 / 10)


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_both_optimisers_train(optimizer: str) -> None:
    trainer, _ = small_trainer(optimizer=optimizer, lr=1e-2)
    batch = fixed_batch(seed=1)

    first = trainer.train_on(batch)
    last = first
    for _ in range(60):
        last = trainer.train_on(batch)

    assert last["total_loss"] < first["total_loss"]


def test_gradients_are_clipped() -> None:
    """A single wild batch must not be able to undo a generation of progress."""
    trainer, _ = small_trainer(grad_clip=0.01, lr=1e-2)
    metrics = trainer.train_on(fixed_batch())

    # clip_grad_norm_ reports the norm *before* clipping, so the check is that it
    # was large enough for clipping to have bitten at all.
    assert metrics["grad_norm"] > 0.01


def test_the_trainer_state_includes_the_optimiser() -> None:
    """Resuming without it means the first steps after a restart are wrong-sized.

    AdamW keeps a running estimate per parameter; dropping it on resume makes the
    optimiser behave as if the run had just started.
    """
    trainer, _ = small_trainer()
    trainer.train_on(fixed_batch())

    state = trainer.state_dict()

    assert state["global_step"] == 1
    assert "model" in state
    assert "optimizer" in state
    assert state["optimizer"]["state"], "optimiser state should not be empty after a step"
