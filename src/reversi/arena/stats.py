"""Turning counts into claims, with the uncertainty attached.

A win rate on its own says nothing about how much to trust it. "19 wins from 20
games" and "950 wins from 1000" are both 95%, and only one of them is evidence.

Two tools here:

* **Wilson intervals** for a single matchup -- how uncertain is this win rate?
* **Bootstrap intervals** for anything derived from a set of games, used by the
  rating fit next door.

**Why Wilson and not the textbook formula.** The interval most people reach for is
``p +/- 1.96 * sqrt(p(1-p)/n)``. At 19 wins from 20 that gives an upper bound of
**1.05** -- a 105% win rate, which is not a thing. It also gives a *zero-width*
interval at 0 or 20 wins, claiming perfect certainty from twenty games. Both
failures happen exactly where this project's results live: near the extremes, at
modest sample sizes.

Wilson has neither problem. It stays inside [0, 1], stays sensible at the
extremes, and is barely more code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Interval", "bootstrap_interval", "wilson_interval"]

# 1.959964 is the two-sided 95% point of the normal distribution. Written out
# rather than imported from scipy so this module has no heavy dependency.
Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a confidence interval around it."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval.

        The usual question is ``excludes(0.5)`` -- "can we say this agent is
        better than even, rather than that it happened to win more?"
        """
        return value < self.low or value > self.high

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson_interval(
    wins: float,
    games: int,
    *,
    z: float = Z_95,
) -> Interval:
    """A confidence interval for a win rate, using the Wilson score method.

    ``wins`` may be fractional, because a draw counts half. ``games`` is the total
    played.

    The method inverts the score test rather than approximating the binomial with
    a normal of the observed variance, which is what keeps it well behaved when
    the observed rate is 0 or 1 -- the case where the textbook interval collapses
    to zero width and claims certainty it has not earned.
    """
    if games <= 0:
        msg = f"need at least one game, got {games}"
        raise ValueError(msg)
    if not 0.0 <= wins <= games:
        msg = f"wins ({wins}) must be between 0 and games ({games})"
        raise ValueError(msg)

    rate = wins / games
    denominator = 1.0 + z * z / games
    centre = (rate + z * z / (2 * games)) / denominator
    spread = (z / denominator) * np.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))

    return Interval(
        point=rate,
        low=float(max(0.0, centre - spread)),
        high=float(min(1.0, centre + spread)),
        confidence=0.95,
    )


def bootstrap_interval(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    resamples: int = 1000,
    rng: np.random.Generator | None = None,
    confidence: float = 0.95,
) -> Interval:
    """A percentile bootstrap interval for a statistic of ``values``.

    Resample the observations with replacement, recompute the statistic each
    time, and read the interval off the spread of those recomputations. It makes
    no assumption about the shape of the underlying distribution, which is why it
    works for things like a rating fit where no closed form exists.

    ``resamples=1000`` is enough for a 95% interval; the percentile estimate is
    itself noisy below a few hundred.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    if resamples < 100:
        msg = f"resamples={resamples} is too few for a stable interval; use at least 100"
        raise ValueError(msg)

    generator = rng if rng is not None else np.random.default_rng(0)
    compute = np.mean if statistic == "mean" else np.median

    draws = generator.integers(0, values.size, size=(resamples, values.size))
    estimates = compute(values[draws], axis=1)

    tail = (1.0 - confidence) / 2.0
    return Interval(
        point=float(compute(values)),
        low=float(np.quantile(estimates, tail)),
        high=float(np.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
    )
