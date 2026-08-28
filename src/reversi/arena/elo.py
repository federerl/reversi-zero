"""Putting every agent on one rating scale, from the whole result matrix at once.

A pile of pairwise win rates is hard to read and easy to misread. "A beat B 60%,
B beat C 70%, A beat C 85%" -- is A improving? By how much? Ratings answer that in
one number per agent.

**Bradley-Terry, not sequential Elo.** Sequential Elo processes matches one at a
time, nudging ratings after each. That makes the answer depend on the *order* the
matches happened to be played, and on an arbitrary K-factor. Play the same games
in a different order and you get different ratings, which is a bad property for a
number meant to be evidence.

Bradley-Terry instead asks: what set of strengths makes the results we actually
observed most likely? It fits all agents simultaneously by maximum likelihood, so
the answer is order-independent and there is no K-factor to pick. Sequential Elo
is reported alongside for familiarity, but this is the authoritative one.

**The model.** Each agent has a strength ``theta``. The chance that i beats j is

    P(i beats j) = 1 / (1 + exp(-(theta_i - theta_j)))

which is the logistic curve -- a difference of zero means an even match, and the
probability rises smoothly with the gap. Converting to the familiar Elo scale is
one multiplication: ``elo = 400 / ln(10) * theta``, chosen so that a 400-point gap
means roughly a 10-to-1 win ratio, exactly as in chess.

**Ratings are relative, so one is pinned.** Only differences are identified --
adding 100 to everyone leaves every predicted result unchanged. Random is fixed
at 0, which makes it the anchor the whole scale is read against and is why the
Random baseline must never change.

**Draws count as half a win to each side.** They carry real information about how
close two agents are, and dropping them would flatter whichever agent draws more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from reversi.arena.stats import Interval
from reversi.errors import ArenaError

__all__ = ["Rating", "RatingTable", "fit_bradley_terry"]

ELO_PER_THETA = 400.0 / math.log(10.0)
"""Converts a Bradley-Terry strength into Elo points. Defined so a 400-point gap
is about a 10:1 win ratio, matching the chess convention people already have
intuitions for."""


@dataclass(frozen=True, slots=True)
class Rating:
    """One agent's place on the scale."""

    name: str
    elo: float
    interval: Interval | None = None
    games: int = 0

    def __str__(self) -> str:
        if self.interval is None:
            return f"{self.name}: {self.elo:+.0f}"
        return f"{self.name}: {self.elo:+.0f} [{self.interval.low:+.0f}, {self.interval.high:+.0f}]"


@dataclass(slots=True)
class RatingTable:
    """Every agent, rated, with the anchor recorded."""

    ratings: list[Rating]
    anchor: str
    results: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)

    def by_name(self) -> dict[str, Rating]:
        return {r.name: r for r in self.ratings}

    def sorted(self) -> list[Rating]:
        return sorted(self.ratings, key=lambda r: -r.elo)

    def stronger(self, later: str, earlier: str) -> bool:
        """True only when the intervals do not overlap.

        The deliberately strict reading, and the one criterion S14 asks for:
        "the later checkpoint's interval lies entirely above the earlier one's".
        Two point estimates in the right order are not evidence when the intervals
        overlap -- that is the difference between "looks better" and "is better".
        """
        table = self.by_name()
        a, b = table[later], table[earlier]
        if a.interval is None or b.interval is None:
            return a.elo > b.elo
        return a.interval.low > b.interval.high

    def describe(self) -> str:
        lines = [f"rating table (anchor: {self.anchor} = 0)"]
        lines.extend(f"  {rating}" for rating in self.sorted())
        return "\n".join(lines)


def fit_bradley_terry(
    results: dict[tuple[str, str], tuple[float, int]],
    *,
    anchor: str = "random",
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
    regularisation: float = 1e-3,
) -> RatingTable:
    """Fit ratings to a set of pairwise results.

    ``results`` maps ``(agent_i, agent_j)`` to ``(score_for_i, games)`` where a
    draw contributes half. Both orderings of a pair may appear; they are summed.

    ``bootstrap`` resamples the *games* -- not the agents -- to put an interval on
    each rating. Zero skips it, which is much faster and fine while iterating.

    ``regularisation`` pulls strengths gently toward zero. Without it, an agent
    that won every one of its games has an infinite maximum-likelihood strength,
    and the fit either fails to converge or returns an absurd number. A small
    penalty makes "won everything" mean "very strong" rather than "infinitely
    strong", which is the honest reading of a finite sample.
    """
    if not results:
        msg = "cannot fit ratings with no results"
        raise ArenaError(msg)

    names = sorted({name for pair in results for name in pair})
    if anchor not in names:
        msg = f"anchor {anchor!r} is not among the agents {names}"
        raise ArenaError(msg)

    index = {name: i for i, name in enumerate(names)}
    anchor_index = index[anchor]

    # (i, j, score_i, games) rows, with both orderings folded together.
    rows: list[tuple[int, int, float, int]] = []
    merged: dict[tuple[str, str], tuple[float, int]] = {}
    for (a, b), (score, games) in results.items():
        if games <= 0:
            continue
        if (b, a) in merged:
            other_score, other_games = merged.pop((b, a))
            merged[(a, b)] = (score + (other_games - other_score), games + other_games)
        else:
            merged[(a, b)] = (score, games)
    for (a, b), (score, games) in merged.items():
        rows.append((index[a], index[b], score, games))

    if not rows:
        msg = "every matchup had zero games"
        raise ArenaError(msg)

    theta = _solve(rows, len(names), anchor_index, regularisation)
    elos = (theta - theta[anchor_index]) * ELO_PER_THETA

    games_played = dict.fromkeys(names, 0)
    for a, b, _score, games in rows:
        games_played[names[a]] += games
        games_played[names[b]] += games

    intervals: dict[str, Interval] = {}
    if bootstrap:
        intervals = _bootstrap_ratings(rows, names, anchor_index, regularisation, bootstrap, rng)

    return RatingTable(
        ratings=[
            Rating(
                name=name,
                elo=float(elos[i]),
                interval=intervals.get(name),
                games=games_played[name],
            )
            for i, name in enumerate(names)
        ],
        anchor=anchor,
        results=dict(merged),
    )


def _solve(
    rows: list[tuple[int, int, float, int]],
    n_agents: int,
    anchor_index: int,
    regularisation: float,
) -> np.ndarray:
    """Maximum likelihood by direct minimisation of the negative log-likelihood."""

    def negative_log_likelihood(theta: np.ndarray) -> float:
        total = regularisation * float(np.dot(theta, theta))
        for i, j, score, games in rows:
            gap = theta[i] - theta[j]
            # log(1 + exp(-gap)) computed stably: for a large positive gap the
            # naive form underflows to log(1) = 0 and loses the gradient.
            log_p_i = -np.logaddexp(0.0, -gap)
            log_p_j = -np.logaddexp(0.0, gap)
            total -= score * log_p_i + (games - score) * log_p_j
        return total

    def gradient(theta: np.ndarray) -> np.ndarray:
        grad = 2.0 * regularisation * theta
        for i, j, score, games in rows:
            gap = theta[i] - theta[j]
            expected = games / (1.0 + np.exp(-gap))
            grad[i] -= score - expected
            grad[j] += score - expected
        return grad

    start = np.zeros(n_agents)
    outcome = minimize(
        negative_log_likelihood,
        start,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    theta = np.asarray(outcome.x, dtype=np.float64)
    return theta - theta[anchor_index]


def _bootstrap_ratings(
    rows: list[tuple[int, int, float, int]],
    names: list[str],
    anchor_index: int,
    regularisation: float,
    resamples: int,
    rng: np.random.Generator | None,
) -> dict[str, Interval]:
    """Refit the whole table on resampled games, and read the spread.

    Resampling *games* rather than agents is the point: the uncertainty being
    estimated is "we played a finite number of games", not "we chose a finite set
    of opponents".
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    samples = np.zeros((resamples, len(names)))

    for draw in range(resamples):
        resampled = [
            (i, j, float(generator.binomial(games, min(max(score / games, 0.0), 1.0))), games)
            for i, j, score, games in rows
        ]
        theta = _solve(resampled, len(names), anchor_index, regularisation)
        samples[draw] = (theta - theta[anchor_index]) * ELO_PER_THETA

    point = _solve(rows, len(names), anchor_index, regularisation)
    point = (point - point[anchor_index]) * ELO_PER_THETA

    return {
        name: Interval(
            point=float(point[i]),
            low=float(np.quantile(samples[:, i], 0.025)),
            high=float(np.quantile(samples[:, i], 0.975)),
        )
        for i, name in enumerate(names)
    }


def sequential_elo(
    matches: list[tuple[str, str, float]],
    *,
    k: float = 32.0,
    start: float = 0.0,
) -> dict[str, float]:
    """The familiar order-dependent Elo update, for comparison only.

    Reported alongside Bradley-Terry because people recognise it, and explicitly
    not authoritative: the answer depends on the order the matches were played
    and on ``k``, neither of which is a property of the agents.
    """
    ratings: dict[str, float] = {}
    for a, b, score_a in matches:
        ratings.setdefault(a, start)
        ratings.setdefault(b, start)
        expected = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
        ratings[a] += k * (score_a - expected)
        ratings[b] -= k * (score_a - expected)
    return ratings
