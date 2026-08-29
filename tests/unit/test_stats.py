"""Confidence intervals and ratings (test matrix T28).

These are the numbers every strength claim in the project rests on, so the tests
lean on two things: agreement with values that can be checked independently, and
the specific cases where the naive alternative breaks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reversi.arena.elo import (
    ELO_PER_THETA,
    fit_bradley_terry,
    sequential_elo,
)
from reversi.arena.stats import bootstrap_interval, wilson_interval
from reversi.errors import ArenaError

# ===========================================================================
# Wilson intervals
# ===========================================================================


def test_wilson_matches_published_values() -> None:
    """Checked against the standard worked example, not against itself."""
    interval = wilson_interval(50, 100)
    assert interval.point == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.4038, abs=5e-4)
    assert interval.high == pytest.approx(0.5962, abs=5e-4)


def test_wilson_stays_inside_zero_and_one_where_the_textbook_formula_does_not() -> None:
    """The reason this method was chosen.

    The usual interval, p +/- 1.96*sqrt(p(1-p)/n), gives an upper bound of 1.046
    at 19 wins from 20 -- a 105% win rate. This project's results live exactly
    there: near the extremes, at modest sample sizes.
    """
    naive_high = 0.95 + 1.959963984540054 * math.sqrt(0.95 * 0.05 / 20)
    assert naive_high > 1.0, "the textbook formula really does overflow here"

    interval = wilson_interval(19, 20)
    assert interval.high <= 1.0
    assert interval.low > 0.5


def test_wilson_keeps_a_real_interval_at_a_perfect_record() -> None:
    """The textbook formula gives zero width at 20/20 -- certainty from 20 games.

    A perfect record is evidence of strength, not proof of perfection, and the
    interval has to say so.
    """
    interval = wilson_interval(20, 20)
    assert interval.point == 1.0
    assert interval.low < 1.0
    assert interval.width > 0.1

    zero = wilson_interval(0, 20)
    assert zero.point == 0.0
    assert zero.high > 0.0


def test_more_games_narrow_the_interval() -> None:
    widths = [wilson_interval(int(0.6 * n), n).width for n in (20, 100, 1000, 10000)]
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] < 0.05


def test_draws_count_as_half_a_win() -> None:
    """Fractional wins are the whole reason `wins` is a float here."""
    interval = wilson_interval(10.5, 20)
    assert interval.point == pytest.approx(0.525)


def test_the_interval_answers_the_question_actually_asked() -> None:
    """ "Is this better than even?" is `excludes(0.5)`, not "is the point above 0.5"."""
    coarse = wilson_interval(12, 20)  # 60%, but only 20 games
    assert coarse.point > 0.5
    assert not coarse.excludes(0.5), "60% from 20 games is not decisive"

    fine = wilson_interval(600, 1000)  # the same 60%, far more games
    assert fine.excludes(0.5)


def test_wilson_rejects_impossible_input() -> None:
    with pytest.raises(ValueError, match="at least one game"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="between 0 and games"):
        wilson_interval(21, 20)


# ===========================================================================
# Bootstrap
# ===========================================================================


def test_the_bootstrap_covers_the_truth_about_as_often_as_it_claims() -> None:
    """A 95% interval should contain the true mean about 95% of the time.

    Checked by simulation rather than asserted: an interval that is wrong about
    its own coverage is worse than no interval, because it looks like rigour.
    """
    rng = np.random.default_rng(0)
    true_mean = 0.3
    covered = 0
    trials = 200

    for _ in range(trials):
        sample = rng.normal(true_mean, 1.0, size=60)
        interval = bootstrap_interval(sample, resamples=200, rng=rng)
        covered += interval.low <= true_mean <= interval.high

    assert 0.88 <= covered / trials <= 0.99, f"covered {covered}/{trials}"


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    values = np.random.default_rng(1).normal(size=50)
    first = bootstrap_interval(values, rng=np.random.default_rng(7))
    again = bootstrap_interval(values, rng=np.random.default_rng(7))
    assert (first.low, first.high) == (again.low, again.high)


def test_the_bootstrap_refuses_too_few_resamples() -> None:
    with pytest.raises(ValueError, match="too few"):
        bootstrap_interval(np.zeros(10), resamples=10)


# ===========================================================================
# Bradley-Terry
# ===========================================================================


def synthetic(true_elo: dict[str, float], games: int, seed: int = 0) -> dict:
    """Results generated from known strengths, so the fit has an answer to find."""
    rng = np.random.default_rng(seed)
    results = {}
    for a in true_elo:
        for b in true_elo:
            if a >= b:
                continue
            probability = 1 / (1 + 10 ** ((true_elo[b] - true_elo[a]) / 400))
            results[(a, b)] = (float(rng.binomial(games, probability)), games)
    return results


def test_it_recovers_ratings_it_was_never_told() -> None:
    """The test that matters: generate games from known strengths, fit, compare."""
    true = {"random": 0.0, "greedy": 200.0, "strong": 500.0, "stronger": 700.0}
    table = fit_bradley_terry(synthetic(true, games=3000), anchor="random")

    fitted = {r.name: r.elo for r in table.ratings}
    for name, expected in true.items():
        assert fitted[name] == pytest.approx(expected, abs=40), (
            f"{name}: fitted {fitted[name]:.0f}, true {expected:.0f}"
        )


def test_the_anchor_is_pinned_at_zero() -> None:
    """Only differences are identified -- adding 100 to everyone changes nothing.

    Pinning Random at 0 is what makes the whole scale interpretable, and is why
    the Random baseline must never change.
    """
    table = fit_bradley_terry(synthetic({"random": 0.0, "a": 300.0}, games=500), anchor="random")
    assert table.by_name()["random"].elo == pytest.approx(0.0, abs=1e-6)
    assert table.anchor == "random"


def test_the_result_does_not_depend_on_the_order_matches_were_played() -> None:
    """The property sequential Elo does not have, and the reason for choosing this."""
    results = synthetic({"random": 0.0, "a": 200.0, "b": 400.0}, games=800)
    shuffled = dict(reversed(list(results.items())))

    first = {r.name: r.elo for r in fit_bradley_terry(results, anchor="random").ratings}
    second = {r.name: r.elo for r in fit_bradley_terry(shuffled, anchor="random").ratings}

    for name in first:
        assert first[name] == pytest.approx(second[name], abs=1e-6)


def test_sequential_elo_does_depend_on_order() -> None:
    """Demonstrating what was rejected, so the choice is not just an assertion."""
    matches = [("a", "b", 1.0)] * 5 + [("b", "c", 1.0)] * 5
    forward = sequential_elo(matches)
    backward = sequential_elo(list(reversed(matches)))
    assert forward != backward


def test_a_perfect_record_gives_a_large_but_finite_rating() -> None:
    """Without regularisation this is infinite and the fit does not converge.

    "Won every game" from a finite sample means "very strong", not "infinitely
    strong", and the number has to reflect that.
    """
    table = fit_bradley_terry({("winner", "random"): (40.0, 40)}, anchor="random")
    elo = table.by_name()["winner"].elo
    assert elo > 300
    assert math.isfinite(elo)
    assert elo < 5000


def test_both_orderings_of_a_pair_are_folded_together() -> None:
    """A tournament may record a vs b and b vs a; they describe one matchup."""
    both = fit_bradley_terry(
        {("a", "random"): (30.0, 40), ("random", "a"): (10.0, 40)}, anchor="random"
    )
    combined = fit_bradley_terry({("a", "random"): (60.0, 80)}, anchor="random")

    assert both.by_name()["a"].elo == pytest.approx(combined.by_name()["a"].elo, abs=1e-6)


def test_bootstrap_intervals_are_produced_and_ordered_sensibly() -> None:
    true = {"random": 0.0, "middle": 250.0, "best": 550.0}
    table = fit_bradley_terry(
        synthetic(true, games=600),
        anchor="random",
        bootstrap=200,
        rng=np.random.default_rng(0),
    )

    for rating in table.ratings:
        assert rating.interval is not None
        assert rating.interval.low <= rating.elo <= rating.interval.high

    # The two clearly separated agents must be distinguishable.
    assert table.stronger("best", "random")
    assert table.stronger("best", "middle")


def test_stronger_requires_the_intervals_not_to_overlap() -> None:
    """The strict reading criterion S14 asks for.

    Two point estimates in the right order are not evidence when their intervals
    overlap. That is the difference between "looks better" and "is better", and
    it is the whole reason for computing intervals at all.
    """
    # Two agents 30 Elo apart, measured over very few games: the point estimates
    # will differ but the intervals will not separate.
    table = fit_bradley_terry(
        synthetic({"random": 0.0, "a": 100.0, "b": 130.0}, games=30, seed=3),
        anchor="random",
        bootstrap=200,
        rng=np.random.default_rng(1),
    )
    ratings = table.by_name()
    assert ratings["b"].interval is not None
    assert ratings["a"].interval is not None

    overlapping = not (
        ratings["b"].interval.low > ratings["a"].interval.high
        or ratings["a"].interval.low > ratings["b"].interval.high
    )
    assert overlapping, "30 Elo from 30 games should not be resolvable"
    assert not table.stronger("b", "a"), "overlapping intervals must not count as evidence"


def test_elo_conversion_matches_the_chess_convention() -> None:
    """A 400-point gap is about a 10:1 win ratio, which is what makes the scale
    readable to anyone who has seen a chess rating."""
    gap_in_theta = 400.0 / ELO_PER_THETA
    probability = 1 / (1 + math.exp(-gap_in_theta))
    assert probability / (1 - probability) == pytest.approx(10.0, rel=1e-6)


def test_nonsense_input_is_refused() -> None:
    with pytest.raises(ArenaError, match="no results"):
        fit_bradley_terry({})
    with pytest.raises(ArenaError, match="not among the agents"):
        fit_bradley_terry({("a", "b"): (1.0, 2)}, anchor="nobody")
    with pytest.raises(ArenaError, match="zero games"):
        fit_bradley_terry({("a", "random"): (0.0, 0)}, anchor="random")
