"""The consistency measurement.

The expensive half of `spread.py` -- playing hundreds of games -- is exercised by
the run itself. What these pin is the arithmetic that turns those games into a
claim, because a dispersion statistic is easy to get subtly wrong and impossible
to eyeball afterwards: a wrong denominator, a wrong degrees-of-freedom, or the
wrong tail all produce a plausible number.

So the tests feed the statistic data whose answer is known in advance.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from reversi.difficulty.levels import LEVELS, level_by_name
from reversi.difficulty.spread import (
    BlockDispersion,
    DropDistribution,
    SpreadReport,
    dispersion_of,
    write_report,
)


def _dispersion(scores: list[float], *, pairs_per_block: int = 10) -> BlockDispersion:
    return dispersion_of(
        scores,
        level="casual",
        opponent="minimax-d4",
        samples_moves=True,
        pairs_per_block=pairs_per_block,
    )


class TestDispersion:
    def test_independent_draws_land_near_one(self) -> None:
        """Data generated *as* independent pairs must not look overdispersed.

        This is the calibration of the statistic itself. If a stream of genuinely
        independent results reports a dispersion far from 1, every number the
        measurement produces is meaningless -- so this is the test that would
        catch a wrong denominator.
        """
        rng = np.random.default_rng(11)
        # 150 pairs, each a fair-ish draw from the same distribution.
        scores = [float(rng.choice([0.0, 0.5, 1.0], p=[0.25, 0.2, 0.55])) for _ in range(150)]

        result = _dispersion(scores)

        assert result.blocks == 15
        assert result.degrees_of_freedom == 14
        assert 0.3 < result.dispersion < 3.0
        assert not result.overdispersed

    def test_a_level_whose_form_comes_and_goes_is_caught(self) -> None:
        """Blocks drawn at alternating strengths must fail the test.

        This is the shape of the complaint: not a level that is weak, but one
        that is a different opponent from one sitting to the next.
        """
        rng = np.random.default_rng(5)
        scores: list[float] = []
        for block in range(15):
            # Every other block plays like a much stronger opponent.
            p_win = 0.95 if block % 2 == 0 else 0.15
            scores += [float(rng.choice([0.0, 1.0], p=[1 - p_win, p_win])) for _ in range(10)]

        result = _dispersion(scores)

        assert result.dispersion > 2.0
        assert result.p_value < 0.01
        assert result.overdispersed

    def test_a_level_that_never_varies_reports_no_ratio(self) -> None:
        """Every pair identical means there is nothing to disperse.

        A deterministic level sweeping a weaker opponent really does produce this,
        and the honest answer is "no variance to explain" rather than a ratio of
        two zeros or a division error.
        """
        result = _dispersion([1.0] * 150)

        assert result.pair_variance == 0.0
        assert math.isnan(result.dispersion)
        assert math.isnan(result.p_value)
        assert not result.overdispersed

    def test_the_unit_is_the_pair_not_the_game(self) -> None:
        """The chance baseline comes from the pairs, and it must.

        Each opening is played twice with the colours swapped, so the two games
        of a pair are anti-correlated. Estimating the baseline from games instead
        would understate the chance variance and manufacture overdispersion out
        of the protocol -- so `expected_variance` has to be the pair variance
        divided by the pairs in a block, exactly.
        """
        rng = np.random.default_rng(3)
        scores = [float(rng.choice([0.0, 0.5, 1.0])) for _ in range(150)]

        result = _dispersion(scores)

        expected = float(np.var(scores, ddof=1)) / 10
        assert result.expected_variance == pytest.approx(expected)
        assert result.dispersion == pytest.approx(result.observed_variance / expected)

    def test_only_the_upper_tail_counts(self) -> None:
        """Varying *less* than chance is not the complaint, and is not flagged.

        Perfectly regular blocks are unusual, so a two-sided test would call them
        a failure. They are not: nobody has ever reported an opponent as
        unpleasantly consistent.
        """
        # Every block identical, but the pairs within a block differ -- so the
        # pair variance is real while the block variance is nearly zero.
        block = [0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 1.0]
        result = _dispersion(block * 15)

        assert result.dispersion < 0.5
        assert result.p_value > 0.99
        assert not result.overdispersed


class TestReport:
    def test_the_report_survives_a_missing_number(self, tmp_path) -> None:
        """A NaN dispersion has to reach the file as null, not crash the writer.

        JSON has no NaN. `json.dumps` emits a bare `NaN` token, which is not
        valid JSON and which every consumer of these reports would reject -- so
        an unmeasurable ratio is written as an absent one.
        """
        report = SpreadReport(
            model="reversi-8x8-gen60.pt",
            drops=[
                DropDistribution(
                    level="max",
                    guard=0.0,
                    control=True,
                    samples=500,
                    forced=3,
                    best_move_fraction=1.0,
                    mean=0.0,
                    median=0.0,
                    p90=0.0,
                    p99=0.0,
                    worst=0.0,
                    near_guard_fraction=0.0,
                )
            ],
            dispersions=[_dispersion([1.0] * 150)],
            elapsed_s=12.5,
        )

        destination = tmp_path / "difficulty_spread.json"
        write_report(destination, report)

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["kind"] == "difficulty_spread"
        assert payload["block_dispersion"][0]["dispersion"] is None
        assert payload["block_dispersion"][0]["p_value"] is None
        assert payload["value_drops"][0]["control"] is True


class TestWhichLevelsAreSubjects:
    def test_exactly_the_sampling_levels_are_the_subject(self) -> None:
        """The subject/control split follows the code, not a list.

        A level is a subject of this measurement when it samples its move, since
        that is the only thing that can make it play differently in the same
        position. If a level's temperature is ever changed, it moves side
        automatically and the report's sample sizes follow.
        """
        subjects = [level.name for level in LEVELS if level.temperature > 0.0]
        controls = [level.name for level in LEVELS if level.temperature <= 0.0]

        assert subjects == ["casual", "club"]
        assert controls == ["strong", "max"]
        # And the two named in the complaint are the two subjects.
        assert level_by_name("casual").top_k is not None
        assert level_by_name("club").top_k is not None
