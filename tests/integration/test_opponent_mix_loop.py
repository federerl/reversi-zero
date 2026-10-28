"""The training loop with fixed opponents in the mix, end to end.

A generation that includes games against Greedy must still close the loop:
the external games become shards like any other, the trainer consumes them,
and the generation's metrics say how many such games were played and how the
network fared in them. Both ways of producing a generation are covered, in
this process and across worker processes, because the worker path builds its
opponents in a fresh interpreter from the config alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.config import Config
from reversi.obs.metrics import MetricsHub, read_jsonl
from reversi.obs.runmeta import RunPaths
from reversi.train.loop import run_training


def mixed(smoke_config: Config, **selfplay: object) -> Config:
    payload = smoke_config.to_dict()
    payload["selfplay"].update({"opponents": ["greedy", "minimax-d2"], "opponent_games": 2})
    payload["selfplay"].update(selfplay)
    return Config.model_validate(payload)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_generation_with_fixed_opponents_closes_the_loop(
    smoke_config: Config, tmp_path: Path, workers: int
) -> None:
    config = mixed(smoke_config, n_workers=workers, games_per_generation=4)
    paths = RunPaths(run_id="mix", root=tmp_path / "run")
    paths.ensure()
    with MetricsHub(paths.metrics, run_id=paths.run_id) as hub:
        reports = run_training(config, paths, metrics=hub, generations=2)

    assert len(reports) == 2
    assert all(report.games == 6 for report in reports), "4 self-play + 2 external"

    rows = read_jsonl(paths.metrics / "selfplay.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert row["opponent_games"] == 2
        assert 0.0 <= row["opponent_score"] <= 1.0
        assert 0.0 <= row["opponent_wipeout_rate"] <= 1.0


def test_pure_self_play_still_reports_no_external_games(
    smoke_config: Config, tmp_path: Path
) -> None:
    paths = RunPaths(run_id="pure", root=tmp_path / "run")
    paths.ensure()
    with MetricsHub(paths.metrics, run_id=paths.run_id) as hub:
        run_training(smoke_config, paths, metrics=hub, generations=1)
    (row,) = read_jsonl(paths.metrics / "selfplay.jsonl")
    assert row["opponent_games"] == 0
