"""Metric recording (backlog T04).

JSONL is the source of truth for every figure in the final report, so the
guarantees tested here are: records survive a kill (flush-per-line), a truncated
file still reads back, and unknown streams are rejected rather than silently
creating a file nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reversi.obs.metrics import STREAMS, JsonlStream, MetricsHub, read_jsonl


def test_stream_writes_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    with JsonlStream(path, run_id="rid") as stream:
        stream.log(generation=0, policy_loss=1.8)
        stream.log(generation=1, policy_loss=1.5)

    records = read_jsonl(path)
    assert [r["policy_loss"] for r in records] == [1.8, 1.5]
    assert all(r["run_id"] == "rid" for r in records)
    assert all("wall_time" in r for r in records)


def test_records_are_flushed_immediately(tmp_path: Path) -> None:
    """An overnight run killed mid-generation must keep what it already logged."""
    path = tmp_path / "selfplay.jsonl"
    stream = JsonlStream(path)
    stream.log(generation=0, games=200)
    assert len(read_jsonl(path)) == 1, "record was still buffered"
    stream.close()


def test_truncated_trailing_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "arena.jsonl"
    with JsonlStream(path) as stream:
        stream.log(generation=0, elo=120.0)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"generation": 1, "elo": 1')  # killed mid-write

    records = read_jsonl(path)
    assert len(records) == 1
    assert records[0]["elo"] == 120.0


def test_read_jsonl_of_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_hub_creates_one_file_per_stream(tmp_path: Path) -> None:
    with MetricsHub(tmp_path, run_id="rid") as hub:
        hub.log("train", generation=1, global_step=100, policy_loss=1.2)
        hub.log("selfplay", generation=1, games_per_s=3.4)
        hub.log("arena", generation=1, elo=88.0)

    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "selfplay.jsonl").is_file()
    assert (tmp_path / "arena.jsonl").is_file()
    assert not (tmp_path / "resource.jsonl").exists(), "streams are created lazily"


def test_hub_stamps_generation_and_step(tmp_path: Path) -> None:
    with MetricsHub(tmp_path, run_id="rid") as hub:
        hub.log("train", generation=3, global_step=1200, value_loss=0.31)

    record = read_jsonl(tmp_path / "train.jsonl")[0]
    assert record["generation"] == 3
    assert record["global_step"] == 1200
    assert record["value_loss"] == 0.31


def test_unknown_stream_is_rejected(tmp_path: Path) -> None:
    with MetricsHub(tmp_path) as hub, pytest.raises(ValueError, match="unknown metric stream"):
        hub.log("losses", generation=0, x=1)


@pytest.mark.parametrize("name", STREAMS)
def test_every_declared_stream_is_usable(tmp_path: Path, name: str) -> None:
    with MetricsHub(tmp_path) as hub:
        hub.log(name, generation=0, value=1.0)
    assert read_jsonl(tmp_path / f"{name}.jsonl")


def test_numpy_scalars_serialise(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    with MetricsHub(tmp_path) as hub:
        hub.log("quality", generation=0, entropy=np.float32(1.75), visits=np.int64(400))

    record = read_jsonl(tmp_path / "quality.jsonl")[0]
    assert record["entropy"] == pytest.approx(1.75)
    assert record["visits"] == 400


def test_appending_to_an_existing_stream_preserves_history(tmp_path: Path) -> None:
    """Resume must extend the metric log, not truncate it."""
    with MetricsHub(tmp_path, run_id="rid") as hub:
        hub.log("train", generation=0, policy_loss=2.0)
    with MetricsHub(tmp_path, run_id="rid") as hub:
        hub.log("train", generation=1, policy_loss=1.7)

    assert len(read_jsonl(tmp_path / "train.jsonl")) == 2
