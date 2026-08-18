"""Metrics recording.

**JSONL is the source of truth.** TensorBoard mirrors scalars for live curves,
but every figure in the README is regenerated from the JSONL files alone
(``scripts/make_plots.py``). Rationale: JSONL is greppable, diffable, and
re-analysable months later; TensorBoard event files are not.

Streams (one file each, under ``runs/<run_id>/metrics/``):

===========  ===========================================================
train        losses, lr, grad norm, throughput
selfplay     games/s, plies, pass rate, branching, worker skew
quality      policy entropy, value MAE, value calibration, Brier
replay       buffer size, age distribution, per-generation composition
arena        W/L/D, Wilson CI, Bradley-Terry Elo, promotion events
resource     GPU util, VRAM, CPU util, RSS, disk
===========  ===========================================================

Every line is stamped with ``run_id``, ``generation``, ``global_step`` (when
known), and ``wall_time``, so streams can be joined after the fact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

STREAMS = ("train", "selfplay", "quality", "replay", "arena", "resource")


class JsonlStream:
    """Append-only JSONL writer.

    Opened in line-buffered append mode and flushed on every record: an overnight
    run that is killed must retain every metric it had already computed.
    """

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, **fields: Any) -> None:
        record: dict[str, Any] = {"wall_time": time.time()}
        if self.run_id is not None:
            record["run_id"] = self.run_id
        record.update(fields)
        self._handle.write(json.dumps(record, default=_json_default) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class MetricsHub:
    """One entry point for all metric streams plus the optional TensorBoard mirror.

    Usage::

        with MetricsHub(paths.metrics, run_id=paths.run_id, tb_dir=paths.tb) as m:
            m.log("train", generation=3, global_step=1200, policy_loss=1.83)
    """

    def __init__(
        self,
        metrics_dir: Path,
        *,
        run_id: str | None = None,
        tb_dir: Path | None = None,
    ) -> None:
        self.metrics_dir = metrics_dir
        self.run_id = run_id
        self._streams: dict[str, JsonlStream] = {}
        self._tb = _open_tensorboard(tb_dir) if tb_dir is not None else None

    def stream(self, name: str) -> JsonlStream:
        if name not in STREAMS:
            msg = f"unknown metric stream {name!r}; expected one of {STREAMS}"
            raise ValueError(msg)
        if name not in self._streams:
            self._streams[name] = JsonlStream(
                self.metrics_dir / f"{name}.jsonl", run_id=self.run_id
            )
        return self._streams[name]

    def log(
        self,
        stream: str,
        *,
        generation: int | None = None,
        global_step: int | None = None,
        tb_prefix: str | None = None,
        **fields: Any,
    ) -> None:
        """Append one record, and mirror numeric scalars to TensorBoard."""
        record: dict[str, Any] = dict(fields)
        if generation is not None:
            record["generation"] = generation
        if global_step is not None:
            record["global_step"] = global_step
        self.stream(stream).log(**record)

        if self._tb is not None:
            step = global_step if global_step is not None else (generation or 0)
            prefix = tb_prefix or stream
            for key, value in fields.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                self._tb.add_scalar(f"{prefix}/{key}", value, step)

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a metric stream back. Malformed trailing lines (from a kill during a
    write) are skipped rather than raising -- the point of JSONL is that a
    truncated file is still 99% usable."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _open_tensorboard(tb_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError:  # pragma: no cover - tensorboard is an optional extra
        return None
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    return str(value)
