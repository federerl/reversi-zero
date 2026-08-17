"""Structured logging.

Two sinks, deliberately different:

* **console** -- human-readable, for watching a run in a terminal or tailing a
  SLURM log.
* **file** (``logs/run.jsonl``) -- one JSON object per line, so a failed
  overnight run can be queried (``jq 'select(.level=="ERROR")'``) instead of
  grepped.

Every record carries ``run_id``, ``component``, and (when set) ``generation``, so
lines from six concurrent self-play workers remain attributable.

``print`` is banned in ``src/`` by lint (ruff T20); use a logger.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONTEXT_KEYS = ("run_id", "component", "generation", "worker_id")

_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)  # fmt: skip


class _ContextFilter(logging.Filter):
    """Stamp every record with the run-level context."""

    def __init__(self, context: dict[str, Any]) -> None:
        super().__init__()
        self.context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line, including any extra=... fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _CONTEXT_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    *,
    log_dir: Path | None = None,
    level: int | str = logging.INFO,
    run_id: str | None = None,
    component: str = "main",
    filename: str = "run.jsonl",
) -> logging.Logger:
    """Configure the ``reversi`` logger tree. Idempotent within a process."""
    root = logging.getLogger("reversi")
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    context = {"run_id": run_id, "component": component}
    context_filter = _ContextFilter(context)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    console.addFilter(context_filter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``reversi`` tree. Use ``__name__`` at call sites."""
    return logging.getLogger(name if name.startswith("reversi") else f"reversi.{name}")
