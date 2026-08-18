"""Observability: run provenance, structured logging, and metric recording.

Nothing in this package may import `reversi.train`, `reversi.selfplay`, or
`reversi.nn` -- it is used by all of them.
"""

from __future__ import annotations

from reversi.obs.logging import get_logger, setup_logging
from reversi.obs.metrics import STREAMS, JsonlStream, MetricsHub, read_jsonl
from reversi.obs.runmeta import RunPaths, environment_info, git_info, init_run, make_run_id

__all__ = [
    "STREAMS",
    "JsonlStream",
    "MetricsHub",
    "RunPaths",
    "environment_info",
    "get_logger",
    "git_info",
    "init_run",
    "make_run_id",
    "read_jsonl",
    "setup_logging",
]
