"""The web API: a stateless server that plays Reversi against you.

Imports ``game``, ``search``, ``nn``, ``difficulty`` and the model loader --
**never** ``train``, ``selfplay`` or ``data``. Shipping the app therefore does
not ship the training pipeline, and no accident can wire a served game into the
replay buffer. There is a test that enforces it.
"""

from __future__ import annotations

from reversi.api.app import build_app

__all__ = ["build_app"]
