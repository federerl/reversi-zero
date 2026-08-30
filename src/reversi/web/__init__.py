"""Build-time artifacts for the web app.

Nothing here runs at serve time or at train time. These modules turn the frozen
Python engine into data that a TypeScript port can be checked against, which is
the only thing standing between a hand-written bitboard routine and a browser
that quietly plays a different game.
"""

from reversi.web.fixtures import FIXTURE_VERSION, write_fixtures

__all__ = ["FIXTURE_VERSION", "write_fixtures"]
