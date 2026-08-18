"""The Reversi rules. Pure Python — no torch, no GPU, no numpy outside symmetry.

This is the layer everything else trusts. If the rules are subtly wrong, the AI
learns a slightly different game and every result afterwards is worthless — and
it fails silently, because nothing crashes and no loss curve looks unusual.

So the rules are implemented twice:

* ``reference.py`` — slow, obvious, readable. The specification.
* ``rules.py`` + ``bitboard.py`` — fast, using two 64-bit integers per board.

Both are written from the rules of Othello rather than from each other, and a
test plays 20,000 random games through both, comparing every move.
"""

from __future__ import annotations
