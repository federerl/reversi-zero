"""Storing the games self-play produces, and sampling them back out to train on.

Three files, in the order data moves through them:

* ``schema.py`` -- what one recorded position looks like, and the checks that
  stop a malformed one from ever reaching disk.
* ``shards.py`` -- one file per generation, plus a manifest listing them with
  checksums. Writes are atomic and corruption is bounded to a single file.
* ``replay.py`` -- the sliding window the trainer samples from, including the
  eightfold symmetry augmentation applied at sampling time.
"""

from __future__ import annotations
