"""Exception hierarchy.

Every failure raised by this package derives from `ReversiError`, so callers can
distinguish "our" errors from bugs. The API layer maps these onto HTTP status
codes: `IllegalMoveError` -> 422, everything else -> 500.

Bare `except:` is banned by lint; catch a specific subclass or let it propagate.
"""

from __future__ import annotations


class ReversiError(Exception):
    """Base class for all errors raised by reversi-zero."""


class ConfigError(ReversiError):
    """A configuration file or override was malformed or inconsistent."""


class IllegalMoveError(ReversiError):
    """An action was applied to a state where it is not legal.

    The message always carries enough information to reconstruct the position,
    because this error appearing anywhere outside a test indicates a contract
    violation (contract C5) rather than user error.
    """


class CheckpointError(ReversiError):
    """A checkpoint could not be written, read, or validated.

    Raised on checksum mismatch, architecture mismatch between a checkpoint and
    the running code, or a missing/corrupt sidecar.
    """


class ReplayError(ReversiError):
    """A replay shard or manifest was missing, corrupt, or inconsistent."""


class WorkerError(ReversiError):
    """A self-play worker process failed.

    The orchestrator re-runs a failed worker's game budget once; a second
    failure in the same generation raises this rather than silently producing a
    short generation.
    """


class ArenaError(ReversiError):
    """An evaluation match or tournament was configured unfairly or failed."""
