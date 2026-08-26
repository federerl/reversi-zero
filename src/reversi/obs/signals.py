"""Turning "you are being killed" into "finish what you are doing, then stop".

When something wants a training run to stop -- Ctrl-C, a shared machine being
rebooted, a scheduler warning that the wall clock is nearly up -- the worst
possible response is to die instantly. The run is likely partway through writing
a shard or a checkpoint, and a file cut off mid-write is exactly the thing that
makes the next resume fail.

So a signal does not stop anything directly. It sets a flag. The training loop
checks that flag at points where stopping is safe -- between generations, where
everything on disk is complete and consistent -- and exits cleanly there.

**The cost of waiting is bounded on purpose.** A generation is sized to finish in
about twenty minutes, so "finish the current one" is a known, small delay rather
than an open-ended one. That sizing is why the number exists.

**The second signal is not ignored.** Press Ctrl-C twice and the process dies
immediately, the way you would expect. Swallowing every signal would leave a user
unable to stop their own program, which is worse than a torn file.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

__all__ = ["StopFlag", "cooperative_stop"]

log = logging.getLogger(__name__)


class StopFlag:
    """A thread-safe "please stop when convenient" flag."""

    __slots__ = ("_event", "_signal")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._signal: str | None = None

    def request(self, reason: str) -> None:
        if not self._event.is_set():
            self._signal = reason
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._signal

    def __bool__(self) -> bool:
        return self.requested

    def __call__(self) -> bool:
        """So it can be passed straight in wherever a predicate is wanted."""
        return self.requested


@contextmanager
def cooperative_stop(
    *, names: tuple[str, ...] = ("SIGINT", "SIGTERM", "SIGUSR1")
) -> Iterator[StopFlag]:
    """Catch stop signals for the duration of the block and expose them as a flag.

    ``SIGUSR1`` is the one a job scheduler sends before a wall-clock limit, when
    it is configured to warn first. It does not exist on Windows, and neither do
    several others, so anything unavailable is skipped rather than raising --
    the same code then runs on a laptop and a cluster.

    Handlers are always restored on the way out, including if the body raises,
    so a caller's own Ctrl-C behaviour is never left broken.
    """
    flag = StopFlag()
    previous: dict[int, object] = {}

    def handle(number: int, frame: FrameType | None) -> None:
        _ = frame
        name = signal.Signals(number).name
        if flag.requested:
            # Already asked once. The user means it -- restore the default
            # behaviour and re-raise so the process actually dies.
            log.warning("%s received again; stopping immediately", name)
            signal.signal(number, signal.SIG_DFL)
            signal.raise_signal(number)
            return
        log.warning(
            "%s received; will stop at the end of the current generation. "
            "Send it again to stop immediately (this may leave a partial file).",
            name,
        )
        flag.request(name)

    for name in names:
        number = getattr(signal, name, None)
        if number is None:
            continue  # not on this platform
        try:
            previous[int(number)] = signal.signal(number, handle)
        except (ValueError, OSError):
            # Not the main thread, or the platform refuses this signal. A run
            # that cannot install a handler still works; it just cannot be asked
            # to stop politely.
            log.debug("could not install a handler for %s", name)

    try:
        yield flag
    finally:
        for number, handler in previous.items():
            try:
                signal.signal(number, handler)  # type: ignore[arg-type]
            except (ValueError, OSError):
                log.debug("could not restore the handler for signal %d", number)
