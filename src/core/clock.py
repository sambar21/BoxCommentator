"""
Injectable clock. All fight-time logic reads now() from here so benchmarks can
replay a fight on a virtual clock instead of waiting in real time.

The time source is per-thread: concurrent fights (batching experiments) each
install their own virtual clock without affecting each other or real-time
request handlers.
"""

import threading
import time
from typing import Callable

_local = threading.local()


def now() -> float:
    source = getattr(_local, "source", None)
    return source() if source is not None else time.time()


def set_source(source: Callable[[], float]):
    """Install a custom time source for the calling thread (e.g. VirtualClock.now)."""
    _local.source = source


def reset():
    """Restore the real wall clock for the calling thread."""
    _local.source = None


class VirtualClock:
    """Manually advanced clock for deterministic, fast replays."""

    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds

    def set(self, t: float):
        self.t = t
