"""The injectable clock.

Every duration in the harness is a difference of two clock readings, so swapping the
clock swaps the entire timing basis. :class:`SystemClock` reads the wall clock for real
sessions; :class:`ManualClock` advances only when told to, which makes a recording a pure
function of seed plus agent behaviour and lets the determinism tests compare two recordings
byte-for-byte with no tolerance window.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """A source of epoch-millisecond timestamps. The single timing primitive."""

    def now_ms(self) -> int:
        """Current time in epoch milliseconds (UTC)."""
        ...


class SystemClock:
    """An epoch clock whose elapsed time comes from the monotonic system clock.

    The wall clock supplies the initial timestamp only. Later NTP or host clock corrections cannot
    make episode durations negative.
    """

    def __init__(self) -> None:
        self._epoch_ms = int(time.time() * 1000)
        self._started_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        return self._epoch_ms + (time.monotonic_ns() - self._started_ns) // 1_000_000


class ManualClock:
    """A :class:`Clock` that starts at a fixed instant and only moves when advanced.

    Tests advance it explicitly (a "slow" agent's ``act`` calls ``advance``), so timeouts
    trip deterministically without a real sleep and timestamps are reproducible.
    """

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def advance(self, ms: int) -> None:
        """Move the clock forward by ``ms`` milliseconds."""
        if ms < 0:
            raise ValueError("cannot advance the clock backwards")
        self._now += ms
