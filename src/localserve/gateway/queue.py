"""Admission control: at most `max_concurrent` requests upstream, at most `max_queue` waiting.

A request beyond `max_concurrent + max_queue` is rejected immediately (HTTP 429) rather than
queued without bound. Unbounded queues turn overload into ever-growing latency for everyone;
a bounded queue keeps latency for admitted requests bounded and pushes back on clients.
"""

from __future__ import annotations

import asyncio


class QueueFull(Exception):
    pass


class Slot:
    """One admitted request. `acquire()` waits for a concurrency slot; `release()` is idempotent."""

    def __init__(self, admission: Admission) -> None:
        self._adm = admission
        self._acquired = False
        self._released = False

    async def acquire(self) -> None:
        await self._adm._sem.acquire()
        self._acquired = True
        self._adm.inflight += 1

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._adm.admitted -= 1
        if self._acquired:
            self._adm.inflight -= 1
            self._adm._sem.release()


class Admission:
    def __init__(self, max_concurrent: int, max_queue: int) -> None:
        if max_concurrent < 1 or max_queue < 0:
            raise ValueError("max_concurrent >= 1 and max_queue >= 0 required")
        self._sem = asyncio.Semaphore(max_concurrent)
        self.capacity = max_concurrent + max_queue
        self.admitted = 0  # waiting + inflight
        self.inflight = 0

    @property
    def waiting(self) -> int:
        return self.admitted - self.inflight

    def admit(self) -> Slot:
        if self.admitted >= self.capacity:
            raise QueueFull
        self.admitted += 1
        return Slot(self)
