from __future__ import annotations

import asyncio
import inspect
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small process-local bounded TTL cache.

    Entries are evicted least-recently-used after expired entries have been
    discarded. Cache operations contain no await points, which keeps them
    atomic under the application's single asyncio event loop.
    """

    def __init__(self, ttl_seconds: float = 600, max_size: int = 1000) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_size <= 0:
            raise ValueError("max_size must be greater than zero")
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: OrderedDict[str, CacheEntry[T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        self._discard_expired(now)
        self._store.pop(key, None)
        while len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(
            value=value, expires_at=now + self.ttl_seconds
        )

    def delete(self, key: str) -> None:
        """Remove a single entry immediately, regardless of TTL."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def _discard_expired(self, now: float) -> None:
        expired = [
            key for key, entry in self._store.items() if entry.expires_at <= now
        ]
        for key in expired:
            self._store.pop(key, None)


class AsyncSingleFlight(Generic[T]):
    """Share one asyncio task among concurrent callers for the same key."""

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Task[T]] = {}
        self._lock = asyncio.Lock()

    async def run(
        self,
        key: str,
        computation: Callable[[], Awaitable[T]],
        *,
        on_role: Callable[[bool], None] | None = None,
    ) -> tuple[T, bool]:
        """Return ``(result, is_leader)`` without propagating caller cancellation.

        The registry lock only protects task creation and removal. The
        computation itself always runs after the lock has been released.
        """
        async with self._lock:
            task = self._in_flight.get(key)
            is_leader = task is None
            if task is None:
                task = asyncio.create_task(self._run_and_cleanup(key, computation))
                task.add_done_callback(self._consume_completion)
                self._in_flight[key] = task

        if on_role is not None:
            on_role(is_leader)
        return await asyncio.shield(task), is_leader

    async def _run_and_cleanup(
        self,
        key: str,
        computation: Callable[[], Awaitable[T]],
    ) -> T:
        current_task = asyncio.current_task()
        try:
            result = computation()
            if not inspect.isawaitable(result):
                raise TypeError("single-flight computation must return an awaitable")
            return await result
        finally:
            async with self._lock:
                if self._in_flight.get(key) is current_task:
                    self._in_flight.pop(key, None)

    async def in_flight_count(self) -> int:
        async with self._lock:
            return len(self._in_flight)

    @staticmethod
    def _consume_completion(task: asyncio.Task[T]) -> None:
        # A shared task may outlive every cancelled caller. Retrieve any
        # exception so asyncio does not report it as unhandled; later awaiters
        # still receive the same exception from the task.
        if not task.cancelled():
            task.exception()
