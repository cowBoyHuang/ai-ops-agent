"""Redis cache interaction for memory module."""

from __future__ import annotations

import os

from cache.redis import RedisAtomicClient


class MemoryCacheStore:
    """Wrap Redis operations required by memory module."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.client = RedisAtomicClient(redis_url=redis_url or os.getenv("REDIS_URL", ""))
        self._memory_fallback: dict[str, str] = {}

    def cache_total_message(self, value: str) -> bool:
        key = "total_message"
        text = str(value)
        if self.client.redis_enabled:
            return self.client.set(key, text)
        self._memory_fallback[key] = text
        return True

    def cache_summary_message(self, value: str) -> bool:
        key = "summary_message"
        text = str(value)
        if self.client.redis_enabled:
            return self.client.set(key, text)
        self._memory_fallback[key] = text
        return True

