"""Redis cache interaction for memory module."""

from __future__ import annotations

import os
from typing import Any

from cache.message_cache_context import MessageCacheContext
from cache.message_cache_store import append_round, get_message_cache
from cache.redis import RedisAtomicClient


class MemoryCacheStore:
    """Wrap Redis operations required by memory module."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.client = RedisAtomicClient(redis_url=redis_url or os.getenv("REDIS_URL", ""))
        self._repeat_chat_fallback: set[str] = set()

    def mark_repeat_chat_id(self, chat_id: str) -> bool:
        chat = str(chat_id or "").strip()
        if not chat:
            return False
        if self.client.redis_enabled:
            ok = self.client.mark_repeat_chat_id(chat)
            if ok:
                return True
        self._repeat_chat_fallback.add(chat)
        return True

    def is_repeat_chat_id(self, chat_id: str) -> bool:
        chat = str(chat_id or "").strip()
        if not chat:
            return False
        if self.client.redis_enabled:
            return self.client.is_repeat_chat_id(chat)
        return chat in self._repeat_chat_fallback

    def cache_message_context(
        self,
        *,
        chat_id: str,
        summary: str,
        user_question: str,
        agent_answer: str,
        tools_context: dict[str, Any],
        user_question_embedding: list[float],
    ) -> bool:
        chat = str(chat_id or "").strip()
        if not chat:
            return False
        return append_round(
            chat_id=chat,
            summary=str(summary or ""),
            message=str(user_question or ""),
            user_message_embedding=[float(item) for item in list(user_question_embedding or [])],
            ai_response=str(agent_answer or ""),
            tools_context=dict(tools_context or {}),
        )

    def get_message_context(self, chat_id: str) -> MessageCacheContext | None:
        chat = str(chat_id or "").strip()
        if not chat:
            return None
        return get_message_cache(chat)

    # ===== backward-compatible methods =====
    def cache_total_message(self, value: str) -> bool:
        return self.cache_message_context(
            chat_id="global",
            summary="",
            user_question=str(value or ""),
            agent_answer="",
            tools_context={},
            user_question_embedding=[],
        )

    def cache_summary_message(self, value: str) -> bool:
        return self.cache_message_context(
            chat_id="global_summary",
            summary=str(value or ""),
            user_question="",
            agent_answer=str(value or ""),
            tools_context={},
            user_question_embedding=[],
        )
