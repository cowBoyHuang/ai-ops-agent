"""Redis cache interaction for memory module."""

from __future__ import annotations

import os
from typing import Any

from cache.message_cache_context import MessageCacheContext
from cache.redis import RedisAtomicClient


class MemoryCacheStore:
    """Wrap Redis operations required by memory module."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.client = RedisAtomicClient(redis_url=redis_url or os.getenv("REDIS_URL", ""))
        self._message_context_fallback: dict[str, MessageCacheContext] = {}
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
        user_question: str,
        agent_answer: str,
        tools_context: dict[str, Any],
        user_question_embedding: list[float],
    ) -> bool:
        chat = str(chat_id or "").strip()
        if not chat:
            return False
        context = MessageCacheContext(
            UserQuestion=str(user_question or ""),
            agentAnswer=str(agent_answer or ""),
            toolsContext=dict(tools_context or {}),
            UserQuestionEmbedding=[float(item) for item in list(user_question_embedding or [])],
        )
        if self.client.redis_enabled:
            ok = self.client.set_message_cache_context(chat, context)
            if ok:
                return True
        self._message_context_fallback[chat] = context
        return True

    def get_message_context(self, chat_id: str) -> MessageCacheContext | None:
        chat = str(chat_id or "").strip()
        if not chat:
            return None
        if self.client.redis_enabled:
            return self.client.get_message_cache_context(chat)
        return self._message_context_fallback.get(chat)

    # ===== backward-compatible methods =====
    def cache_total_message(self, value: str) -> bool:
        return self.cache_message_context(
            chat_id="global",
            user_question=str(value or ""),
            agent_answer="",
            tools_context={},
            user_question_embedding=[],
        )

    def cache_summary_message(self, value: str) -> bool:
        return self.cache_message_context(
            chat_id="global_summary",
            user_question="",
            agent_answer=str(value or ""),
            tools_context={},
            user_question_embedding=[],
        )
