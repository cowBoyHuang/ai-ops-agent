"""Shared store for message_cache_context_{chat_id}."""

from __future__ import annotations

import os

from cache.message_cache_context import MessageCacheContext, RoundMessageContext
from cache.redis import RedisAtomicClient

_REDIS = RedisAtomicClient(redis_url=os.getenv("REDIS_URL", ""))
_FALLBACK: dict[str, MessageCacheContext] = {}
_MAX_ROUNDS = 50


def _chat_id(chat_id: str) -> str:
    return str(chat_id or "").strip()


def _trim_rounds(cache: MessageCacheContext) -> None:
    if len(cache.rounds) > _MAX_ROUNDS:
        del cache.rounds[: len(cache.rounds) - _MAX_ROUNDS]


def get_message_cache(chat_id: str) -> MessageCacheContext | None:
    chat = _chat_id(chat_id)
    if not chat:
        return None

    if _REDIS.redis_enabled:
        key = _REDIS.build_message_cache_context_key(chat)
        raw = _REDIS.get(key)
        if raw:
            parsed = MessageCacheContext.from_json(raw)
            if parsed is not None:
                return parsed
    return _FALLBACK.get(chat)


def set_message_cache(chat_id: str, cache: MessageCacheContext) -> bool:
    chat = _chat_id(chat_id)
    if not chat:
        return False
    _trim_rounds(cache)

    if _REDIS.redis_enabled:
        key = _REDIS.build_message_cache_context_key(chat)
        if _REDIS.set(key, cache.to_json()):
            return True
    _FALLBACK[chat] = cache
    return True


def append_round(
    *,
    chat_id: str,
    summary: str,
    message: str,
    user_message_embedding: list[float],
    ai_response: str,
    tools_context: dict[str, object],
) -> bool:
    chat = _chat_id(chat_id)
    if not chat:
        return False

    cache = get_message_cache(chat) or MessageCacheContext()
    cache.summary = str(summary or cache.summary or "")
    cache.rounds.append(
        RoundMessageContext(
            message=str(message or ""),
            userMessageEmbedding=[float(item) for item in list(user_message_embedding or [])],
            aiResponse=str(ai_response or ""),
            toolsContext=dict(tools_context or {}),
        )
    )
    return set_message_cache(chat, cache)
