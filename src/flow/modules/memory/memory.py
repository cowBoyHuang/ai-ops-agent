"""Memory module orchestration."""

from __future__ import annotations

from typing import Any

from db.db_store import ChatDBStore
from flow.modules.memory.cache_store import MemoryCacheStore
from flow.modules.memory.summary import summarize_with_llm_placeholder

_TRACE_ROWS: dict[str, list[dict[str, Any]]] = {}
_CACHE_STORE = MemoryCacheStore()
_DB_STORE = ChatDBStore()


def _to_float_list(values: Any) -> list[float]:
    rows: list[float] = []
    for item in list(values or []):
        try:
            rows.append(float(item))
        except (TypeError, ValueError):
            continue
    return rows


def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    chat_id = str(context.get("chat_id") or "")
    user_id = str(context.get("user_id") or "anonymous")
    message = str(context.get("message") or "")

    response_message = str((context.get("response") or {}).get("message") or "")

    total_message = f"user:{message}\nassistant:{response_message}".strip()
    summary_message = summarize_with_llm_placeholder(
        total_message=total_message,
        summary_message=str(context.get("summary_message") or ""),
    )

    # Redis cache keys:
    # - repeat_chat_id_{chat_id} -> 1
    # - message_cache_context_{chat_id} -> MessageCacheContext JSON
    _CACHE_STORE.mark_repeat_chat_id(chat_id)
    _CACHE_STORE.cache_message_context(
        chat_id=chat_id,
        summary=summary_message,
        user_question=message,
        agent_answer=response_message,
        tools_context={
            "tool_result": dict(context.get("tool_result") or {}),
            "merged_evidence": dict(context.get("merged_evidence") or {}),
            "route": context.get("route"),
        },
        user_question_embedding=_to_float_list(context.get("UserQuestionEmbedding")),
    )

    # MySQL tables:
    # - total_message(id, chat_id, role, content)
    # - summary_message(id, user_id, chat_id, content)
    user_message_row_id = _DB_STORE.create_total_message(chat_id=chat_id or "unknown_chat", role="user", content=message)
    assistant_message_row_id = 0
    if response_message:
        assistant_message_row_id = _DB_STORE.create_total_message(
            chat_id=chat_id or "unknown_chat",
            role="assistant",
            content=response_message,
        )

    summary_row_id = _DB_STORE.create_summary_message(
        user_id=user_id,
        chat_id=chat_id or "unknown_chat",
        content=summary_message,
    )

    traces = _TRACE_ROWS.setdefault(chat_id, [])
    traces.append(
        {
            "status": context.get("status"),
            "route": context.get("route"),
            "retry_count": context.get("retry_count", 0),
            "error_code": context.get("error_code", ""),
        }
    )
    context["persisted"] = True
    context["memory_row_id"] = summary_row_id
    context["total_message_row_ids"] = [row_id for row_id in [user_message_row_id, assistant_message_row_id] if row_id]
    context["summary_message"] = summary_message
    return context
