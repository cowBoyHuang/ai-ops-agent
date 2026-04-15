"""Memory module orchestration."""

from __future__ import annotations

from typing import Any

from flow.modules.context_build.context_build import remember_message
from flow.modules.duplicate_detect.duplicate_detect import remember_qa
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
    state = dict(payload)
    chat_id = str(state.get("chat_id") or "")
    user_id = str(state.get("user_id") or "anonymous")
    message = str(state.get("message") or "")
    remember_message(chat_id, message)

    response_message = str((state.get("response") or {}).get("message") or "")
    if response_message:
        remember_qa(message, response_message)

    total_message = f"user:{message}\nassistant:{response_message}".strip()
    summary_message = summarize_with_llm_placeholder(
        total_message=total_message,
        summary_message=str(state.get("summary_message") or ""),
    )

    # Redis cache keys:
    # - repeat_chat_id_{chat_id} -> 1
    # - message_cache_context_{chat_id} -> MessageCacheContext JSON
    _CACHE_STORE.mark_repeat_chat_id(chat_id)
    _CACHE_STORE.cache_message_context(
        chat_id=chat_id,
        user_question=message,
        agent_answer=response_message,
        tools_context={
            "tool_result": dict(state.get("tool_result") or {}),
            "merged_evidence": dict(state.get("merged_evidence") or {}),
            "route": state.get("route"),
        },
        user_question_embedding=_to_float_list(state.get("UserQuestionEmbedding")),
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
            "status": state.get("status"),
            "route": state.get("route"),
            "retry_count": state.get("retry_count", 0),
            "error_code": state.get("error_code", ""),
        }
    )
    state["persisted"] = True
    state["memory_row_id"] = summary_row_id
    state["total_message_row_ids"] = [row_id for row_id in [user_message_row_id, assistant_message_row_id] if row_id]
    state["summary_message"] = summary_message
    return state
