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

    _CACHE_STORE.cache_total_message(total_message)
    _CACHE_STORE.cache_summary_message(summary_message)
    row_id = _DB_STORE.insert_message(
        chat_id=chat_id or "unknown_chat",
        user_id=user_id,
        total_message=total_message,
        summary_message=summary_message,
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
    state["memory_row_id"] = row_id
    state["summary_message"] = summary_message
    return state

