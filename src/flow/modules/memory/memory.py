"""Memory module orchestration."""

from __future__ import annotations

from typing import Any

from cache.message_cache_store import clear_message_cache_fallback
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


def _clip(value: Any, max_len: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _pick_trace_order(context: dict[str, Any]) -> tuple[str, str]:
    structured = dict(context.get("structured_context") or {})
    rewrite = dict(context.get("query_rewrite") or structured.get("query_rewrite") or {})
    extracted = dict(rewrite.get("extracted_entities") or {})
    trace_id = str(
        rewrite.get("trace_id")
        or extracted.get("trace_id")
        or structured.get("trace_id")
        or context.get("trace_id")
        or ""
    ).strip()
    order_id = str(
        rewrite.get("order_id")
        or extracted.get("order_id")
        or structured.get("order_id")
        or context.get("order_id")
        or context.get("orderNo")
        or ""
    ).strip()
    return trace_id, order_id


def _build_tools_context_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    execution = dict(context.get("execution") or {})
    analysis = dict(context.get("analysis") or {})
    rewrite = dict(context.get("query_rewrite") or {})
    required_results = dict(execution.get("required_answer_results") or {})
    goal_reports = [dict(item or {}) for item in list(execution.get("goal_reports") or [])]
    last_goal = dict(goal_reports[-1] or {}) if goal_reports else {}
    action_chain = [dict(item or {}) for item in list(last_goal.get("action_chain") or [])]
    last_action = dict(action_chain[-1] or {}) if action_chain else {}
    trace_id, order_id = _pick_trace_order(context)

    conclusion = str(
        context.get("root_cause")
        or analysis.get("root_cause")
        or analysis.get("reply")
        or last_goal.get("goal_conclusion")
        or ""
    ).strip()
    goal_conclusions: list[dict[str, str]] = []
    for row in goal_reports[-5:]:
        objective = str(row.get("goal_objective") or "").strip()
        goal_conclusion = str(row.get("goal_conclusion") or "").strip()
        if not objective and not goal_conclusion:
            continue
        goal_conclusions.append(
            {
                "objective": _clip(objective, 180),
                "conclusion": _clip(goal_conclusion, 300),
            }
        )

    snapshot = {
        "trace_id": trace_id,
        "order_id": order_id,
        "query_rewrite": {
            "normalized_query": _clip(rewrite.get("normalized_query"), 300),
            "trace_id": trace_id,
            "order_id": order_id,
            "keywords": list(rewrite.get("keywords") or [])[:8],
        },
        "required_answer_resolved": dict(required_results.get("resolved") or {}),
        "last_tool": {
            "tool_name": str(last_action.get("tool_name") or "").strip(),
            "conclusion": str(last_action.get("conclusion") or "").strip(),
            "result_summary": _clip(last_action.get("result_summary"), 500),
            "error": _clip(last_action.get("error"), 240),
        },
        "goal_conclusions": goal_conclusions,
        "conclusion": _clip(conclusion, 600),
        "route": str(context.get("route") or "").strip(),
    }
    return snapshot


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
    tools_context_snapshot = _build_tools_context_snapshot(context)
    _CACHE_STORE.cache_message_context(
        chat_id=chat_id,
        summary=summary_message,
        user_question=message,
        agent_answer=response_message,
        tools_context={
            "tool_result": dict(context.get("tool_result") or {}),
            "merged_evidence": dict(context.get("merged_evidence") or {}),
            **tools_context_snapshot,
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
            "error_code": context.get("error_code", ""),
        }
    )
    context["persisted"] = True
    context["memory_row_id"] = summary_row_id
    context["total_message_row_ids"] = [row_id for row_id in [user_message_row_id, assistant_message_row_id] if row_id]
    context["summary_message"] = summary_message
    return context


def clear_persistent_data() -> dict[str, Any]:
    db_deleted = _DB_STORE.clear_data_only()
    redis_deleted = _CACHE_STORE.client.clear_chat_cache_data()
    repeat_chat_fallback_cleared = _CACHE_STORE.clear_repeat_chat_fallback()
    message_cache_fallback_cleared = clear_message_cache_fallback()
    trace_rows_cleared = len(_TRACE_ROWS)
    _TRACE_ROWS.clear()
    return {
        "db_enabled": _DB_STORE.enabled,
        "db_deleted": db_deleted,
        "redis_enabled": _CACHE_STORE.client.redis_enabled,
        "redis_deleted": redis_deleted,
        "repeat_chat_fallback_cleared": repeat_chat_fallback_cleared,
        "message_cache_fallback_cleared": message_cache_fallback_cleared,
        "trace_rows_cleared": trace_rows_cleared,
    }
