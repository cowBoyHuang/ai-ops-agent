"""子图状态构建节点。

业务职责：
- 仅从上游 `context` 搬运字段到 AgentState。
- 对缺失字段补固定默认值，不做推导/提取/清洗等业务逻辑。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState


def _pick_question(raw_context: dict[str, Any], state: dict[str, Any], structured_context: dict[str, Any]) -> str:
    for key in ("question", "message", "query", "content"):
        value = raw_context.get(key)
        if value is not None and str(value).strip():
            return str(value)
    value = state.get("question")
    if value is not None and str(value).strip():
        return str(value)
    value = structured_context.get("question")
    if value is not None and str(value).strip():
        return str(value)
    return ""


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建子图公共状态（仅字段搬运 + 固定默认值）。"""
    state: AgentState = dict(payload)
    raw_context = dict(state.get("context") or {})
    structured_context = dict(state.get("structured_context") or {})

    question = _pick_question(raw_context, state, structured_context)
    state["question"] = question
    state["structured_context"] = {
        **structured_context,
        "question": question,
        "order_id": str(raw_context.get("order_id") or structured_context.get("order_id") or ""),
        "request_id": str(raw_context.get("request_id") or structured_context.get("request_id") or ""),
        "simulate_tool_timeout_once": bool(
            raw_context.get("simulate_tool_timeout_once") or structured_context.get("simulate_tool_timeout_once")
        ),
    }

    state["conversation_context"] = list(
        raw_context.get("conversation_context") or state.get("conversation_context") or []
    )
    state["retry_count"] = raw_context.get("retry_count", state.get("retry_count", 0))
    state["max_retries"] = raw_context.get("max_retries", raw_context.get("max_retry", state.get("max_retries", 2)))
    state["replan_count"] = raw_context.get("replan_count", state.get("replan_count", 0))
    state["max_replan"] = raw_context.get("max_replan", state.get("max_replan", 2))
    state["tool_call_count"] = raw_context.get("tool_call_count", state.get("tool_call_count", 0))
    state["max_tool_calls"] = raw_context.get("max_tool_calls", state.get("max_tool_calls", 6))
    state["current_step_index"] = raw_context.get("current_step_index", state.get("current_step_index", 0))
    state["tool_history"] = raw_context.get("tool_history", state.get("tool_history", []))
    state["execution_history"] = raw_context.get("execution_history", state.get("execution_history", {}))
    state["intermediate_results"] = raw_context.get("intermediate_results", state.get("intermediate_results", {}))
    state["extracted_keywords"] = raw_context.get("extracted_keywords", state.get("extracted_keywords", []))
    state["intent_retry_count"] = raw_context.get("intent_retry_count", state.get("intent_retry_count", 0))
    state["intent_history_prompt"] = str(
        raw_context.get("intent_history_prompt") or state.get("intent_history_prompt") or ""
    )
    state["intent_retry_results"] = raw_context.get("intent_retry_results", state.get("intent_retry_results", []))
    return dict(state)
