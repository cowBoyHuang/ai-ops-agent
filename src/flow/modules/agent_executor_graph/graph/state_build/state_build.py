"""子图状态构建节点。

业务职责：
- 仅从上游 `context` 搬运字段到 AgentState。
- 对缺失字段补固定默认值，不做推导/提取/清洗等业务逻辑。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from cache.message_cache_context import MessageCacheContext, RoundMessageContext
from flow.modules.agent_executor_graph.agent_state import AgentState

_TRACE_ID_PATTERN = re.compile(
    r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)(?=$|[^A-Za-z0-9_\.\-])",
    re.IGNORECASE,
)
_ORDER_ID_PATTERN = re.compile(r"\b(?:xep|sid|fod|hpv)[A-Za-z0-9]{6,}\b", re.IGNORECASE)


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


def _pick_upstream_value(
    raw_context: dict[str, Any],
    structured_context: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = raw_context.get(key)
        if value is not None and str(value).strip():
            return value
    for key in keys:
        value = structured_context.get(key)
        if value is not None and str(value).strip():
            return value
    return ""


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _restore_message_context(value: Any) -> MessageCacheContext:
    if isinstance(value, MessageCacheContext):
        return value
    if isinstance(value, dict):
        return MessageCacheContext.from_dict(value)
    if isinstance(value, str):
        return MessageCacheContext.from_json(value) or MessageCacheContext()
    return MessageCacheContext()


def _clip(value: Any, max_len: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _extract_ids_from_text(text: str) -> tuple[str, str]:
    raw = str(text or "")
    trace_match = _TRACE_ID_PATTERN.search(raw)
    order_match = _ORDER_ID_PATTERN.search(raw)
    return (
        str(trace_match.group(0) if trace_match else "").strip(),
        str(order_match.group(0) if order_match else "").strip(),
    )


def _round_tools_context_to_history_lines(round_row: RoundMessageContext) -> list[str]:
    tools_context = dict(round_row.toolsContext or {})
    if not tools_context:
        return []
    rows: list[str] = []
    raw_text = json.dumps(tools_context, ensure_ascii=False, default=str)
    trace_id, order_id = _extract_ids_from_text(raw_text)
    if trace_id or order_id:
        rows.append(f"历史继承上下文：trace_id={trace_id} order_id={order_id}".strip())

    resolved = dict(tools_context.get("required_answer_resolved") or {})
    if resolved:
        rows.append(f"历史字段结果：{_clip(json.dumps(resolved, ensure_ascii=False), 500)}")

    last_tool = dict(tools_context.get("last_tool") or {})
    if last_tool:
        tool_name = str(last_tool.get("tool_name") or "").strip()
        summary = _clip(last_tool.get("result_summary"), 320)
        conclusion = str(last_tool.get("conclusion") or "").strip()
        if tool_name or summary or conclusion:
            rows.append(
                f"历史工具结果：tool={tool_name} summary={summary} conclusion={_clip(conclusion, 180)}".strip()
            )

    conclusion = str(tools_context.get("conclusion") or "").strip()
    if conclusion:
        rows.append(f"历史结论：{_clip(conclusion, 400)}")
    return rows


def _round_to_history_line(round_row: RoundMessageContext) -> str:
    user_text = str(round_row.message or "").strip()
    assistant_text = str(round_row.aiResponse or "").strip()
    if user_text and assistant_text:
        return f"用户：{user_text}\n助手：{assistant_text}"
    if user_text:
        return f"用户：{user_text}"
    if assistant_text:
        return f"助手：{assistant_text}"
    return ""


def _extract_recent_history_from_message_context(raw_context: dict[str, Any], *, limit: int = 3) -> list[str]:
    context = _restore_message_context(raw_context.get("message_context"))
    if not context.rounds:
        return []
    rows: list[str] = []
    for item in list(context.rounds)[-max(1, limit) :]:
        line = _round_to_history_line(item)
        if line:
            rows.append(line)
        rows.extend(_round_tools_context_to_history_lines(item))
    return rows


def _extract_ids_from_message_context(raw_context: dict[str, Any]) -> tuple[str, str]:
    context = _restore_message_context(raw_context.get("message_context"))
    if not context.rounds:
        return "", ""
    trace_id = ""
    order_id = ""
    for item in reversed(list(context.rounds)):
        if not trace_id:
            trace_id, _ = _extract_ids_from_text(str(item.message or ""))
        if not order_id:
            _, order_id = _extract_ids_from_text(str(item.message or ""))
        if not trace_id:
            trace_id, _ = _extract_ids_from_text(str(item.aiResponse or ""))
        if not order_id:
            _, order_id = _extract_ids_from_text(str(item.aiResponse or ""))
        if not trace_id or not order_id:
            tools_text = json.dumps(dict(item.toolsContext or {}), ensure_ascii=False, default=str)
            t_trace, t_order = _extract_ids_from_text(tools_text)
            trace_id = trace_id or t_trace
            order_id = order_id or t_order
        if trace_id and order_id:
            break
    return trace_id, order_id


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建子图公共状态（仅字段搬运 + 固定默认值）。"""
    state: AgentState = dict(payload)
    raw_context = dict(state.get("context") or {})
    structured_context = dict(state.get("structured_context") or {})
    raw_query_rewrite = dict(raw_context.get("query_rewrite") or {})
    raw_query_rewrite_entities = dict(raw_query_rewrite.get("extracted_entities") or {})
    raw_query_rewrite_window = dict(raw_query_rewrite.get("time_window") or {})

    question = _pick_question(raw_context, state, structured_context)
    history_trace_id, history_order_id = _extract_ids_from_message_context(raw_context)
    trace_id = str(
        raw_context.get("trace_id")
        or raw_context.get("traceId")
        or raw_query_rewrite.get("trace_id")
        or raw_query_rewrite_entities.get("trace_id")
        or structured_context.get("trace_id")
        or history_trace_id
        or ""
    ).strip()
    order_id = str(
        raw_context.get("order_id")
        or raw_context.get("orderNo")
        or raw_context.get("orderId")
        or raw_query_rewrite.get("order_id")
        or raw_query_rewrite_entities.get("order_id")
        or structured_context.get("order_id")
        or history_order_id
        or ""
    ).strip()
    begin_time = _pick_upstream_value(
        raw_context,
        structured_context,
        ("begin_time", "beginTime", "start_time", "startTime"),
    )
    end_time = _pick_upstream_value(
        raw_context,
        structured_context,
        ("end_time", "endTime", "finish_time", "finishTime"),
    )
    begin_time = str(begin_time or raw_query_rewrite_window.get("begin_time") or "").strip()
    end_time = str(end_time or raw_query_rewrite_window.get("end_time") or "").strip()
    state["question"] = question
    state["structured_context"] = {
        **structured_context,
        "question": question,
        "trace_id": trace_id,
        "order_id": order_id,
        "request_id": str(raw_context.get("request_id") or structured_context.get("request_id") or ""),
        "begin_time": begin_time,
        "end_time": end_time,
        "query_rewrite": raw_query_rewrite or dict(structured_context.get("query_rewrite") or {}),
        "simulate_tool_timeout_once": bool(
            raw_context.get("simulate_tool_timeout_once") or structured_context.get("simulate_tool_timeout_once")
        ),
    }

    state["conversation_context"] = list(
        raw_context.get("conversation_context")
        or state.get("conversation_context")
        or structured_context.get("recent_messages")
        or _extract_recent_history_from_message_context(raw_context)
        or []
    )
    state["retry_count"] = raw_context.get("retry_count", state.get("retry_count", 0))
    state["max_retries"] = raw_context.get("max_retries", raw_context.get("max_retry", state.get("max_retries", 0)))
    state["replan_count"] = raw_context.get("replan_count", state.get("replan_count", 0))
    state["max_replan"] = raw_context.get("max_replan", state.get("max_replan", _env_int("AIOPS_MAX_REPLAN", 0)))
    state["max_insufficient_rounds"] = raw_context.get(
        "max_insufficient_rounds", state.get("max_insufficient_rounds", 2)
    )
    state["tool_call_count"] = raw_context.get("tool_call_count", state.get("tool_call_count", 0))
    # Plan-ReAct 会在同一步内做有限重试，默认预算需覆盖“多步 + 重规划”场景，避免过早触发预算熔断。
    state["max_tool_calls"] = raw_context.get("max_tool_calls", state.get("max_tool_calls", 24))
    state["in_place_retry_count"] = raw_context.get("in_place_retry_count", state.get("in_place_retry_count", 0))
    state["current_step_index"] = raw_context.get("current_step_index", state.get("current_step_index", 0))
    state["current_plan"] = raw_context.get("current_plan", state.get("current_plan", state.get("plan_steps", [])))
    state["original_plan"] = raw_context.get("original_plan", state.get("original_plan", []))
    state["needs_adjustment"] = bool(raw_context.get("needs_adjustment", state.get("needs_adjustment", False)))
    state["adjustment_type"] = str(raw_context.get("adjustment_type") or state.get("adjustment_type") or "")
    state["proposed_changes"] = raw_context.get("proposed_changes", state.get("proposed_changes", {}))
    state["pending_insertions"] = raw_context.get("pending_insertions", state.get("pending_insertions", []))
    state["adjustment_history"] = raw_context.get("adjustment_history", state.get("adjustment_history", []))
    state["tool_history"] = raw_context.get("tool_history", state.get("tool_history", []))
    state["execution_history"] = raw_context.get("execution_history", state.get("execution_history", {}))
    state["current_step_result"] = raw_context.get("current_step_result", state.get("current_step_result", {}))
    state["newly_discovered_clues"] = raw_context.get("newly_discovered_clues", state.get("newly_discovered_clues", []))
    state["intermediate_results"] = raw_context.get("intermediate_results", state.get("intermediate_results", {}))
    state["extracted_keywords"] = raw_context.get("extracted_keywords", state.get("extracted_keywords", []))
    state["intent_retry_count"] = raw_context.get("intent_retry_count", state.get("intent_retry_count", 0))
    state["intent_history_prompt"] = str(
        raw_context.get("intent_history_prompt") or state.get("intent_history_prompt") or ""
    )
    state["intent_retry_results"] = raw_context.get("intent_retry_results", state.get("intent_retry_results", []))
    state["query_rewrite"] = raw_context.get("query_rewrite", state.get("query_rewrite", {}))
    state["plan"] = raw_context.get("plan", state.get("plan", {}))
    state["knowledge_context"] = raw_context.get("knowledge_context", state.get("knowledge_context", {}))
    state["execution"] = raw_context.get("execution", state.get("execution", {}))
    if isinstance(state["execution"], dict):
        state["execution"].setdefault("max_insufficient_rounds", int(state.get("max_insufficient_rounds") or 2))
        state["execution"].setdefault("max_act_times", _env_int("AIOPS_REACTOR_MAX_ACT_TIMES", 3))
        state["execution"].setdefault("goal_reports", [])
        state["execution"].setdefault("reactor_runtime", {})
    state["evaluation"] = raw_context.get("evaluation", state.get("evaluation", {}))
    state["rejected_hypothesis"] = raw_context.get(
        "rejected_hypothesis", state.get("rejected_hypothesis", [])
    )
    state["replan_reason"] = str(raw_context.get("replan_reason") or state.get("replan_reason") or "")
    return dict(state)
