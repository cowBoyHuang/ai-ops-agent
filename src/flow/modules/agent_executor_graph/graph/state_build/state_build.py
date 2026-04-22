"""子图状态构建节点。

业务职责：
- 从上游上下文提取基础业务字段（order_id/request_id）。
- 初始化重试、重规划、工具调用等执行计数器。
- 统一补齐子图执行所需默认字段，避免后续节点重复初始化。
"""

from __future__ import annotations

import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_ORDER_ID_RE = re.compile(r"(?:订单|order[_\s-]?id|order)(?:[:=：\s#-]*)?(\d{4,20})", re.IGNORECASE)
_REQUEST_ID_RE = re.compile(r"(?:request[_\s-]?id|req[_\s-]?id)(?:[:=：\s#-]*)?([a-zA-Z0-9_.:-]{6,128})", re.IGNORECASE)


def _as_int(value: Any, default: int) -> int:
    """把外部输入安全转成 int，异常时回退默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_with_regex(text: str, pattern: re.Pattern[str]) -> str:
    """按正则提取结构化字段，未命中返回空字符串。"""
    matched = pattern.search(text)
    if not matched:
        return ""
    return str(matched.group(1) or "").strip()


def _pick_message(state: dict[str, Any]) -> str:
    """优先读取 context.message，不存在时回退到 message。"""
    ctx = state.get("context")
    if isinstance(ctx, dict):
        text = str(ctx.get("message") or "").strip()
        if text:
            return text
    return str(state.get("message") or "").strip()


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建子图公共状态。

    入参：
    - payload: AgentState，包含上游 flow 传入的 context/message 与结构化上下文。

    出参：
    - AgentState: 写入业务 ID 与执行控制字段默认值。
    """
    state: AgentState = dict(payload)
    structured_context = dict(state.get("structured_context") or {})
    message = _pick_message(state)

    order_id = str(
        state.get("order_id")
        or structured_context.get("order_id")
        or _extract_with_regex(message, _ORDER_ID_RE)
    )
    request_id = str(
        state.get("request_id")
        or structured_context.get("request_id")
        or _extract_with_regex(message, _REQUEST_ID_RE)
    )

    state["order_id"] = order_id
    state["request_id"] = request_id
    state["retry_count"] = max(0, _as_int(state.get("retry_count"), 0))
    state["max_retry"] = max(0, _as_int(state.get("max_retry", state.get("max_retries")), 2))
    state["max_retries"] = state["max_retry"]
    state["replan_count"] = max(0, _as_int(state.get("replan_count"), 0))
    state["max_replan"] = max(0, _as_int(state.get("max_replan"), 2))
    state["tool_call_count"] = max(0, _as_int(state.get("tool_call_count"), 0))
    state["max_tool_calls"] = max(1, _as_int(state.get("max_tool_calls"), 6))
    state["current_step_index"] = max(0, _as_int(state.get("current_step_index"), 0))
    state["tool_history"] = [dict(item) for item in list(state.get("tool_history") or [])]
    state["intent_retry_count"] = max(0, _as_int(state.get("intent_retry_count"), 0))
    state["intent_history_prompt"] = str(state.get("intent_history_prompt") or "")
    state["intent_retry_results"] = [dict(item) for item in list(state.get("intent_retry_results") or [])]
    state["structured_context"] = {
        **structured_context,
        "order_id": order_id,
        "request_id": request_id,
    }
    return dict(state)
