"""意图识别节点。

业务职责：
- 识别是否为运维排障问题（OPS_ANALYSIS）。
- 从问题中提取 order_id / request_id。
- 初始化 retry/replan/tool 调用计数字段，保障后续循环可控。
"""

from __future__ import annotations

import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_ORDER_ID_RE = re.compile(r"(?:订单|order[_\s-]?id|order)(?:[:=：\s#-]*)?(\d{4,20})", re.IGNORECASE)
_REQUEST_ID_RE = re.compile(r"(?:request[_\s-]?id|req[_\s-]?id)(?:[:=：\s#-]*)?([a-zA-Z0-9_.:-]{6,128})", re.IGNORECASE)
_OPS_HINTS = ("失败", "异常", "超时", "报错", "告警", "日志", "error", "timeout")


def _as_int(value: Any, default: int) -> int:
    """把外部输入安全转成 int，异常时回退默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_question(text: str) -> str:
    """问题归一化：去除首尾空白并压缩连续空格。"""
    return re.sub(r"\s+", " ", str(text or "").strip())


def _extract_with_regex(text: str, pattern: re.Pattern[str]) -> str:
    """按正则提取结构化字段，未命中返回空字符串。"""
    matched = pattern.search(text)
    if not matched:
        return ""
    return str(matched.group(1) or "").strip()


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行意图识别与上下文字段初始化。

    入参：
    - payload: AgentState，通常包含 message/structured_context。

    返参：
    - AgentState: 写入 intent_type、结构化 ID 字段及循环控制默认值。
    """
    state: AgentState = dict(payload)
    context = dict(state.get("structured_context") or {})

    question = str(state.get("question") or state.get("normalized_question") or state.get("message") or "").strip()
    normalized_question = _normalize_question(question)
    history = state.get("conversation_context") or context.get("recent_messages") or []
    conversation_context = [str(item) for item in history if str(item).strip()]

    # 运维关键词命中优先判定为 OPS_ANALYSIS，其次才按长度走 UNKNOWN/GENERAL_QA。
    lower_question = normalized_question.lower()
    if not normalized_question:
        intent_type = "UNKNOWN"
    elif any(hint in lower_question for hint in _OPS_HINTS):
        intent_type = "OPS_ANALYSIS"
    elif len(normalized_question) <= 3:
        intent_type = "UNKNOWN"
    else:
        intent_type = "GENERAL_QA"

    # 从问题文本中抽取业务 ID，后续工具调用直接复用。
    order_id = str(state.get("order_id") or context.get("order_id") or _extract_with_regex(normalized_question, _ORDER_ID_RE))
    request_id = str(state.get("request_id") or context.get("request_id") or _extract_with_regex(normalized_question, _REQUEST_ID_RE))

    state["question"] = question or normalized_question
    state["normalized_question"] = normalized_question
    state["conversation_context"] = conversation_context
    state["intent_type"] = intent_type

    state["structured_context"] = {
        **context,
        "question": state["question"],
        "normalized_question": normalized_question,
        "conversation_context": conversation_context,
        "order_id": order_id,
        "request_id": request_id,
    }
    state["order_id"] = order_id
    state["request_id"] = request_id

    state["retry_count"] = max(0, _as_int(state.get("retry_count"), 0))
    state["max_retry"] = max(0, _as_int(state.get("max_retry", state.get("max_retries")), 2))
    # 兼容旧字段 max_retries，避免外部调用方依赖断裂。
    state["max_retries"] = state["max_retry"]
    state["replan_count"] = max(0, _as_int(state.get("replan_count"), 0))
    state["max_replan"] = max(0, _as_int(state.get("max_replan"), 2))
    state["tool_call_count"] = max(0, _as_int(state.get("tool_call_count"), 0))
    state["max_tool_calls"] = max(1, _as_int(state.get("max_tool_calls"), 6))
    state["current_step_index"] = max(0, _as_int(state.get("current_step_index"), 0))
    state["tool_history"] = [dict(item) for item in list(state.get("tool_history") or [])]
    state["route"] = "rag_retrieve"
    return dict(state)
