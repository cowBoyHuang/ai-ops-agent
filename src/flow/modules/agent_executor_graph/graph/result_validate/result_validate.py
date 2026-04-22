"""结果验证节点。

业务职责：
- 判断分析结果是否可直接返回（SUCCESS）。
- 识别是否需要重试工具（NEED_RETRY）。
- 识别是否需要重规划（NEED_REPLAN）。
- 超预算或明显失败时标记 FAIL。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState


def _has_retryable_error(error_text: str) -> bool:
    """是否为可重试错误（网络抖动、超时、限流等）。"""
    lowered = str(error_text or "").lower()
    retry_hints = ("timeout", "network", "connection", "temporarily", "503", "429")
    return any(token in lowered for token in retry_hints)


def _has_uncertain_answer(root_cause: str, solution: str) -> bool:
    """分析文案是否表达了“不确定/信息不足”。"""
    text = f"{root_cause} {solution}".lower()
    uncertain_hints = ("无法确定", "不确定", "未知", "unknown", "need more", "更多信息")
    return any(token in text for token in uncertain_hints)


def _to_confidence(value: Any) -> float:
    """统一把置信度转换为 0~1 的浮点值。"""
    text = str(value or "").strip().lower()
    if text == "high":
        return 0.9
    if text == "medium":
        return 0.65
    if text == "low":
        return 0.35
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """验证本轮分析结果并输出 analysis_status。

    入参：
    - payload: AgentState，需包含 analysis/tool_result/merged_evidence 等字段。

    返参：
    - AgentState: 写入 analysis_status，并统一路由到 retry_router。
    """
    state: AgentState = dict(payload)
    analysis = dict(state.get("analysis") or {})
    tool_result = dict(state.get("tool_result") or {})
    merged_evidence = dict(state.get("merged_evidence") or {})

    root_cause = str(state.get("root_cause") or analysis.get("root_cause") or "").strip()
    solution = str(state.get("solution") or analysis.get("reply") or "").strip()
    confidence = _to_confidence(state.get("confidence") or analysis.get("confidence"))
    tool_ok = bool(tool_result.get("ok", True))
    tool_error = str(tool_result.get("error") or "")

    tool_call_count = int(state.get("tool_call_count") or 0)
    max_tool_calls = max(1, int(state.get("max_tool_calls") or 6))
    current_step_index = int(state.get("current_step_index") or 0)
    plan_steps = list(state.get("plan_steps") or [])
    has_more_plan_steps = current_step_index < len(plan_steps)
    logs = list(merged_evidence.get("logs") or [])
    knowledge = list(merged_evidence.get("knowledge") or [])
    has_evidence = bool(logs or knowledge)

    # 判定顺序遵循“先硬限制，再可重试，再成功，再重规划”。
    if tool_call_count >= max_tool_calls:
        status = "FAIL"
    elif not tool_ok and _has_retryable_error(tool_error):
        status = "NEED_RETRY"
    elif confidence > 0.7 and bool(root_cause):
        status = "SUCCESS"
    elif not has_evidence:
        status = "NEED_REPLAN"
    elif _has_uncertain_answer(root_cause, solution):
        status = "NEED_REPLAN"
    elif has_more_plan_steps:
        status = "NEED_RETRY"
    elif not root_cause:
        status = "NEED_REPLAN"
    else:
        status = "NEED_REPLAN"

    state["analysis_status"] = status
    state["route"] = "retry_router"
    return dict(state)
