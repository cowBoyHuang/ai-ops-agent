"""重试/重规划路由节点。

业务职责：
- 根据 analysis_status 决定下一跳（executor/planner/finish/fallback）。
- 控制 retry_count、replan_count、tool_call_count 三类预算。
- 保证预算耗尽时强制走 fallback，防止无限循环。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_FALLBACK_MESSAGE = "暂未能自动定位问题，请联系人工排查。"


def _as_int(value: Any, default: int) -> int:
    """把外部输入安全转换成整数。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行 Retry + Replan 路由判断。

    入参：
    - payload: AgentState，需包含 analysis_status 与各类计数器。

    返参：
    - AgentState: 更新计数器并写入 route，供图条件边继续执行。
    """
    state: AgentState = dict(payload)
    analysis_status = str(state.get("analysis_status") or "FAIL")
    retry_count = _as_int(state.get("retry_count"), 0)
    max_retry = max(0, _as_int(state.get("max_retries"), 2))
    replan_count = _as_int(state.get("replan_count"), 0)
    max_replan = max(0, _as_int(state.get("max_replan"), 2))
    tool_call_count = _as_int(state.get("tool_call_count"), 0)
    max_tool_calls = max(1, _as_int(state.get("max_tool_calls"), 6))
    current_step_index = _as_int(state.get("current_step_index"), 0)
    plan_steps = list(state.get("current_plan") or state.get("plan_steps") or [])
    has_more_plan_steps = current_step_index < len(plan_steps)
    tool_result = dict(state.get("tool_result") or {})
    tool_ok = bool(tool_result.get("ok", True))

    # 成功直接收口到 finish。
    if analysis_status == "SUCCESS":
        state["route"] = "finish"
        return dict(state)

    # 工具预算硬限制：达到上限直接 fallback，避免死循环。
    if tool_call_count >= max_tool_calls:
        state["route"] = "fallback"
        return dict(state)

    # NEED_RETRY：优先继续执行计划，其次重规划，预算耗尽时 fallback。
    if analysis_status == "NEED_RETRY":
        if not tool_ok and retry_count < max_retry:
            state["retry_count"] = retry_count + 1
            state["route"] = "executor"
            return dict(state)
        if has_more_plan_steps:
            state["route"] = "executor"
            return dict(state)
        if replan_count < max_replan:
            state["replan_count"] = replan_count + 1
            state["current_step_index"] = 0
            state["route"] = "planner"
            return dict(state)
        state["route"] = "fallback"
        return dict(state)

    # NEED_REPLAN：优先重规划；预算耗尽时尝试继续步骤，否则 fallback。
    if analysis_status == "NEED_REPLAN":
        if replan_count < max_replan:
            state["replan_count"] = replan_count + 1
            state["current_step_index"] = 0
            state["route"] = "planner"
            return dict(state)
        if has_more_plan_steps:
            state["route"] = "executor"
            return dict(state)
        state["route"] = "fallback"
        return dict(state)

    # FAIL 或未知状态：按“可重试 -> 可重规划 -> fallback”的顺序兜底。
    if retry_count >= max_retry and replan_count >= max_replan:
        state["route"] = "fallback"
        return dict(state)
    if not tool_ok and retry_count < max_retry:
        state["retry_count"] = retry_count + 1
        state["route"] = "executor"
        return dict(state)
    if replan_count < max_replan:
        state["replan_count"] = replan_count + 1
        state["current_step_index"] = 0
        state["route"] = "planner"
        return dict(state)

    state["analysis"] = {
        **dict(state.get("analysis") or {}),
        "reply": _FALLBACK_MESSAGE,
    }
    state["final_answer"] = _FALLBACK_MESSAGE
    state["route"] = "fallback"
    return dict(state)
