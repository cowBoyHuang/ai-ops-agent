"""规划节点（Planner）。

业务职责：
- 根据意图和问题生成可执行步骤 plan_steps。
- Replan 时切换到更激进的排障步骤模板。
- 必要时重置 current_step_index，确保从新计划首步执行。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.graph.agent_state import AgentState


def _as_int(value: Any, default: int) -> int:
    """把输入转换成 int，异常时使用默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """生成或重生成排障计划。

    入参：
    - payload: AgentState，至少包含 question、intent_type、replan_count。

    返参：
    - AgentState: 写入 plan_steps/current_step_index/plan，并路由到 tool_router。
    """
    state: AgentState = dict(payload)
    question = str(state.get("normalized_question") or state.get("question") or state.get("message") or "").strip()
    intent_type = str(state.get("intent_type") or "UNKNOWN")
    replan_count = _as_int(state.get("replan_count"), 0)
    planner_reset = bool(state.pop("planner_reset", False))

    # 正常首轮计划：按问题类型给出不同步骤模板。
    if intent_type == "OPS_ANALYSIS":
        plan_steps = [
            "query_order_logs",
            "query_service_logs",
            "analyze_error",
        ]
    elif intent_type == "GENERAL_QA":
        plan_steps = [
            "collect_general_context",
            "answer_question",
        ]
    else:
        plan_steps = [
            "collect_missing_context",
            "query_order_logs",
            "analyze_error",
        ]

    # Replan 时直接替换为“最新日志+依赖日志+分析”路径。
    if replan_count > 0:
        plan_steps = [
            "query_latest_error_logs",
            "query_dependency_logs",
            "analyze_error",
        ]

    # 当明确要求重置，或首次生成计划时，从第 0 步开始。
    if planner_reset or not state.get("plan_steps"):
        state["current_step_index"] = 0

    state["plan_steps"] = plan_steps
    state["plan"] = {
        "question": question,
        "intent_type": intent_type,
        "steps": plan_steps,
        "replan_count": replan_count,
        "tool_goal": "补齐分析证据并收敛根因",
    }
    state["route"] = "tool_router"
    return dict(state)
