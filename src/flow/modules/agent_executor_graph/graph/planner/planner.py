"""规划节点（Planner）。

业务职责：
- 根据意图和问题生成可执行步骤 plan_steps。
- Replan 时切换到更激进的排障步骤模板。
- 必要时重置 current_step_index，确保从新计划首步执行。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.plan_step import PlanStep


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
    - AgentState: 写入 plan_steps/current_step_index，并路由到 tool_router。
    """
    state: AgentState = dict(payload)
    question = str(state.get("question") or "").strip()
    intent_type = str(state.get("intent_type") or "UNKNOWN")
    replan_count = _as_int(state.get("replan_count"), 0)

    # 正常首轮计划：按问题类型给出结构化步骤模板。
    if intent_type == "OPS_ANALYSIS":
        plan_steps: list[PlanStep] = [
            {
                "action_type": "tool_call",
                "tool_name": "log_query",
                "params": {},
            },
            {
                "action_type": "tool_call",
                "tool_name": "dependency_log_query",
                "params": {},
            },
            {
                "action_type": "merge_evidence",
                "tool_name": None,
                "params": {},
            },
        ]
    elif intent_type == "GENERAL_QA":
        plan_steps = [
            {
                "action_type": "tool_call",
                "tool_name": "knowledge_lookup",
                "params": {},
            },
            {
                "action_type": "merge_evidence",
                "tool_name": None,
                "params": {},
            },
        ]
    else:
        plan_steps = [
            {
                "action_type": "tool_call",
                "tool_name": "knowledge_lookup",
                "params": {},
            },
            {
                "action_type": "tool_call",
                "tool_name": "log_query",
                "params": {},
            },
            {
                "action_type": "merge_evidence",
                "tool_name": None,
                "params": {},
            },
        ]

    # Replan 时直接替换为“最新日志+依赖日志+分析”路径。
    if replan_count > 0:
        plan_steps = [
            {
                "action_type": "tool_call",
                "tool_name": "log_query",
                "params": {},
            },
            {
                "action_type": "tool_call",
                "tool_name": "dependency_log_query",
                "params": {},
            },
            {
                "action_type": "merge_evidence",
                "tool_name": None,
                "params": {},
            },
        ]

    # 首次生成计划时，从第 0 步开始。
    if not state.get("plan_steps"):
        state["current_step_index"] = 0

    state["plan_steps"] = plan_steps
    _ = question
    state["route"] = "tool_router"
    return dict(state)
