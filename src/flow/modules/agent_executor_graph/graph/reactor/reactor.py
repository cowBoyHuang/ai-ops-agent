"""Reactor 节点：根据 observer 建议对当前计划做局部调整。"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_MAX_INSERTIONS_PER_REACT = 3


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    current_plan = [dict(item) for item in list(state.get("current_plan") or state.get("plan_steps") or [])]
    current_step_index = _as_int(state.get("current_step_index"), 0)
    needs_adjustment = bool(state.get("needs_adjustment"))
    adjustment_type = str(state.get("adjustment_type") or "")
    pending_insertions = [dict(item) for item in list(state.get("pending_insertions") or [])[:_MAX_INSERTIONS_PER_REACT]]
    proposed_changes = dict(state.get("proposed_changes") or {})

    if not needs_adjustment:
        state["route"] = "executor"
        return dict(state)

    if adjustment_type == "global":
        # 全局调整交由 planner 重新规划，此节点不处理。
        state["route"] = "planner"
        return dict(state)

    before_len = len(current_plan)
    insert_pos = max(0, min(current_step_index, len(current_plan)))
    if pending_insertions:
        current_plan[insert_pos:insert_pos] = pending_insertions

    adjustment_history = [dict(item) for item in list(state.get("adjustment_history") or [])]
    record = {
        "type": "local",
        "reason": str(proposed_changes.get("reason") or "observer_local_adjustment"),
        "insert_pos": insert_pos,
        "inserted_steps": pending_insertions,
        "before_len": before_len,
        "after_len": len(current_plan),
    }
    adjustment_history.append(record)

    state["current_plan"] = current_plan
    state["plan_steps"] = current_plan
    state["adjustment_applied"] = record
    state["adjustment_history"] = adjustment_history
    state["needs_adjustment"] = False
    state["adjustment_type"] = ""
    state["proposed_changes"] = {}
    state["pending_insertions"] = []
    state["route"] = "executor"
    return dict(state)
