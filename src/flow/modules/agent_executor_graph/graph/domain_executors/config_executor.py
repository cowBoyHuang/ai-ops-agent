"""Config domain executor placeholder."""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.domain_executors.common import find_goal


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    investigation = dict(state.get("investigation") or {})
    pending = dict(investigation.get("pending_execution") or {})
    goal = find_goal(investigation)
    goal_id = str(goal.get("id") or investigation.get("current_goal_id") or "").strip()
    attempt = int(pending.get("attempt") or 1)
    investigation["last_executor_result"] = {
        "executor": "ConfigExecutor",
        "result_id": f"exec_{goal_id}_{attempt}",
        "goal_id": goal_id,
        "goal_complete": False,
        "status": "unsupported",
        "summary": "",
        "facts": {},
        "evidence": [],
        "artifacts": [],
        "confidence": 0.0,
        "error": "capability_not_supported",
    }
    state["investigation"] = investigation
    state["route"] = "plan_controller"
    return dict(state)


__all__ = ["run"]
