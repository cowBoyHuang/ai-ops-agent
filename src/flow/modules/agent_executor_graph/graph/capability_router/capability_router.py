"""Capability Router: map investigation capabilities to domain executors."""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_CAPABILITY_MAP: dict[str, tuple[str, list[str]]] = {
    "runtime_evidence": (
        "LogExecutor",
        ["queryLog", "dependency_log_query", "getFlightCreateOrderResult", "getCreateOrderResult"],
    ),
    "code_analysis": (
        "CodeExecutor",
        ["searchMethod", "locateCode", "analyzeCodeFromLogs", "analyzeCodeForBusinessConsult"],
    ),
    "business_validation": (
        "KnowledgeExecutor",
        ["rag_parent_chunk_query", "knowledge_lookup"],
    ),
    "config_analysis": (
        "ConfigExecutor",
        [],
    ),
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _find_goal(plan: dict[str, Any], goal_id: str) -> dict[str, Any]:
    for item in list(plan.get("goals") or []):
        row = dict(item or {})
        if str(row.get("id") or "").strip() == goal_id:
            return row
    return {}


def _next_result_id(investigation: dict[str, Any], prefix: str, goal_id: str) -> str:
    consumed = list(investigation.get("consumed_result_ids") or [])
    count = sum(1 for item in consumed if str(item).startswith(f"{prefix}_{goal_id}_")) + 1
    return f"{prefix}_{goal_id}_{count}"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    investigation = dict(state.get("investigation") or {})
    plan = dict(investigation.get("plan") or {})
    goal_id = str(investigation.get("current_goal_id") or "").strip()
    goal = _find_goal(plan, goal_id)
    capability = str(goal.get("required_capability") or "").strip()
    result_id = _next_result_id(investigation, "route", goal_id or "unknown")

    if capability not in _CAPABILITY_MAP:
        route_result = {
            "executor": "",
            "result_id": result_id,
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
    else:
        executor, allowed_tools = _CAPABILITY_MAP[capability]
        route_result = {
            "result_id": result_id,
            "goal_id": goal_id,
            "required_capability": capability,
            "executor": executor,
            "allowed_tools": allowed_tools,
            "ok": True,
            "error": "",
        }

    events = [dict(item or {}) for item in list(investigation.get("events") or [])]
    events.append(
        {
            "type": "router",
            "message": "capability routed",
            "payload": {"goal_id": goal_id, "capability": capability, "result_id": result_id},
        }
    )
    investigation["events"] = events
    investigation["last_route_result"] = route_result
    state["investigation"] = investigation
    state["route"] = "plan_controller"
    return dict(state)


__all__ = ["run"]
