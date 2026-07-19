"""Plan Controller for complete-plan investigation execution."""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_RETRYABLE_ERRORS = {"tool_error", "empty_evidence", "invalid_executor_result"}
_DEFAULT_REPLAN_TRIGGERS = {
    "goal_unexecutable",
    "evidence_conflict",
    "missing_required_context",
    "capability_not_supported",
}
_EXECUTOR_ROUTE = {
    "LogExecutor": "log_executor",
    "CodeExecutor": "code_executor",
    "KnowledgeExecutor": "knowledge_executor",
    "ConfigExecutor": "config_executor",
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _goals(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item or {}) for item in list(plan.get("goals") or [])]


def _goal_by_id(plan: dict[str, Any], goal_id: str) -> dict[str, Any]:
    for item in _goals(plan):
        if str(item.get("id") or "").strip() == goal_id:
            return item
    return {}


def _required_goal_ids(plan: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id") or "").strip()
        for item in _goals(plan)
        if bool(item.get("required", True)) and str(item.get("id") or "").strip()
    ]


def _ensure_runtime(investigation: dict[str, Any]) -> dict[str, Any]:
    plan = dict(investigation.get("plan") or {})
    goals = _goals(plan)
    status = dict(investigation.get("goal_status") or {})
    for item in goals:
        goal_id = str(item.get("id") or "").strip()
        if goal_id:
            status.setdefault(goal_id, "pending")
    current = str(investigation.get("current_goal_id") or "").strip()
    if not current and goals:
        current = str(goals[0].get("id") or "").strip()
    investigation["current_goal_id"] = current
    investigation["goal_status"] = status
    investigation.setdefault("evidence", [])
    investigation.setdefault("events", [])
    investigation.setdefault("pending_execution", {})
    investigation.setdefault("last_route_result", {})
    investigation.setdefault("last_executor_result", {})
    investigation.setdefault("consumed_result_ids", [])
    investigation.setdefault("retry_counts_by_goal", {})
    investigation.setdefault("max_retries_per_goal", 2)
    investigation.setdefault("replan_count", 0)
    investigation.setdefault("max_replans", 1)
    investigation.setdefault("failure_reason", "")
    return investigation


def _is_consumed(investigation: dict[str, Any], result: dict[str, Any]) -> bool:
    result_id = str(result.get("result_id") or "").strip()
    return bool(result_id and result_id in set(str(item) for item in list(investigation.get("consumed_result_ids") or [])))


def _mark_consumed(investigation: dict[str, Any], result: dict[str, Any]) -> None:
    result_id = str(result.get("result_id") or "").strip()
    if not result_id:
        return
    rows = [str(item) for item in list(investigation.get("consumed_result_ids") or [])]
    if result_id not in rows:
        rows.append(result_id)
    investigation["consumed_result_ids"] = rows


def _append_event(
    investigation: dict[str, Any],
    message: str,
    payload: dict[str, Any],
    *,
    event_type: str = "controller",
) -> None:
    events = [dict(item or {}) for item in list(investigation.get("events") or [])]
    events.append({"type": event_type, "message": message, "payload": payload})
    investigation["events"] = events


def _persist_evidence(investigation: dict[str, Any], goal: dict[str, Any], result: dict[str, Any]) -> None:
    rows = [dict(item or {}) for item in list(investigation.get("evidence") or [])]
    rows.append(
        {
            "goal_id": str(result.get("goal_id") or ""),
            "capability": str(goal.get("required_capability") or ""),
            "executor": str(result.get("executor") or ""),
            "summary": str(result.get("summary") or ""),
            "facts": dict(result.get("facts") or {}),
            "evidence": list(result.get("evidence") or []),
            "artifacts": list(result.get("artifacts") or []),
            "confidence": result.get("confidence", 0.0),
            "status": str(result.get("status") or ""),
            "error": str(result.get("error") or ""),
        }
    )
    investigation["evidence"] = rows


def _sync_legacy_evidence_graph(state: AgentState, investigation: dict[str, Any], goal: dict[str, Any], result: dict[str, Any]) -> None:
    execution = dict(state.get("execution") or {})
    graph = dict(execution.get("evidence_graph") or {})
    plan = dict(investigation.get("plan") or {})
    graph.setdefault("hypothesis", str(plan.get("hypothesis") or ""))
    rows = [dict(item or {}) for item in list(graph.get("evidence") or [])]
    rows.append(
        {
            "objective": str(goal.get("goal") or ""),
            "skill": str(result.get("executor") or ""),
            "observation": str(result.get("summary") or ""),
            "summary": str(result.get("summary") or ""),
            "conclusion": "supports" if str(result.get("status") or "") == "succeeded" else "insufficient",
            "raw_result": {
                "ok": str(result.get("status") or "") == "succeeded",
                "evidence": [str(item) for item in list(result.get("evidence") or [])],
                "effective_info": {"summary": str(result.get("summary") or ""), "facts": dict(result.get("facts") or {})},
            },
        }
    )
    graph["evidence"] = rows
    graph["supported"] = None
    execution["evidence_graph"] = graph
    state["execution"] = execution


def _dependencies_satisfied(goal: dict[str, Any], status: dict[str, Any]) -> bool:
    for dep in list(goal.get("depends_on") or []):
        if str(status.get(str(dep)) or "") != "succeeded":
            return False
    return True


def _next_runnable_goal(plan: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    for item in sorted(_goals(plan), key=lambda row: int(row.get("priority") or 0)):
        goal_id = str(item.get("id") or "").strip()
        if not goal_id or str(status.get(goal_id) or "pending") != "pending":
            continue
        if _dependencies_satisfied(item, status):
            return item
    return {}


def _all_required_succeeded(plan: dict[str, Any], status: dict[str, Any]) -> bool:
    required = _required_goal_ids(plan)
    return bool(required) and all(str(status.get(goal_id) or "") == "succeeded" for goal_id in required)


def _consume_supported_route(state: AgentState, investigation: dict[str, Any], route_result: dict[str, Any]) -> bool:
    if _is_consumed(investigation, route_result) or route_result.get("ok") is not True:
        return False
    goal_id = str(route_result.get("goal_id") or "").strip()
    executor = str(route_result.get("executor") or "").strip()
    _mark_consumed(investigation, route_result)
    attempt = _as_int(dict(investigation.get("retry_counts_by_goal") or {}).get(goal_id), 0) + 1
    investigation["pending_execution"] = {"goal_id": goal_id, "executor": executor, "attempt": attempt}
    investigation["last_route_result"] = {}
    status = dict(investigation.get("goal_status") or {})
    status[goal_id] = "running"
    investigation["goal_status"] = status
    _append_event(investigation, "route consumed", {"goal_id": goal_id, "executor": executor})
    state["route"] = _EXECUTOR_ROUTE.get(executor, "fallback")
    return True


def _consume_goal_result(state: AgentState, investigation: dict[str, Any], result: dict[str, Any]) -> bool:
    if not result or _is_consumed(investigation, result):
        return False
    plan = dict(investigation.get("plan") or {})
    goal_id = str(result.get("goal_id") or investigation.get("current_goal_id") or "").strip()
    goal = _goal_by_id(plan, goal_id)
    _mark_consumed(investigation, result)
    status = dict(investigation.get("goal_status") or {})
    result_status = str(result.get("status") or "failed").strip()
    error = str(result.get("error") or "").strip()
    goal_required = bool(goal.get("required", True))

    if result_status == "succeeded" or bool(result.get("goal_complete")):
        status[goal_id] = "succeeded"
        _persist_evidence(investigation, goal, result)
        _sync_legacy_evidence_graph(state, investigation, goal, result)
    elif result_status == "unsupported" and not goal_required:
        status[goal_id] = "skipped"
    else:
        status[goal_id] = "failed"
        if result.get("summary") or result.get("evidence") or result.get("facts"):
            _persist_evidence(investigation, goal, result)

    investigation["goal_status"] = status
    investigation["pending_execution"] = {}
    investigation["last_executor_result"] = {}
    if dict(investigation.get("last_route_result") or {}).get("status") == "unsupported":
        investigation["last_route_result"] = {}
    investigation["failure_reason"] = error
    _append_event(investigation, "goal result consumed", {"goal_id": goal_id, "status": status.get(goal_id), "error": error})
    return True


def _can_retry(investigation: dict[str, Any], goal_id: str) -> bool:
    retries = dict(investigation.get("retry_counts_by_goal") or {})
    return _as_int(retries.get(goal_id), 0) < _as_int(investigation.get("max_retries_per_goal"), 2)


def _can_retry_error(investigation: dict[str, Any], goal_id: str, error: str) -> bool:
    retries = dict(investigation.get("retry_counts_by_goal") or {})
    if error == "invalid_executor_result":
        return _as_int(retries.get(goal_id), 0) < 1
    return error in _RETRYABLE_ERRORS and _can_retry(investigation, goal_id)


def _increment_retry(investigation: dict[str, Any], goal_id: str) -> None:
    retries = dict(investigation.get("retry_counts_by_goal") or {})
    retries[goal_id] = _as_int(retries.get(goal_id), 0) + 1
    investigation["retry_counts_by_goal"] = retries


def _can_replan(investigation: dict[str, Any], error: str) -> bool:
    plan = dict(investigation.get("plan") or {})
    triggers = [str(item).strip() for item in list(plan.get("replan_triggers") or []) if str(item).strip()]
    trigger_set = set(triggers or _DEFAULT_REPLAN_TRIGGERS)
    return error in trigger_set and _as_int(investigation.get("replan_count"), 0) < _as_int(investigation.get("max_replans"), 1)


def _prepare_replan(state: AgentState, investigation: dict[str, Any], *, reason: str, failed_goal_id: str) -> None:
    plan = dict(investigation.get("plan") or {})
    goal_status = dict(investigation.get("goal_status") or {})
    evidence = [dict(item or {}) for item in list(investigation.get("evidence") or [])]
    failed_goal = _goal_by_id(plan, failed_goal_id)
    replan_index = _as_int(investigation.get("replan_count"), 0) + 1
    _append_event(
        investigation,
        "replan accepted",
        {
            "replan_index": replan_index,
            "reason": str(reason or ""),
            "failed_goal_id": failed_goal_id,
            "failed_goal": failed_goal,
            "previous_plan": plan,
            "goal_status": goal_status,
            "evidence_count": len(evidence),
        },
        event_type="replan",
    )
    investigation["replan_count"] = _as_int(investigation.get("replan_count"), 0) + 1
    investigation["pending_execution"] = {}
    investigation["last_route_result"] = {}
    investigation["last_executor_result"] = {}
    investigation["retry_counts_by_goal"] = {}
    investigation["failure_reason"] = str(reason or "")
    state["replan_reason"] = str(reason or "")
    state["replan_context"] = {
        "failure_reason": str(reason or ""),
        "failed_goal_id": failed_goal_id,
        "failed_goal": failed_goal,
        "previous_plan": plan,
        "goal_status": goal_status,
        "evidence": evidence,
    }


def _mark_blocked_pending_goals(investigation: dict[str, Any], reason: str) -> None:
    plan = dict(investigation.get("plan") or {})
    status = dict(investigation.get("goal_status") or {})
    blocked: list[str] = []
    for goal in _goals(plan):
        goal_id = str(goal.get("id") or "").strip()
        if not goal_id or str(status.get(goal_id) or "pending") != "pending":
            continue
        if _dependencies_satisfied(goal, status):
            continue
        status[goal_id] = "failed" if bool(goal.get("required", True)) else "skipped"
        blocked.append(goal_id)
    if blocked:
        investigation["goal_status"] = status
        _append_event(investigation, "blocked goals marked", {"goal_ids": blocked, "reason": reason})


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    investigation = _ensure_runtime(dict(state.get("investigation") or {}))
    if not investigation.get("plan"):
        state["route"] = "planner"
        state["investigation"] = investigation
        return dict(state)

    route_result = dict(investigation.get("last_route_result") or {})
    if route_result and route_result.get("goal_id") == investigation.get("current_goal_id"):
        if route_result.get("ok") is True and _consume_supported_route(state, investigation, route_result):
            state["investigation"] = investigation
            return dict(state)
        if route_result.get("status") == "unsupported":
            _consume_goal_result(state, investigation, route_result)

    executor_result = dict(investigation.get("last_executor_result") or {})
    if executor_result and executor_result.get("goal_id") == investigation.get("current_goal_id"):
        _consume_goal_result(state, investigation, executor_result)

    plan = dict(investigation.get("plan") or {})
    status = dict(investigation.get("goal_status") or {})
    current_goal_id = str(investigation.get("current_goal_id") or "").strip()
    current_status = str(status.get(current_goal_id) or "pending")
    error = str(investigation.get("failure_reason") or "").strip()

    if _all_required_succeeded(plan, status):
        state["route"] = "summary"
        state["investigation"] = investigation
        return dict(state)

    if current_status == "failed":
        goal = _goal_by_id(plan, current_goal_id)
        if _can_retry_error(investigation, current_goal_id, error):
            _increment_retry(investigation, current_goal_id)
            status[current_goal_id] = "pending"
            investigation["goal_status"] = status
            state["route"] = "capability_router"
        elif error == "capability_not_supported" and not bool(goal.get("required", True)):
            status[current_goal_id] = "skipped"
            investigation["goal_status"] = status
            state["route"] = "plan_controller"
        elif _can_replan(investigation, error):
            _prepare_replan(state, investigation, reason=error, failed_goal_id=current_goal_id)
            state["route"] = "planner"
        else:
            state["route"] = "fallback"
        state["investigation"] = investigation
        return dict(state)

    next_goal = _next_runnable_goal(plan, status)
    if next_goal:
        goal_id = str(next_goal.get("id") or "").strip()
        investigation["current_goal_id"] = goal_id
        status[goal_id] = "pending"
        investigation["goal_status"] = status
        state["route"] = "capability_router"
    else:
        _mark_blocked_pending_goals(investigation, "no_runnable_goal")
        if _as_int(investigation.get("replan_count"), 0) < _as_int(investigation.get("max_replans"), 1):
            _prepare_replan(state, investigation, reason="no_runnable_goal", failed_goal_id=current_goal_id)
            state["route"] = "planner"
        else:
            state["route"] = "fallback"
    state["investigation"] = investigation
    return dict(state)


__all__ = ["run"]
