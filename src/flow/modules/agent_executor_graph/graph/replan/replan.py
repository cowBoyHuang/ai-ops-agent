"""RePlan 节点。"""

from __future__ import annotations

import logging

from flow.modules.agent_executor_graph.agent_state import AgentState

_LOGGER = logging.getLogger(__name__)


def run(payload: dict[str, object]) -> dict[str, object]:
    state: AgentState = dict(payload)
    replan_context = dict(state.get("replan_context") or {})
    evaluation = dict(state.get("evaluation") or {})
    reason = str(
        replan_context.get("failure_reason")
        or state.get("replan_reason")
        or evaluation.get("reason")
        or "当前证据不支持该假设"
    ).strip()
    current_hypothesis = str(dict(state.get("plan") or {}).get("hypothesis") or "").strip()

    rejected = [str(item).strip() for item in list(state.get("rejected_hypothesis") or []) if str(item).strip()]
    if current_hypothesis and current_hypothesis not in rejected:
        rejected.append(current_hypothesis)

    state["replan_reason"] = reason
    state["rejected_hypothesis"] = rejected
    state["replan_count"] = int(state.get("replan_count") or 0) + 1
    state["route"] = "planner"
    _LOGGER.info(
        "replan triggered count=%d current_hypothesis=%s rejected_count=%d reason=%s route=planner",
        int(state.get("replan_count") or 0),
        current_hypothesis,
        len(rejected),
        reason,
    )
    return dict(state)
