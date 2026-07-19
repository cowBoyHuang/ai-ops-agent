"""Knowledge domain executor."""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.domain_executors.common import execute_domain_goal, find_goal

_ALLOWED_TOOLS = ["rag_parent_chunk_query", "knowledge_lookup"]


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    investigation = dict(state.get("investigation") or {})
    pending = dict(investigation.get("pending_execution") or {})
    result = execute_domain_goal(
        executor="KnowledgeExecutor",
        prompt_name="knowledge_executor_react_system_prompt.txt",
        question=str(state.get("question") or ""),
        current_goal=find_goal(investigation),
        allowed_tools=_ALLOWED_TOOLS,
        existing_evidence=[dict(item or {}) for item in list(investigation.get("evidence") or [])],
        structured_context=dict(state.get("structured_context") or {}),
        attempt=int(pending.get("attempt") or 1),
    )
    investigation["last_executor_result"] = result
    state["investigation"] = investigation
    state["route"] = "plan_controller"
    return dict(state)


__all__ = ["run", "execute_domain_goal"]
