"""Agent executor subgraph entry."""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_subgraph.subgraph.build_langgraph_subgraph import build_langgraph_subgraph


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    if state.get("pipeline_stop"):
        return state
    chain = build_langgraph_subgraph()
    return chain.invoke(state)

