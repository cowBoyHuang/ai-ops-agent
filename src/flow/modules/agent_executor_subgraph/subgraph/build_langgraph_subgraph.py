"""Subgraph builder."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from flow.modules.agent_executor_subgraph.subgraph.analysis_execute.analysis_execute import run as analysis_execute_run
from flow.modules.agent_executor_subgraph.subgraph.evidence_merge.evidence_merge import run as evidence_merge_run
from flow.modules.agent_executor_subgraph.subgraph.intent_decide.intent_decide import run as intent_decide_run
from flow.modules.agent_executor_subgraph.subgraph.rag_retrieve.rag_retrieve import run as rag_retrieve_run
from flow.modules.agent_executor_subgraph.subgraph.reply_clarify.reply_clarify import run as reply_clarify_run
from flow.modules.agent_executor_subgraph.subgraph.reply_degrade.reply_degrade import run as reply_degrade_run
from flow.modules.agent_executor_subgraph.subgraph.reply_success.reply_success import run as reply_success_run
from flow.modules.agent_executor_subgraph.subgraph.result_validate.result_validate import run as result_validate_run
from flow.modules.agent_executor_subgraph.subgraph.retry_router.retry_router import run as retry_router_run


def _route(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    route = str(state.get("route") or "")
    if route == "reply_clarify":
        return reply_clarify_run(state)
    if route == "reply_degrade":
        return reply_degrade_run(state)
    if route == "rag_retrieve":
        state = rag_retrieve_run(state)
        state = evidence_merge_run(state)
        state = analysis_execute_run(state)
        state = result_validate_run(state)
        while str(state.get("route")) == "retry_router":
            state = retry_router_run(state)
            if str(state.get("route")) == "analysis_execute":
                state = analysis_execute_run(state)
                state = result_validate_run(state)
                continue
            break
        if str(state.get("route")) == "reply_success":
            return reply_success_run(state)
        return reply_degrade_run(state)
    return reply_degrade_run(state)


def build_langgraph_subgraph() -> Runnable:
    """Build runnable subgraph chain with routing."""
    return RunnableLambda(intent_decide_run) | RunnableLambda(_route)

