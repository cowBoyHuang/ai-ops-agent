"""LangGraph 主图构建器（Hypothesis Driven Loop）。"""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.capability_router.capability_router import run as capability_router_run
from flow.modules.agent_executor_graph.graph.domain_executors.code_executor import run as code_executor_run
from flow.modules.agent_executor_graph.graph.domain_executors.config_executor import run as config_executor_run
from flow.modules.agent_executor_graph.graph.domain_executors.knowledge_executor import run as domain_knowledge_executor_run
from flow.modules.agent_executor_graph.graph.domain_executors.log_executor import run as log_executor_run
from flow.modules.agent_executor_graph.graph.fixed_flow_execute.fixed_flow_execute import run as fixed_flow_execute_run
from flow.modules.agent_executor_graph.graph.intent_decide.intent_decide import run as intent_decide_run
from flow.modules.agent_executor_graph.graph.knowledge_retrieve.knowledge_retrieve import run as knowledge_retrieve_run
from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as plan_controller_run
from flow.modules.agent_executor_graph.graph.planner.planner import run as planner_run
from flow.modules.agent_executor_graph.graph.query_rewrite.query_rewrite import run as query_rewrite_run
from flow.modules.agent_executor_graph.graph.state_build.state_build import run as state_build_run
from flow.modules.agent_executor_graph.graph.summary.summary import run as summary_run

_FALLBACK_MESSAGE = "暂未能自动定位问题，请联系人工排查。"
_LOGGER = logging.getLogger(__name__)


def _finish_node(payload: AgentState) -> AgentState:
    state: AgentState = dict(payload)
    analysis = dict(state.get("analysis") or {})
    root_cause = str(state.get("root_cause") or analysis.get("root_cause") or "").strip()
    solution = str(state.get("solution") or analysis.get("reply") or "").strip()
    analysis_reply = str(analysis.get("reply") or "").strip()
    intent_type = str(state.get("intent_type") or "").strip()
    is_business_consult = intent_type == "SYSTEM_LOGIC_CONSULT"

    if is_business_consult:
        result_summary = solution or analysis_reply or (f"业务结论：{root_cause}" if root_cause else "业务分析完成")
    else:
        if root_cause and solution:
            result_summary = f"问题根因：{root_cause}。建议：{solution}"
        elif solution:
            result_summary = solution
        elif root_cause:
            result_summary = f"问题根因：{root_cause}"
        else:
            result_summary = "分析完成"

    # 最终对用户返回只输出结论，不回显用户问题本身。
    final_answer = result_summary

    state["final_answer"] = final_answer
    state["status"] = "finished"
    state["route"] = "finish"
    state["analysis"] = {
        **analysis,
        "reply": analysis_reply or result_summary,
    }
    state["response"] = {
        "chatId": state.get("chat_id") or "",
        "status": "finished",
        "message": final_answer,
    }
    return state


def _fallback_node(payload: AgentState) -> AgentState:
    state: AgentState = dict(payload)
    analysis = dict(state.get("analysis") or {})
    state["final_answer"] = _FALLBACK_MESSAGE
    state["status"] = "degraded"
    state["route"] = "fallback"
    state["analysis"] = {
        **analysis,
        "reply": str(analysis.get("reply") or _FALLBACK_MESSAGE),
    }
    state["response"] = {
        "chatId": state.get("chat_id") or "",
        "status": "degraded",
        "message": _FALLBACK_MESSAGE,
    }
    return state


def _route_after_intent_decide(state: dict[str, Any]) -> str:
    route = str(state.get("route") or "fallback")
    supported = {"intent_decide", "query_rewrite", "fixed_flow_execute", "fallback"}
    if route in supported:
        return route
    return "fallback"


def _route_after_knowledge_retrieve(state: dict[str, Any]) -> str:
    route = str(state.get("route") or "planner")
    if route in {"planner", "plan_controller"}:
        return route
    return "fallback"


def _route_after_plan_controller(state: dict[str, Any]) -> str:
    route = str(state.get("route") or "").strip()
    if route in {
        "planner",
        "capability_router",
        "log_executor",
        "code_executor",
        "knowledge_executor",
        "config_executor",
        "summary",
        "fallback",
    }:
        return route
    _LOGGER.info("route_after_plan_controller route=%s -> fallback", route)
    return "fallback"


@lru_cache(maxsize=1)
def build_langgraph_graph() -> Runnable:
    graph = StateGraph(AgentState)

    graph.add_node("state_build", state_build_run)
    graph.add_node("intent_decide", intent_decide_run)
    graph.add_node("fixed_flow_execute", fixed_flow_execute_run)
    graph.add_node("query_rewrite", query_rewrite_run)
    graph.add_node("knowledge_retrieve", knowledge_retrieve_run)
    graph.add_node("planner", planner_run)
    graph.add_node("plan_controller", plan_controller_run)
    graph.add_node("capability_router", capability_router_run)
    graph.add_node("log_executor", log_executor_run)
    graph.add_node("code_executor", code_executor_run)
    graph.add_node("knowledge_executor", domain_knowledge_executor_run)
    graph.add_node("config_executor", config_executor_run)
    graph.add_node("summary", summary_run)
    graph.add_node("finish", _finish_node)
    graph.add_node("fallback", _fallback_node)

    graph.add_edge(START, "state_build")
    graph.add_edge("state_build", "intent_decide")
    graph.add_conditional_edges(
        "intent_decide",
        _route_after_intent_decide,
        {
            "intent_decide": "intent_decide",
            "query_rewrite": "query_rewrite",
            "fixed_flow_execute": "fixed_flow_execute",
            "fallback": "fallback",
        },
    )

    graph.add_edge("fixed_flow_execute", "finish")

    graph.add_edge("query_rewrite", "knowledge_retrieve")
    graph.add_conditional_edges(
        "knowledge_retrieve",
        _route_after_knowledge_retrieve,
        {
            "planner": "planner",
            "plan_controller": "plan_controller",
            "fallback": "fallback",
        },
    )
    graph.add_edge("planner", "plan_controller")
    graph.add_conditional_edges(
        "plan_controller",
        _route_after_plan_controller,
        {
            "planner": "planner",
            "capability_router": "capability_router",
            "log_executor": "log_executor",
            "code_executor": "code_executor",
            "knowledge_executor": "knowledge_executor",
            "config_executor": "config_executor",
            "summary": "summary",
            "fallback": "fallback",
        },
    )
    graph.add_edge("capability_router", "plan_controller")
    graph.add_edge("log_executor", "plan_controller")
    graph.add_edge("code_executor", "plan_controller")
    graph.add_edge("knowledge_executor", "plan_controller")
    graph.add_edge("config_executor", "plan_controller")
    graph.add_edge("summary", "finish")

    graph.add_edge("finish", END)
    graph.add_edge("fallback", END)
    return graph.compile()
