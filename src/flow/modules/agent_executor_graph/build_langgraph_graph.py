"""LangGraph 单主图构建器。

业务目标：
- 定义固定主链路：state_build -> intent -> rag -> planner -> tool -> merge -> analysis -> validate -> retry_router。
- 在 retry_router 做条件分流，形成 Retry/Replan/Finish/Fallback 四类出口。
- 保证图最终一定进入 END，避免无限循环。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.analysis_execute.analysis_execute import run as analysis_execute_run
from flow.modules.agent_executor_graph.graph.evidence_merge.evidence_merge import run as evidence_merge_run
from flow.modules.agent_executor_graph.graph.fixed_flow_execute.fixed_flow_execute import run as fixed_flow_execute_run
from flow.modules.agent_executor_graph.graph.intent_decide.intent_decide import run as intent_decide_run
from flow.modules.agent_executor_graph.graph.planner.planner import run as planner_run
from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import run as rag_retrieve_run
from flow.modules.agent_executor_graph.graph.result_validate.result_validate import run as result_validate_run
from flow.modules.agent_executor_graph.graph.retry_router.retry_router import run as retry_router_run
from flow.modules.agent_executor_graph.graph.state_build.state_build import run as state_build_run
from flow.modules.agent_executor_graph.graph.tool_execute.tool_execute import run as tool_execute_run
from flow.modules.agent_executor_graph.graph.tool_router.tool_router import run as tool_router_run

_FALLBACK_MESSAGE = "暂未能自动定位问题，请联系人工排查。"


class _GraphRuntimeState(TypedDict, total=False):
    """LangGraph 运行时状态模式（避免未声明键在运行时被过滤）。"""

    message: str
    context: dict[str, Any]
    query: str
    chat_id: str
    chatId: str
    user_id: str
    userId: str
    error: str
    error_code: str
    result: bool

    question: str
    normalized_question: str
    conversation_context: list[str]
    intent_type: str
    intent_recognition: dict[str, Any]
    intent_history_prompt: str
    intent_retry_results: list[dict[str, Any]]
    intent_retry_count: int
    structured_context: dict[str, Any]
    order_id: str
    request_id: str

    rag_docs: list[dict[str, Any]]
    rag_scores: list[float]

    plan_steps: list[dict[str, Any]]
    current_step_index: int

    tool_name: str
    tool_params: dict[str, Any]
    tool_result: dict[str, Any]
    tool_history: list[dict[str, Any]]
    tool_call_count: int
    max_tool_calls: int

    merged_evidence: dict[str, Any]

    analysis: dict[str, Any]
    root_cause: str
    confidence: float
    solution: str
    analysis_status: str

    retry_count: int
    max_retry: int
    max_retries: int
    replan_count: int
    max_replan: int

    final_answer: str
    route: str
    status: str
    response: dict[str, Any]
    planner_reset: bool
    fixed_flow_hit: bool
    simulate_tool_timeout_once: bool
    _simulate_tool_timeout_used: bool


def _finish_node(payload: AgentState) -> AgentState:
    """成功收口节点。

    入参：
    - payload: 已通过 result_validate 判定为 SUCCESS 的状态。

    返参：
    - AgentState: 写入 final_answer/response/status=finished。
    """
    state: AgentState = dict(payload)
    analysis = dict(state.get("analysis") or {})
    root_cause = str(state.get("root_cause") or analysis.get("root_cause") or "").strip()
    solution = str(state.get("solution") or analysis.get("reply") or "").strip()

    # 对外文案优先包含“根因 + 建议”，便于排障同学直接使用。
    if root_cause and solution:
        final_answer = f"问题根因：{root_cause}。建议：{solution}"
    elif solution:
        final_answer = solution
    elif root_cause:
        final_answer = f"问题根因：{root_cause}"
    else:
        final_answer = "分析完成"

    state["final_answer"] = final_answer
    state["status"] = "finished"
    state["route"] = "finish"
    state["analysis"] = {
        **analysis,
        "reply": str(analysis.get("reply") or final_answer),
    }
    state["response"] = {
        "chatId": state.get("chat_id") or state.get("chatId") or "",
        "status": "finished",
        "message": final_answer,
    }
    return state


def _fallback_node(payload: AgentState) -> AgentState:
    """失败兜底节点。

    触发场景：
    - 超过 retry/replan/tool 调用预算。
    - 路由异常或状态异常。
    """
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
        "chatId": state.get("chat_id") or state.get("chatId") or "",
        "status": "degraded",
        "message": _FALLBACK_MESSAGE,
    }
    return state


def _route_after_retry_router(state: dict[str, Any]) -> str:
    """根据 retry_router 写入的 route 决定下一跳。

    仅允许白名单节点，非法值统一降级到 fallback，避免图跑飞。
    """
    route = str(state.get("route") or "fallback")
    supported = {"tool_router", "tool_execute", "planner", "finish", "fallback"}
    if route in supported:
        return route
    return "fallback"


def _route_after_intent_decide(state: dict[str, Any]) -> str:
    route = str(state.get("route") or "rag_retrieve")
    if route == "intent_decide":
        return "intent_decide"
    if route == "fallback":
        return "fallback"
    if route == "fixed_flow_execute":
        return "fixed_flow_execute"
    return "rag_retrieve"


@lru_cache(maxsize=1)
def build_langgraph_graph() -> Runnable:
    """构建并编译主图。

    返回：
    - Runnable: 可直接 `invoke(state)` 的执行图对象。
    """

    # 运行时状态模式使用内部完整字段集合，避免键被过滤影响路由。
    graph = StateGraph(_GraphRuntimeState)
    graph.add_node("state_build", state_build_run)
    graph.add_node("intent_decide", intent_decide_run)
    graph.add_node("fixed_flow_execute", fixed_flow_execute_run)
    graph.add_node("rag_retrieve", rag_retrieve_run)
    graph.add_node("planner", planner_run)
    graph.add_node("tool_router", tool_router_run)
    graph.add_node("tool_execute", tool_execute_run)
    graph.add_node("evidence_merge", evidence_merge_run)
    graph.add_node("analysis_execute", analysis_execute_run)
    graph.add_node("result_validate", result_validate_run)
    graph.add_node("retry_router", retry_router_run)
    graph.add_node("finish", _finish_node)
    graph.add_node("fallback", _fallback_node)

    # 主路径：执行一轮“检索-规划-工具-分析-验证”。
    graph.add_edge(START, "state_build")
    graph.add_edge("state_build", "intent_decide")
    graph.add_conditional_edges(
        "intent_decide",
        _route_after_intent_decide,
        {
            "intent_decide": "intent_decide",
            "rag_retrieve": "rag_retrieve",
            "fixed_flow_execute": "fixed_flow_execute",
            "fallback": "fallback",
        },
    )
    graph.add_edge("fixed_flow_execute", "finish")
    graph.add_edge("rag_retrieve", "planner")
    graph.add_edge("planner", "tool_router")
    graph.add_edge("tool_router", "tool_execute")
    graph.add_edge("tool_execute", "evidence_merge")
    graph.add_edge("evidence_merge", "analysis_execute")
    graph.add_edge("analysis_execute", "result_validate")
    graph.add_edge("result_validate", "retry_router")

    # 条件分流：retry_router 决定进入 retry/replan/finish/fallback。
    graph.add_conditional_edges(
        "retry_router",
        _route_after_retry_router,
        {
            "tool_router": "tool_router",
            "tool_execute": "tool_execute",
            "planner": "planner",
            "finish": "finish",
            "fallback": "fallback",
        },
    )

    # 统一终止出口。
    graph.add_edge("finish", END)
    graph.add_edge("fallback", END)
    return graph.compile()
