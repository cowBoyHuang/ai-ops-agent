from __future__ import annotations

from typing import Any

import pytest

from flow.modules.agent_executor_graph.agent_executor_graph import run as agent_executor_run
from flow.modules.agent_executor_graph.build_langgraph_graph import _finish_node
import flow.modules.agent_executor_graph.build_langgraph_graph as graph_builder


def _build_payload(message: str, **kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "chat_id": "chat_test_graph",
        "user_id": "u_test_graph",
        "message": message,
        "structured_context": {"question": message},
    }
    base.update(kwargs)
    return base


def test_agent_graph_ops_analysis_new_chain_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _intent_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["intent_type"] = "OPS_ANALYSIS"
        state["route"] = "query_rewrite"
        return state

    def _query_rewrite_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["query_rewrite"] = {"normalized_query": state.get("question") or ""}
        state["route"] = "knowledge_retrieve"
        return state

    def _knowledge_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["knowledge_context"] = {"domain_docs": [], "case_docs": [], "code_docs": []}
        state["route"] = "planner" if not dict(state.get("investigation") or {}).get("plan") else "plan_controller"
        return state

    def _planner_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["investigation"] = {
            "plan": {
                "plan_id": "plan_001",
                "hypothesis": "MQ异常",
                "goals": [{"id": "g1", "goal": "确认MQ发送状态", "required": True, "required_capability": "runtime_evidence"}],
            },
            "goal_status": {"g1": "pending"},
            "current_goal_id": "g1",
        }
        state["route"] = "plan_controller"
        return state

    def _plan_controller_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["route"] = "summary"
        return state

    def _summary_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["root_cause"] = "MQ发送状态异常"
        state["solution"] = "检查MQ发送链路"
        state["route"] = "finish"
        return state

    monkeypatch.setattr(graph_builder, "intent_decide_run", _intent_run)
    monkeypatch.setattr(graph_builder, "query_rewrite_run", _query_rewrite_run)
    monkeypatch.setattr(graph_builder, "knowledge_retrieve_run", _knowledge_run)
    monkeypatch.setattr(graph_builder, "planner_run", _planner_run)
    monkeypatch.setattr(graph_builder, "plan_controller_run", _plan_controller_run)
    monkeypatch.setattr(graph_builder, "summary_run", _summary_run)
    graph_builder.build_langgraph_graph.cache_clear()

    state = agent_executor_run(_build_payload("订单12345创建失败 timeout traceId=abc123"))
    assert state["status"] == "finished"
    assert state["intent_type"] == "OPS_ANALYSIS"
    assert state["route"] == "finish"


def test_agent_graph_uses_plan_controller_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    def _intent_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["intent_type"] = "OPS_ANALYSIS"
        state["route"] = "query_rewrite"
        return state

    def _query_rewrite_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["query_rewrite"] = {"normalized_query": state.get("question") or ""}
        state["route"] = "knowledge_retrieve"
        return state

    def _knowledge_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["knowledge_context"] = {"domain_docs": [], "case_docs": [], "code_docs": []}
        state["route"] = "planner" if not dict(state.get("investigation") or {}).get("plan") else "plan_controller"
        return state

    calls: list[str] = []

    def _planner_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        calls.append("planner")
        state["investigation"] = {
            **dict(state.get("investigation") or {}),
            "plan": {
                "plan_id": "plan_001",
                "hypothesis": "MQ异常",
                "goals": [{"id": "g1", "goal": "确认消费状态", "required": True, "required_capability": "runtime_evidence"}],
            },
            "goal_status": {"g1": "pending"},
            "current_goal_id": "g1",
        }
        state["route"] = "plan_controller"
        return state

    def _plan_controller_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        calls.append("plan_controller")
        state["route"] = "summary"
        return state

    def _summary_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        calls.append("summary")
        state["root_cause"] = "MQ异常"
        state["solution"] = "检查MQ"
        state["route"] = "finish"
        return state

    monkeypatch.setattr(graph_builder, "intent_decide_run", _intent_run)
    monkeypatch.setattr(graph_builder, "query_rewrite_run", _query_rewrite_run)
    monkeypatch.setattr(graph_builder, "knowledge_retrieve_run", _knowledge_run)
    monkeypatch.setattr(graph_builder, "planner_run", _planner_run)
    monkeypatch.setattr(graph_builder, "plan_controller_run", _plan_controller_run)
    monkeypatch.setattr(graph_builder, "summary_run", _summary_run)
    graph_builder.build_langgraph_graph.cache_clear()

    state = agent_executor_run(
        _build_payload(
            "订单12345创建失败",
            max_replan=2,
        )
    )
    assert state["status"] == "finished"
    assert calls == ["planner", "plan_controller", "summary"]


def test_agent_graph_unknown_intent_direct_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _intent_run(payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload)
        state["intent_type"] = "UNKNOWN_INTENT"
        state["route"] = "fallback"
        return state

    monkeypatch.setattr(graph_builder, "intent_decide_run", _intent_run)
    graph_builder.build_langgraph_graph.cache_clear()

    state = agent_executor_run(_build_payload("这是什么问题"))
    assert state["status"] == "degraded"
    assert state["route"] == "fallback"


def test_finish_node_uses_business_template_for_business_consult() -> None:
    state = _finish_node(
        {
            "chat_id": "chat_business_1",
            "intent_type": "SYSTEM_LOGIC_CONSULT",
            "root_cause": "这段内容不应该作为根因包装输出",
            "solution": "业务结论：当前问题属于订单同步与支付校验流程。",
            "analysis": {"reply": "业务结论：当前问题属于订单同步与支付校验流程。"},
        }
    )
    message = str(dict(state.get("response") or {}).get("message") or "")
    assert "问题根因：" not in message
    assert "建议：" not in message
    assert "业务结论：" in message


def test_finish_node_returns_conclusion_without_echoing_question() -> None:
    state = _finish_node(
        {
            "chat_id": "chat_ops_1",
            "question": "为什么这个traceId生单失败？",
            "intent_type": "OPS_ANALYSIS",
            "root_cause": "总单汇总阶段返回subErrorCode=39",
            "solution": "定位总单汇总模块并补充失败分支日志",
            "analysis": {},
        }
    )
    message = str(dict(state.get("response") or {}).get("message") or "")
    assert "用户问题：" not in message
    assert "为什么这个traceId生单失败？" not in message
    assert "问题根因：" in message
