from __future__ import annotations

import pytest

from flow.modules.agent_executor_graph.graph.knowledge_retrieve.knowledge_retrieve import run as knowledge_retrieve_run


def test_knowledge_retrieve_supports_pre_plan_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.knowledge_retrieve.knowledge_retrieve.query_parent_docs_from_rag",
        lambda **kwargs: (
            [],
            [],
            [{"path": "/tmp/domain.md", "content": "业务流程说明"}],
        ),
    )

    state = knowledge_retrieve_run(
        {
            "question": "订单失败",
            "query_rewrite": {"normalized_query": "订单失败", "keywords": ["生单失败"]},
        }
    )
    context = dict(state.get("knowledge_context") or {})
    assert "domain_docs" in context
    assert "case_docs" in context
    assert "code_docs" in context
    assert state.get("route") == "planner"


def test_knowledge_retrieve_uses_hypothesis_and_routes_to_planner_for_legacy_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.knowledge_retrieve.knowledge_retrieve.query_parent_docs_from_rag",
        lambda **kwargs: (
            [],
            [],
            [{"path": "/tmp/case.md", "content": "故障案例：MQ消费失败"}],
        ),
    )

    state = knowledge_retrieve_run(
        {
            "question": "订单失败",
            "query_rewrite": {"normalized_query": "订单失败", "keywords": ["MQ"]},
            "plan": {
                "hypothesis": "MQ异常",
                "investigation_goals": ["确认MQ发送状态"],
            },
            "execution": {"goal_index": 0},
        }
    )
    basis = dict(dict(state.get("knowledge_context") or {}).get("query_basis") or {})
    assert basis.get("hypothesis") == "MQ异常"
    assert basis.get("objective") == "确认MQ发送状态"
    assert state.get("route") == "planner"


def test_knowledge_retrieve_routes_existing_investigation_to_plan_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.knowledge_retrieve.knowledge_retrieve.query_parent_docs_from_rag",
        lambda **kwargs: (
            [],
            [],
            [{"path": "/tmp/case.md", "content": "故障案例：MQ消费失败"}],
        ),
    )

    state = knowledge_retrieve_run(
        {
            "question": "订单失败",
            "query_rewrite": {"normalized_query": "订单失败", "keywords": ["MQ"]},
            "investigation": {
                "plan": {
                    "hypothesis": "MQ异常",
                    "goals": [{"id": "g1", "goal": "确认MQ发送状态", "required_capability": "runtime_evidence"}],
                },
                "current_goal_id": "g1",
            },
        }
    )
    assert state.get("route") == "plan_controller"
