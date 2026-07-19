from __future__ import annotations

import json

import pytest

from flow.modules.agent_executor_graph.graph.planner.planner import run as planner_run


def test_planner_outputs_hypothesis_and_investigation_goals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "MQ异常",
                "investigation_goals": ["确认MQ发送状态", "确认MQ消费状态"],
                "required_answers": [{"field": "bizErrorCode", "question": "给出 bizErrorCode", "required": True}],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "MQ积压",
            "intent_type": "OPS_ANALYSIS",
            "knowledge_context": {
                "domain_docs": [{"path": "/tmp/domain.md", "content": "MQ链路说明"}],
                "case_docs": [{"path": "/tmp/case.md", "content": "MQ消费失败案例"}],
                "code_docs": [],
            },
            "structured_context": {},
        }
    )

    assert dict(state.get("plan") or {}).get("hypothesis") == "MQ异常"
    goals = list(dict(state.get("plan") or {}).get("investigation_goals") or [])
    assert goals[0] == "确认MQ发送状态"
    required_answers = list(dict(state.get("plan") or {}).get("required_answers") or [])
    assert str(dict(required_answers[0]).get("field") or "") == "bizErrorCode"
    assert state.get("route") == "plan_controller"


def test_planner_avoids_rejected_hypothesis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "支付异常",
                "investigation_goals": ["确认支付回调状态"],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "订单失败",
            "rejected_hypothesis": ["MQ异常", "支付异常"],
            "replan_reason": "支付链路证据不足",
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )
    assert dict(state.get("plan") or {}).get("hypothesis") not in {"MQ异常", "支付异常"}


def test_planner_resets_execution_pointer_and_retry_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "下游异常",
                "investigation_goals": ["确认下游接口返回"],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "订单失败",
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "execution": {
                "goal_index": 3,
                "objective_retry_count": 1,
                "insufficient_round_count": 1,
            },
            "structured_context": {},
        }
    )
    execution = dict(state.get("execution") or {})
    assert execution.get("goal_index") == 0
    assert execution.get("objective_retry_count") == 0
    assert execution.get("insufficient_round_count") == 0


def test_planner_derives_required_answers_from_question_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "下游异常",
                "investigation_goals": ["核实失败返回字段"],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "这个请求失败的bizErrorCode是多少？",
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )
    required_answers = list(dict(state.get("plan") or {}).get("required_answers") or [])
    assert any(str(dict(item).get("field") or "") == "bizErrorCode" for item in required_answers)


def test_planner_derives_generic_required_field_without_fixed_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "状态聚合异常",
                "investigation_goals": ["确认返回字段取值"],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "请给出 createOrderStatus 字段的真实取值",
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )
    required_answers = list(dict(state.get("plan") or {}).get("required_answers") or [])
    assert any(str(dict(item).get("field") or "") == "createOrderStatus" for item in required_answers)


def test_planner_filters_irrelevant_code_field_when_question_only_asks_passenger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "生单失败由规则拦截触发",
                "investigation_goals": [
                    "获取并确认字段 intercepted_passenger：被拦截的乘机人是谁",
                    "获取并确认字段 bizErrorCode：本次生单失败对应的业务错误编码是什么",
                ],
                "required_answers": [
                    {"field": "intercepted_passenger", "question": "被拦截的乘机人是谁", "required": True},
                    {"field": "bizErrorCode", "question": "本次生单失败对应的业务错误编码是什么", "required": True},
                ],
            },
            ensure_ascii=False,
        ),
    )

    state = planner_run(
        {
            "question": "被拦截的乘机人是谁",
            "query_rewrite": {
                "normalized_query": "被拦截的乘机人是谁",
                "keywords": ["被拦截", "乘机人", "是谁"],
            },
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )

    plan = dict(state.get("plan") or {})
    goals = [str(item) for item in list(plan.get("investigation_goals") or [])]
    required_answers = [dict(item or {}) for item in list(plan.get("required_answers") or [])]
    fields = [str(item.get("field") or "") for item in required_answers]

    assert "intercepted_passenger" in fields
    assert "bizErrorCode" not in fields
    assert len(goals) == 1
    assert "intercepted_passenger" in goals[0]
