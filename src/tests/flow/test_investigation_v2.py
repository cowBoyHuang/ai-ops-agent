from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow.modules.agent_executor_graph.graph.planner.planner import run as planner_run


def _plan() -> dict[str, object]:
    return {
        "plan_id": "plan_001",
        "hypothesis": "生单链路校验失败",
        "goals": [
            {
                "id": "g1",
                "goal": "定位异常发生阶段",
                "required_capability": "runtime_evidence",
                "priority": 1,
                "required": True,
                "success_criteria": ["明确失败日志"],
                "expected_evidence": ["log_event"],
                "depends_on": [],
            },
            {
                "id": "g2",
                "goal": "验证业务规则",
                "required_capability": "business_validation",
                "priority": 2,
                "required": True,
                "success_criteria": ["找到规则说明"],
                "expected_evidence": ["business_doc"],
                "depends_on": ["g1"],
            },
        ],
        "finish_criteria": ["根因明确"],
        "replan_triggers": ["missing_required_context", "capability_not_supported"],
    }


def test_planner_writes_complete_investigation_plan_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(_plan(), ensure_ascii=False),
    )

    state = planner_run(
        {
            "question": "为什么订单失败",
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )

    investigation = dict(state.get("investigation") or {})
    plan = dict(investigation.get("plan") or {})
    assert plan.get("plan_id") == "plan_001"
    assert plan.get("hypothesis") == "生单链路校验失败"
    assert list(plan.get("finish_criteria") or []) == ["根因明确"]
    assert list(plan.get("replan_triggers") or []) == ["missing_required_context", "capability_not_supported"]
    first_goal = dict(list(plan.get("goals") or [])[0])
    assert first_goal["required_capability"] == "runtime_evidence"
    assert first_goal["required"] is True
    assert state.get("route") == "plan_controller"


def test_planner_prompt_requests_investigation_plan_v2_and_forbids_tool_params() -> None:
    prompt_dir = Path(__file__).resolve().parents[2] / "llm" / "prompts"
    system_prompt = (prompt_dir / "planner_system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = (prompt_dir / "planner_user_prompt.txt").read_text(encoding="utf-8")
    combined = f"{system_prompt}\n{user_prompt}"

    assert "InvestigationPlanV2" in combined
    assert "required_capability" in combined
    assert "finish_criteria" in combined
    assert "replan_triggers" in combined
    assert "tool_name" in combined
    assert "工具参数" in combined


def test_domain_executor_prompt_limits_action_to_allowed_tools() -> None:
    prompt_dir = Path(__file__).resolve().parents[2] / "llm" / "prompts"
    user_prompt = (prompt_dir / "domain_executor_react_user_prompt.txt").read_text(encoding="utf-8")
    log_prompt = (prompt_dir / "log_executor_react_system_prompt.txt").read_text(encoding="utf-8")

    assert "Allowed Tools" in user_prompt
    assert "只能选择 Allowed Tools" in user_prompt
    assert "不要修改 plan" in user_prompt
    assert "queryLog" in log_prompt
    assert "searchMethod" not in log_prompt


def test_planner_keeps_v2_goals_aligned_after_required_answer_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.planner.planner.chat_with_llm",
        lambda **_: json.dumps(
            {
                "hypothesis": "生单失败由规则拦截触发",
                "goals": [
                    {
                        "id": "g1",
                        "goal": "获取并确认字段 intercepted_passenger：被拦截的乘机人是谁",
                        "required_capability": "runtime_evidence",
                        "priority": 1,
                        "required": True,
                        "success_criteria": ["拿到乘机人"],
                        "expected_evidence": ["log_event"],
                        "depends_on": [],
                    },
                    {
                        "id": "g2",
                        "goal": "获取并确认字段 bizErrorCode：业务错误编码是什么",
                        "required_capability": "runtime_evidence",
                        "priority": 2,
                        "required": True,
                        "success_criteria": ["拿到错误码"],
                        "expected_evidence": ["log_event"],
                        "depends_on": [],
                    },
                ],
                "required_answers": [
                    {"field": "intercepted_passenger", "question": "被拦截的乘机人是谁", "required": True},
                    {"field": "bizErrorCode", "question": "业务错误编码是什么", "required": True},
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
                "keywords": ["被拦截", "乘机人"],
            },
            "knowledge_context": {"domain_docs": [], "case_docs": [], "code_docs": []},
            "structured_context": {},
        }
    )

    legacy_goals = list(dict(state.get("plan") or {}).get("investigation_goals") or [])
    v2_goals = list(dict(dict(state.get("investigation") or {}).get("plan") or {}).get("goals") or [])
    assert len(legacy_goals) == 1
    assert len(v2_goals) == 1
    assert "intercepted_passenger" in str(dict(v2_goals[0]).get("goal") or "")


def test_capability_router_maps_runtime_evidence_to_log_executor() -> None:
    from flow.modules.agent_executor_graph.graph.capability_router.capability_router import run as router_run

    state = router_run(
        {
            "investigation": {
                "current_goal_id": "g1",
                "plan": {"goals": [{"id": "g1", "required_capability": "runtime_evidence"}]},
            }
        }
    )
    route = dict(dict(state.get("investigation") or {}).get("last_route_result") or {})
    assert route["executor"] == "LogExecutor"
    assert "queryLog" in list(route["allowed_tools"])
    assert state["route"] == "plan_controller"


@pytest.mark.parametrize(
    ("capability", "executor"),
    [
        ("runtime_evidence", "LogExecutor"),
        ("code_analysis", "CodeExecutor"),
        ("business_validation", "KnowledgeExecutor"),
        ("config_analysis", "ConfigExecutor"),
    ],
)
def test_capability_router_maps_all_supported_capabilities(capability: str, executor: str) -> None:
    from flow.modules.agent_executor_graph.graph.capability_router.capability_router import run as router_run

    state = router_run(
        {
            "investigation": {
                "current_goal_id": "g1",
                "plan": {"goals": [{"id": "g1", "required_capability": capability}]},
            }
        }
    )
    result = dict(dict(state.get("investigation") or {}).get("last_route_result") or {})
    assert result["executor"] == executor
    assert result["ok"] is True


def test_capability_router_unknown_capability_returns_unsupported_result() -> None:
    from flow.modules.agent_executor_graph.graph.capability_router.capability_router import run as router_run

    state = router_run(
        {
            "investigation": {
                "current_goal_id": "g1",
                "plan": {"goals": [{"id": "g1", "required_capability": "unknown_capability"}]},
            }
        }
    )
    result = dict(dict(state.get("investigation") or {}).get("last_route_result") or {})
    assert result["status"] == "unsupported"
    assert result["error"] == "capability_not_supported"
    assert result["result_id"].startswith("route_g1_")


def test_plan_controller_consumes_supported_route_and_sets_pending_execution() -> None:
    from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as controller_run

    state = controller_run(
        {
            "investigation": {
                "plan": {"goals": [{"id": "g1", "required": True, "required_capability": "runtime_evidence"}]},
                "current_goal_id": "g1",
                "goal_status": {"g1": "pending"},
                "last_route_result": {
                    "result_id": "route_g1_1",
                    "goal_id": "g1",
                    "executor": "LogExecutor",
                    "allowed_tools": ["queryLog"],
                    "ok": True,
                    "error": "",
                },
                "consumed_result_ids": [],
            }
        }
    )
    investigation = dict(state.get("investigation") or {})
    assert dict(investigation.get("pending_execution") or {}).get("executor") == "LogExecutor"
    assert investigation.get("last_route_result") == {}
    assert "route_g1_1" in list(investigation.get("consumed_result_ids") or [])
    assert state["route"] == "log_executor"


def test_plan_controller_persists_executor_result_and_finishes_required_goals() -> None:
    from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as controller_run

    state = controller_run(
        {
            "investigation": {
                "plan": {"goals": [{"id": "g1", "required": True, "required_capability": "runtime_evidence"}]},
                "current_goal_id": "g1",
                "goal_status": {"g1": "running"},
                "pending_execution": {"goal_id": "g1", "executor": "LogExecutor", "attempt": 1},
                "last_executor_result": {
                    "executor": "LogExecutor",
                    "result_id": "exec_g1_1",
                    "goal_id": "g1",
                    "goal_complete": True,
                    "status": "succeeded",
                    "summary": "命中失败日志",
                    "facts": {"errorCode": "10321"},
                    "evidence": [{"type": "log_event", "source": "queryLog", "content": "failed", "confidence": 0.9}],
                    "artifacts": [],
                    "confidence": 0.9,
                    "error": "",
                },
                "consumed_result_ids": [],
                "evidence": [],
            }
        }
    )
    investigation = dict(state.get("investigation") or {})
    assert dict(investigation.get("goal_status") or {})["g1"] == "succeeded"
    assert list(investigation.get("evidence") or [])[0]["facts"]["errorCode"] == "10321"
    assert state["route"] == "summary"


def test_plan_controller_routes_failed_required_goal_to_replan_with_budget() -> None:
    from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as controller_run

    state = controller_run(
        {
            "investigation": {
                "plan": {
                    "goals": [{"id": "g1", "required": True, "required_capability": "runtime_evidence"}],
                    "replan_triggers": ["missing_required_context"],
                },
                "current_goal_id": "g1",
                "goal_status": {"g1": "running"},
                "pending_execution": {"goal_id": "g1", "executor": "LogExecutor", "attempt": 1},
                "last_executor_result": {
                    "executor": "LogExecutor",
                    "result_id": "exec_g1_1",
                    "goal_id": "g1",
                    "goal_complete": False,
                    "status": "failed",
                    "summary": "",
                    "facts": {},
                    "evidence": [],
                    "artifacts": [],
                    "confidence": 0.0,
                    "error": "missing_required_context",
                },
                "consumed_result_ids": [],
                "evidence": [],
                "replan_count": 0,
                "max_replans": 1,
            }
        }
    )
    investigation = dict(state.get("investigation") or {})
    assert state["route"] == "planner"
    assert investigation["replan_count"] == 1
    assert investigation.get("pending_execution") == {}
    assert state.get("replan_reason") == "missing_required_context"
    assert dict(state.get("replan_context") or {}).get("failed_goal_id") == "g1"
    events = [dict(item or {}) for item in list(investigation.get("events") or [])]
    assert any(item.get("type") == "replan" and item.get("message") == "replan accepted" for item in events)


def test_plan_controller_advances_to_next_goal_after_dependency_succeeds() -> None:
    from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as controller_run

    state = controller_run(
        {
            "investigation": {
                "plan": {
                    "goals": [
                        {"id": "g1", "required": True, "required_capability": "runtime_evidence"},
                        {
                            "id": "g2",
                            "required": True,
                            "required_capability": "business_validation",
                            "depends_on": ["g1"],
                            "priority": 2,
                        },
                    ]
                },
                "current_goal_id": "g1",
                "goal_status": {"g1": "running", "g2": "pending"},
                "last_executor_result": {
                    "executor": "LogExecutor",
                    "result_id": "exec_g1_1",
                    "goal_id": "g1",
                    "goal_complete": True,
                    "status": "succeeded",
                    "summary": "命中失败日志",
                    "facts": {},
                    "evidence": [],
                    "artifacts": [],
                    "confidence": 0.8,
                    "error": "",
                },
                "consumed_result_ids": [],
                "evidence": [],
            }
        }
    )
    investigation = dict(state.get("investigation") or {})
    assert dict(investigation.get("goal_status") or {})["g1"] == "succeeded"
    assert investigation.get("current_goal_id") == "g2"
    assert state["route"] == "capability_router"


def test_plan_controller_retries_invalid_executor_result_once_then_fallback() -> None:
    from flow.modules.agent_executor_graph.graph.plan_controller.plan_controller import run as controller_run

    first = controller_run(
        {
            "investigation": {
                "plan": {"goals": [{"id": "g1", "required": True, "required_capability": "runtime_evidence"}]},
                "current_goal_id": "g1",
                "goal_status": {"g1": "running"},
                "last_executor_result": {
                    "executor": "LogExecutor",
                    "result_id": "exec_g1_1",
                    "goal_id": "g1",
                    "goal_complete": False,
                    "status": "failed",
                    "summary": "",
                    "facts": {},
                    "evidence": [],
                    "artifacts": [],
                    "confidence": 0.0,
                    "error": "invalid_executor_result",
                },
                "consumed_result_ids": [],
                "retry_counts_by_goal": {},
                "max_retries_per_goal": 2,
                "max_replans": 0,
            }
        }
    )
    assert first["route"] == "capability_router"
    assert dict(dict(first.get("investigation") or {}).get("retry_counts_by_goal") or {})["g1"] == 1

    second = controller_run(
        {
            **first,
            "investigation": {
                **dict(first.get("investigation") or {}),
                "goal_status": {"g1": "running"},
                "last_executor_result": {
                    "executor": "LogExecutor",
                    "result_id": "exec_g1_2",
                    "goal_id": "g1",
                    "goal_complete": False,
                    "status": "failed",
                    "summary": "",
                    "facts": {},
                    "evidence": [],
                    "artifacts": [],
                    "confidence": 0.0,
                    "error": "invalid_executor_result",
                },
            },
        }
    )
    assert second["route"] == "fallback"


def test_log_domain_executor_receives_only_log_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from flow.modules.agent_executor_graph.graph.domain_executors.log_executor import run as log_executor_run

    captured: dict[str, object] = {}

    def _fake_execute_domain_goal(**kwargs):
        captured.update(kwargs)
        return {
            "executor": "LogExecutor",
            "result_id": "exec_g1_1",
            "goal_id": "g1",
            "goal_complete": True,
            "status": "succeeded",
            "summary": "ok",
            "facts": {},
            "evidence": [],
            "artifacts": [],
            "confidence": 0.8,
            "error": "",
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.domain_executors.log_executor.execute_domain_goal",
        _fake_execute_domain_goal,
    )
    state = log_executor_run(
        {
            "question": "为什么失败",
            "investigation": {
                "current_goal_id": "g1",
                "plan": {"goals": [{"id": "g1", "goal": "定位异常", "required_capability": "runtime_evidence"}]},
                "pending_execution": {"goal_id": "g1", "executor": "LogExecutor", "attempt": 1},
            },
        }
    )
    allowed = list(captured.get("allowed_tools") or [])
    assert allowed
    assert set(allowed) <= {"queryLog", "dependency_log_query", "getFlightCreateOrderResult", "getCreateOrderResult"}
    assert dict(dict(state.get("investigation") or {}).get("last_executor_result") or {})["status"] == "succeeded"
