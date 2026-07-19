from __future__ import annotations

import pytest

from flow.modules.agent_executor_graph.graph.reactor.reactor import _validate_querylog_params
from flow.modules.agent_executor_graph.graph.reactor.reactor import run as reactor_run


def _base_state() -> dict:
    return {
        "question": "订单失败，bizErrorCode 是什么",
        "plan": {
            "hypothesis": "生单失败由业务规则触发",
            "investigation_goals": ["获取并确认字段 bizErrorCode：该请求生单失败原因对应的错误码是什么？"],
            "required_answers": [
                {"field": "bizErrorCode", "question": "该请求生单失败原因对应的错误码是什么？", "required": True}
            ],
        },
        "execution": {
            "goal_index": 0,
            "max_act_times": 3,
            "evidence_graph": {"hypothesis": "生单失败由业务规则触发", "evidence": [], "supported": None},
        },
        "structured_context": {
            "begin_time": "2026-06-14T17:20:42+08:00",
            "end_time": "2026-06-14T19:20:42+08:00",
        },
    }


def test_reactor_required_field_missing_marks_goal_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        lambda **kwargs: {
            "skill": "queryLog",
            "tool_name": "log_query",
            "params": {
                "log_method": "queryLog",
                "app_code": "f_tts_trade_core",
                "logname": "tts",
                "match_phrase_list": ["trace123"],
                "match_list": [],
            },
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        lambda **kwargs: {
            "tool": "log_query",
            "ok": True,
            "error": "",
            "evidence": ["status=404, subErrorCode=39, errMsg=年龄限制"],
            "effective_info": {"summary": "仅出现 subErrorCode=39，未出现 bizErrorCode 字段", "facts": {}},
            "log_hit_count": 1,
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(_base_state())
    report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})

    assert state.get("route") == "observer"
    assert str(report.get("goal_status") or "") == "failed"
    assert "required field bizErrorCode unresolved" in str(report.get("failure_reason") or "")
    assert int(report.get("act_times") or 0) == 3


def test_reactor_required_field_resolved_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        lambda **kwargs: {
            "skill": "queryLog",
            "tool_name": "log_query",
            "params": {
                "log_method": "queryLog",
                "app_code": "f_tts_trade_core",
                "logname": "tts",
                "match_phrase_list": ["trace123"],
                "match_list": [],
            },
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        lambda **kwargs: {
            "tool": "log_query",
            "ok": True,
            "error": "",
            "evidence": ['{"bizErrorCode":"39","errMsg":"年龄限制"}'],
            "effective_info": {"summary": "bizErrorCode=39", "facts": {"bizErrorCode": "39"}},
            "log_hit_count": 1,
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(_base_state())
    report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})

    assert state.get("route") == "observer"
    assert str(report.get("goal_status") or "") == "success"
    assert "bizErrorCode=39" in str(report.get("goal_conclusion") or "")


def test_reactor_last_attempt_forces_querylog_fallback_with_only_phrase_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_decide_skill_with_llm(**kwargs):
        if bool(kwargs.get("force_querylog")):
            return {
                "skill": "queryLog",
                "tool_name": "log_query",
                "params": {
                    "log_method": "queryLog",
                    "app_code": "f_tts_trade_core",
                    "logname": "tts",
                    "match_phrase_list": [
                        "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
                        "xep260614182042451380394",
                    ],
                    "match_list": [],
                },
            }
        return {
            "skill": "getCreateOrderResult",
            "tool_name": "log_query",
            "params": {
                "log_method": "getCreateOrderResult",
                "match_phrase_list": ["生单返回结果", "trace123"],
                "match_list": ["失败原因", "bizErrorCode"],
            },
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        _fake_decide_skill_with_llm,
    )

    calls: list[dict[str, object]] = []

    def _fake_execute_tool_call(*, tool_name: str, params: dict[str, object], state: dict[str, object]) -> dict[str, object]:
        calls.append({"tool_name": tool_name, "params": dict(params)})
        # 前两轮均无命中，第三轮由 reactor 强制兜底 queryLog
        if len(calls) < 3:
            return {
                "tool": "log_query",
                "ok": False,
                "error": "no hits",
                "evidence": [],
                "effective_info": {"summary": "未检索到日志命中", "facts": {}},
                "log_hit_count": 0,
            }
        return {
            "tool": "log_query",
            "ok": True,
            "error": "",
            "evidence": ['{"subErrorCode":"39"}'],
            "effective_info": {"summary": "subErrorCode=39", "facts": {"subErrorCode": "39"}},
            "log_hit_count": 1,
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        _fake_execute_tool_call,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(
        {
            **_base_state(),
            "query_rewrite": {
                "trace_id": "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
                "order_id": "xep260614182042451380394",
            },
        }
    )

    assert state.get("route") == "observer"
    assert len(calls) == 3
    assert str(calls[-1].get("tool_name") or "") == "queryLog"
    third = dict(calls[-1].get("params") or {})
    assert "log_method" not in third
    assert list(third.get("match_list") or []) == []
    phrase_list = list(third.get("match_phrase_list") or [])
    assert "生单返回结果" not in phrase_list
    assert "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0" in phrase_list
    assert "xep260614182042451380394" in phrase_list


def test_reactor_overrides_llm_context_params_with_required_tool_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        lambda **kwargs: {
            "skill": "getFlightCreateOrderResult",
            "tool_name": "log_query",
            "params": {
                "log_method": "getFlightCreateOrderResult",
                "trace_id": "wrong_trace",
                "begin_time": "2026-06-16T00:00:00+08:00",
                "end_time": "2026-06-16T23:59:59+08:00",
            },
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._build_required_tool_params",
        lambda state: {
            "trace_id": "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
            "traceId": "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
            "order_id": "xep260614182042451380394",
            "orderNo": "xep260614182042451380394",
            "begin_time": "2026-06-14T17:20:42+08:00",
            "end_time": "2026-06-14T19:20:42+08:00",
            "app_code": "",
            "logname": "",
        },
    )

    captured: list[dict[str, object]] = []

    def _fake_execute_tool_call(*, tool_name: str, params: dict[str, object], state: dict[str, object]) -> dict[str, object]:
        captured.append(dict(params))
        return {
            "tool": "log_query",
            "ok": True,
            "error": "",
            "evidence": ['{"bizErrorCode":"39"}'],
            "effective_info": {"summary": "bizErrorCode=39", "facts": {"bizErrorCode": "39"}},
            "log_hit_count": 1,
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        _fake_execute_tool_call,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(_base_state())
    assert state.get("route") == "observer"
    assert captured
    params = dict(captured[0] or {})
    assert params.get("trace_id") == "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0"
    assert params.get("order_id") == "xep260614182042451380394"
    assert params.get("begin_time") == "2026-06-14T17:20:42+08:00"
    assert params.get("end_time") == "2026-06-14T19:20:42+08:00"


def test_reactor_semantic_fallback_extends_to_n_plus_one_and_forces_final_querylog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_decide_skill_with_llm(**kwargs):
        if bool(kwargs.get("force_querylog")):
            return {
                "skill": "queryLog",
                "tool_name": "log_query",
                "params": {
                    "log_method": "queryLog",
                    "app_code": "f_tts_trade_core",
                    "logname": "tts",
                    "match_phrase_list": [
                        "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
                        "xep260614182042451380394",
                    ],
                    "match_list": [],
                },
            }
        return {
            "skill": "getFlightCreateOrderResult",
            "tool_name": "log_query",
            "params": {
                "log_method": "getFlightCreateOrderResult",
                "match_phrase_list": ["trace123"],
                "match_list": ["bizErrorCode"],
            },
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        _fake_decide_skill_with_llm,
    )

    calls: list[dict[str, object]] = []

    def _fake_execute_tool_call(*, tool_name: str, params: dict[str, object], state: dict[str, object]) -> dict[str, object]:
        calls.append({"tool_name": tool_name, "params": dict(params)})
        if len(calls) == 1:
            return {
                "tool": "log_query",
                "ok": True,
                "error": "",
                "evidence": ["errorCode=404"],
                "effective_info": {"summary": "errorCode=404", "facts": {}},
                "log_hit_count": 1,
            }
        return {
            "tool": "log_query",
            "ok": True,
            "error": "",
            "evidence": ['{"bizErrorCode":"39"}'],
            "effective_info": {"summary": "bizErrorCode=39", "facts": {"bizErrorCode": "39"}},
            "log_hit_count": 1,
        }

    def _fake_extract_required_field_value(*, field: str, raw_result: dict, result_summary: str, tool_params: dict) -> tuple[str, str]:
        # 第 1 轮仅语义命中 -> 需要继续跑到 n+1 最后一轮兜底 queryLog。
        if len(calls) == 1:
            return "404", "semantic_fallback"
        return "39", "exact_field"

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        _fake_execute_tool_call,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.reactor.reactor._extract_required_field_value",
        _fake_extract_required_field_value,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(
        {
            **_base_state(),
            "query_rewrite": {
                "trace_id": "flight_supply_open_api_260614.182042.10.77.55.20.341.451380394_0",
                "order_id": "xep260614182042451380394",
            },
        }
    )
    report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})

    assert state.get("route") == "observer"
    assert str(report.get("goal_status") or "") == "success"
    # n=3，语义命中后允许执行到 n+1=4。
    assert int(report.get("act_times") or 0) == 4
    assert len(calls) == 4
    assert str(calls[-1].get("tool_name") or "") == "queryLog"
    final_params = dict(calls[-1].get("params") or {})
    assert "log_method" not in final_params
    assert list(final_params.get("match_list") or []) == []


def test_querylog_validation_accepts_zvp_order_no_in_match_phrase_list() -> None:
    error = _validate_querylog_params(
        tool_name="queryLog",
        tool_params={
            "app_code": "f_tts_trade_core",
            "logname": "tts",
            "match_phrase_list": ["zvp260621120318083"],
            "match_list": [],
        },
    )
    assert error == ""


def test_reactor_keeps_required_field_after_semantic_hit_even_if_final_fallback_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_decide_skill_with_llm(**kwargs):
        if bool(kwargs.get("force_querylog")):
            return {
                "skill": "queryLog",
                "tool_name": "log_query",
                "params": {
                    "log_method": "queryLog",
                    "match_phrase_list": ["trace123"],
                    "match_list": [],
                },
            }
        return {
            "skill": "getFlightCreateOrderResult",
            "tool_name": "log_query",
            "params": {
                "log_method": "getFlightCreateOrderResult",
                "match_phrase_list": ["trace123"],
                "match_list": [],
            },
        }

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._decide_skill_with_llm",
        _fake_decide_skill_with_llm,
    )

    call_count = {"n": 0}

    def _fake_execute_tool_call(*, tool_name: str, params: dict[str, object], state: dict[str, object]) -> dict[str, object]:
        call_count["n"] += 1
        return {
            "tool": tool_name,
            "ok": True,
            "error": "",
            "evidence": ['{"errorCode":"404"}'],
            "effective_info": {"summary": "errorCode=404", "facts": {}},
            "log_hit_count": 1,
        }

    def _fake_extract_required_field_value(*, field: str, raw_result: dict, result_summary: str, tool_params: dict) -> tuple[str, str]:
        if call_count["n"] == 1:
            return "39", "semantic_fallback"
        return "", "none"

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._execute_tool_call",
        _fake_execute_tool_call,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.reactor.reactor._extract_required_field_value",
        _fake_extract_required_field_value,
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.executor._infer_conclusion",
        lambda **kwargs: "supports",
    )

    state = reactor_run(_base_state())
    report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})

    assert state.get("route") == "observer"
    assert str(report.get("goal_status") or "") == "success"
    assert "bizErrorCode=39" in str(report.get("goal_conclusion") or "")
