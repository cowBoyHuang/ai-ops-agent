from __future__ import annotations

import datetime as dt

from flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor import (
    _as_datetime,
    _clip_log_text,
    run as run_log_sub_executor,
)
from log.log import EsResult


def _patch_invoke_tool(monkeypatch, captured: dict[str, object], rows: list[EsResult] | None = None) -> None:
    def _fake_invoke_tool(name: str, args: dict[str, object]):
        captured["name"] = name
        captured.update(args)
        return rows if rows is not None else [EsResult(score=1.0, content="line1")]

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor.invoke_tool",
        _fake_invoke_tool,
    )


def test_log_executor_dispatches_to_registered_tool(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(monkeypatch, captured)
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "ok", "keywords": [], "facts": {}},
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "log_query",
            "params": {
                "log_method": "getCreateOrderResult",
                "app_code": "f_tts_trade_order",
                "logname": "ttsorder.log",
                "match_phrase_list": ["ops_slugger_260101.120000.10.0.0.1.1.1_0"],
                "match_list": ["生单返回结果"],
            },
        },
        state={"question": "订单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert captured.get("name") == "getCreateOrderResult"
    assert "log_method" not in captured


def test_log_executor_keeps_only_identifier_terms_for_querylog(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(monkeypatch, captured)
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "ok", "keywords": [], "facts": {}},
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "log_query",
            "params": {
                "log_method": "queryLog",
                "app_code": "f_tts_trade_order",
                "logname": "ttsorder",
                "match_phrase_list": ["生单返回结果"],
                "match_list": ["生单返回结果"],
            },
        },
        state={"question": "ops_slugger_260101.120000.10.0.0.1.1.1_0 订单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert captured.get("name") == "queryLog"
    phrase_list = list(captured.get("match_phrase_list") or [])
    assert "生单返回结果" not in phrase_list
    assert "ops_slugger_260101.120000.10.0.0.1.1.1_0" in phrase_list
    assert list(captured.get("match_list") or []) == []


def test_querylog_requires_trace_or_order_identifier(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(monkeypatch, captured)
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "ok", "keywords": [], "facts": {}},
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "queryLog",
            "params": {
                "app_code": "f_tts_trade_core",
                "logname": "tts",
                "match_phrase_list": [],
                "match_list": ["生单失败", "bizErrorCode"],
            },
        },
        state={"question": "为什么生单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is False
    assert "queryLog requires trace_id/order_no in match_phrase_list" in str(out.get("error") or "")
    assert captured == {}


def test_querylog_accepts_sid_order_identifier_and_clears_match_list(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(monkeypatch, captured)
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "ok", "keywords": [], "facts": {}},
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "queryLog",
            "params": {
                "app_code": "f_tts_trade_core",
                "logname": "tts",
                "order_id": "sid260614103651007",
                "match_phrase_list": [],
                "match_list": ["生单失败", "bizErrorCode"],
            },
        },
        state={"question": "为什么生单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert captured.get("name") == "queryLog"
    assert "sid260614103651007" in list(captured.get("match_phrase_list") or [])
    assert list(captured.get("match_list") or []) == []


def test_log_executor_uses_fixed_scope_tool_without_upstream_app_and_log(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(monkeypatch, captured)
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "ok", "keywords": [], "facts": {}},
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "getFlightCreateOrderResult",
            "params": {
                "trace_id": "ops_slugger_260101.120000.10.0.0.1.1.1_0",
            },
        },
        state={"question": "机票生单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert captured.get("name") == "getFlightCreateOrderResult"
    assert captured.get("trace_id") == "ops_slugger_260101.120000.10.0.0.1.1.1_0"
    assert "app_code" not in captured
    assert "logname" not in captured


def test_log_executor_returns_registry_structured_error(monkeypatch) -> None:
    def _fake_invoke_tool(name: str, args: dict[str, object]):
        _ = args
        return {"tool": name, "ok": False, "error": "boom", "evidence": []}

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor.invoke_tool",
        _fake_invoke_tool,
    )
    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "queryLog",
            "params": {
                "app_code": "f_tts_trade_core",
                "logname": "tts",
                "match_phrase_list": ["ops_slugger_260101.120000.10.0.0.1.1.1_0"],
                "match_list": [],
            },
        },
        state={"question": "为什么生单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out == {"tool": "queryLog", "ok": False, "error": "boom", "evidence": []}


def test_log_executor_clips_single_log_row_to_10000_chars() -> None:
    raw = "a" * 1600
    clipped = _clip_log_text(raw)
    assert len(clipped) == 1600


def test_as_datetime_supports_relative_now_format() -> None:
    now_value = _as_datetime("now")
    before_value = _as_datetime("now-2h")
    assert now_value is not None
    assert before_value is not None
    assert now_value >= before_value


def test_log_executor_uses_code_index_when_log_result_not_decisive(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(
        monkeypatch,
        captured,
        rows=[EsResult(score=1.0, content="at OrderUtil.java:1945 unknown failure")],
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {"summary": "日志证据不足", "keywords": ["OrderUtil"], "facts": {}},
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor.analyze_code_from_logs",
        lambda **kwargs: {
            "ok": True,
            "mode": "locateCode",
            "summary": "定位到方法 setTmpMapJsonValue(1938-1958)",
            "current_method": {"methodName": "setTmpMapJsonValue", "startLine": 1938, "endLine": 1958},
            "caller": [],
            "callee": [],
            "logs": [],
            "matched_methods": [],
            "evidence": [
                "[code_index] locateCode class=OrderUtil line=1945",
                "[code_index] 定位到方法 setTmpMapJsonValue(1938-1958)",
            ],
            "error": "",
        },
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "getCreateOrderResult",
            "params": {
                "match_phrase_list": ["ops_slugger_260101.120000.10.0.0.1.1.1_0"],
                "match_list": ["unknown failure"],
            },
        },
        state={"question": "为什么下单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert bool(dict(out.get("code_analysis") or {}).get("ok")) is True
    evidence = [str(item) for item in list(out.get("evidence") or [])]
    assert any("[code_index]" in row for row in evidence)


def test_log_executor_skips_code_index_when_log_has_direct_fact(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_invoke_tool(
        monkeypatch,
        captured,
        rows=[EsResult(score=1.0, content='{"bizErrorCode":"39","errMsg":"年龄限制"}')],
    )
    called = {"value": False}

    def _fake_analyze_code_from_logs(**kwargs):
        _ = kwargs
        called["value"] = True
        return {"ok": False, "summary": "should not call"}

    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor._extract_effective_info",
        lambda *args, **kwargs: {
            "summary": "bizErrorCode=39",
            "keywords": [],
            "facts": {"bizErrorCode": "39"},
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor.analyze_code_from_logs",
        _fake_analyze_code_from_logs,
    )

    now = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    out = run_log_sub_executor(
        step={
            "tool_name": "getCreateOrderResult",
            "params": {
                "match_phrase_list": ["ops_slugger_260101.120000.10.0.0.1.1.1_0"],
                "match_list": ["bizErrorCode"],
            },
        },
        state={"question": "为什么下单失败"},
        structured_context={
            "begin_time": (now - dt.timedelta(minutes=30)).isoformat(),
            "end_time": now.isoformat(),
        },
    )

    assert out["ok"] is True
    assert called["value"] is False
