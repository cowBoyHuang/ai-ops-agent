from __future__ import annotations

import datetime as dt

from log.log import EsResult
from tool.registry import invoke_tool


def _time_range() -> tuple[dt.datetime, dt.datetime]:
    end = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    begin = end - dt.timedelta(minutes=30)
    return begin, end


def test_flight_create_order_tool_uses_fixed_app_and_log(monkeypatch) -> None:
    begin, end = _time_range()
    captured: dict[str, object] = {}

    def _fake_query_external_logs(**kwargs):
        captured.update(kwargs)
        return [EsResult(score=1.0, content="ok")]

    monkeypatch.setattr("log.log.query_external_logs", _fake_query_external_logs)

    rows = invoke_tool(
        "getFlightCreateOrderResult",
        {
            "trace_id": "ops_slugger_260101.120000.xxx",
            "begin_time": begin.isoformat(),
            "end_time": end.isoformat(),
        },
    )

    assert len(rows) == 1
    assert captured.get("app_code") == "f_tts_trade_core"
    assert captured.get("logname") == "tts"
    content = dict(captured.get("content") or {})
    phrase = list(content.get("match_phrase_list") or [])
    assert "单程生单结果" in phrase


def test_create_order_tool_uses_fixed_app_and_log(monkeypatch) -> None:
    begin, end = _time_range()
    captured: dict[str, object] = {}

    def _fake_query_external_logs(**kwargs):
        captured.update(kwargs)
        return [EsResult(score=1.0, content="ok")]

    monkeypatch.setattr("log.log.query_external_logs", _fake_query_external_logs)

    rows = invoke_tool(
        "getCreateOrderResult",
        {
            "trace_id": "ops_slugger_260101.120000.xxx",
            "begin_time": begin.isoformat(),
            "end_time": end.isoformat(),
        },
    )

    assert len(rows) == 1
    assert captured.get("app_code") == "f_tts_trade_order"
    assert captured.get("logname") == "ttsorder"
    content = dict(captured.get("content") or {})
    phrase = list(content.get("match_phrase_list") or [])
    assert "生单返回结果" in phrase
    assert "ops_slugger_260101.120000.xxx" in phrase


def test_query_log_tool_dispatches_generic_query(monkeypatch) -> None:
    begin, end = _time_range()
    captured: dict[str, object] = {}

    def _fake_query_external_logs(**kwargs):
        captured.update(kwargs)
        return [EsResult(score=1.0, content="ok")]

    monkeypatch.setattr("log.log.query_external_logs", _fake_query_external_logs)

    rows = invoke_tool(
        "queryLog",
        {
            "app_code": "f_tts_trade_order",
            "logname": "ttsorder.log",
            "begin_time": begin.isoformat(),
            "end_time": end.isoformat(),
            "match_phrase_list": ["ops_slugger_260101.120000.xxx"],
            "match_list": ["生单失败"],
        },
    )

    assert len(rows) == 1
    content = dict(captured.get("content") or {})
    assert list(content.get("match_list") or []) == ["生单失败"]
