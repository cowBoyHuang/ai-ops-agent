from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from log.log import EsResult
from tool.registry import build_tool_schemas_for_prompt, get_all_tools, get_tool, invoke_tool

EXPECTED_TOOLS = {
    "queryLog",
    "dependency_log_query",
    "getFlightCreateOrderResult",
    "getCreateOrderResult",
    "indexProject",
    "searchMethod",
    "locateCode",
    "analyzeCodeFromLogs",
    "analyzeCodeForBusinessConsult",
    "rag_parent_chunk_query",
    "knowledge_lookup",
}


def test_registry_loads_expected_tools_with_unique_names() -> None:
    names = [tool.name for tool in get_all_tools()]
    assert set(names) >= EXPECTED_TOOLS
    assert len(names) == len(set(names))


def test_prompt_schema_comes_from_tool_descriptions() -> None:
    schemas = build_tool_schemas_for_prompt()
    by_name = {row["tool_name"]: row for row in schemas}
    assert "机票子单生单结果" in by_name["getFlightCreateOrderResult"]["description"]
    assert "乘机人" in by_name["getFlightCreateOrderResult"]["description"]
    assert "年龄" in by_name["getFlightCreateOrderResult"]["description"]
    assert "特殊产品" in by_name["getFlightCreateOrderResult"]["description"]
    assert "优先使用" in by_name["getFlightCreateOrderResult"]["description"]
    assert "总单生单结果" in by_name["getCreateOrderResult"]["description"]
    assert "不适合回答具体乘机人" in by_name["getCreateOrderResult"]["description"]
    assert "机票子单内部明细" in by_name["getCreateOrderResult"]["description"]
    assert "match_list=[]" in by_name["queryLog"]["description"]
    assert "扩展兜底" in by_name["queryLog"]["description"]
    assert "乘机人" in by_name["queryLog"]["description"]
    assert "年龄" in by_name["queryLog"]["description"]
    assert "业务文档" in by_name["analyzeCodeForBusinessConsult"]["description"]
    assert "实际代码" in by_name["analyzeCodeForBusinessConsult"]["description"]


def test_registry_invokes_query_log_tool(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_query_external_logs(**kwargs):
        captured.update(kwargs)
        return [EsResult(score=1.0, content="ok")]

    monkeypatch.setattr("log.log.query_external_logs", _fake_query_external_logs)
    begin = dt.datetime(2026, 1, 1, 11, 30, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    rows = invoke_tool(
        "queryLog",
        {
            "app_code": "f_tts_trade_order",
            "logname": "ttsorder",
            "begin_time": begin.isoformat(),
            "end_time": end.isoformat(),
            "match_phrase_list": ["ops_slugger_260101.120000.xxx"],
            "match_list": [],
        },
    )

    assert len(rows) == 1
    assert captured["app_code"] == "f_tts_trade_order"
    assert captured["logname"] == "ttsorder"
    assert get_tool("queryLog").name == "queryLog"


def test_invoke_tool_returns_structured_error_for_unknown_tool() -> None:
    out = invoke_tool("missingTool", {"x": 1})
    assert out == {
        "tool": "missingTool",
        "ok": False,
        "error": "unsupported tool: missingTool",
        "evidence": [],
    }


def test_production_runtime_has_no_old_string_router_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    production_files = [
        *root.glob("flow/**/*.py"),
        *root.glob("tool/**/*.py"),
        *root.glob("log/**/*.py"),
    ]
    offenders: list[str] = []
    for path in production_files:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "execute_log_query_method" in text or "execute_code_index_method" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_prompt_schemas_do_not_expose_old_router_names() -> None:
    rendered = json.dumps(build_tool_schemas_for_prompt(), ensure_ascii=False)
    assert "execute_log_query_method" not in rendered
    assert "execute_code_index_method" not in rendered
