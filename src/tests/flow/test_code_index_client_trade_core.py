from __future__ import annotations

from pathlib import Path
from typing import Any

from tool.code_index_client import (
    analyze_code_for_business_consult,
    analyze_code_from_logs,
    search_method,
)
from tool.registry import invoke_tool


def _trade_core_method_row() -> dict[str, Any]:
    return {
        "methodId": 33586,
        "className": "ReceiveOrderBuilder",
        "fullClassName": "com.qunar.flight.trade.core.adapter.receive.ReceiveOrderBuilder",
        "methodName": "getAsynRoundBackNote",
        "signature": "getAsynRoundBackNote(CreateOrderBean createOrderBean)",
        "filePath": "/Users/zhicheng.huang/code/qunar/tts-trade-core/provider/src/main/java/com/qunar/flight/trade/core/adapter/receive/ReceiveOrderBuilder.java",
        "startLine": 529,
        "endLine": 539,
    }


class TestTradeCoreGetAsynRoundBackNoteCodeAnalysis:
    def test_search_method_returns_trade_core_get_asyn_round_back_note(self, monkeypatch) -> None:
        expected = _trade_core_method_row()

        def _fake_request_json(*, method: str, path: str, params=None, json_body=None):
            assert method == "GET"
            assert path == "/searchMethod"
            assert params == {"keyword": "getAsynRoundBackNote"}
            assert json_body is None
            return True, [expected], ""

        monkeypatch.setattr("tool.code_index_client._request_json", _fake_request_json)

        out = search_method("getAsynRoundBackNote")

        assert out["ok"] is True
        methods = list(out.get("methods") or [])
        assert len(methods) == 1
        row = dict(methods[0] or {})
        assert row.get("className") == "ReceiveOrderBuilder"
        assert row.get("methodName") == "getAsynRoundBackNote"
        assert "tts-trade-core" in str(row.get("filePath") or "")
        assert int(row.get("startLine") or 0) == 529
        assert int(row.get("endLine") or 0) == 539

    def test_analyze_code_from_logs_hits_get_asyn_round_back_note_by_search(self, monkeypatch) -> None:
        expected = _trade_core_method_row()
        searched_keywords: list[str] = []

        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": f"miss {class_name}:{line}"},
        )

        def _fake_search_method(keyword: str) -> dict[str, Any]:
            searched_keywords.append(str(keyword))
            if str(keyword) == "getAsynRoundBackNote":
                return {"ok": True, "methods": [expected], "error": ""}
            return {"ok": True, "methods": [], "error": ""}

        monkeypatch.setattr("tool.code_index_client.search_method", _fake_search_method)

        result = analyze_code_from_logs(
            question="请分析 f_tts_trade_core 中 getAsynRoundBackNote 的实现行为",
            evidence_rows=["下单链路日志未给出直接失败原因"],
            extra_keywords=["getAsynRoundBackNote"],
        )

        assert result["ok"] is True
        assert result["mode"] == "searchMethod"
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "ReceiveOrderBuilder"
        assert method.get("methodName") == "getAsynRoundBackNote"
        assert "getAsynRoundBackNote" in searched_keywords
        evidence_rows = [str(item) for item in list(result.get("evidence") or [])]
        assert any("searchMethod keyword=getAsynRoundBackNote" in row for row in evidence_rows)

    def test_analyze_code_for_business_consult_locate_hits_get_asyn_round_back_note(self, monkeypatch) -> None:
        expected = _trade_core_method_row()

        def _fake_locate_code(class_name: str, line: int) -> dict[str, Any]:
            assert class_name == "ReceiveOrderBuilder"
            assert line == 529
            return {
                "ok": True,
                "result": {
                    "method": {
                        "methodId": expected["methodId"],
                        "methodName": expected["methodName"],
                        "startLine": expected["startLine"],
                        "endLine": expected["endLine"],
                    },
                    "caller": [{"methodName": "buildReceiveOrder"}],
                    "callee": [{"methodName": "fillRoundBackNote"}],
                    "logs": [],
                },
                "error": "",
            }

        monkeypatch.setattr("tool.code_index_client.locate_code", _fake_locate_code)

        result = analyze_code_for_business_consult(
            question="f_tts_trade_core 的 getAsynRoundBackNote 处理逻辑是什么？",
            structured_context={"class_name": "ReceiveOrderBuilder", "line": 529},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "locateCode"
        current_method = dict(result.get("current_method") or {})
        assert current_method.get("methodName") == "getAsynRoundBackNote"
        assert int(current_method.get("startLine") or 0) == 529
        assert int(current_method.get("endLine") or 0) == 539
        assert len(list(result.get("caller") or [])) == 1
        assert len(list(result.get("callee") or [])) == 1

    def test_search_method_tool_dispatches_search_method(self, monkeypatch) -> None:
        expected = _trade_core_method_row()
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [expected], "error": ""},
        )

        out = invoke_tool("searchMethod", {"keyword": "getAsynRoundBackNote"})

        assert out["ok"] is True
        methods = list(out.get("methods") or [])
        assert methods
        assert str(dict(methods[0] or {}).get("methodName") or "") == "getAsynRoundBackNote"

    def test_analyze_business_tool_dispatches_analyze_business(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "tool.code_index_client.analyze_code_for_business_consult",
            lambda **kwargs: {"ok": True, "mode": "locateCode", "summary": "定位到方法 getAsynRoundBackNote"},
        )
        out = invoke_tool(
            "analyzeCodeForBusinessConsult",
            {
                "question": "生单特殊产品拦截入口在哪",
                "structured_context": {"class_name": "ReceiveOrderBuilder", "line": 529},
                "evidence_rows": [],
            },
        )
        assert out["ok"] is True
        assert out["mode"] == "locateCode"


class TestLocalCodeIndexFallback:
    def test_analyze_logs_falls_back_to_local_line_context(self, monkeypatch, tmp_path: Path) -> None:
        source_root = tmp_path / "tts-trade-core"
        source_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/service/impl/SingleCreateOrderServiceImpl.java"
        )
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.service.impl;",
                    "",
                    "public class SingleCreateOrderServiceImpl {",
                    "    public SingleOrderCreateResponse createOrder(SingleOrderCreateRequest request) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_from_logs(
            question="根据日志定位代码",
            evidence_rows=["SingleCreateOrderServiceImpl.java:4"],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_locate_line"
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "SingleCreateOrderServiceImpl"
        assert method.get("methodName") == "createOrder"
        assert int(method.get("startLine") or 0) == 4
        assert int(method.get("endLine") or 0) == 6

    def test_analyze_business_falls_back_to_local_symbol_declaration(self, monkeypatch, tmp_path: Path) -> None:
        source_root = tmp_path / "tts-trade-core"
        enum_path = source_root / "provider/src/main/java/com/qunar/flight/trade/core/enums/BizErrorCode.java"
        enum_path.parent.mkdir(parents=True)
        enum_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.enums;",
                    "",
                    "public enum BizErrorCode {",
                    "    SUCCESS(\"00_000_000000_0000\", \"成功\");",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="bizErrorCode在哪个类声明？类路径是什么",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_symbol"
        symbol = dict(result.get("current_symbol") or {})
        assert symbol.get("symbolName") == "BizErrorCode"
        assert symbol.get("kind") == "enum"
        assert symbol.get("fullClassName") == "com.qunar.flight.trade.core.enums.BizErrorCode"
        assert str(symbol.get("filePath") or "").endswith("BizErrorCode.java")
        assert int(symbol.get("line") or 0) == 3

    def test_analyze_business_falls_back_to_local_business_entry(self, monkeypatch, tmp_path: Path) -> None:
        source_root = tmp_path / "tts-trade-core"
        single_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/service/impl/SingleCreateOrderServiceImpl.java"
        )
        double_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/service/impl/DoubleCreateOrderServiceImpl.java"
        )
        single_path.parent.mkdir(parents=True)
        single_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.service.impl;",
                    "",
                    "public class SingleCreateOrderServiceImpl {",
                    "    public SingleOrderCreateResponse createOrder(SingleOrderCreateRequest request) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        double_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.service.impl;",
                    "",
                    "public class DoubleCreateOrderServiceImpl {",
                    "    public OrderCreateResponse createOrder(OrderCreateRequest request) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="帮我找到生单入口代码",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_business_entry"
        methods = [dict(item or {}) for item in list(result.get("matched_methods") or [])]
        assert {row.get("className") for row in methods} == {
            "SingleCreateOrderServiceImpl",
            "DoubleCreateOrderServiceImpl",
        }
        assert all(row.get("methodName") == "createOrder" for row in methods)

    def test_analyze_business_expands_order_creation_synonym_to_local_business_entry(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        source_root = tmp_path / "tts-trade-core"
        source_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/service/impl/SingleCreateOrderServiceImpl.java"
        )
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.service.impl;",
                    "",
                    "public class SingleCreateOrderServiceImpl {",
                    "    public SingleOrderCreateResponse createOrder(SingleOrderCreateRequest request) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="订单创建入口在哪个方法？",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_business_entry"
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "SingleCreateOrderServiceImpl"
        assert method.get("methodName") == "createOrder"

    def test_analyze_business_falls_back_to_any_local_method_name(self, monkeypatch, tmp_path: Path) -> None:
        source_root = tmp_path / "tts-trade-core"
        source_path = source_root / "provider/src/main/java/com/qunar/flight/trade/core/service/OrderHelper.java"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.service;",
                    "",
                    "public class OrderHelper {",
                    "    public void calculateTotalFare() {",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="calculateTotalFare在哪里实现？",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_method"
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "OrderHelper"
        assert method.get("methodName") == "calculateTotalFare"

    def test_analyze_business_falls_back_to_local_full_text_match(self, monkeypatch, tmp_path: Path) -> None:
        source_root = tmp_path / "tts-trade-core"
        source_path = source_root / "provider/src/main/resources/flow/create-order-flow.yml"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "\n".join(
                [
                    "name: create-order-flow",
                    "steps:",
                    "  - specialLocalMasterOnlyMarker",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="specialLocalMasterOnlyMarker在哪里配置？",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_text"
        text_matches = [dict(item or {}) for item in list(result.get("text_matches") or [])]
        assert text_matches
        assert str(text_matches[0].get("filePath") or "").endswith("create-order-flow.yml")
        assert int(text_matches[0].get("line") or 0) == 3

    def test_analyze_business_finds_chinese_flow_description_in_local_source(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        source_root = tmp_path / "tts-trade-core"
        source_path = source_root / "provider/src/main/resources/flow/flow-single.xml"
        java_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/component/SpecialProductRuleInterceptor.java"
        )
        source_path.parent.mkdir(parents=True)
        java_path.parent.mkdir(parents=True)
        source_path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    "<qflow:flow>",
                    '    <qflow:stage name="priceValidate" desc="特殊产品校验">',
                    '        <qflow:component id="specialProductRuleComp" desc="特殊产品"/>',
                    "    </qflow:stage>",
                    "</qflow:flow>",
                ]
            ),
            encoding="utf-8",
        )
        java_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.component;",
                    "",
                    "import com.qunar.flight.trade.engine.annotation.QFlowComponent;",
                    "",
                    '@QFlowComponent(value = "specialProductRuleComp")',
                    "public class SpecialProductRuleInterceptor {",
                    "    public CreateOrderResult execute(CreateOrderBean bean) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="特殊产品校验是哪个文件呢？哪个入口呢？",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_text"
        text_matches = [dict(item or {}) for item in list(result.get("text_matches") or [])]
        assert text_matches
        assert str(text_matches[0].get("filePath") or "").endswith("flow-single.xml")
        assert int(text_matches[0].get("line") or 0) == 3
        assert "特殊产品校验" in str(text_matches[0].get("text") or "")
        symbol = dict(result.get("current_symbol") or {})
        assert symbol.get("symbolName") == "SpecialProductRuleInterceptor"
        assert symbol.get("componentId") == "specialProductRuleComp"
        assert str(symbol.get("filePath") or "").endswith("SpecialProductRuleInterceptor.java")
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "SpecialProductRuleInterceptor"
        assert method.get("methodName") == "execute"

    def test_analyze_business_maps_pnr_creation_term_to_strategy_component(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        source_root = tmp_path / "tts-trade-core"
        flow_path = source_root / "provider/src/main/resources/flow/flow-connect.xml"
        java_path = (
            source_root
            / "provider/src/main/java/com/qunar/flight/trade/core/component/pnr/pnrasyn/NormalPolicyPnrOrderServiceImpl.java"
        )
        flow_path.parent.mkdir(parents=True)
        java_path.parent.mkdir(parents=True)
        flow_path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    "<qflow:flow>",
                    '    <qflow:component id="commonRiskEstimationCheckComp" desc="异步调用打标系统校验"/>',
                    '    <qflow:component id="strategyNormalPnrOrderServiceComp" desc="策略生编"/>',
                    "</qflow:flow>",
                ]
            ),
            encoding="utf-8",
        )
        java_path.write_text(
            "\n".join(
                [
                    "package com.qunar.flight.trade.core.component.pnr.pnrasyn;",
                    "",
                    "import com.qunar.flight.trade.engine.annotation.QFlowComponent;",
                    "",
                    '@QFlowComponent("strategyNormalPnrOrderServiceComp")',
                    "public class NormalPolicyPnrOrderServiceImpl {",
                    "    public AsyncOrderRet execute(ReceiveOrder data, FlowContext flowContext) {",
                    "        return null;",
                    "    }",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AIOPS_CODE_REPO_ROOTS", str(source_root))
        monkeypatch.setattr(
            "tool.code_index_client.locate_code",
            lambda class_name, line: {"ok": False, "result": {}, "error": "code-index unavailable"},
        )
        monkeypatch.setattr(
            "tool.code_index_client.search_method",
            lambda keyword: {"ok": True, "methods": [], "error": ""},
        )

        result = analyze_code_for_business_consult(
            question="生编是调用哪个接口呢？调用位置在哪",
            structured_context={},
            evidence_rows=[],
        )

        assert result["ok"] is True
        assert result["mode"] == "local_text"
        text_matches = [dict(item or {}) for item in list(result.get("text_matches") or [])]
        assert text_matches
        assert str(text_matches[0].get("text") or "").endswith('desc="策略生编"/>')
        symbol = dict(result.get("current_symbol") or {})
        assert symbol.get("symbolName") == "NormalPolicyPnrOrderServiceImpl"
        assert symbol.get("componentId") == "strategyNormalPnrOrderServiceComp"
        method = dict(result.get("current_method") or {})
        assert method.get("className") == "NormalPolicyPnrOrderServiceImpl"
        assert method.get("methodName") == "execute"
