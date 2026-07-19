from __future__ import annotations

from flow.modules.agent_executor_graph.graph.fixed_flow_execute.fixed_flow_execute import run as fixed_flow_execute_run


def test_fixed_flow_execute_business_consult_uses_business_code_skill(monkeypatch) -> None:
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.fixed_flow_execute.fixed_flow_execute.run_business_code_consult_skill",
        lambda **kwargs: {
            "skill_name": "business_code_consult",
            "knowledge_context": {"domain_docs": [{"path": "doc-a", "content": "规则A"}], "case_docs": [], "code_docs": []},
            "merged_evidence": {
                "logs": [],
                "knowledge": [{"text": "规则A"}],
                "code": [{"text": "[code_index] 定位到方法 paySuccess"}],
            },
            "evidence_context": "业务文档证据+代码证据",
            "code_analysis": {"ok": True, "summary": "定位到方法 paySuccess"},
        },
    )
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.fixed_flow_execute.fixed_flow_execute.analyze_with_llm",
        lambda **kwargs: {
            "root_cause": "当前实现以 paySuccess 方法为准",
            "confidence": "high",
            "reply": "根据业务文档与实际代码，当前规则由 paySuccess 方法执行。",
        },
    )

    state = fixed_flow_execute_run(
        {
            "question": "支付成功后会触发哪些业务逻辑？",
            "intent_type": "SYSTEM_LOGIC_CONSULT",
            "structured_context": {},
        }
    )

    assert state.get("route") == "finish"
    assert "根据业务文档与实际代码" in str(dict(state.get("analysis") or {}).get("reply") or "")
    assert dict(state.get("knowledge_context") or {}).get("domain_docs")
    assert dict(state.get("merged_evidence") or {}).get("code")
    assert str(dict(state.get("structured_context") or {}).get("evidence_context") or "").strip()
