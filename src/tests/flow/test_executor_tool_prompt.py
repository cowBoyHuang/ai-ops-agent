from __future__ import annotations

import json

from flow.modules.agent_executor_graph.graph.executor import executor


def test_executor_prompt_uses_registry_tool_schemas_not_skill_catalog(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_render_prompt(template: str, **kwargs):
        _ = template
        for key, value in kwargs.items():
            captured[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return "prompt"

    monkeypatch.setattr(executor, "render_prompt", _fake_render_prompt)
    monkeypatch.setattr(executor, "load_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(
        executor,
        "chat_with_llm",
        lambda **kwargs: '{"action":{"tool_name":"queryLog","params":{"match_phrase_list":["ops_slugger_260101.120000.xxx"],"match_list":[]}}}',
    )

    out = executor._decide_skill_with_llm(
        state={"question": "订单失败", "plan": {"hypothesis": "h", "investigation_goals": ["g"]}},
        hypothesis="h",
        objective="g",
        current_evidence={},
        evidence_rows=[],
        retry_count=0,
    )

    assert out["tool_name"] == "queryLog"
    rendered = json.dumps(captured, ensure_ascii=False)
    assert "tool_schemas_json" in captured
    assert "execute_log_query_method" not in rendered
    assert "execute_code_index_method" not in rendered
    assert "skill_catalog_json" not in captured
