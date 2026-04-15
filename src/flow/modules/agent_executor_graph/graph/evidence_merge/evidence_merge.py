"""证据融合节点。

业务职责：
- 合并 RAG 检索结果和工具结果。
- 统一产出 merged_evidence（logs/knowledge/context）。
- 生成 evidence_context 文本，供分析节点直接喂给大模型。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.graph.agent_state import AgentState


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行证据融合。

    入参：
    - payload: AgentState，需包含 rag_docs/tool_result/结构化上下文字段。

    返参：
    - AgentState: 写入 merged_evidence 和 structured_context.evidence_context。
    """
    state: AgentState = dict(payload)
    docs = [dict(item) for item in list(state.get("rag_docs") or [])]
    tool_result = dict(state.get("tool_result") or {})
    tool_evidence = list(tool_result.get("evidence") or [])

    merged = {
        "logs": [],
        "knowledge": [],
        "context": {
            "order_id": state.get("order_id") or "",
            "trace_id": state.get("trace_id") or "",
            "request_id": state.get("request_id") or "",
            "intent_type": state.get("intent_type") or "UNKNOWN",
        },
    }

    # 检索证据按分数降序，优先保留高相关片段。
    docs.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    for item in docs[:6]:
        merged["knowledge"].append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "score": float(item.get("score") or 0.0),
                "text": str(item.get("text") or ""),
            }
        )

    # 工具证据按工具类型归类到 logs 或 knowledge。
    tool_name = str(tool_result.get("tool") or "")
    for idx, text in enumerate(tool_evidence, start=1):
        record = {
            "id": f"tool-{idx}",
            "source": tool_name or "tool",
            "score": 1.0,
            "text": str(text),
        }
        if "log" in tool_name:
            merged["logs"].append(record)
        else:
            merged["knowledge"].append(record)

    # 生成纯文本证据，便于分析节点直接拼 Prompt。
    evidence_text_rows: list[str] = []
    for item in merged["logs"]:
        evidence_text_rows.append(str(item.get("text") or ""))
    for item in merged["knowledge"]:
        evidence_text_rows.append(str(item.get("text") or ""))

    state["merged_evidence"] = merged
    state["structured_context"] = {
        **dict(state.get("structured_context") or {}),
        "evidence_context": "\n".join(row for row in evidence_text_rows if row),
        "merged_evidence_summary": {
            "logs_count": len(merged["logs"]),
            "knowledge_count": len(merged["knowledge"]),
        },
    }
    state["route"] = "analysis_execute"
    return dict(state)
