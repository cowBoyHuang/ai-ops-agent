"""RAG 检索节点。

业务职责：
- 根据问题生成候选知识片段（BM25 + 向量召回占位）。
- OPS_ANALYSIS 场景补充运维排障规则片段。
- 输出 rag_docs/rag_scores，供 planner 与分析节点使用。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行检索步骤。

    入参：
    - payload: AgentState，需包含 normalized_question 或 question。

    返参：
    - AgentState: 写入 rag_docs/rag_scores，并路由到 planner。
    """
    state: AgentState = dict(payload)
    question = str(state.get("normalized_question") or state.get("question") or "")
    intent_type = str(state.get("intent_type") or "UNKNOWN")

    # 当前为占位实现：模拟关键词召回与语义召回各一条。
    docs: list[dict[str, Any]] = []
    if question:
        docs.append(
            {
                "id": "bm25-1",
                "score": 0.82,
                "source": "bm25",
                "text": f"关键词召回：{question[:64]}",
            }
        )
        docs.append(
            {
                "id": "vec-1",
                "score": 0.78,
                "source": "vector",
                "text": f"语义召回：{question[:64]}",
            }
        )
    # 运维问题补充固定排障知识，避免仅靠问题文本信息不足。
    if intent_type == "OPS_ANALYSIS":
        docs.append(
            {
                "id": "ops-guide-1",
                "score": 0.71,
                "source": "knowledge",
                "text": "排障建议：先确认trace链路，再核对核心服务日志与超时告警。",
            }
        )

    # 分数降序，确保后续优先使用高相关证据。
    docs.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    state["rag_docs"] = docs
    state["rag_scores"] = [float(item.get("score") or 0.0) for item in docs]
    state["route"] = "planner"
    return dict(state)
