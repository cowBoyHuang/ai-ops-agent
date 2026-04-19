"""分析执行节点。

业务职责：
- 汇总证据文本（优先使用结构化上下文里的 evidence_context）。
- 调用统一 LLM 方法生成根因与建议。
- 标准化置信度字段，供验证节点统一判断。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import analyze_with_llm

_CONFIDENCE_MAP = {
    "high": 0.9,
    "medium": 0.65,
    "low": 0.35,
}


def _coerce_confidence(value: Any) -> float:
    """把置信度转换为 float。

    兼容输入：
    - 文本 high/medium/low
    - 数字字符串或数值类型
    """
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value or "").strip().lower()
    if text in _CONFIDENCE_MAP:
        return _CONFIDENCE_MAP[text]
    try:
        return float(text)
    except ValueError:
        return 0.0


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行分析步骤。

    入参：
    - payload: AgentState，需包含 question/merged_evidence 等字段。

    返参：
    - AgentState: 写入 analysis/root_cause/solution/confidence，并路由到 result_validate。
    """
    state: AgentState = dict(payload)
    context = dict(state.get("structured_context") or {})
    merged_evidence = dict(state.get("merged_evidence") or {})

    # 证据优先级：先用上游已拼好的 evidence_context，避免重复拼接。
    if context.get("evidence_context"):
        evidence = str(context.get("evidence_context") or "")
    else:
        # 回退逻辑：从 merged_evidence 的 logs/knowledge 中拼接证据。
        logs = [str(item.get("text") or "") for item in list(merged_evidence.get("logs") or [])]
        knowledge = [str(item.get("text") or "") for item in list(merged_evidence.get("knowledge") or [])]
        evidence = "\n".join(row for row in [*logs, *knowledge] if row)

    question = str(state.get("question") or context.get("question") or state.get("message") or "")
    # 统一走 llm.llm.analyze_with_llm，便于后续做模型配置收口。
    analysis = analyze_with_llm(question=question, evidence=evidence)
    root_cause = str(analysis.get("root_cause") or "").strip()
    solution = str(analysis.get("reply") or "").strip()
    confidence = _coerce_confidence(analysis.get("confidence"))

    state["analysis"] = analysis
    state["root_cause"] = root_cause
    state["solution"] = solution
    state["confidence"] = confidence
    state["route"] = "result_validate"
    return dict(state)
