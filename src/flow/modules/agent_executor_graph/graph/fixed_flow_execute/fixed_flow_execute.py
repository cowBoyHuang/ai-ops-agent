"""固定流程执行节点。"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm, load_prompt

_SYSTEM_PROMPT_FILE = "fixed_flow_execute_system_prompt.txt"


def _fallback_reply(intent_type: str, question: str) -> str:
    if intent_type == "ORDER_INFO_QUERY":
        return f"已收到你的订单信息查询请求：{question}。请提供订单号或更具体的查询项。"
    if intent_type == "SYSTEM_LOGIC_CONSULT":
        return f"已收到你的系统逻辑咨询：{question}。建议补充具体场景或规则点，便于准确回答。"
    return f"已收到你的问题：{question}。"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    question = str(state.get("normalized_question") or state.get("question") or state.get("message") or "").strip()
    intent_type = str(state.get("intent_type") or "UNKNOWN")

    system_prompt = load_prompt(_SYSTEM_PROMPT_FILE, default="")
    reply = chat_with_llm(question=question, system_prompt=system_prompt)
    if not reply:
        reply = _fallback_reply(intent_type, question)

    analysis = dict(state.get("analysis") or {})
    analysis["reply"] = reply
    analysis["intent_type"] = intent_type

    confidence = 0.8
    try:
        confidence = float((state.get("intent_recognition") or {}).get("confidence") or confidence)
    except (TypeError, ValueError):
        confidence = 0.8

    state["analysis"] = analysis
    state["solution"] = reply
    state["root_cause"] = str(state.get("root_cause") or "")
    state["confidence"] = max(0.0, min(confidence, 1.0))
    state["analysis_status"] = "SUCCESS"
    state["fixed_flow_hit"] = True
    state["route"] = "finish"
    return dict(state)
