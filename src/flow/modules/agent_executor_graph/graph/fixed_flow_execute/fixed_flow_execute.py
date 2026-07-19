"""固定流程执行节点。"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.fixed_flow_execute.business_code_consult_skill import (
    run as run_business_code_consult_skill,
)
from llm.llm import analyze_with_llm, chat_with_llm, load_prompt

_SYSTEM_PROMPT_FILE = "fixed_flow_execute_system_prompt.txt"
_BUSINESS_SYSTEM_PROMPT_FILE = "analysis_business_consult_system_prompt.txt"
_BUSINESS_USER_PROMPT_FILE = "analysis_business_consult_user_prompt.txt"


def _fallback_reply(intent_type: str, question: str) -> str:
    if intent_type == "SYSTEM_LOGIC_CONSULT":
        return f"已收到你的业务咨询：{question}。建议补充具体场景或规则点，便于准确回答。"
    return f"已收到你的问题：{question}。"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    question = str(state.get("question") or "").strip()
    intent_type = str(state.get("intent_type") or "SYSTEM_LOGIC_CONSULT")

    if intent_type == "SYSTEM_LOGIC_CONSULT":
        structured_context = dict(state.get("structured_context") or {})
        skill_result = run_business_code_consult_skill(
            question=question,
            structured_context=structured_context,
        )
        evidence_context = str(skill_result.get("evidence_context") or "").strip()
        llm_result = analyze_with_llm(
            question=question,
            evidence=evidence_context,
            system_prompt_file=_BUSINESS_SYSTEM_PROMPT_FILE,
            user_prompt_file=_BUSINESS_USER_PROMPT_FILE,
        )
        reply = str(llm_result.get("reply") or "").strip()
        if not reply:
            reply = _fallback_reply(intent_type, question)

        analysis = dict(state.get("analysis") or {})
        analysis["reply"] = reply
        analysis["intent_type"] = intent_type
        analysis["root_cause"] = str(llm_result.get("root_cause") or "").strip()
        analysis["business_skill"] = {
            "skill_name": str(skill_result.get("skill_name") or "business_code_consult"),
            "code_analysis": dict(skill_result.get("code_analysis") or {}),
        }

        confidence_raw = llm_result.get("confidence")
        confidence = 0.75
        if isinstance(confidence_raw, str):
            lowered = confidence_raw.strip().lower()
            if lowered == "high":
                confidence = 0.9
            elif lowered == "medium":
                confidence = 0.7
            elif lowered == "low":
                confidence = 0.4
        else:
            try:
                confidence = float(confidence_raw or confidence)
            except (TypeError, ValueError):
                confidence = 0.75

        merged_evidence = dict(skill_result.get("merged_evidence") or {})
        knowledge_context = dict(skill_result.get("knowledge_context") or {})
        state["knowledge_context"] = knowledge_context
        state["merged_evidence"] = merged_evidence
        state["structured_context"] = {
            **structured_context,
            "knowledge_context": knowledge_context,
            "evidence_context": evidence_context,
        }
        state["analysis"] = analysis
        state["solution"] = reply
        state["root_cause"] = str(llm_result.get("root_cause") or "")
        state["confidence"] = max(0.0, min(confidence, 1.0))
        state["analysis_status"] = "SUCCESS"
        state["route"] = "finish"
        return dict(state)

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
    state["route"] = "finish"
    return dict(state)
