"""Prompt build + LLM call + parse (placeholder)."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    context = dict(state.get("structured_context") or {})
    evidence = str(context.get("evidence_context") or "")
    question = str(context.get("question") or state.get("message") or "")
    analysis = {
        "root_cause": "inventory-service timeout" if "timeout" in (question + evidence).lower() else "可能是下游依赖异常",
        "confidence": "high" if "timeout" in (question + evidence).lower() else "medium",
        "reply": "根因初步定位完成，建议先检查下游服务超时与连接池。",
    }
    state["analysis"] = analysis
    state["route"] = "result_validate"
    return state

