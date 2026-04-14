"""Evidence fusion and context building."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    docs = list(state.get("rag_docs") or [])
    docs.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    merged = docs[:4]
    state["merged_evidence"] = merged
    state["structured_context"] = {
        **dict(state.get("structured_context") or {}),
        "evidence_context": "\n".join(str(item.get("text") or "") for item in merged),
    }
    state["route"] = "analysis_execute"
    return state

