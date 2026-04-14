"""Decide continue / clarify / degrade."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    question = str((state.get("structured_context") or {}).get("question") or "")
    if len(question) < 4:
        state["route"] = "reply_clarify"
        return state
    if "无法" in question or "不知道" in question:
        state["route"] = "reply_degrade"
        return state
    state["route"] = "rag_retrieve"
    return state

