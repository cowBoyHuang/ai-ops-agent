"""Validate analysis output."""

from __future__ import annotations

from typing import Any

_VALID_CONFIDENCE = {"high", "medium", "low"}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    analysis = dict(state.get("analysis") or {})
    root_cause = str(analysis.get("root_cause") or "").strip()
    confidence = str(analysis.get("confidence") or "").lower().strip()
    ok = bool(root_cause) and confidence in _VALID_CONFIDENCE
    state["analysis_ok"] = ok
    state["route"] = "reply_success" if ok else "retry_router"
    return state

