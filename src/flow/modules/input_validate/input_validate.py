"""Input validate module."""

from __future__ import annotations

from typing import Any

_MAX_MESSAGE_LEN = 4000


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    message = str(state.get("message") or "")
    if not message:
        state["pipeline_stop"] = True
        state["status"] = "failed"
        state["error_code"] = "EMPTY_MESSAGE"
        state["error"] = "message is required"
        return state
    if len(message) > _MAX_MESSAGE_LEN:
        state["pipeline_stop"] = True
        state["status"] = "failed"
        state["error_code"] = "MESSAGE_TOO_LONG"
        state["error"] = f"message length exceeds {_MAX_MESSAGE_LEN}"
        return state
    # 敏感词检查占位：当前先透传
    state.setdefault("sensitive_check", {"passed": True, "mode": "placeholder"})
    return state

