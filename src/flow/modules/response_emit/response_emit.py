"""Response emit module."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    if "response" in state and isinstance(state["response"], dict):
        state["response"].setdefault("chatId", state.get("chat_id", ""))
        return state

    message = ""
    if state.get("duplicate_hit"):
        message = str(state.get("duplicate_answer") or "")
    elif state.get("analysis", {}).get("reply"):
        message = str(state["analysis"]["reply"])
    elif state.get("error"):
        message = str(state.get("error"))
    else:
        message = "任务处理中"

    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": state.get("status", "running"),
        "message": message,
    }
    return state

