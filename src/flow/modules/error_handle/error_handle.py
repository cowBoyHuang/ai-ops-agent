"""Error handle module."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    if not state.get("error"):
        state["error"] = "unexpected error"
    state["error_code"] = state.get("error_code") or "INTERNAL_ERROR"
    state["status"] = "failed"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "failed",
        "message": f"[{state['error_code']}] {state['error']}",
    }
    return state

