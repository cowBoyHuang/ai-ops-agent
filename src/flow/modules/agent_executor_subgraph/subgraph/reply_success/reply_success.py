"""Build success reply."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    analysis = dict(state.get("analysis") or {})
    message = str(analysis.get("reply") or "分析完成")
    state["status"] = "finished"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "finished",
        "message": message,
    }
    return state

