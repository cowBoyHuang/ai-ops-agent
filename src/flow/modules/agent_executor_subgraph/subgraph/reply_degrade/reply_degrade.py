"""Build degrade reply."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    state["status"] = "degraded"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "degraded",
        "message": "当前自动分析可信度不足，已降级为人工介入建议。",
    }
    return state

