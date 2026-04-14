"""Build clarify reply."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    state["status"] = "waiting_input"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "waiting_input",
        "message": "信息不足，请补充 traceId、报错时间和服务名。",
    }
    return state

