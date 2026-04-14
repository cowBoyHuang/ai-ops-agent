"""Context build module."""

from __future__ import annotations

from typing import Any

_RECENT_CONTEXT: dict[str, list[str]] = {}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    if state.get("pipeline_stop"):
        return state

    chat_id = str(state.get("chat_id") or "")
    history = _RECENT_CONTEXT.get(chat_id, [])[-5:]
    message = str(state.get("message") or "")
    structured_context = {
        "chat_id": chat_id,
        "recent_messages": history,
        "intent_summary": message[:120],
        "question": state.get("normalized_question") or message,
    }
    state["structured_context"] = structured_context
    return state


def remember_message(chat_id: str, message: str) -> None:
    rows = _RECENT_CONTEXT.setdefault(chat_id, [])
    rows.append(message)

