"""Request ingest module."""

from __future__ import annotations

from typing import Any
from uuid import uuid4


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    state["chat_id"] = str(state.get("chat_id") or state.get("chatId") or f"chat_{uuid4().hex[:8]}")
    state["user_id"] = str(state.get("user_id") or state.get("userId") or "anonymous")
    state["message"] = str(state.get("message") or state.get("query") or "").strip()
    state.setdefault("status", "running")
    state.setdefault("retry_count", 0)
    state.setdefault("max_retries", 2)
    state.setdefault("pipeline_stop", False)
    state.setdefault("error_code", "")
    state.setdefault("error", "")
    return state

