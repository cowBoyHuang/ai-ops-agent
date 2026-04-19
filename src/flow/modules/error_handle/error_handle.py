"""Error handle module."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    if not context.get("error"):
        context["error"] = "unexpected error"
    context["error_code"] = context.get("error_code") or "INTERNAL_ERROR"
    context["status"] = "failed"
    context["response"] = {
        "chatId": context.get("chat_id", ""),
        "status": "failed",
        "message": f"[{context['error_code']}] {context['error']}",
    }
    return context
