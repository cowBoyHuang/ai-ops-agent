"""Response emit module."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    if "response" in context and isinstance(context["response"], dict):
        context["response"].setdefault("chatId", context.get("chat_id", ""))
        return context

    message = ""
    if context.get("analysis", {}).get("reply"):
        message = str(context["analysis"]["reply"])
    elif context.get("error"):
        message = str(context.get("error"))
    else:
        message = "任务处理中"

    context["response"] = {
        "chatId": context.get("chat_id", ""),
        "status": context.get("status", "running"),
        "message": message,
    }
    return context
