"""Error handle module."""

from __future__ import annotations

import logging
from typing import Any

_LOG = logging.getLogger(__name__)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    if not context.get("error"):
        context["error"] = "unexpected error"
    context["error_code"] = context.get("error_code") or "INTERNAL_ERROR"
    _LOG.error("error_handle.apply error_code=%s", str(context.get("error_code") or ""))
    context["status"] = "failed"
    context["response"] = {
        "chatId": context.get("chat_id", ""),
        "status": "failed",
        "message": f"[{context['error_code']}] {context['error']}",
    }
    return context
