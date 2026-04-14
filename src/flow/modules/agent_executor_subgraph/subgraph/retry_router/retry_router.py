"""Retry routing module."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 2))
    if retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        state["route"] = "analysis_execute"
        return state
    state["route"] = "reply_degrade"
    return state

