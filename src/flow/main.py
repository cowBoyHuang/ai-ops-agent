from __future__ import annotations

from typing import Any

from flow.flow import run as flow_run


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Public flow entrypoint used by web adapters."""
    return flow_run(payload)
