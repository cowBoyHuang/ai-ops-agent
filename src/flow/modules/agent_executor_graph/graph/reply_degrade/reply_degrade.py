"""降级回复节点。

业务职责：
- 当自动分析失败或可信度不足时，返回人工介入提示。
"""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建降级回复。

    入参：
    - payload: 任意状态字典。

    返参：
    - dict[str, Any]: 写入 degraded 状态和 response。
    """
    state = dict(payload)
    state["status"] = "degraded"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "degraded",
        "message": "当前自动分析可信度不足，已降级为人工介入建议。",
    }
    return state
