"""澄清回复节点。

业务职责：
- 当信息不足时返回补充信息提示（traceId、时间范围、服务名）。
"""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建澄清回复。

    入参：
    - payload: 任意状态字典。

    返参：
    - dict[str, Any]: 写入 waiting_input 状态和 response。
    """
    state = dict(payload)
    state["status"] = "waiting_input"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "waiting_input",
        "message": "信息不足，请补充 traceId、报错时间和服务名。",
    }
    return state
