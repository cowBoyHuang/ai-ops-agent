"""成功回复节点。

业务职责：
- 从 analysis.reply 读取最终回复文案。
- 对外标记任务 finished。
"""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """构建成功回复。

    入参：
    - payload: 任意状态字典，通常包含 analysis。

    返参：
    - dict[str, Any]: 写入 finished 状态和 response。
    """
    state = dict(payload)
    analysis = dict(state.get("analysis") or {})
    message = str(analysis.get("reply") or "分析完成")
    state["status"] = "finished"
    state["response"] = {
        "chatId": state.get("chat_id", ""),
        "status": "finished",
        "message": message,
    }
    return state
