"""Input validate module."""

from __future__ import annotations

from typing import Any

_MAX_MESSAGE_LEN = 4000


# 方法注释（业务）:
# - 业务：输入校验节点，仅判断消息基础合法性（不做敏感操作判断）。
# - 入参：`payload`(dict[str, Any])=上游传入上下文，核心字段为 `message`。
# - 出参：`dict[str, Any]`，返回更新后的上下文（失败时包含错误码并短路）。
# - 逻辑：
#   1) 校验消息非空与长度上限；
#   2) 不执行黑名单与 LLM 敏感拦截；
#   3) 写入已关闭敏感校验标记并进入下游。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    message = str(context.get("message") or "")
    if not message:
        context["status"] = "finished"
        context["error_code"] = "EMPTY_MESSAGE"
        context["error"] = "message is required"
        return context
    if len(message) > _MAX_MESSAGE_LEN:
        context["status"] = "finished"
        context["error_code"] = "MESSAGE_TOO_LONG"
        context["error"] = f"message length exceeds {_MAX_MESSAGE_LEN}"
        return context

    context["sensitive_check"] = {
        "passed": True,
        "mode": "disabled",
        "reason": "sensitive check disabled",
    }
    return context
