"""Input validate module."""

from __future__ import annotations

from typing import Any

from llm.llm import check_sensitive_operation_with_llm

_MAX_MESSAGE_LEN = 4000
_DANGEROUS_COMMAND_BLACKLIST = (
    "rm -rf",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "shutdown -h",
    "poweroff",
    "reboot",
    "chmod 777 /",
    "chown -r root",
    "curl | sh",
    "wget | sh",
)


def _is_llm_check_unavailable(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    if text in {"llm check unavailable", "llm unavailable", "llm not available"}:
        return True
    return ("llm" in text and "unavailable" in text) or ("llm" in text and "not available" in text)


# 方法注释（业务）:
# - 业务：在校验失败时统一写入失败状态和错误信息。
# - 入参：`context`(dict)=流程上下文；`error_code`(str)=错误码；`error`(str)=错误文案；
#         `sensitive_check`(dict)=敏感校验详情。
# - 出参：`dict[str, Any]`，返回已标记失败状态的上下文。
# - 逻辑：设置 `status/error_code/error/sensitive_check` 后返回。
def _mark_validate_failed(context: dict[str, Any], error_code: str, error: str, sensitive_check: dict[str, Any]) -> dict[str, Any]:
    context["status"] = "finished"
    context["error_code"] = error_code
    context["error"] = error
    context["sensitive_check"] = sensitive_check
    return context


# 方法注释（业务）:
# - 业务：执行危险命令黑名单匹配。
# - 入参：`message`(str)=用户输入文本。
# - 出参：`list[str]`，返回命中的黑名单命令列表（可能为空）。
# - 逻辑：将文本转小写后，按包含关系匹配 `_DANGEROUS_COMMAND_BLACKLIST`。
def _match_blacklist(message: str) -> list[str]:
    lowered = str(message or "").lower()
    return [cmd for cmd in _DANGEROUS_COMMAND_BLACKLIST if cmd in lowered]


# 方法注释（业务）:
# - 业务：输入校验节点，判断消息合法性与敏感性，决定是否允许下游继续执行。
# - 入参：`payload`(dict[str, Any])=上游传入上下文，核心字段为 `message`。
# - 出参：`dict[str, Any]`，返回更新后的上下文（失败时包含错误码并短路）。
# - 逻辑：
#   1) 校验消息非空与长度上限；
#   2) 黑名单匹配危险命令；
#   3) 黑名单通过后调用大模型做敏感操作判断；
#   4) 任一失败则标记失败并短路，全部通过则写入通过标记。
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

    blacklist_hits = _match_blacklist(message)
    if blacklist_hits:
        return _mark_validate_failed(
            context=context,
            error_code="SENSITIVE_BLACKLIST_BLOCKED",
            error=f"sensitive command blocked by blacklist: {blacklist_hits[0]}",
            sensitive_check={
                "passed": False,
                "mode": "blacklist",
                "reason": "matched dangerous command",
                "hits": blacklist_hits,
            },
        )

    llm_check = check_sensitive_operation_with_llm(message)
    llm_reason = str(llm_check.get("reason") or "")
    if _is_llm_check_unavailable(llm_reason):
        # LLM 风险校验不可用时降级放行，仍保留黑名单拦截能力。
        context["sensitive_check"] = {
            "passed": True,
            "mode": "blacklist+llm_degraded",
            "reason": "llm unavailable, skipped llm check",
        }
        return context

    if not bool(llm_check.get("passed")):
        return _mark_validate_failed(
            context=context,
            error_code="SENSITIVE_LLM_BLOCKED",
            error=str(llm_reason or "llm sensitive check blocked"),
            sensitive_check={
                "passed": False,
                "mode": "llm",
                "reason": llm_reason,
            },
        )

    context["sensitive_check"] = {
        "passed": True,
        "mode": "blacklist+llm",
        "reason": "passed",
    }
    return context
