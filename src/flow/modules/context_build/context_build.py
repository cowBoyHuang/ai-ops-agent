"""Context build module."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any
import unicodedata
from uuid import uuid4

from cache.message_cache_context import MessageCacheContext
from cache.message_cache_store import get_message_cache


# 方法注释（业务）:
# - 业务：生成请求追踪 ID（仅日志用途）。
# - 入参：无。
# - 出参：`str`，格式为 `yyyyMMdd + 4位随机后缀`。
# - 逻辑：取当前日期并拼接 `uuid4().hex[:4]`。
def _generate_request_id() -> str:
    date_text = datetime.now().strftime("%Y%m%d")
    suffix = uuid4().hex[:4]
    return f"{date_text}{suffix}"


# 方法注释（业务）:
# - 业务：兼容多种入参键名，提取用户问题原文。
# - 入参：`context`(dict[str, Any])=请求上下文。
# - 出参：`Any`，返回 `message/query/content` 中首个有效值，不存在则返回空字符串。
# - 逻辑：按 `message -> query -> content` 顺序取第一个非空文本。
def _pick_message_input(context: dict[str, Any]) -> Any:
    for key in ("message", "query", "content"):
        value = context.get(key)
        if value is None:
            continue
        if str(value).strip() == "":
            continue
        return value
    return ""


# 方法注释（业务）:
# - 业务：对输入文本做统一标准化清洗。
# - 入参：`text`(Any)=待处理文本。
# - 出参：`str`，标准化后的文本。
# - 逻辑：全角转半角、去换行制表符、去标点、合并空白、去首尾空白并转小写。
def _normalize_message(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower()


# 方法注释（业务）:
# - 业务：将缓存中的值还原为 `MessageCacheContext` 对象。
# - 入参：`cache_value`(Any)=缓存原值，可能是对象/JSON字符串/字典。
# - 出参：`MessageCacheContext`，可安全供下游模块读取。
# - 逻辑：按类型分支还原；解析失败或不支持类型时返回空对象。
def _restore_message_context(cache_value: Any) -> MessageCacheContext:
    if isinstance(cache_value, MessageCacheContext):
        return cache_value
    if isinstance(cache_value, str):
        return MessageCacheContext.from_json(cache_value) or MessageCacheContext()
    if isinstance(cache_value, dict):
        return MessageCacheContext.from_dict(cache_value)
    return MessageCacheContext()


# 方法注释（业务）:
# - 业务：上下文构建节点，统一会话字段、规范消息、加载缓存上下文。
# - 入参：`payload`(dict[str, Any])=请求原始上下文，兼容 camelCase/snake_case 与 message/query/content。
# - 出参：`dict[str, Any]`，返回标准化后的流程上下文。
# - 逻辑：
#   1) 统一 `chat_id/user_id`；
#   2) 生成 `requestId`；
#   3) 标准化 `message`；
#   4) 读取缓存并恢复为 `MessageCacheContext` 放入 `message_context`；
#   5) 补齐流程默认控制字段（初始状态为 `init`）。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    context["chat_id"] = str(context.get("chat_id") or context.get("chatId") or f"chat_{uuid4().hex[:8]}")
    context["user_id"] = str(context.get("user_id") or context.get("userId") or "anonymous")
    # requestId 仅用于日志打印追踪，不参与业务路由字段。
    context["requestId"] = _generate_request_id()
    context["message"] = _normalize_message(_pick_message_input(context))
    cache = get_message_cache(context["chat_id"])
    context["message_context"] = _restore_message_context(cache)
    context.setdefault("status", "init")
    context.setdefault("error_code", "")
    context.setdefault("error", "")
    return context
