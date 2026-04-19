"""Duplicate detect module."""

from __future__ import annotations

from typing import Any

from cache.message_cache_context import MessageCacheContext, RoundMessageContext
from embedding.embedding import cosine_similarity, text_embedding

_EMBEDDING_DUPLICATE_THRESHOLD = 0.95


# 方法注释（业务）:
# - 业务：将任意缓存输入统一转换为 `MessageCacheContext` 对象。
# - 入参：`value`(Any)=缓存值，可能为对象/JSON字符串/字典。
# - 出参：`MessageCacheContext`，规范化后的缓存对象。
# - 逻辑：按类型还原；无法还原时返回空对象。
def _as_message_context(value: Any) -> MessageCacheContext:
    if isinstance(value, MessageCacheContext):
        return value
    if isinstance(value, str):
        return MessageCacheContext.from_json(value) or MessageCacheContext()
    if isinstance(value, dict):
        return MessageCacheContext.from_dict(value)
    return MessageCacheContext()


# 方法注释（业务）:
# - 业务：读取单轮缓存中的历史回答文本。
# - 入参：`row`(RoundMessageContext)=单轮缓存对象。
# - 出参：`str`，历史回答字符串（去首尾空白）。
# - 逻辑：直接读取 `row.aiResponse` 并标准化为空字符串兜底。
def _row_ai_response(row: RoundMessageContext) -> str:
    return str(row.aiResponse or "").strip()


# 方法注释（业务）:
# - 业务：读取单轮缓存中的用户问题 embedding 向量。
# - 入参：`row`(RoundMessageContext)=单轮缓存对象。
# - 出参：`list[float]`，embedding 向量；异常类型时返回空数组。
# - 逻辑：直接取 `row.userMessageEmbedding`，仅在类型正确时返回。
def _row_embedding(row: RoundMessageContext) -> list[float]:
    embedding = row.userMessageEmbedding
    return embedding if isinstance(embedding, list) else []


# 方法注释（业务）:
# - 业务：在历史缓存中查找可复用回答。
# - 入参：`message_context`(Any)=历史缓存；`embedding`(list[float])=当前问题向量。
# - 出参：`str | None`，命中则返回历史回答，未命中返回 `None`。
# - 逻辑：倒序遍历历史轮次，计算余弦相似度，达到阈值立即返回对应回答。
def _find_duplicate_ai_response(message_context: Any, embedding: list[float]) -> str | None:
    rounds = _as_message_context(message_context).rounds
    for row in reversed(rounds):
        response = _row_ai_response(row)
        if not response:
            continue

        if embedding:
            score = cosine_similarity(embedding, _row_embedding(row))
            if score >= _EMBEDDING_DUPLICATE_THRESHOLD:
                return response

    return None


# 方法注释（业务）:
# - 业务：重复问题检测节点，命中时直接复用缓存回答并短路流程。
# - 入参：`payload`(dict[str, Any])=上游上下文，核心字段为 `message` 与 `message_context`。
# - 出参：`dict[str, Any]`，返回更新后的上下文（命中则 `status=finished` 并写入 `response`）。
# - 逻辑：
#   1) 计算当前消息 embedding；
#   2) 倒序与缓存向量做余弦相似度匹配；
#   3) 命中则写入历史回答并结束，否则继续下游。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)

    message = str(context.get("message") or "")
    embedding = text_embedding(message, dim=512)
    message_context = context.get("message_context")

    duplicate_answer = _find_duplicate_ai_response(message_context, embedding)
    if duplicate_answer is not None:
        context["status"] = "finished"
        context["error_code"] = ""
        context["error"] = ""
        context["response"] = {
            "chatId": context.get("chat_id", ""),
            "status": "finished",
            "message": duplicate_answer,
        }
        return context

    return context
