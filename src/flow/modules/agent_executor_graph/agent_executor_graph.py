"""Agent 执行主图入口。

业务职责：
- 接收上游 flow 传入的状态。
- 否则进入 LangGraph 主图执行完整排障流程。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.errors import GraphRecursionError

from flow.modules.agent_executor_graph.build_langgraph_graph import build_langgraph_graph

_LOG = logging.getLogger(__name__)
_FALLBACK_MESSAGE = "暂未能自动定位问题，请联系人工排查。"
_MIN_RECURSION_LIMIT = 64
_MAX_RECURSION_LIMIT = 240


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compute_recursion_limit(payload: dict[str, Any]) -> int:
    """根据预算估算合理递归上限，避免默认 25 过低导致误报。"""
    max_tool_calls = max(1, _as_int(payload.get("max_tool_calls"), 6))
    max_replan = max(0, _as_int(payload.get("max_replan"), 2))
    max_retries = max(0, _as_int(payload.get("max_retries"), _as_int(payload.get("max_retry"), 2)))
    estimated = 16 + max_tool_calls * 4 + max_replan * 14 + max_retries * 6
    return max(_MIN_RECURSION_LIMIT, min(_MAX_RECURSION_LIMIT, estimated))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行 Agent 主图。

    入参：
    - payload: 上游统一状态字典，至少包含用户问题、会话信息等字段。

    返参：
    - dict[str, Any]: 主图执行后的完整状态，包含分析结果、路由状态、回复内容等。
    """
    context = dict(payload)
    # 保留上游原始上下文，字段映射统一下沉到 state_build 节点处理。
    context["context"] = dict(payload)
    chain = build_langgraph_graph()
    recursion_limit = _compute_recursion_limit(context)
    _LOG.info("agent_graph.invoke.start recursion_limit=%s", recursion_limit)
    try:
        out = chain.invoke(context, config={"recursion_limit": recursion_limit})
        _LOG.info("agent_graph.invoke.end route=%s", str((out or {}).get("route") or ""))
        return out
    except GraphRecursionError as exc:
        _LOG.warning("agent_graph.invoke.recursion_limit: %s", exc)
        analysis = dict(context.get("analysis") or {})
        chat_id = str(context.get("chat_id") or "")
        return {
            **context,
            "error": str(exc),
            "error_code": "GRAPH_RECURSION_LIMIT",
            "status": "degraded",
            "route": "fallback",
            "final_answer": _FALLBACK_MESSAGE,
            "analysis": {
                **analysis,
                "reply": str(analysis.get("reply") or _FALLBACK_MESSAGE),
            },
            "response": {
                "chatId": chat_id,
                "status": "degraded",
                "message": _FALLBACK_MESSAGE,
            },
        }
