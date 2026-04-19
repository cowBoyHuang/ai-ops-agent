"""Agent 执行主图入口。

业务职责：
- 接收上游 flow 传入的状态。
- 否则进入 LangGraph 主图执行完整排障流程。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.build_langgraph_graph import build_langgraph_graph


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行 Agent 主图。

    入参：
    - payload: 上游统一状态字典，至少包含用户问题、会话信息等字段。

    返参：
    - dict[str, Any]: 主图执行后的完整状态，包含分析结果、路由状态、回复内容等。
    """
    context = dict(payload)
    chain = build_langgraph_graph()
    return chain.invoke(context)
