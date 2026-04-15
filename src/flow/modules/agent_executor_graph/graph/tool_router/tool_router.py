"""工具路由节点。

业务职责：
- 根据当前 plan_steps 与步骤下标，选择下一步工具。
- 生成工具调用参数，减少下游工具节点分支复杂度。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.graph.agent_state import AgentState


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行工具路由。

    入参：
    - payload: AgentState，需包含 plan_steps/current_step_index/结构化上下文。

    返参：
    - AgentState: 写入 tool_name/tool_params/tool_route，并路由到 tool_execute。
    """
    state: AgentState = dict(payload)
    steps = [str(item) for item in list(state.get("plan_steps") or [])]
    step_index = int(state.get("current_step_index") or 0)
    context = dict(state.get("structured_context") or {})

    # 当步骤已经执行完，路由到 none，后续由验证/路由节点决定收口或重规划。
    if step_index >= len(steps):
        tool_name = "none"
        tool_params: dict[str, Any] = {}
        route_reason = "计划步骤已执行完成"
    else:
        # 规则路由：优先按步骤语义选择日志、依赖、知识工具。
        step = steps[step_index].lower()
        if "log" in step:
            tool_name = "log_query"
            tool_params = {
                "query": state.get("normalized_question") or state.get("question") or "",
                "trace_id": state.get("trace_id") or context.get("trace_id") or "",
                "order_id": state.get("order_id") or context.get("order_id") or "",
                "step": steps[step_index],
            }
            route_reason = "计划步骤要求查询日志"
        elif "dependency" in step:
            tool_name = "dependency_log_query"
            tool_params = {
                "query": state.get("normalized_question") or state.get("question") or "",
                "request_id": state.get("request_id") or context.get("request_id") or "",
                "step": steps[step_index],
            }
            route_reason = "计划步骤要求查询依赖服务"
        elif "context" in step:
            tool_name = "knowledge_lookup"
            tool_params = {
                "query": state.get("normalized_question") or state.get("question") or "",
                "step": steps[step_index],
            }
            route_reason = "计划步骤要求补充上下文"
        else:
            tool_name = "knowledge_lookup"
            tool_params = {
                "query": state.get("normalized_question") or state.get("question") or "",
                "step": steps[step_index],
            }
            route_reason = "默认知识检索策略"

    state["tool_name"] = tool_name
    state["tool_params"] = tool_params
    state["tool_route"] = {
        "tool_name": tool_name,
        "tool_params": tool_params,
        "step_index": step_index,
        "step_total": len(steps),
        "reason": route_reason,
    }
    state["route"] = "tool_execute"
    return dict(state)
