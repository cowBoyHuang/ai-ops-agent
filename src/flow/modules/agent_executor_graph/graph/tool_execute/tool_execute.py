"""工具执行节点。

业务职责：
- 执行 tool_router 选出的工具（当前为占位实现）。
- 记录 tool_history，便于审计与重试。
- 维护 tool_call_count/current_step_index，用于循环控制。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.graph.agent_state import AgentState


def _as_int(value: Any, default: int) -> int:
    """把输入安全转换为 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tool_success(tool: str, evidence: list[str]) -> dict[str, Any]:
    """构建成功工具结果结构。"""
    return {
        "tool": tool,
        "ok": True,
        "error": "",
        "evidence": evidence,
    }


def _tool_failed(tool: str, error: str) -> dict[str, Any]:
    """构建失败工具结果结构。"""
    return {
        "tool": tool,
        "ok": False,
        "error": error,
        "evidence": [],
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行工具并写回状态。

    入参：
    - payload: AgentState，需包含 tool_name/tool_params 与循环计数字段。

    返参：
    - AgentState: 写入 tool_result/tool_history，并路由到 evidence_merge。
    """
    state: AgentState = dict(payload)
    tool_name = str(state.get("tool_name") or "none")
    tool_params = dict(state.get("tool_params") or {})
    question = str(state.get("normalized_question") or state.get("question") or "")
    tool_call_count = _as_int(state.get("tool_call_count"), 0)
    max_tool_calls = max(1, _as_int(state.get("max_tool_calls"), 6))

    # 超过工具预算时直接记失败，由 retry_router 统一决定后续。
    if tool_call_count >= max_tool_calls:
        tool_result = _tool_failed(tool_name, "max_tool_calls_exceeded")
    # 测试开关：仅第一次调用模拟网络超时，用于验证 retry 分支。
    elif bool(state.get("simulate_tool_timeout_once")) and not bool(state.get("_simulate_tool_timeout_used")):
        state["_simulate_tool_timeout_used"] = True
        tool_result = _tool_failed(tool_name, "network timeout")
    # 占位工具实现：按 tool_name 构造模拟证据，保持字段结构稳定。
    elif tool_name == "log_query":
        order_id = str(tool_params.get("order_id") or state.get("order_id") or "")
        trace_id = str(tool_params.get("trace_id") or state.get("trace_id") or "")
        evidence_rows = [
            f"log_query命中：{question[:64]}",
            f"trace_id={trace_id or 'N/A'} order_id={order_id or 'N/A'}",
        ]
        tool_result = _tool_success("log_query", evidence_rows)
    elif tool_name == "dependency_log_query":
        request_id = str(tool_params.get("request_id") or state.get("request_id") or "")
        tool_result = _tool_success(
            "dependency_log_query",
            [f"依赖调用日志：{question[:64]}", f"request_id={request_id or 'N/A'}"],
        )
    elif tool_name == "knowledge_lookup":
        tool_result = _tool_success("knowledge_lookup", [f"知识库证据：{question[:64]}"])
    else:
        tool_result = _tool_success("none", [])

    # 有实际工具动作才推进调用计数与计划步进。
    if tool_name != "none":
        state["tool_call_count"] = tool_call_count + 1
        state["current_step_index"] = _as_int(state.get("current_step_index"), 0) + 1

    state["tool_result"] = tool_result
    history = [dict(item) for item in list(state.get("tool_history") or [])]
    history.append(
        {
            "idx": len(history) + 1,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "ok": bool(tool_result.get("ok")),
            "error": str(tool_result.get("error") or ""),
        }
    )
    state["tool_history"] = history
    state["route"] = "evidence_merge"
    return dict(state)
