"""Main flow chain built with LangChain runnables."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from langchain_core.runnables import Runnable, RunnableLambda

from flow.modules.agent_executor_subgraph.agent_executor_subgraph import run as agent_executor_subgraph_run
from flow.modules.context_build.context_build import run as context_build_run
from flow.modules.duplicate_detect.duplicate_detect import run as duplicate_detect_run
from flow.modules.error_handle.error_handle import run as error_handle_run
from flow.modules.input_validate.input_validate import run as input_validate_run
from flow.modules.memory.memory import run as memory_run
from flow.modules.observability.observability import run as observability_run
from flow.modules.request_ingest.request_ingest import run as request_ingest_run
from flow.modules.response_emit.response_emit import run as response_emit_run


class FlowShortCircuitError(RuntimeError):
    """短路异常：节点返回 result=False 或空值时抛出。"""


def _ensure_shared_context(payload: dict[str, Any]) -> dict[str, Any]:
    """注入全链路共享上下文对象（同一引用）。"""
    state = dict(payload)
    ctx = state.get("_chain_ctx")
    if not isinstance(ctx, dict):
        ctx = {
            "request_id": f"req_{uuid4().hex[:12]}",
            "started_at": datetime.utcnow().isoformat(timespec="seconds"),
            "visited_nodes": [],
        }
    ctx["chat_id"] = str(state.get("chat_id") or state.get("chatId") or "")
    ctx["user_id"] = str(state.get("user_id") or state.get("userId") or "")
    state["_chain_ctx"] = ctx
    return state


def _validate_module_result(module_name: str, before: dict[str, Any], after: Any) -> dict[str, Any]:
    """统一校验模块返回值，并强制继承共享上下文。"""
    if not isinstance(after, dict):
        raise TypeError(f"{module_name} must return dict, got {type(after).__name__}")
    state = dict(after)
    state["_chain_ctx"] = before["_chain_ctx"]
    state.setdefault("status", str(before.get("status") or "running"))
    return state


def _is_empty_result(value: Any) -> bool:
    if value is None:
        return True
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _assert_node_result(module_name: str, state: dict[str, Any]) -> None:
    """
    每个节点执行后统一校验 `result`：
    - `result` 为 False
    - `result` 为空（None/空字符串/空集合）
    即抛出短路异常。
    """
    if "result" not in state:
        state["result"] = True
    result_value = state.get("result")
    if _is_empty_result(result_value):
        raise FlowShortCircuitError(f"{module_name} result is empty or false")


def _module_node(module_name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    包装模块节点：
    - 注入/复用同一个 `_chain_ctx`
    - 记录访问节点
    - 校验返回
    - 根据短路规则提前返回（仅保留 error/emit/observability 收尾节点）
    """

    def _wrapped(payload: dict[str, Any]) -> dict[str, Any]:
        incoming = _ensure_shared_context(payload)
        ctx = incoming["_chain_ctx"]
        if isinstance(ctx, dict):
            ctx.setdefault("visited_nodes", []).append(module_name)
            ctx["current_node"] = module_name

        result = fn(incoming)
        state = _validate_module_result(module_name, incoming, result)
        _assert_node_result(module_name, state)
        return state

    return _wrapped


def _error_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """仅在存在 error 时执行 error_handle。"""
    state = dict(payload)
    if state.get("error"):
        return error_handle_run(state)
    return state


def build_langchain_main_chain() -> Runnable:
    """主链路：显式挂载全部模块节点。"""
    return (
        RunnableLambda(_module_node("request_ingest", request_ingest_run))
        | RunnableLambda(_module_node("input_validate", input_validate_run))
        | RunnableLambda(_module_node("duplicate_detect", duplicate_detect_run))
        | RunnableLambda(_module_node("context_build", context_build_run))
        | RunnableLambda(_module_node("agent_executor_subgraph", agent_executor_subgraph_run))
        | RunnableLambda(_module_node("error_handle", _error_gate))
        | RunnableLambda(_module_node("response_emit", response_emit_run))
        | RunnableLambda(_module_node("observability", observability_run))
        | RunnableLambda(_module_node("memory", memory_run))
    )


def build_langchain_error_chain() -> Runnable:
    """异常链路：仅进行错误映射。"""
    return RunnableLambda(_ensure_shared_context) | RunnableLambda(_module_node("error_handle", error_handle_run))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    try:
        return build_langchain_main_chain().invoke(state)
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)
        # 主链路按 result 触发短路后，异常链路需继续执行兜底返回，
        # 因此在进入异常链前将 result 置为可通过状态。
        state["result"] = True
        return build_langchain_error_chain().invoke(state)

