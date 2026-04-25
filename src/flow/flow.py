"""Main flow chain built with LangChain runnables."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.runnables import Runnable, RunnableLambda

from flow.modules.agent_executor_graph.agent_executor_graph import run as agent_executor_graph_run
from flow.modules.context_build.context_build import run as context_build_run
from flow.modules.duplicate_detect.duplicate_detect import run as duplicate_detect_run
from flow.modules.error_handle.error_handle import run as error_handle_run
from flow.modules.input_validate.input_validate import run as input_validate_run
from flow.modules.memory.memory import run as memory_run
from flow.modules.response_emit.response_emit import run as response_emit_run

_MOD_LOG = logging.getLogger("aiops.flow.module")


def _validate_module_result(module_name: str, before: dict[str, Any], after: Any) -> dict[str, Any]:
    """统一校验模块返回值。"""
    if not isinstance(after, dict):
        raise TypeError(f"{module_name} must return dict, got {type(after).__name__}")
    context = dict(after)
    context.setdefault("status", str(before.get("status") or "init"))
    return context


def _module_node(module_name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    包装模块节点：
    - 校验返回
    - 根据短路规则提前返回
    """

    def _wrapped(payload: dict[str, Any]) -> dict[str, Any]:
        incoming = dict(payload)
        # 全局停止规则：status=finished 时不再处理后续节点。
        if str(incoming.get("status") or "").lower() == "finished":
            _MOD_LOG.info("module.skip name=%s reason=status_finished", module_name)
            return incoming

        _MOD_LOG.info("module.enter name=%s", module_name)
        result = fn(incoming)
        context = _validate_module_result(module_name, incoming, result)
        _MOD_LOG.info("module.exit name=%s status=%s", module_name, str(context.get("status") or ""))
        return context

    return _wrapped


def build_langchain_main_chain() -> Runnable:
    """主链路：显式挂载业务模块节点。"""
    return (
        RunnableLambda(_module_node("input_validate", input_validate_run))
        | RunnableLambda(_module_node("context_build", context_build_run))
        | RunnableLambda(_module_node("duplicate_detect", duplicate_detect_run))
        | RunnableLambda(_module_node("agent_executor_graph", agent_executor_graph_run))
        | RunnableLambda(_module_node("response_emit", response_emit_run))
        | RunnableLambda(_module_node("memory", memory_run))
    )


def build_langchain_error_chain() -> Runnable:
    """异常链路：仅进行错误映射。"""
    return RunnableLambda(error_handle_run)
