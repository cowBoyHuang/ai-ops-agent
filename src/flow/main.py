from __future__ import annotations

from typing import Any

from flow.flow import build_langchain_error_chain, build_langchain_main_chain


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Flow 中间层入口：主链路异常时切换到错误链路。"""
    context = dict(payload)
    try:
        # 主流程：支持节点业务短路（短路由模块自身状态控制，不依赖异常）。
        return build_langchain_main_chain().invoke(context)
    except Exception as exc:  # noqa: BLE001
        # 异常流程：网络失败、运行时错误、非法状态等统一映射到错误链路。
        context["error"] = str(exc)
        return build_langchain_error_chain().invoke(context)
