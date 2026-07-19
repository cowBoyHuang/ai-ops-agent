"""Annotated log tools used by the agent runtime."""

from __future__ import annotations

import datetime as dt

from langchain_core.tools import tool

from log.log import (
    dependency_log_query,
    get_create_order_result,
    get_flight_create_order_result,
    query_log,
)


@tool(
    "queryLog",
    description=(
        "通用底层日志查询技能。必须传 app_code、logname、begin_time、end_time、match_phrase_list。"
        "兜底 queryLog 必须满足 match_phrase_list 至少包含 traceId 或 orderNo，且 match_list=[]；"
        "match_phrase_list 只允许真实可落库检索的精确标识，禁止占位符或业务短语。"
    ),
)
def query_log_tool(
    app_code: str,
    logname: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    match_phrase_list: list[str] | None = None,
    match_list: list[str] | None = None,
) -> list:
    """Query logs with explicit app/log scope and structured terms."""
    return query_log(
        app_code=app_code,
        logname=logname,
        begin_time=begin_time,
        end_time=end_time,
        match_phrase_list=match_phrase_list or [],
        match_list=match_list or [],
    )


@tool(
    "dependency_log_query",
    description=(
        "依赖链路日志查询工具。用于跨服务或依赖侧日志取证；必须传 app_code、logname、begin_time、end_time。"
        "当用于 queryLog 兜底语义时，match_phrase_list 仍只能放 traceId/orderNo，match_list 必须为 []。"
    ),
)
def dependency_log_query_tool(
    app_code: str,
    logname: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    match_phrase_list: list[str] | None = None,
    match_list: list[str] | None = None,
) -> list:
    """Query dependency logs with explicit app/log scope."""
    return dependency_log_query(
        app_code=app_code,
        logname=logname,
        begin_time=begin_time,
        end_time=end_time,
        match_phrase_list=match_phrase_list or [],
        match_list=match_list or [],
    )


@tool(
    "getFlightCreateOrderResult",
    description=(
        "机票子单生单结果专用日志技能。固定 app_code=f_tts_trade_core、logname=tts，"
        "不允许上层覆盖 app/log。适合定位机票子单执行、校验、路由、创建阶段的失败语义与业务错误码来源。"
        "入参只需要 trace_id、begin_time、end_time。"
    ),
)
def get_flight_create_order_result_tool(
    trace_id: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
) -> list:
    """Query fixed-scope flight sub-order creation result logs."""
    return get_flight_create_order_result(
        trace_id=trace_id,
        begin_time=begin_time,
        end_time=end_time,
    )


@tool(
    "getCreateOrderResult",
    description=(
        "总单生单结果专用日志技能。固定 app_code=f_tts_trade_order、logname=ttsorder，"
        "不允许上层覆盖 app/log。适合确认最终返回态、聚合后的错误口径，以及返回给调用方的总单生单结果。"
        "入参只需要 trace_id、begin_time、end_time。"
    ),
)
def get_create_order_result_tool(
    trace_id: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
) -> list:
    """Query fixed-scope aggregate order creation result logs."""
    return get_create_order_result(
        trace_id=trace_id,
        begin_time=begin_time,
        end_time=end_time,
    )


LOG_TOOLS = [
    query_log_tool,
    dependency_log_query_tool,
    get_flight_create_order_result_tool,
    get_create_order_result_tool,
]
