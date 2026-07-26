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
        "通用底层日志查询技能，用于固定业务技能仍不足时的扩展兜底检索。"
        "必须传 app_code、logname、begin_time、end_time、match_phrase_list；"
        "match_phrase_list 至少包含 traceId 或 orderNo，且兜底 queryLog 必须 match_list=[]。"
        "match_phrase_list 只允许真实可落库检索的精确标识，禁止占位符或业务短语。"
        "当需要继续查乘机人、旅客、年龄、特殊产品、被拦截原因等明细时，"
        "优先在 app_code=f_tts_trade_core、logname=tts 上扩展检索；"
        "只有确认要查总单聚合返回口径时才查 f_tts_trade_order/ttsorder。"
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
        "依赖链路日志查询工具，用于跨服务、网关、外部依赖或调用方/被调方日志取证。"
        "必须传 app_code、logname、begin_time、end_time。"
        "当用于 queryLog 兜底语义时，match_phrase_list 仍只能放 traceId/orderNo，match_list 必须为 []。"
        "适合确认 trace 是否跨服务传递、依赖侧是否返回错误；"
        "不适合替代机票子单内的乘机人、年龄、特殊产品校验明细查询。"
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
        "当用户追问具体乘机人、旅客、姓名、证件、年龄、儿童年龄限制、特殊产品校验、"
        "被拦截对象、哪位乘机人不满足规则、机票子单内部明细时必须优先使用本工具。"
        "本工具查询子单链路的“单程生单结果”，比总单工具更适合回答乘机人级别失败原因。"
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
        "适用于先判断这笔单最终是否失败、失败文案是什么、总单聚合错误码是什么。"
        "不适合回答具体乘机人、旅客、年龄、证件、被拦截对象等机票子单内部明细；"
        "遇到这些追问应改用 getFlightCreateOrderResult，必要时再用 queryLog 扩展兜底。"
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
