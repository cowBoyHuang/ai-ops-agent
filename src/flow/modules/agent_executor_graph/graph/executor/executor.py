"""Executor 节点：按 Plan-ReAct 模式执行当前计划子任务。"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.executor.sub_executor import (
    run_code_sub_executor,
    run_log_sub_executor,
)
from llm.llm import chat_with_llm, load_prompt, render_prompt

_LOGGER = logging.getLogger(__name__)
_ALLOWED_TOOL_NAMES = {"log_query", "dependency_log_query", "knowledge_lookup", "code_clone", "code_pull", "none"}
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{2,64}")
_MAX_LLM_HISTORY_ROWS = 4
_MAX_SUMMARY_LEN = 300
_MAX_REACT_RETRY_PER_STEP = 3
_LOCAL_TZ = dt.timezone(dt.timedelta(hours=8))
_XEP_ORDER_PATTERN = re.compile(r"\bxep\s*(\d{6})(\d{6})\d*\b", re.IGNORECASE)
_OPS_SLUGGER_PATTERN = re.compile(r"\bops[\s_.-]*slugger[\s_.-]*(\d{6})[\s_.-]*(\d{6})\b", re.IGNORECASE)
_GENERIC_DT_PATTERN = re.compile(r"\b(\d{6})[\s_.-]+(\d{6})\b")
_TRACE_ID_PATTERN = re.compile(r"[a-z]+[_-]slugger[_a-z0-9\.\-]+(?=$|[^A-Za-z0-9_\.\-])", re.IGNORECASE)
_TRACE_KEY_PATTERN = re.compile(r"\btrace[_-]?id\b\s*[:=]?\s*([A-Za-z0-9_.:\-]{4,128})", re.IGNORECASE)
_ORDER_KEY_PATTERN = re.compile(
    r"(?:\border[_-]?(?:id|no)\b|订单号|订单id|订单ID|子单号)\s*[:：=]?\s*([A-Za-z0-9_.:\-]{4,128})",
    re.IGNORECASE,
)
_ORDER_TOKEN_PATTERN = re.compile(r"\bxep\d{12,}\b", re.IGNORECASE)
_SERVICE_TO_APP_CODE = {
    "order-service": "f_tts_trade_order",
    "tts-trade-order": "f_tts_trade_order",
    "f_tts_trade_order": "f_tts_trade_order",
    "trade-order": "f_tts_trade_order",
    "core-service": "f_tts_trade_core",
    "tts-trade-core": "f_tts_trade_core",
    "f_tts_trade_core": "f_tts_trade_core",
    "trade-core": "f_tts_trade_core",
}
_APP_CODE_TO_LOGNAME = {
    "f_tts_trade_order": "ttsorder.log",
    "f_tts_trade_core": "tts.log",
}
_DEPENDENCY_APP_CODE_FALLBACK = {
    "f_tts_trade_order": "f_tts_trade_core",
    "f_tts_trade_core": "f_tts_trade_order",
}
_REACT_TOOL_NAME_MAP = {
    "query_log": "log_query",
    "log_query": "log_query",
    "query_dependency_log": "dependency_log_query",
    "dependency_log_query": "dependency_log_query",
    "knowledge_lookup": "knowledge_lookup",
    "knowledge_search": "knowledge_lookup",
    "fetch_code": "code_pull",
    "code_pull": "code_pull",
    "clone_code": "code_clone",
    "code_clone": "code_clone",
    "none": "none",
    "final_answer": "none",
}
_REACT_TOOL_SCHEMAS = [
    {
        "name": "query_log",
        "description": "查询指定服务在给定时间范围内的运行日志，用于定位异常堆栈与错误关键词。",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string", "description": "微服务名，如 order-service"},
                "time_range": {"type": "string", "description": "时间范围，如 last_10m / last_1h"},
                "keyword": {"type": "string", "description": "日志关键词，如 Exception / Timeout"},
                "match_phrase_list": {"type": "array", "items": {"type": "string"}},
                "match_list": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["service_name", "time_range"],
        },
    },
    {
        "name": "query_dependency_log",
        "description": "查询依赖服务日志，用于沿调用链继续排查。",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string"},
                "time_range": {"type": "string"},
                "keyword": {"type": "string"},
                "match_phrase_list": {"type": "array", "items": {"type": "string"}},
                "match_list": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["service_name", "time_range"],
        },
    },
    {
        "name": "fetch_code",
        "description": "根据仓库、文件路径和行号拉取代码片段，用于结合日志定位根因。",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "description": "仓库名"},
                "file_path": {"type": "string", "description": "文件相对路径"},
                "line_number": {"type": "integer", "description": "行号"},
                "git_url": {"type": "string", "description": "可选仓库地址"},
            },
            "required": ["repo_name", "file_path", "line_number"],
        },
    },
    {
        "name": "knowledge_lookup",
        "description": "检索知识库或排障文档，补充规则与经验性证据。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词"},
            },
            "required": ["query"],
        },
    },
]
_DEFAULT_REACT_SYSTEM_PROMPT = (
    "你是排障系统的执行者（ReAct Agent）。"
    "你会收到当前子任务、上一步观察结果和可用工具列表。"
    "请先思考再决定是否调用工具。只输出 JSON 对象，字段必须包含："
    "thought(string), action(object|null), final_answer(string), advance_plan_step(boolean)。"
    "当证据足够时可直接给 final_answer 并将 action 置为 null。"
)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_text(text: Any, max_len: int = _MAX_SUMMARY_LEN) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_success(tool: str, evidence: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"tool": tool, "ok": True, "error": "", "evidence": evidence}
    if extra:
        payload.update(extra)
    return payload


def _tool_failed(tool: str, error: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"tool": tool, "ok": False, "error": str(error or "unknown_error"), "evidence": []}
    if extra:
        payload.update(extra)
    return payload


def _execute_tool_call(
    tool_name: str,
    tool_params: dict[str, Any],
    state: dict[str, Any],
    structured_context: dict[str, Any],
) -> dict[str, Any]:
    normalized_tool = str(tool_name or "none").strip()
    if normalized_tool not in _ALLOWED_TOOL_NAMES:
        return _tool_failed(normalized_tool or "none", f"unsupported tool: {normalized_tool}")

    tool_call_count = _as_int(state.get("tool_call_count"), 0)
    max_tool_calls = max(1, _as_int(state.get("max_tool_calls"), 8))
    question = str(state.get("question") or "")
    if tool_call_count >= max_tool_calls:
        return _tool_failed(normalized_tool, "max_tool_calls_exceeded", extra={"budget_exhausted": True})

    if bool(structured_context.get("simulate_tool_timeout_once")) and not bool(
        structured_context.get("_simulate_tool_timeout_used")
    ):
        structured_context["_simulate_tool_timeout_used"] = True
        return _tool_failed(normalized_tool, "network timeout")

    if normalized_tool in {"log_query", "dependency_log_query"}:
        begin_time = _pick_non_empty_value(tool_params, ("begin_time", "beginTime", "start_time", "startTime"))
        end_time = _pick_non_empty_value(tool_params, ("end_time", "endTime", "finish_time", "finishTime"))
        effective_context = dict(structured_context or {})
        if begin_time and end_time:
            effective_context["begin_time"] = begin_time
            effective_context["end_time"] = end_time
        return run_log_sub_executor(
            step={"tool_name": normalized_tool, "params": tool_params},
            state=state,
            structured_context=effective_context,
        )

    if normalized_tool == "knowledge_lookup":
        return _tool_success("knowledge_lookup", [f"知识库证据：{question[:64]}"])

    if normalized_tool in {"code_clone", "code_pull"}:
        return run_code_sub_executor(
            step={"tool_name": normalized_tool, "params": tool_params},
            state=state,
            structured_context=structured_context,
        )

    return _tool_success("none", [])


def _build_step_history_preview(execution_history: dict[str, Any]) -> str:
    rows: list[str] = []
    keys = sorted(execution_history.keys(), key=lambda item: _as_int(str(item).split("_")[-1], 0))
    for key in keys[-_MAX_LLM_HISTORY_ROWS:]:
        item = dict(execution_history.get(key) or {})
        step = dict(item.get("step") or {})
        executed_step = dict(item.get("executed_step") or {})
        raw_result = dict(item.get("raw_result") or {})
        processed = dict(item.get("processed") or {})
        rows.append(
            " | ".join(
                [
                    key,
                    f"plan_action={step.get('action_type') or 'tool_call'}",
                    f"tool={executed_step.get('tool_name') or raw_result.get('tool') or 'none'}",
                    f"ok={bool(raw_result.get('ok'))}",
                    f"summary={_clip_text(processed.get('summary'), 120)}",
                ]
            )
        )
    return "\n".join(rows).strip() or "无历史步骤"


def _fallback_keywords(step: dict[str, Any], raw_result: dict[str, Any]) -> list[str]:
    rows = [
        str(step.get("tool_name") or ""),
        str(raw_result.get("error") or ""),
        " ".join(str(item) for item in list(raw_result.get("evidence") or [])[:3]),
    ]
    merged = " ".join(rows)
    tokens = {token for token in _KEYWORD_PATTERN.findall(merged) if len(token) >= 3}
    return sorted(tokens)[:20]


def _process_step_result_with_llm(
    step: dict[str, Any],
    raw_result: dict[str, Any],
    execution_history: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = load_prompt("plan_execute_step_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "plan_execute_step_user_prompt.txt",
        step_json=json.dumps(step, ensure_ascii=False),
        result_json=json.dumps(raw_result, ensure_ascii=False),
        history_preview=_build_step_history_preview(execution_history),
    )
    fallback = {
        "summary": _clip_text(raw_result.get("error") or "步骤执行完成"),
        "extracted_keywords": _fallback_keywords(step, raw_result),
        "structured_facts": {},
        "retry_current_step": False,
        "retry_params": {},
        "continue_execution": bool(raw_result.get("ok")),
        "next_step_guidance": "",
    }
    if not user_prompt:
        return fallback
    llm_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    parsed = _parse_json_object(llm_output)
    if not parsed:
        return fallback
    keywords = [str(item).strip() for item in list(parsed.get("extracted_keywords") or []) if str(item).strip()]
    return {
        "summary": _clip_text(parsed.get("summary") or fallback["summary"]),
        "extracted_keywords": sorted(set(keywords))[:30],
        "structured_facts": dict(parsed.get("structured_facts") or {}),
        "retry_current_step": bool(parsed.get("retry_current_step")),
        "retry_params": dict(parsed.get("retry_params") or {}),
        "continue_execution": bool(parsed.get("continue_execution", bool(raw_result.get("ok")))),
        "next_step_guidance": _clip_text(parsed.get("next_step_guidance"), 200),
    }


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _extract_exact_identifiers(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    hits: list[str] = []
    for match in _TRACE_ID_PATTERN.findall(raw):
        value = str(match or "").strip()
        if value and value not in hits:
            hits.append(value)
    for pattern in (_TRACE_KEY_PATTERN, _ORDER_KEY_PATTERN):
        for match in pattern.findall(raw):
            value = str(match or "").strip()
            if value and value not in hits:
                hits.append(value)
    for match in _ORDER_TOKEN_PATTERN.findall(raw):
        value = str(match or "").strip()
        if value and value not in hits:
            hits.append(value)
    return hits


def _collect_forced_match_phrase_terms(params: dict[str, Any], state: dict[str, Any], step: dict[str, Any]) -> list[str]:
    structured_context = dict(state.get("structured_context") or {})
    candidates = [
        str(params.get("trace_id") or ""),
        str(params.get("traceId") or ""),
        str(params.get("order_id") or ""),
        str(params.get("orderId") or ""),
        str(params.get("order_no") or ""),
        str(params.get("orderNo") or ""),
        str(params.get("request_id") or ""),
        str(params.get("requestId") or ""),
        str(structured_context.get("trace_id") or ""),
        str(structured_context.get("traceId") or ""),
        str(structured_context.get("order_id") or ""),
        str(structured_context.get("orderId") or ""),
        str(structured_context.get("order_no") or ""),
        str(structured_context.get("orderNo") or ""),
        str(structured_context.get("request_id") or ""),
        str(structured_context.get("requestId") or ""),
        str(params.get("query") or ""),
        " ".join(_normalize_str_list(params.get("keywords"))),
        str(params.get("keyword") or ""),
        " ".join(_normalize_str_list(params.get("match_list"))),
        str(step.get("subtask") or ""),
        str(step.get("hypothesis") or ""),
        str(step.get("success_criteria") or ""),
        str(structured_context.get("user_query") or ""),
        str(state.get("question") or ""),
    ]
    forced_terms: list[str] = []
    for text in candidates:
        for token in _extract_exact_identifiers(text):
            if token not in forced_terms:
                forced_terms.append(token)
    return forced_terms


def _to_time_window(time_range: str) -> tuple[str, str] | None:
    text = str(time_range or "").strip().lower()
    if not text:
        return None
    minutes_map = {
        "last_10m": 10,
        "10m": 10,
        "last_30m": 30,
        "30m": 30,
        "last_1h": 60,
        "1h": 60,
        "last_2h": 120,
        "2h": 120,
    }
    minutes = minutes_map.get(text)
    if minutes is None:
        return None
    now = dt.datetime.now(_LOCAL_TZ)
    begin = now - dt.timedelta(minutes=minutes)
    return begin.isoformat(), now.isoformat()


def _build_time_from_yymmdd_hhmmss(yymmdd: str, hhmmss: str) -> dt.datetime | None:
    if len(yymmdd) != 6 or len(hhmmss) != 6:
        return None
    try:
        year = 2000 + int(yymmdd[0:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        hour = int(hhmmss[0:2])
        minute = int(hhmmss[2:4])
        second = int(hhmmss[4:6])
        return dt.datetime(year, month, day, hour, minute, second, tzinfo=_LOCAL_TZ)
    except ValueError:
        return None


def _parse_datetime_text(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _extract_event_time_from_text(text: str) -> dt.datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    for pattern in (_XEP_ORDER_PATTERN, _OPS_SLUGGER_PATTERN, _GENERIC_DT_PATTERN):
        matched = pattern.search(raw)
        if not matched:
            continue
        event_time = _build_time_from_yymmdd_hhmmss(str(matched.group(1)), str(matched.group(2)))
        if event_time is not None:
            return event_time
    return None


def _pick_non_empty_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _infer_log_event_time(params: dict[str, Any], state: dict[str, Any], step: dict[str, Any]) -> dt.datetime | None:
    previous_step_result = dict(state.get("current_step_result") or {})
    previous_raw_result = dict(previous_step_result.get("raw_result") or {})
    previous_evidence = " ".join(str(item) for item in list(previous_raw_result.get("evidence") or [])[:5])
    query_terms = " ".join(
        [
            *_normalize_str_list(params.get("match_phrase_list")),
            *_normalize_str_list(params.get("match_list")),
            str(params.get("query") or ""),
            str(params.get("keyword") or ""),
        ]
    )
    structured_context = dict(state.get("structured_context") or {})
    candidates = [
        query_terms,
        str(step.get("subtask") or ""),
        str(step.get("hypothesis") or ""),
        str(step.get("success_criteria") or ""),
        str(structured_context.get("user_query") or ""),
        str(state.get("question") or ""),
        previous_evidence,
    ]
    for item in candidates:
        event_time = _extract_event_time_from_text(item)
        if event_time is not None:
            return event_time
    return None


def _ensure_log_query_time_window(params: dict[str, Any], state: dict[str, Any], step: dict[str, Any]) -> None:
    event_time = _infer_log_event_time(params, state, step)
    if event_time is not None:
        params["begin_time"] = (event_time - dt.timedelta(hours=1)).isoformat()
        params["end_time"] = (event_time + dt.timedelta(hours=1)).isoformat()
        return

    time_range = _pick_non_empty_value(params, ("time_range", "timeRange", "window", "time_window"))
    range_window = _to_time_window(time_range) if time_range else None
    if range_window:
        params["begin_time"], params["end_time"] = range_window
        return

    begin_time = _pick_non_empty_value(params, ("begin_time", "beginTime", "start_time", "startTime"))
    end_time = _pick_non_empty_value(params, ("end_time", "endTime", "finish_time", "finishTime"))
    alias_window: tuple[str, str] | None = None
    if begin_time and not end_time:
        alias_window = _to_time_window(begin_time)
    elif end_time and not begin_time:
        alias_window = _to_time_window(end_time)
    if alias_window:
        params["begin_time"], params["end_time"] = alias_window
        return

    begin_dt = _parse_datetime_text(begin_time)
    end_dt = _parse_datetime_text(end_time)
    if begin_dt is not None and end_dt is not None:
        params["begin_time"] = begin_dt.isoformat()
        params["end_time"] = end_dt.isoformat()
        return

    structured_context = dict(state.get("structured_context") or {})
    if begin_dt is None:
        begin_time = _pick_non_empty_value(structured_context, ("begin_time", "beginTime", "start_time", "startTime"))
    if end_dt is None:
        end_time = _pick_non_empty_value(structured_context, ("end_time", "endTime", "finish_time", "finishTime"))
    begin_dt = begin_dt or _parse_datetime_text(begin_time)
    end_dt = end_dt or _parse_datetime_text(end_time)

    now = dt.datetime.now(_LOCAL_TZ)
    if begin_dt is None:
        begin_dt = now - dt.timedelta(hours=1)
    if end_dt is None:
        end_dt = now + dt.timedelta(hours=1)
    params["begin_time"] = begin_dt.isoformat()
    params["end_time"] = end_dt.isoformat()


def _pick_recent_log_app_code(state: dict[str, Any]) -> str:
    history = list(state.get("tool_history") or [])
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        if tool_name not in {"log_query", "dependency_log_query"}:
            continue
        params = dict(item.get("tool_params") or {})
        app_code = str(params.get("app_code") or "").strip().lower()
        if app_code in _APP_CODE_TO_LOGNAME:
            return app_code
    return ""


def _infer_app_code_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    core_tokens = ("f_tts_trade_core", "trade_core", "trade-core", "tts.log", "core-service", "tts-trade-core")
    order_tokens = ("f_tts_trade_order", "trade_order", "trade-order", "ttsorder.log", "order-service", "tts-trade-order")
    if any(token in lowered for token in core_tokens):
        return "f_tts_trade_core"
    if any(token in lowered for token in order_tokens):
        return "f_tts_trade_order"
    return ""


def _infer_log_query_target(
    *,
    tool_name: str,
    params: dict[str, Any],
    state: dict[str, Any],
    step: dict[str, Any],
) -> None:
    structured_context = dict(state.get("structured_context") or {})
    explicit_app_code = str(params.get("app_code") or params.get("appCode") or "").strip().lower()
    app_code = str(
        explicit_app_code
        or structured_context.get("app_code")
        or ""
    ).strip().lower()
    logname = str(params.get("logname") or structured_context.get("logname") or "").strip().lower()

    service_name = str(params.get("service_name") or params.get("service") or "").strip().lower()
    if service_name and not app_code:
        app_code = str(_SERVICE_TO_APP_CODE.get(service_name) or "").strip().lower()

    if not app_code:
        merged_text = " ".join(
            [
                str(step.get("subtask") or ""),
                str(step.get("hypothesis") or ""),
                str(step.get("success_criteria") or ""),
                str(state.get("question") or ""),
                " ".join(_normalize_str_list(params.get("match_phrase_list"))),
                " ".join(_normalize_str_list(params.get("match_list"))),
                str(params.get("query") or ""),
            ]
        )
        app_code = _infer_app_code_from_text(merged_text)

    recent_app_code = _pick_recent_log_app_code(state) if tool_name == "dependency_log_query" else ""
    if tool_name == "dependency_log_query" and recent_app_code and not explicit_app_code:
        if not app_code or app_code == recent_app_code:
            app_code = str(_DEPENDENCY_APP_CODE_FALLBACK.get(recent_app_code) or app_code)

    if not app_code:
        app_code = "f_tts_trade_core" if tool_name == "dependency_log_query" else "f_tts_trade_order"

    if not logname:
        logname = str(_APP_CODE_TO_LOGNAME.get(app_code) or "")

    params["app_code"] = app_code
    if logname:
        params["logname"] = logname


def _normalize_react_tool_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return ""
    lowered = name.lower()
    if lowered in _ALLOWED_TOOL_NAMES:
        return lowered
    return str(_REACT_TOOL_NAME_MAP.get(lowered) or "")


def _normalize_react_params(
    tool_name: str,
    raw_params: dict[str, Any],
    state: dict[str, Any],
    step: dict[str, Any],
) -> dict[str, Any]:
    params = dict(raw_params or {})
    if tool_name in {"log_query", "dependency_log_query"}:
        phrase_terms = _normalize_str_list(params.get("match_phrase_list"))
        fuzzy_terms = _normalize_str_list(params.get("match_list"))
        forced_phrase_terms = _collect_forced_match_phrase_terms(params=params, state=state, step=step)
        for item in forced_phrase_terms:
            if item not in phrase_terms:
                phrase_terms.append(item)
        params["match_phrase_list"] = phrase_terms
        params["match_list"] = fuzzy_terms

        service_name = str(params.get("service_name") or params.get("service") or "").strip().lower()
        if service_name and not str(params.get("app_code") or "").strip():
            mapped_code = _SERVICE_TO_APP_CODE.get(service_name, "")
            if mapped_code:
                params["app_code"] = mapped_code
        _infer_log_query_target(tool_name=tool_name, params=params, state=state, step=step)

        _ensure_log_query_time_window(params, state, step)

    if tool_name in {"code_pull", "code_clone"}:
        file_path = str(params.get("file_path") or "").strip()
        line_number = params.get("line_number")
        if file_path and line_number and not str(params.get("search_hint") or "").strip():
            params["search_hint"] = f"{file_path}:{line_number}"
        if not str(params.get("repo_name") or "").strip():
            repo_name = str(params.get("repo") or params.get("service_name") or "").strip()
            if repo_name:
                params["repo_name"] = repo_name
    return params


def _build_react_subtask_payload(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": str(step.get("action_type") or "tool_call"),
        "subtask": str(step.get("subtask") or ""),
        "hypothesis": str(step.get("hypothesis") or ""),
        "success_criteria": str(step.get("success_criteria") or ""),
        "suggested_tools": list(step.get("suggested_tools") or []),
        "planned_tool": str(step.get("tool_name") or ""),
        "planned_params": dict(step.get("params") or {}),
    }


def _infer_default_tool_for_subtask(step: dict[str, Any], state: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(step.get("subtask") or ""),
            str(step.get("hypothesis") or ""),
            str(step.get("success_criteria") or ""),
            str(state.get("question") or ""),
        ]
    ).lower()
    log_hints = (
        "日志",
        "log",
        "trace",
        "traceid",
        "ops_slugger",
        "xep",
        "生单",
        "ordercreateresultmap",
        "单程生单结果",
        "往返生单结果",
    )
    code_hints = ("代码", "code", "file", "文件", "行号", "方法", "类")
    if any(token in text for token in log_hints):
        return "log_query"
    if any(token in text for token in code_hints):
        return "code_pull"
    return "knowledge_lookup"


def _default_react_decision(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    action_type = str(step.get("action_type") or "tool_call")
    if action_type == "tool_call":
        tool_name = str(step.get("tool_name") or "knowledge_lookup")
        params = dict(step.get("params") or {})
        return {
            "thought": "按既定计划执行当前步骤。",
            "action": {"tool_name": tool_name, "params": params},
            "internal_tool_name": tool_name if tool_name in _ALLOWED_TOOL_NAMES else "knowledge_lookup",
            "action_params": params,
            "final_answer": "",
            "advance_plan_step": True,
            "parse_ok": False,
            "raw_output": "",
        }

    suggested_tools = [str(item).strip() for item in list(step.get("suggested_tools") or []) if str(item).strip()]
    selected_tool = _normalize_react_tool_name(suggested_tools[0] if suggested_tools else "")
    if not selected_tool:
        selected_tool = _infer_default_tool_for_subtask(step, state)
    return {
        "thought": "先执行当前子任务的首选工具，再根据观察结果动态调整。",
        "action": {"tool_name": suggested_tools[0] if suggested_tools else selected_tool, "params": {}},
        "internal_tool_name": selected_tool,
        "action_params": {},
        "final_answer": "",
        "advance_plan_step": True,
        "parse_ok": False,
        "raw_output": "",
    }


def _decide_react_action(
    *,
    step: dict[str, Any],
    state: dict[str, Any],
    execution_history: dict[str, Any],
) -> dict[str, Any]:
    fallback = _default_react_decision(step, state)
    fallback["action_params"] = _normalize_react_params(
        str(fallback.get("internal_tool_name") or ""),
        dict(fallback.get("action_params") or {}),
        state,
        step,
    )
    system_prompt = load_prompt("executor_react_system_prompt.txt", default=_DEFAULT_REACT_SYSTEM_PROMPT)
    previous_observation = dict(state.get("current_step_result") or {})
    user_prompt = render_prompt(
        "executor_react_user_prompt.txt",
        question=str(state.get("question") or ""),
        current_subtask_json=json.dumps(_build_react_subtask_payload(step), ensure_ascii=False),
        previous_observation_json=json.dumps(previous_observation, ensure_ascii=False),
        tool_schemas_json=json.dumps(_REACT_TOOL_SCHEMAS, ensure_ascii=False),
        history_preview=_build_step_history_preview(execution_history),
    )
    if not user_prompt:
        return fallback

    llm_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    parsed = _parse_json_object(llm_output)
    if not parsed:
        fallback["raw_output"] = llm_output
        return fallback

    action_payload = parsed.get("action")
    tool_name = ""
    params: dict[str, Any] = {}
    if isinstance(action_payload, dict):
        tool_name = str(
            action_payload.get("tool_name")
            or action_payload.get("name")
            or action_payload.get("function")
            or ""
        ).strip()
        params = dict(action_payload.get("params") or action_payload.get("arguments") or {})
    else:
        tool_name = str(parsed.get("tool_name") or "").strip()
        params = dict(parsed.get("params") or {})

    final_answer = str(parsed.get("final_answer") or parsed.get("answer") or "").strip()
    advance_plan_step = bool(parsed.get("advance_plan_step", True))
    if bool(parsed.get("continue_current_subtask")):
        advance_plan_step = False

    internal_tool_name = _normalize_react_tool_name(tool_name)
    if not internal_tool_name and final_answer:
        internal_tool_name = "none"
    if not internal_tool_name and not final_answer:
        fallback["raw_output"] = llm_output
        return fallback
    if internal_tool_name == "none" and not final_answer:
        fallback["raw_output"] = llm_output
        return fallback
    if internal_tool_name == "none":
        advance_plan_step = True

    normalized_params = _normalize_react_params(internal_tool_name, params, state, step)
    return {
        "thought": str(parsed.get("thought") or parsed.get("reasoning") or "").strip(),
        "action": {"tool_name": tool_name, "params": params},
        "internal_tool_name": internal_tool_name,
        "action_params": normalized_params,
        "final_answer": final_answer,
        "advance_plan_step": advance_plan_step,
        "parse_ok": True,
        "raw_output": llm_output,
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    current_plan = list(state.get("current_plan") or state.get("plan_steps") or [])
    current_step_index = _as_int(state.get("current_step_index"), 0)
    _LOGGER.info("executor.run step_index=%d plan_len=%d", current_step_index, len(current_plan))
    execution_history = dict(state.get("execution_history") or {})
    intermediate_results = dict(state.get("intermediate_results") or {})
    extracted_keywords = {str(item).strip() for item in list(state.get("extracted_keywords") or []) if str(item).strip()}
    structured_context = dict(state.get("structured_context") or {})

    if current_step_index >= len(current_plan):
        state["current_plan"] = current_plan
        state["plan_steps"] = current_plan
        state["current_step_result"] = {
            "step_index": current_step_index,
            "step": {},
            "raw_result": _tool_success("none", []),
            "processed": {"summary": "no step to execute", "extracted_keywords": [], "structured_facts": {}},
        }
        state["newly_discovered_clues"] = []
        state["route"] = "observer"
        return dict(state)

    raw_step = current_plan[current_step_index]
    step = dict(raw_step) if isinstance(raw_step, dict) else {}
    step.setdefault("action_type", "tool_call")
    action_type = str(step.get("action_type") or "tool_call")

    if action_type == "merge_evidence":
        raw_result = _tool_success("none", [], extra={"action_type": "merge_evidence"})
        processed = {
            "summary": "merge_evidence marker",
            "extracted_keywords": [],
            "structured_facts": {},
            "retry_current_step": False,
            "retry_params": {},
            "continue_execution": True,
            "next_step_guidance": "",
        }
        executed_step = {"action_type": "merge_evidence", "tool_name": None, "params": {}}
        react_decision = {
            "thought": "当前步骤是证据归并标记，无需调用工具。",
            "action": None,
            "final_answer": "",
            "advance_plan_step": True,
            "parse_ok": True,
            "raw_output": "",
        }
    else:
        react_decision = _decide_react_action(step=step, state=state, execution_history=execution_history)
        tool_name = str(react_decision.get("internal_tool_name") or "none")
        tool_params = dict(react_decision.get("action_params") or {})
        tool_params.setdefault("query", state.get("question") or "")
        tool_params.setdefault("order_id", structured_context.get("order_id") or "")
        tool_params.setdefault("request_id", structured_context.get("request_id") or "")
        executed_step = {"action_type": "tool_call", "tool_name": tool_name, "params": tool_params}

        final_answer = str(react_decision.get("final_answer") or "").strip()
        if tool_name == "none":
            evidence = [f"[react_final_answer] {final_answer}"] if final_answer else []
            raw_result = _tool_success("none", evidence, extra={"react_final_answer": final_answer})
        else:
            raw_result = _execute_tool_call(
                tool_name=tool_name,
                tool_params=tool_params,
                state=state,
                structured_context=structured_context,
            )
            state["tool_call_count"] = _as_int(state.get("tool_call_count"), 0) + 1

        processed = _process_step_result_with_llm(step=executed_step, raw_result=raw_result, execution_history=execution_history)
        if final_answer and not str(processed.get("summary") or "").strip():
            processed["summary"] = _clip_text(final_answer)

        state["tool_name"] = tool_name
        state["tool_params"] = tool_params

        history = [dict(item) for item in list(state.get("tool_history") or [])]
        history.append(
            {
                "idx": len(history) + 1,
                "tool_name": tool_name,
                "tool_params": tool_params,
                "ok": bool(raw_result.get("ok")),
                "error": str(raw_result.get("error") or ""),
                "thought": str(react_decision.get("thought") or ""),
                "advance_plan_step": bool(react_decision.get("advance_plan_step", True)),
            }
        )
        state["tool_history"] = history

    step_key = f"step_{current_step_index}"
    execution_history[step_key] = {
        "index": current_step_index,
        "step": step,
        "executed_step": executed_step,
        "react_decision": react_decision,
        "raw_result": raw_result,
        "processed": processed,
    }
    intermediate_results[step_key] = {
        "summary": str(processed.get("summary") or ""),
        "structured_facts": dict(processed.get("structured_facts") or {}),
        "tool_ok": bool(raw_result.get("ok")),
    }
    clues = [str(item).strip() for item in list(processed.get("extracted_keywords") or []) if str(item).strip()]
    extracted_keywords.update(clues)

    advance_plan_step = bool(react_decision.get("advance_plan_step", True))
    in_place_retry_count = _as_int(state.get("in_place_retry_count"), 0)
    if advance_plan_step:
        in_place_retry_count = 0
        next_step_index = current_step_index + 1
    else:
        in_place_retry_count += 1
        if in_place_retry_count >= _MAX_REACT_RETRY_PER_STEP:
            next_step_index = current_step_index + 1
            in_place_retry_count = 0
        else:
            next_step_index = current_step_index

    state["in_place_retry_count"] = in_place_retry_count
    state["execution_history"] = execution_history
    state["intermediate_results"] = intermediate_results
    state["extracted_keywords"] = sorted(extracted_keywords)
    state["current_plan"] = current_plan
    state["plan_steps"] = current_plan
    state["current_step_result"] = {
        "step_index": current_step_index,
        "step": step,
        "executed_step": executed_step,
        "react_decision": react_decision,
        "raw_result": raw_result,
        "processed": processed,
    }
    state["newly_discovered_clues"] = clues
    state["tool_result"] = raw_result
    state["structured_context"] = {
        **structured_context,
        "react_last_step": {
            "step_index": current_step_index,
            "step": step,
            "executed_step": executed_step,
            "decision": {
                "thought": str(react_decision.get("thought") or ""),
                "action": react_decision.get("action"),
                "final_answer": str(react_decision.get("final_answer") or ""),
                "advance_plan_step": bool(react_decision.get("advance_plan_step", True)),
            },
            "observation": raw_result,
        },
    }
    state["current_step_index"] = next_step_index
    state["route"] = "observer"
    _LOGGER.info(
        "executor 单步执行完成: step=%d tool=%s ok=%s advance=%s",
        current_step_index,
        str(executed_step.get("tool_name") or "none"),
        bool(raw_result.get("ok")),
        bool(react_decision.get("advance_plan_step", True)),
    )
    return dict(state)
