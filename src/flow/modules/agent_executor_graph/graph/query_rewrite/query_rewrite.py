"""Query rewrite 节点。"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm, load_prompt, render_prompt

_TRACE_ID_PATTERN = re.compile(
    r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+|f_athena_gateway_[a-z0-9_.\-]+)(?=$|[^A-Za-z0-9_\.\-])",
    re.IGNORECASE,
)
_ORDER_ID_PATTERN = re.compile(r"\b(?:xep|sid|fod|hpv)[A-Za-z0-9]{6,}\b", re.IGNORECASE)
_YYMMDD_HHMMSS_PATTERN = re.compile(r"(?<!\d)(\d{6})[\s_.-]+(\d{6})(?!\d)")
_QUERY_REWRITE_SYSTEM_PROMPT = "query_rewrite_system_prompt.txt"
_QUERY_REWRITE_USER_PROMPT = "query_rewrite_user_prompt.txt"
_LOGGER = logging.getLogger(__name__)
_PLACEHOLDER_ENTITY_VALUES = {
    "traceid",
    "requestid",
    "spanid",
    "orderid",
    "orderno",
    "suborderno",
    "id",
    "订单号",
    "订单id",
    "子单号",
}

_KEYWORD_STOPWORDS = {
    "traceid",
    "trace_id",
    "ops",
    "slugger",
    "xep",
    "订单",
    "问题",
    "失败",
    "异常",
}


def _clip(value: Any, max_len: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _round_tools_context_lines(tools_context: dict[str, Any]) -> list[str]:
    if not isinstance(tools_context, dict):
        return []
    rows: list[str] = []
    raw_text = json.dumps(tools_context, ensure_ascii=False, default=str)
    trace_match = _TRACE_ID_PATTERN.search(raw_text)
    order_match = _ORDER_ID_PATTERN.search(raw_text)
    trace_id = str(trace_match.group(0) if trace_match else "").strip()
    order_id = str(order_match.group(0) if order_match else "").strip()
    if trace_id or order_id:
        rows.append(f"历史继承上下文：trace_id={trace_id} order_id={order_id}".strip())

    resolved = dict(tools_context.get("required_answer_resolved") or {})
    if resolved:
        rows.append(f"历史字段结果：{_clip(json.dumps(resolved, ensure_ascii=False), 500)}")

    last_tool = dict(tools_context.get("last_tool") or {})
    if last_tool:
        tool_name = str(last_tool.get("tool_name") or "").strip()
        summary = _clip(last_tool.get("result_summary"), 320)
        conclusion = str(last_tool.get("conclusion") or "").strip()
        if tool_name or summary or conclusion:
            rows.append(
                f"历史工具结果：tool={tool_name} summary={summary} conclusion={_clip(conclusion, 180)}".strip()
            )

    conclusion = str(tools_context.get("conclusion") or "").strip()
    if conclusion:
        rows.append(f"历史结论：{_clip(conclusion, 400)}")
    return rows


def _build_history_context_text(state: dict[str, Any], *, limit: int = 4) -> str:
    rows: list[str] = []
    conversation_rows = [str(item).strip() for item in list(state.get("conversation_context") or []) if str(item).strip()]
    if conversation_rows:
        rows.extend(conversation_rows[-max(1, limit) :])

    raw_context = dict(state.get("context") or {})
    message_context = raw_context.get("message_context")
    round_rows: list[Any] = []
    if hasattr(message_context, "rounds"):
        round_rows = list(getattr(message_context, "rounds") or [])
    elif isinstance(message_context, dict):
        round_rows = list(message_context.get("rounds") or [])
    for row in round_rows[-max(1, limit) :]:
        item = dict(row or {}) if isinstance(row, dict) else {
            "message": getattr(row, "message", ""),
            "aiResponse": getattr(row, "aiResponse", ""),
            "toolsContext": getattr(row, "toolsContext", {}),
        }
        user_text = str(item.get("message") or "").strip()
        ai_text = str(item.get("aiResponse") or "").strip()
        tools_context = dict(item.get("toolsContext") or {})
        if user_text:
            rows.append(f"用户：{user_text}")
        if ai_text:
            rows.append(f"助手：{ai_text}")
        rows.extend(_round_tools_context_lines(tools_context))

    compact: list[str] = []
    seen: set[str] = set()
    for text in rows[-max(1, limit * 2) :]:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        compact.append(key)
    return "\n".join(compact) if compact else "无"


def _pick_question(state: dict[str, Any]) -> str:
    question = str(state.get("question") or "").strip()
    if question:
        return question
    structured = dict(state.get("structured_context") or {})
    return str(structured.get("question") or "").strip()


def _extract_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", question)
    keywords: list[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if not normalized or normalized in _KEYWORD_STOPWORDS:
            continue
        if normalized in {item.lower() for item in keywords}:
            continue
        keywords.append(token.strip())
        if len(keywords) >= 8:
            break
    return keywords


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


def _is_placeholder_entity_value(value: Any) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", raw).lower()
    return normalized in _PLACEHOLDER_ENTITY_VALUES


def _sanitize_llm_entity_value(value: Any, source_text: str) -> str:
    text = str(value or "").strip()
    if not text or _is_placeholder_entity_value(text):
        return ""
    if text not in source_text:
        return ""
    return text


def _to_iso_time(yymmdd: str, hhmmss: str) -> str:
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone(dt.timedelta(hours=8))).isoformat()


def _infer_time_window(question: str) -> dict[str, str]:
    matched = _YYMMDD_HHMMSS_PATTERN.search(question)
    if not matched:
        return {}
    try:
        event_time = dt.datetime.fromisoformat(_to_iso_time(str(matched.group(1)), str(matched.group(2))))
    except ValueError:
        return {}
    return {
        "begin_time": (event_time - dt.timedelta(hours=1)).isoformat(),
        "end_time": (event_time + dt.timedelta(hours=1)).isoformat(),
    }


def _normalize_keywords(value: Any, *, fallback_question: str) -> list[str]:
    if isinstance(value, list):
        merged = ",".join(str(item) for item in value)
        return _extract_keywords(merged)
    if isinstance(value, str):
        return _extract_keywords(value)
    return _extract_keywords(fallback_question)


def _normalize_time_window(parsed: dict[str, Any]) -> dict[str, str]:
    window = dict(parsed.get("time_window") or {}) if isinstance(parsed.get("time_window"), dict) else {}
    begin_time = str(parsed.get("begin_time") or window.get("begin_time") or "").strip()
    end_time = str(parsed.get("end_time") or window.get("end_time") or "").strip()
    if not begin_time and not end_time:
        return {}
    return {
        "begin_time": begin_time,
        "end_time": end_time,
    }


def _build_regex_rewrite(question: str, *, history_context: str = "") -> dict[str, Any]:
    search_text = f"{question}\n{history_context}".strip()
    trace_match = _TRACE_ID_PATTERN.search(search_text)
    order_match = _ORDER_ID_PATTERN.search(search_text)
    time_window = _infer_time_window(search_text)
    return {
        "normalized_query": question,
        "trace_id": str(trace_match.group(0) if trace_match else "").strip(),
        "order_id": str(order_match.group(0) if order_match else "").strip(),
        "keywords": _extract_keywords(question),
        "time_window": time_window,
        "source": "rule_fallback",
    }


def _rewrite_with_remote_llm(question: str, *, history_context: str) -> dict[str, Any]:
    question_text = str(question or "").strip()
    if not question_text:
        return {}

    system_prompt = load_prompt(_QUERY_REWRITE_SYSTEM_PROMPT, default="")
    user_prompt = (
        render_prompt(
            _QUERY_REWRITE_USER_PROMPT,
            question=question_text,
            history_context=history_context or "无",
        )
        or question_text
    )
    raw_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt, scene="query_rewrite")
    parsed = _parse_json_object(raw_output)
    if not isinstance(parsed, dict):
        _LOGGER.info("query_rewrite.remote_llm.parse_failed raw_len=%d", len(str(raw_output or "")))
        return {}

    normalized_query = str(parsed.get("normalized_query") or parsed.get("query") or "").strip()
    source_text = f"{question_text}\n{history_context or ''}"
    trace_id = _sanitize_llm_entity_value(parsed.get("trace_id") or parsed.get("traceId"), source_text)
    order_id = _sanitize_llm_entity_value(parsed.get("order_id") or parsed.get("orderId"), source_text)
    keywords = _normalize_keywords(parsed.get("keywords"), fallback_question=normalized_query or question_text)
    time_window = _normalize_time_window(parsed)
    return {
        "normalized_query": normalized_query or question_text,
        "trace_id": trace_id,
        "order_id": order_id,
        "keywords": keywords,
        "time_window": time_window,
        "source": "remote_llm",
    }


def _merge_rewrite(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged["normalized_query"] = str(primary.get("normalized_query") or fallback.get("normalized_query") or "").strip()
    merged["trace_id"] = str(primary.get("trace_id") or fallback.get("trace_id") or "").strip()
    merged["order_id"] = str(primary.get("order_id") or fallback.get("order_id") or "").strip()
    merged["keywords"] = list(primary.get("keywords") or fallback.get("keywords") or [])
    primary_window = dict(primary.get("time_window") or {})
    fallback_window = dict(fallback.get("time_window") or {})
    merged["time_window"] = {
        "begin_time": str(primary_window.get("begin_time") or fallback_window.get("begin_time") or "").strip(),
        "end_time": str(primary_window.get("end_time") or fallback_window.get("end_time") or "").strip(),
    }
    merged["source"] = str(primary.get("source") or fallback.get("source") or "rule_fallback")
    return merged


def _sanitize_existing_rewrite(rewrite: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(rewrite)
    if _is_placeholder_entity_value(sanitized.get("trace_id")):
        sanitized["trace_id"] = ""
    if _is_placeholder_entity_value(sanitized.get("order_id")):
        sanitized["order_id"] = ""
    return sanitized


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    existing_rewrite = _sanitize_existing_rewrite(dict(state.get("query_rewrite") or {}))
    if existing_rewrite and (
        str(existing_rewrite.get("normalized_query") or "").strip()
        or list(existing_rewrite.get("rewritten_queries") or [])
    ):
        time_window = dict(existing_rewrite.get("time_window") or {})
        structured = dict(state.get("structured_context") or {})
        state["structured_context"] = {
            **structured,
            "trace_id": str(structured.get("trace_id") or existing_rewrite.get("trace_id") or "").strip(),
            "order_id": str(structured.get("order_id") or existing_rewrite.get("order_id") or "").strip(),
            "begin_time": str(structured.get("begin_time") or time_window.get("begin_time") or "").strip(),
            "end_time": str(structured.get("end_time") or time_window.get("end_time") or "").strip(),
            "query_rewrite": existing_rewrite,
        }
        state["route"] = "knowledge_retrieve"
        _LOGGER.info(
            "query_rewrite.skip use_existing source=%s rewritten_count=%d",
            str(existing_rewrite.get("source") or ""),
            len(list(existing_rewrite.get("rewritten_queries") or [])),
        )
        return dict(state)

    question = _pick_question(state)
    history_context = _build_history_context_text(state)
    fallback_rewrite = _build_regex_rewrite(question, history_context=history_context)
    llm_rewrite = _rewrite_with_remote_llm(question, history_context=history_context)
    rewrite = _merge_rewrite(llm_rewrite, fallback_rewrite) if llm_rewrite else fallback_rewrite
    time_window = dict(rewrite.get("time_window") or {})

    structured = dict(state.get("structured_context") or {})
    begin_time = str(structured.get("begin_time") or time_window.get("begin_time") or "").strip()
    end_time = str(structured.get("end_time") or time_window.get("end_time") or "").strip()
    trace_id = str(structured.get("trace_id") or rewrite.get("trace_id") or "").strip()
    order_id = str(structured.get("order_id") or rewrite.get("order_id") or "").strip()

    state["query_rewrite"] = rewrite
    state["structured_context"] = {
        **structured,
        "trace_id": trace_id,
        "order_id": order_id,
        "begin_time": begin_time,
        "end_time": end_time,
        "query_rewrite": rewrite,
    }
    _LOGGER.info(
        "query_rewrite.finished source=%s trace_id=%s order_id=%s begin_time=%s end_time=%s keywords=%s",
        str(rewrite.get("source") or ""),
        trace_id,
        order_id,
        begin_time,
        end_time,
        list(rewrite.get("keywords") or []),
    )
    state["route"] = "knowledge_retrieve"
    return dict(state)
