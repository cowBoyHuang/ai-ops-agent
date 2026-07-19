"""日志子执行器：调用外部日志接口并提取关键信息。"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any

from llm.llm import chat_with_llm
from log.log import EsResult
from tool.code_index_client import analyze_code_from_logs
from tool.registry import invoke_tool

_MAX_LOG_ROWS = 8
_MAX_LOG_EVIDENCE_CHARS = 10000
_LOGGER = logging.getLogger(__name__)
_TRACE_ID_PATTERN = re.compile(r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)", re.IGNORECASE)
_RELATIVE_NOW_PATTERN = re.compile(r"^now(?:\s*([+-])\s*(\d+)\s*([smhd]))?$", re.IGNORECASE)
_TRACE_KEY_PATTERN = re.compile(r"\btrace[_-]?id\b\s*[:=]?\s*([A-Za-z0-9_.:\-]{4,128})", re.IGNORECASE)
_ORDER_KEY_PATTERN = re.compile(
    r"(?:\border[_-]?(?:id|no)\b|订单号|订单id|订单ID|子单号)\s*[:：=]?\s*([A-Za-z0-9_.:\-]{4,128})",
    re.IGNORECASE,
)
_ORDER_TOKEN_PATTERN = re.compile(r"\bxep\d{12,}\b", re.IGNORECASE)
_ORDER_GENERIC_PATTERN = re.compile(r"\b(?:xep|sid|hpv)[A-Za-z0-9]{6,}\b", re.IGNORECASE)
_ASCII_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{6,128}$")
_PLACEHOLDER_TOKEN_PATTERN = re.compile(r"(?:\bxxx\b|placeholder|tbd|todo|待补充|示例)", re.IGNORECASE)
_BIZ_CODE_PATTERN = re.compile(r"\b\d{2}_[0-9A-Z]{3,}_[0-9A-Z]{3,}_[0-9]{4}\b")
_DECISIVE_HINTS = (
    "校验不通过",
    "人数限制",
    "很抱歉",
    "生单失败",
    "失败",
    "超时",
    "timeout",
    "connection refused",
    "连接失败",
    '"errormsg":"',
    '"errmsg":"',
    "failres",
    "block_reason",
)
_DIRECT_FACT_KEYS = (
    "bizErrorCode",
    "subErrorCode",
    "refSubErrorCode",
    "errorCode",
    "errMsg",
    "errorMsg",
    "block_reason",
    "reason",
    "failRes",
)
_DIRECT_SUMMARY_TOKENS = (
    "bizerrorcode",
    "suberrorcode",
    "refsuberrorcode",
    "errormsg",
    "errmsg",
    "block_reason",
    "校验不通过",
    "很抱歉",
    "connection refused",
    "timeout",
)
_LOG_TOOL_ALIASES = {
    "querylog": "queryLog",
    "query_log": "queryLog",
    "log_query": "queryLog",
    "getcreateorderresult": "getCreateOrderResult",
    "get_create_order_result": "getCreateOrderResult",
    "getflightcreateorderresult": "getFlightCreateOrderResult",
    "get_flight_create_order_result": "getFlightCreateOrderResult",
    "dependency_log_query": "dependency_log_query",
    "query_dependency_log": "dependency_log_query",
}
_TOOLS_REQUIRING_UPSTREAM_SCOPE = {"queryLog", "dependency_log_query"}
_FIXED_SCOPE_LOG_TOOLS = {"getCreateOrderResult", "getFlightCreateOrderResult"}


def _as_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    relative_match = _RELATIVE_NOW_PATTERN.fullmatch(text)
    if relative_match:
        sign = str(relative_match.group(1) or "-").strip()
        amount_text = str(relative_match.group(2) or "0").strip()
        unit = str(relative_match.group(3) or "s").strip().lower()
        try:
            amount = int(amount_text or "0")
        except ValueError:
            amount = 0
        base = dt.datetime.now().astimezone()
        if amount <= 0:
            return base
        delta_by_unit = {
            "s": dt.timedelta(seconds=amount),
            "m": dt.timedelta(minutes=amount),
            "h": dt.timedelta(hours=amount),
            "d": dt.timedelta(days=amount),
        }
        delta = delta_by_unit.get(unit, dt.timedelta(seconds=amount))
        return base - delta if sign != "+" else base + delta
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _pick_upstream_value(
    params: dict[str, Any],
    structured_context: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in structured_context and structured_context.get(key) is not None and str(structured_context.get(key)).strip():
            return structured_context.get(key)
    for key in keys:
        if key in params and params.get(key) is not None and str(params.get(key)).strip():
            return params.get(key)
    return None


def _resolve_registered_log_tool_name(tool_name: str, params: dict[str, Any]) -> str:
    raw = str(params.pop("log_method", "") or tool_name or "queryLog").strip()
    return _LOG_TOOL_ALIASES.get(raw.lower(), raw)


def _extract_backup_keywords(question: str) -> list[str]:
    text = str(question or "").strip()
    if not text:
        return []
    trace_hits = _TRACE_ID_PATTERN.findall(text)
    if trace_hits:
        return [trace_hits[0], "生单请求参数为"]
    short = text[:80].strip()
    return [short] if short else []


def _normalize_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _is_placeholder_token(value: str) -> bool:
    return bool(_PLACEHOLDER_TOKEN_PATTERN.search(str(value or "").strip()))


def _is_valid_trace_token(value: str) -> bool:
    text = str(value or "").strip()
    if not text or _is_placeholder_token(text):
        return False
    if not _TRACE_ID_PATTERN.search(text):
        return False
    return bool(re.search(r"\d{6}", text))


def _sanitize_match_phrase_terms(terms: list[str]) -> list[str]:
    sanitized: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if not text or _is_placeholder_token(text):
            continue
        if not _is_trace_or_order_identifier(text):
            continue
        if text not in sanitized:
            sanitized.append(text)
        extracted = _extract_forced_phrase_terms([text])
        for token in extracted:
            if not _is_trace_or_order_identifier(token):
                continue
            if token not in sanitized:
                sanitized.append(token)
    return sanitized


def _split_terms_for_query(terms: list[str]) -> tuple[list[str], list[str]]:
    phrase_terms: list[str] = []
    fuzzy_terms: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        if _TRACE_ID_PATTERN.search(text):
            if text not in phrase_terms:
                phrase_terms.append(text)
            continue
        if text not in fuzzy_terms:
            fuzzy_terms.append(text)
    return phrase_terms, fuzzy_terms


def _extract_forced_phrase_terms(texts: list[str]) -> list[str]:
    results: list[str] = []
    for text in texts:
        raw = str(text or "").strip()
        if not raw:
            continue
        for match in _TRACE_ID_PATTERN.findall(raw):
            value = str(match or "").strip()
            if _is_valid_trace_token(value) and value not in results:
                results.append(value)
        for pattern in (_TRACE_KEY_PATTERN, _ORDER_KEY_PATTERN):
            for match in pattern.findall(raw):
                value = str(match or "").strip()
                if (
                    value
                    and not _is_placeholder_token(value)
                    and _is_trace_or_order_identifier(value)
                    and value not in results
                ):
                    results.append(value)
        for match in _ORDER_TOKEN_PATTERN.findall(raw):
            value = str(match or "").strip()
            if value and not _is_placeholder_token(value) and value not in results:
                results.append(value)
    return results


def _score_row_for_evidence(text: str) -> int:
    lowered = str(text or "").lower()
    if not lowered:
        return -99
    score = 0
    for token in _DECISIVE_HINTS:
        if token in lowered:
            score += 6
    if '"success":false' in lowered or '"resultok":false' in lowered:
        score += 2
    if "需为" in text and "人" in text:
        score += 8
    if "bizerrorcode" in lowered and _BIZ_CODE_PATTERN.search(text):
        has_human_reason = any(token in lowered for token in ("errormsg", "errmsg", "failres", "block_reason", "很抱歉", "失败"))
        if not has_human_reason:
            score -= 6
    return score


def _clip_log_text(text: Any, max_len: int = _MAX_LOG_EVIDENCE_CHARS) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    marker = "\n...\n"
    head = max_len // 2
    tail = max_len - head - len(marker)
    if tail <= 0:
        return raw[:max_len]
    return f"{raw[:head]}{marker}{raw[-tail:]}"


def _clip_for_log(text: Any, max_len: int = 180) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


def _pick_trace_id_for_dispatch(params: dict[str, Any], phrase_terms: list[str]) -> str:
    explicit = str(params.get("trace_id") or params.get("traceId") or "").strip()
    if explicit:
        return explicit
    for token in phrase_terms:
        text = str(token or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if "slugger" in lowered or "flight_supply_open_api" in lowered:
            return text
    for token in phrase_terms:
        text = str(token or "").strip()
        if len(text) >= 12 and any(ch.isdigit() for ch in text):
            return text
    return ""


def _is_trace_or_order_identifier(value: str) -> bool:
    text = str(value or "").strip()
    if not text or _is_placeholder_token(text):
        return False
    if _TRACE_ID_PATTERN.search(text):
        return True
    if _ORDER_TOKEN_PATTERN.search(text):
        return True
    if _ORDER_GENERIC_PATTERN.search(text):
        return True
    if not _ASCII_ID_PATTERN.fullmatch(text):
        return False
    lowered = text.lower()
    if lowered.startswith(("xep", "sid", "hpv", "trace", "order")):
        return True
    if any(ch in text for ch in ("_", "-", ".")) and any(ch.isdigit() for ch in text):
        return True
    return False


def _select_rows_for_evidence(rows: list[EsResult]) -> list[EsResult]:
    if len(rows) <= _MAX_LOG_ROWS:
        return rows

    scored: list[tuple[int, int]] = []
    for idx, item in enumerate(rows):
        scored.append((_score_row_for_evidence(str(item.content or "")), idx))

    selected_indices: list[int] = []
    # 先挑“可常识判断”的关键失败行。
    for score, idx in sorted(scored, key=lambda pair: (-pair[0], pair[1])):
        if score < 8:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= _MAX_LOG_ROWS:
            break

    # 再按原顺序补足上下文。
    for idx in range(len(rows)):
        if len(selected_indices) >= _MAX_LOG_ROWS:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    return [rows[idx] for idx in selected_indices]


def _extract_effective_info(tool_name: str, query_word: str, rows: list[EsResult]) -> dict[str, Any]:
    log_rows = [_clip_log_text(item.content or "") for item in rows]
    if not log_rows:
        return {
            "summary": "未检索到日志命中",
            "keywords": [],
            "facts": {},
        }
    system_prompt = (
        "你是日志排障助手。请从日志中提取有效信息，返回 JSON，字段："
        "summary(字符串), keywords(字符串数组), facts(对象)。"
    )
    user_prompt = (
        f"工具: {tool_name}\n"
        f"查询关键词: {query_word}\n"
        f"日志内容:\n{json.dumps(log_rows, ensure_ascii=False)}"
    )
    raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary") or "日志提取完成"),
                "keywords": [str(item).strip() for item in list(parsed.get("keywords") or []) if str(item).strip()],
                "facts": dict(parsed.get("facts") or {}),
            }
    except Exception:
        pass
    return {
        "summary": "日志提取完成",
        "keywords": [],
        "facts": {},
    }


def _log_has_direct_answer(rows: list[EsResult], extracted: dict[str, Any]) -> bool:
    if not rows:
        return False
    facts = dict(extracted.get("facts") or {})
    for key in _DIRECT_FACT_KEYS:
        value = str(facts.get(key) or "").strip()
        if value:
            return True
    summary = str(extracted.get("summary") or "").strip()
    lowered_summary = summary.lower()
    if "未检索到日志命中" in summary:
        return False
    if any(token in lowered_summary for token in _DIRECT_SUMMARY_TOKENS):
        return True
    merged = "\n".join(str(item.content or "") for item in rows).lower()
    return any(token in merged for token in _DIRECT_SUMMARY_TOKENS)


def run(*, step: dict[str, Any], state: dict[str, Any], structured_context: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(step.get("tool_name") or "queryLog")
    params = dict(step.get("params") or {})
    registered_tool = _resolve_registered_log_tool_name(tool_name, params)
    normalized_method = registered_tool.lower()
    begin_raw = _pick_upstream_value(
        params,
        structured_context,
        ("begin_time", "beginTime", "start_time", "startTime"),
    )
    end_raw = _pick_upstream_value(
        params,
        structured_context,
        ("end_time", "endTime", "finish_time", "finishTime"),
    )
    _LOGGER.info(
        "log_executor start tool=%s begin_time=%s end_time=%s app_code=%s logname=%s",
        tool_name,
        _clip_for_log(begin_raw),
        _clip_for_log(end_raw),
        _clip_for_log(params.get("app_code") or structured_context.get("app_code") or ""),
        _clip_for_log(params.get("logname") or structured_context.get("logname") or ""),
    )
    if begin_raw is None or end_raw is None:
        _LOGGER.warning("log_executor missing time window: tool=%s", tool_name)
        return {
            "tool": tool_name,
            "ok": False,
            "error": "missing begin_time/end_time from upstream",
            "evidence": [],
        }
    begin_time = _as_datetime(begin_raw)
    end_time = _as_datetime(end_raw)
    if begin_time is None or end_time is None:
        _LOGGER.warning("log_executor invalid time format: tool=%s begin=%s end=%s", tool_name, begin_raw, end_raw)
        return {
            "tool": tool_name,
            "ok": False,
            "error": "invalid begin_time/end_time format from upstream",
            "evidence": [],
        }
    requested_app_code = str(params.get("app_code") or structured_context.get("app_code") or "").strip()
    requested_logname = str(params.get("logname") or structured_context.get("logname") or "").strip()
    app_code = requested_app_code
    logname = requested_logname
    raw_phrase_list = params.get("match_phrase_list")
    raw_match_list = params.get("match_list")
    match_phrase_list = _sanitize_match_phrase_terms(_normalize_terms(raw_phrase_list))
    match_list = _normalize_terms(raw_match_list)
    forced_terms = _extract_forced_phrase_terms(
        [
            str(state.get("question") or ""),
            str(params.get("query") or ""),
            str(params.get("keyword") or ""),
            " ".join(_normalize_terms(params.get("keywords"))),
            " ".join(match_list),
            str(params.get("trace_id") or ""),
            str(params.get("traceId") or ""),
            str(params.get("order_id") or ""),
            str(params.get("orderId") or ""),
            str(params.get("order_no") or ""),
            str(params.get("orderNo") or ""),
            str(structured_context.get("trace_id") or ""),
            str(structured_context.get("traceId") or ""),
            str(structured_context.get("order_id") or ""),
            str(structured_context.get("orderId") or ""),
            str(structured_context.get("order_no") or ""),
            str(structured_context.get("orderNo") or ""),
            str(state.get("trace_id") or ""),
            str(state.get("traceId") or ""),
            str(state.get("order_id") or ""),
            str(state.get("orderId") or ""),
            str(state.get("order_no") or ""),
            str(state.get("orderNo") or ""),
        ]
    )
    for token in forced_terms:
        if token not in match_phrase_list:
            match_phrase_list.append(token)
    explicit_id_candidates = [
        params.get("trace_id"),
        params.get("traceId"),
        params.get("order_id"),
        params.get("orderId"),
        params.get("order_no"),
        params.get("orderNo"),
        structured_context.get("trace_id"),
        structured_context.get("traceId"),
        structured_context.get("order_id"),
        structured_context.get("orderId"),
        structured_context.get("order_no"),
        structured_context.get("orderNo"),
        state.get("trace_id"),
        state.get("traceId"),
        state.get("order_id"),
        state.get("orderId"),
        state.get("order_no"),
        state.get("orderNo"),
    ]
    for token in explicit_id_candidates:
        text = str(token or "").strip()
        if not _is_trace_or_order_identifier(text):
            continue
        if text not in match_phrase_list:
            match_phrase_list.append(text)

    # 兜底 queryLog 强约束：
    # - match_phrase_list 必须至少包含一个 traceId/orderNo 类标识；
    # - match_list 必须为空（禁止模糊扩召回）。
    if registered_tool == "queryLog":
        strict_phrase_list: list[str] = []
        for token in [*match_phrase_list, *forced_terms]:
            text = str(token or "").strip()
            if not text or not _is_trace_or_order_identifier(text):
                continue
            if text not in strict_phrase_list:
                strict_phrase_list.append(text)
        match_phrase_list = strict_phrase_list
        match_list = []
        if not match_phrase_list:
            _LOGGER.warning("log_executor queryLog requires trace_id/order_no in match_phrase_list")
            return {
                "tool": tool_name,
                "ok": False,
                "error": "queryLog requires trace_id/order_no in match_phrase_list",
                "evidence": [],
            }

    query_word_for_prompt = " ".join([*match_phrase_list, *match_list]).strip()

    if not query_word_for_prompt:
        _LOGGER.warning("log_executor missing query terms: tool=%s", tool_name)
        return {
            "tool": tool_name,
            "ok": False,
            "error": "missing match_phrase_list/match_list for log executor",
            "evidence": [],
        }

    # 仅通用 queryLog / 依赖日志要求上游透传 app/log，业务技能方法使用固定作用域。
    if registered_tool in _TOOLS_REQUIRING_UPSTREAM_SCOPE and (not app_code or not logname):
        extracted = _extract_effective_info(
            tool_name,
            query_word_for_prompt,
            [EsResult(score=0.0, content=f"fallback-log: {query_word_for_prompt}")],
        )
        _LOGGER.info(
            "log_executor degraded fallback tool=%s query=%s reason=missing app_code/logname",
            tool_name,
            _clip_for_log(query_word_for_prompt, 240),
        )
        return {
            "tool": tool_name,
            "ok": True,
            "error": "",
            "evidence": [f"[summary] {str(extracted.get('summary') or '')}", f"fallback-log: {query_word_for_prompt}"],
            "effective_info": extracted,
            "log_hit_count": 1,
            "degraded": True,
        }

    query_payload = {
        "match_phrase_list": [str(item).strip() for item in match_phrase_list if str(item).strip()],
        "match_list": [str(item).strip() for item in match_list if str(item).strip()],
    }
    dispatch_trace_id = _pick_trace_id_for_dispatch(params, query_payload.get("match_phrase_list") or [])
    _LOGGER.info(
        "log_executor query tool=%s requested_app_code=%s requested_logname=%s effective_app_code=%s effective_logname=%s trace_id=%s match_phrase_list=%s match_list=%s",
        registered_tool,
        _clip_for_log(requested_app_code, 80),
        _clip_for_log(requested_logname, 80),
        _clip_for_log(app_code, 80),
        _clip_for_log(logname, 80),
        _clip_for_log(dispatch_trace_id, 100),
        [_clip_for_log(item, 80) for item in query_payload.get("match_phrase_list") or []],
        [_clip_for_log(item, 80) for item in query_payload.get("match_list") or []],
    )
    if registered_tool in _FIXED_SCOPE_LOG_TOOLS:
        tool_args = {
            "trace_id": dispatch_trace_id,
            "begin_time": begin_time,
            "end_time": end_time,
        }
    else:
        tool_args = {
            "app_code": app_code,
            "logname": logname,
            "begin_time": begin_time,
            "end_time": end_time,
            "match_phrase_list": query_payload.get("match_phrase_list"),
            "match_list": query_payload.get("match_list"),
        }
    try:
        rows = invoke_tool(registered_tool, tool_args)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("log_executor query failed tool=%s error=%s", registered_tool, exc)
        return {"tool": registered_tool, "ok": False, "error": str(exc), "evidence": []}
    if isinstance(rows, dict) and rows.get("ok") is False:
        return {
            "tool": str(rows.get("tool") or registered_tool),
            "ok": False,
            "error": str(rows.get("error") or ""),
            "evidence": list(rows.get("evidence") or []),
        }

    evidence_rows = _select_rows_for_evidence(rows)
    extracted = _extract_effective_info(tool_name, query_word_for_prompt, evidence_rows)
    evidence = [_clip_log_text(item.content or "") for item in evidence_rows]
    if extracted.get("summary"):
        evidence.insert(0, f"[summary] {str(extracted['summary'])}")
    code_analysis: dict[str, Any] = {}
    if not _log_has_direct_answer(evidence_rows, extracted):
        code_analysis = analyze_code_from_logs(
            question=str(state.get("question") or ""),
            evidence_rows=evidence,
            extra_keywords=[str(item or "") for item in list(extracted.get("keywords") or [])],
        )
        if bool(code_analysis.get("ok")):
            evidence.extend([str(item) for item in list(code_analysis.get("evidence") or []) if str(item).strip()])
            facts = dict(extracted.get("facts") or {})
            facts["code_index_context"] = {
                "mode": str(code_analysis.get("mode") or ""),
                "current_method": dict(code_analysis.get("current_method") or {}),
                "caller_count": len(list(code_analysis.get("caller") or [])),
                "callee_count": len(list(code_analysis.get("callee") or [])),
            }
            extracted["facts"] = facts
            code_summary = str(code_analysis.get("summary") or "").strip()
            if code_summary:
                log_summary = str(extracted.get("summary") or "").strip()
                extracted["summary"] = f"{log_summary}；代码补充：{code_summary}" if log_summary else code_summary
    _LOGGER.info(
        "log_executor done tool=%s log_hit_count=%d selected_rows=%d summary=%s",
        registered_tool,
        len(rows),
        len(evidence_rows),
        _clip_for_log(extracted.get("summary"), 220),
    )
    return {
        "tool": registered_tool,
        "ok": True,
        "error": "",
        "evidence": evidence,
        "effective_info": extracted,
        "log_hit_count": len(rows),
        "code_analysis": code_analysis,
    }
