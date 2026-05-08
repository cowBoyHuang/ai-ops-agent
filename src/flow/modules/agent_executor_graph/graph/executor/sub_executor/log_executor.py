"""日志子执行器：调用外部日志接口并提取关键信息。"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from llm.llm import chat_with_llm
from log.log import EsResult, query_external_logs

_MAX_LOG_ROWS = 8
_TRACE_ID_PATTERN = re.compile(r"[a-z]+[_-]slugger[_a-z0-9\.\-]+(?=$|[^A-Za-z0-9_\.\-])", re.IGNORECASE)
_TRACE_KEY_PATTERN = re.compile(r"\btrace[_-]?id\b\s*[:=]?\s*([A-Za-z0-9_.:\-]{4,128})", re.IGNORECASE)
_ORDER_KEY_PATTERN = re.compile(
    r"(?:\border[_-]?(?:id|no)\b|订单号|订单id|订单ID|子单号)\s*[:：=]?\s*([A-Za-z0-9_.:\-]{4,128})",
    re.IGNORECASE,
)
_ORDER_TOKEN_PATTERN = re.compile(r"\bxep\d{12,}\b", re.IGNORECASE)
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


def _as_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
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
            if value and value not in results:
                results.append(value)
        for pattern in (_TRACE_KEY_PATTERN, _ORDER_KEY_PATTERN):
            for match in pattern.findall(raw):
                value = str(match or "").strip()
                if value and value not in results:
                    results.append(value)
        for match in _ORDER_TOKEN_PATTERN.findall(raw):
            value = str(match or "").strip()
            if value and value not in results:
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
    log_rows = [str(item.content or "") for item in rows]
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


def run(*, step: dict[str, Any], state: dict[str, Any], structured_context: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(step.get("tool_name") or "log_query")
    params = dict(step.get("params") or {})
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
    if begin_raw is None or end_raw is None:
        return {
            "tool": tool_name,
            "ok": False,
            "error": "missing begin_time/end_time from upstream",
            "evidence": [],
        }
    begin_time = _as_datetime(begin_raw)
    end_time = _as_datetime(end_raw)
    if begin_time is None or end_time is None:
        return {
            "tool": tool_name,
            "ok": False,
            "error": "invalid begin_time/end_time format from upstream",
            "evidence": [],
        }
    app_code = str(params.get("app_code") or structured_context.get("app_code") or "").strip()
    logname = str(params.get("logname") or structured_context.get("logname") or "").strip()
    raw_phrase_list = params.get("match_phrase_list")
    raw_match_list = params.get("match_list")
    match_phrase_list = _normalize_terms(raw_phrase_list)
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
        ]
    )
    for token in forced_terms:
        if token not in match_phrase_list:
            match_phrase_list.append(token)

    query_word_for_prompt = " ".join([*match_phrase_list, *match_list]).strip()

    if not query_word_for_prompt:
        return {
            "tool": tool_name,
            "ok": False,
            "error": "missing match_phrase_list/match_list for log executor",
            "evidence": [],
        }

    # 参数不足时保留可执行能力：退化为本地日志摘要，避免执行链路被硬中断。
    if not app_code or not logname:
        extracted = _extract_effective_info(
            tool_name,
            query_word_for_prompt,
            [EsResult(score=0.0, content=f"fallback-log: {query_word_for_prompt}")],
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
    try:
        rows = query_external_logs(
            app_code=app_code,
            logname=logname,
            begin_time=begin_time,
            end_time=end_time,
            content=query_payload,
        )
    except Exception as exc:  # noqa: BLE001
        return {"tool": tool_name, "ok": False, "error": str(exc), "evidence": []}

    evidence_rows = _select_rows_for_evidence(rows)
    extracted = _extract_effective_info(tool_name, query_word_for_prompt, evidence_rows)
    evidence = [str(item.content or "") for item in evidence_rows]
    if extracted.get("summary"):
        evidence.insert(0, f"[summary] {str(extracted['summary'])}")
    return {
        "tool": tool_name,
        "ok": True,
        "error": "",
        "evidence": evidence,
        "effective_info": extracted,
        "log_hit_count": len(rows),
    }
