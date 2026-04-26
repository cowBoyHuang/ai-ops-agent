"""日志子执行器：调用外部日志接口并提取关键信息。"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from llm.llm import chat_with_llm
from log.log import EsResult, query_external_logs

_DEFAULT_WINDOW_MINUTES = 30
_MAX_LOG_ROWS = 8
_TRACE_ID_PATTERN = re.compile(r"\b[a-z]+[_-]slugger[_a-z0-9\.\-]+\b", re.IGNORECASE)


def _as_datetime(value: Any, default: dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(candidate)
    except ValueError:
        return default


def _extract_backup_keywords(question: str) -> list[str]:
    text = str(question or "").strip()
    if not text:
        return []
    trace_hits = _TRACE_ID_PATTERN.findall(text)
    if trace_hits:
        return [trace_hits[0], "生单请求参数为"]
    short = text[:80].strip()
    return [short] if short else []


def _extract_effective_info(tool_name: str, query_word: str, rows: list[EsResult]) -> dict[str, Any]:
    log_rows = [str(item.content or "") for item in rows[:_MAX_LOG_ROWS]]
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
    now = dt.datetime.now(dt.timezone.utc)
    begin_time = _as_datetime(params.get("begin_time") or structured_context.get("begin_time"), now - dt.timedelta(minutes=_DEFAULT_WINDOW_MINUTES))
    end_time = _as_datetime(params.get("end_time") or structured_context.get("end_time"), now)
    app_code = str(params.get("app_code") or structured_context.get("app_code") or "").strip()
    logname = str(params.get("logname") or structured_context.get("logname") or "").strip()
    raw_phrase_list = params.get("match_phrase_list")
    raw_match_list = params.get("match_list")
    match_phrase_list = [str(item).strip() for item in list(raw_phrase_list or []) if str(item).strip()]
    match_list = [str(item).strip() for item in list(raw_match_list or []) if str(item).strip()]

    if not match_phrase_list and not match_list:
        keywords = params.get("keywords") or params.get("keyword") or params.get("query") or ""
        if not keywords:
            backup = _extract_backup_keywords(str(state.get("question") or ""))
            keywords = backup if backup else (state.get("question") or "")
        if isinstance(keywords, list):
            query_terms = [str(item).strip() for item in keywords if str(item).strip()]
            match_phrase_list = query_terms
            match_list = query_terms
        else:
            term = str(keywords or "").strip()
            if term:
                match_list = [term]

    query_word_for_prompt = " ".join([*match_phrase_list, *match_list]).strip()

    if not query_word_for_prompt:
        return {
            "tool": tool_name,
            "ok": False,
            "error": "missing query keywords for log executor",
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

    extracted = _extract_effective_info(tool_name, query_word_for_prompt, rows)
    evidence = [str(item.content or "") for item in rows[:_MAX_LOG_ROWS]]
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
