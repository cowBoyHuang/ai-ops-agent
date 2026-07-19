"""Client utilities for local Java code index service."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

from tool.local_code_index import analyze_local_code, locate_line_local

_LOGGER = logging.getLogger(__name__)
_DEFAULT_BASE_URL = "http://127.0.0.1:18080"
_DEFAULT_TIMEOUT_SEC = 8.0
_MAX_LOCATE_HINTS = 4
_MAX_SEARCH_KEYWORDS = 6
_CLASS_LINE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\.(?:java|kt):(\d+)\b")
_SIMPLE_CLASS_LINE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+):(\d+)\b")
_TOKEN_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,64})\b")
_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "true",
    "false",
    "null",
    "none",
    "query",
    "trace",
    "order",
    "error",
    "failed",
    "failure",
    "status",
    "result",
    "create",
    "flight",
    "code",
    "line",
}


def _clip(value: Any, max_len: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _base_url() -> str:
    value = str(os.getenv("AIOPS_CODE_INDEX_BASE_URL", _DEFAULT_BASE_URL)).strip()
    return (value or _DEFAULT_BASE_URL).rstrip("/")


def _timeout_sec() -> float:
    raw = str(os.getenv("AIOPS_CODE_INDEX_TIMEOUT_SEC", _DEFAULT_TIMEOUT_SEC)).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TIMEOUT_SEC
    if value <= 0:
        value = _DEFAULT_TIMEOUT_SEC
    return value


def _request_json(
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[bool, Any, str]:
    url = f"{_base_url()}{path}"
    started = time.perf_counter()
    _LOGGER.info(
        "code_index.request.start method=%s path=%s params=%s has_json=%s",
        method,
        path,
        _clip(params, 300),
        bool(json_body),
    )
    try:
        with httpx.Client(timeout=_timeout_sec()) as client:
            response = client.request(method=method, url=url, params=params, json=json_body)
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - started) * 1000
        _LOGGER.warning(
            "code_index.request.error method=%s path=%s duration_ms=%.2f error=%s",
            method,
            path,
            duration_ms,
            _clip(exc),
        )
        return False, None, str(exc)

    if response.status_code >= 400:
        duration_ms = (time.perf_counter() - started) * 1000
        _LOGGER.warning(
            "code_index.request.http_error method=%s path=%s status=%s duration_ms=%.2f body=%s",
            method,
            path,
            response.status_code,
            duration_ms,
            _clip(response.text, 300),
        )
        return False, None, f"http_{response.status_code}: {response.text[:300]}"
    try:
        payload = response.json()
        duration_ms = (time.perf_counter() - started) * 1000
        _LOGGER.info(
            "code_index.request.end method=%s path=%s status=%s duration_ms=%.2f",
            method,
            path,
            response.status_code,
            duration_ms,
        )
        return True, payload, ""
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - started) * 1000
        _LOGGER.warning(
            "code_index.request.invalid_json method=%s path=%s duration_ms=%.2f error=%s",
            method,
            path,
            duration_ms,
            _clip(exc),
        )
        return False, None, f"invalid_json: {exc}"


def index_project(project_path: str) -> dict[str, Any]:
    path = str(project_path or "").strip()
    if not path:
        return {"ok": False, "error": "empty project_path"}
    ok, payload, error = _request_json(
        method="POST",
        path="/index",
        json_body={"projectPath": path},
    )
    if not ok:
        return {"ok": False, "error": error}
    row = dict(payload or {})
    status = str(row.get("status") or "").strip().upper()
    return {
        "ok": status == "SUCCESS" or bool(row),
        "payload": row,
        "error": "" if status == "SUCCESS" else str(row.get("errorMessage") or ""),
    }


def search_method(keyword: str) -> dict[str, Any]:
    text = str(keyword or "").strip()
    if not text:
        return {"ok": False, "methods": [], "error": "empty keyword"}
    ok, payload, error = _request_json(
        method="GET",
        path="/searchMethod",
        params={"keyword": text},
    )
    if not ok:
        return {"ok": False, "methods": [], "error": error}
    methods = [dict(item or {}) for item in list(payload or []) if isinstance(item, dict)]
    return {"ok": True, "methods": methods, "error": ""}


def locate_code(class_name: str, line: int) -> dict[str, Any]:
    name = str(class_name or "").strip()
    line_no = int(line or 0)
    if not name or line_no <= 0:
        return {"ok": False, "result": {}, "error": "invalid class_name/line"}
    ok, payload, error = _request_json(
        method="GET",
        path="/locateCode",
        params={"class": name, "line": line_no},
    )
    if not ok:
        return {"ok": False, "result": {}, "error": error}
    return {"ok": True, "result": dict(payload or {}), "error": ""}


def _extract_class_line_hints(texts: list[str]) -> list[tuple[str, int]]:
    hints: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for text in list(texts or []):
        raw = str(text or "").strip()
        if not raw:
            continue
        for class_name, line_text in _CLASS_LINE_PATTERN.findall(raw):
            hint = (str(class_name).strip(), int(str(line_text).strip() or "0"))
            if hint[1] <= 0 or hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
        for class_name, line_text in _SIMPLE_CLASS_LINE_PATTERN.findall(raw):
            hint = (str(class_name).strip(), int(str(line_text).strip() or "0"))
            if hint[1] <= 0 or hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
    return hints[:_MAX_LOCATE_HINTS]


def _collect_keywords(question: str, extra_keywords: list[str] | None = None) -> list[str]:
    rows = [str(question or "").strip(), *[str(item or "").strip() for item in list(extra_keywords or [])]]
    keywords: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row:
            continue
        for token in _TOKEN_PATTERN.findall(row):
            text = str(token or "").strip()
            lowered = text.lower()
            if not text or lowered in _STOPWORDS:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(text)
    return keywords[:_MAX_SEARCH_KEYWORDS]


def _build_locate_summary(result: dict[str, Any]) -> str:
    method = dict(result.get("method") or {})
    if not method:
        return "code_index 定位完成，但未返回 method 信息"
    method_name = str(method.get("methodName") or "").strip() or str(method.get("method") or "").strip()
    start_line = int(method.get("startLine") or 0)
    end_line = int(method.get("endLine") or 0)
    if start_line > 0 and end_line > 0:
        return f"定位到方法 {method_name}({start_line}-{end_line})"
    return f"定位到方法 {method_name}"


def analyze_code_from_logs(
    *,
    question: str,
    evidence_rows: list[str],
    extra_keywords: list[str] | None = None,
) -> dict[str, Any]:
    hints = _extract_class_line_hints(evidence_rows)
    for class_name, line_no in hints:
        located = locate_code(class_name, line_no)
        if not located.get("ok"):
            located = locate_line_local(class_name, line_no)
            if located.get("ok"):
                result = dict(located.get("result") or {})
                method = dict(result.get("method") or {})
                symbol = dict(result.get("symbol") or {})
                line = dict(result.get("line") or {})
                if method:
                    summary = (
                        f"本地源码定位到方法 {method.get('methodName')} "
                        f"({method.get('startLine')}-{method.get('endLine')})"
                    )
                else:
                    summary = f"本地源码定位到行 {line.get('filePath')}:{line.get('line')}"
                return {
                    "ok": True,
                    "mode": "local_locate_line",
                    "summary": summary,
                    "current_method": method,
                    "current_symbol": symbol,
                    "caller": [],
                    "callee": [],
                    "logs": [],
                    "matched_methods": [method] if method else [],
                    "text_matches": [line] if line else [],
                    "evidence": [
                        f"[local_code_index] locateLine class={class_name} line={line_no}",
                        f"[local_code_index] {summary}",
                    ],
                    "error": "",
                }
            continue
        result = dict(located.get("result") or {})
        method = dict(result.get("method") or {})
        if not method:
            continue
        summary = _build_locate_summary(result)
        return {
            "ok": True,
            "mode": "locateCode",
            "summary": summary,
            "current_method": method,
            "caller": list(result.get("caller") or []),
            "callee": list(result.get("callee") or []),
            "logs": list(result.get("logs") or []),
            "matched_methods": [],
            "evidence": [
                f"[code_index] locateCode class={class_name} line={line_no}",
                f"[code_index] {summary}",
            ],
            "error": "",
        }

    keywords = _collect_keywords(question, extra_keywords=extra_keywords)
    for keyword in keywords:
        searched = search_method(keyword)
        if not searched.get("ok"):
            continue
        methods = [dict(item or {}) for item in list(searched.get("methods") or []) if isinstance(item, dict)]
        if not methods:
            continue
        top = dict(methods[0] or {})
        class_name = str(top.get("className") or top.get("fullClassName") or "").strip()
        method_name = str(top.get("methodName") or "").strip()
        signature = str(top.get("signature") or "").strip()
        summary = f"代码检索命中 {class_name}.{method_name} {signature}".strip()
        return {
            "ok": True,
            "mode": "searchMethod",
            "summary": summary,
            "current_method": top,
            "caller": [],
            "callee": [],
            "logs": [],
            "matched_methods": methods[:5],
            "evidence": [
                f"[code_index] searchMethod keyword={keyword}",
                f"[code_index] {summary}",
            ],
            "error": "",
        }

    local_result = analyze_local_code(
        question=question,
        extra_keywords=[str(item or "").strip() for item in list(extra_keywords or []) if str(item or "").strip()],
    )
    if bool(local_result.get("ok")):
        return local_result

    return {
        "ok": False,
        "mode": "none",
        "summary": "code_index 未命中可用代码上下文",
        "current_method": {},
        "current_symbol": {},
        "caller": [],
        "callee": [],
        "logs": [],
        "matched_methods": [],
        "evidence": [],
        "error": "no locate/search result",
    }


def analyze_code_for_business_consult(
    *,
    question: str,
    structured_context: dict[str, Any] | None = None,
    evidence_rows: list[str] | None = None,
) -> dict[str, Any]:
    context = dict(structured_context or {})
    class_name = str(context.get("class_name") or context.get("className") or "").strip()
    line_no = int(context.get("line") or context.get("line_no") or 0)
    if class_name and line_no > 0:
        located = locate_code(class_name, line_no)
        if located.get("ok"):
            result = dict(located.get("result") or {})
            method = dict(result.get("method") or {})
            if method:
                summary = _build_locate_summary(result)
                return {
                    "ok": True,
                    "mode": "locateCode",
                    "summary": summary,
                    "current_method": method,
                    "caller": list(result.get("caller") or []),
                    "callee": list(result.get("callee") or []),
                    "logs": list(result.get("logs") or []),
                    "matched_methods": [],
                    "evidence": [
                        f"[code_index] locateCode class={class_name} line={line_no}",
                        f"[code_index] {summary}",
                    ],
                    "error": "",
                }

    return analyze_code_from_logs(
        question=question,
        evidence_rows=[str(item or "") for item in list(evidence_rows or [])],
        extra_keywords=[str(context.get("code_keyword") or "").strip()],
    )
