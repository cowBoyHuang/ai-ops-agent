"""Utilities for cleaning noisy log content before retrieval diagnosis."""

from __future__ import annotations

import re

_ISO_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TRACE_SPAN_REQUEST_ID_PATTERN = re.compile(
    r"\b((?:trace|span|request)[-_]?id)\s*[:=\s]*[a-fA-F0-9\-]{8,}\b",
    re.IGNORECASE,
)
_BUSINESS_ID_PATTERN = re.compile(
    r"\b((?:user|order|customer)[-_]?id)\s*[:=\s]*[a-zA-Z0-9\-_]{5,}\b",
    re.IGNORECASE,
)
_HEX_PATTERN = re.compile(r"\b[a-fA-F0-9]{16,}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_log_content(text: str) -> str:
    """清理日志内容，移除动态和无关信息。"""
    if not text:
        return ""

    cleaned = str(text)
    cleaned = _ISO_TIMESTAMP_PATTERN.sub("", cleaned)
    cleaned = _DATETIME_PATTERN.sub("", cleaned)
    cleaned = _IP_PATTERN.sub("[IP]", cleaned)
    cleaned = _TRACE_SPAN_REQUEST_ID_PATTERN.sub(r"\1:[ID]", cleaned)
    cleaned = _BUSINESS_ID_PATTERN.sub(r"\1:[ID]", cleaned)
    cleaned = _HEX_PATTERN.sub("[HEX]", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
