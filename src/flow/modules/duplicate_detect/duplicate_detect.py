"""Duplicate detect module."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_HISTORY_BY_MD5: dict[str, str] = {}
_HISTORY_BY_TOKENS: list[tuple[set[str], str]] = []


def _normalize_question(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s\u4e00-\u9fff]", "", lowered)
    return lowered


def _token_jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    if state.get("pipeline_stop"):
        return state

    normalized = _normalize_question(str(state.get("message") or ""))
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    state["normalized_question"] = normalized

    if digest in _HISTORY_BY_MD5:
        state["duplicate_hit"] = True
        state["duplicate_answer"] = _HISTORY_BY_MD5[digest]
        state["pipeline_stop"] = True
        state["status"] = "finished"
        return state

    query_tokens = set(normalized.split())
    for tokens, answer in _HISTORY_BY_TOKENS:
        if _token_jaccard(tokens, query_tokens) >= 0.85:
            state["duplicate_hit"] = True
            state["duplicate_answer"] = answer
            state["pipeline_stop"] = True
            state["status"] = "finished"
            return state

    state["duplicate_hit"] = False
    state["pipeline_stop"] = False
    return state


def remember_qa(question: str, answer: str) -> None:
    normalized = _normalize_question(question)
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    _HISTORY_BY_MD5[digest] = answer
    _HISTORY_BY_TOKENS.append((set(normalized.split()), answer))

