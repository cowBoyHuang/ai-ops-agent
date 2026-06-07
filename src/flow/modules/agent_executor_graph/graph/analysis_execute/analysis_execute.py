"""分析执行节点。

业务职责：
- 汇总证据文本（优先使用结构化上下文里的 evidence_context）。
- 调用统一 LLM 方法生成根因与建议。
- 标准化置信度字段，供验证节点统一判断。
"""

from __future__ import annotations

import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import analyze_with_llm

_CONFIDENCE_MAP = {
    "high": 0.9,
    "medium": 0.65,
    "low": 0.35,
}
_FAILURE_HINTS = (
    "校验不通过",
    "生单失败",
    "order_failed",
    "errormsg",
    "success\":false",
    "resultok\":false",
    "block_reason",
)
_DETAIL_HINTS = (
    "n_flight",
    "childcount",
    "ordercreateresultmap",
    "nonsuccessordercreateresultmap",
    "refsuberrmsg",
    "refsuberrormsg",
    "exception",
    ".java:",
    ".kt:",
    ".py:",
)
_REASON_PATTERNS = (
    re.compile(r'"(?:errorMsg|errMsg|failRes|block_reason|reason|msg)"\s*:\s*"([^"]{4,220})"'),
    re.compile(r"(?:校验不通过|失败)[^:：]{0,24}[:：]\s*(?:\([^,，]*[,，])?([^)\n]{4,220})"),
)
_REF_SUB_REASON_PATTERN = re.compile(r'"(?:refSubErrMsg|refSubErrorMsg)"\s*:\s*"([^"]{4,220})"')
_CHILD_COUNT_PATTERN = re.compile(r'"childCount"\s*:\s*(\d+)')
_STACK_LOCATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+\.(?:java|kt|py):\d+)\b")
_MAX_LOG_ROWS_FOR_LLM = 12
_MAX_LOG_EVIDENCE_CHARS = 24000
_MAX_LOG_ROW_CHARS = 2200
_MAX_BUSINESS_EVIDENCE_CHARS = 60000
_BUSINESS_CONSULT_INTENT = "SYSTEM_LOGIC_CONSULT"
_BUSINESS_ANALYSIS_SYSTEM_PROMPT = "analysis_business_consult_system_prompt.txt"
_BUSINESS_ANALYSIS_USER_PROMPT = "analysis_business_consult_user_prompt.txt"
_SYSTEM_BUSINESS_COLON_PATTERN = re.compile(r"^\s*([a-z][a-z0-9_-]{1,63})\s*[：:]\s*(.+)$", re.IGNORECASE)
_SYSTEM_INLINE_PATTERN = re.compile(r"\b([a-z][a-z0-9_]{2,63})\b(?=\s*系统)", re.IGNORECASE)
_MAX_SYSTEM_BUSINESS_ROWS = 20
_BINARY_QUESTION_HINTS = (
    "吗",
    "是否",
    "是不是",
    "算不算",
    "属不属于",
    "属于吗",
    "能否",
    "可否",
)


def _coerce_confidence(value: Any) -> float:
    """把置信度转换为 float。

    兼容输入：
    - 文本 high/medium/low
    - 数字字符串或数值类型
    """
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value or "").strip().lower()
    if text in _CONFIDENCE_MAP:
        return _CONFIDENCE_MAP[text]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _confidence_label(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _normalize_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("text") or "").strip() for item in rows if str(item.get("text") or "").strip()]


def _clip_log_row(text: str, max_chars: int = _MAX_LOG_ROW_CHARS) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    marker = "\n...\n"
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    if tail <= 0:
        return raw[:max_chars]
    return f"{raw[:head]}{marker}{raw[-tail:]}"


def _extract_decisive_log_reason(rows: list[str]) -> str:
    for raw in rows:
        text = str(raw or "").strip()
        if not text:
            continue
        normalized = text.replace('\\"', '"')
        lowered = normalized.lower()
        if not any(token in lowered for token in _FAILURE_HINTS):
            continue
        for pattern in _REASON_PATTERNS:
            matched = pattern.search(normalized)
            if matched:
                reason = str(matched.group(1) or "").strip(" ,，:：'\"")
                if reason:
                    return reason
    return ""


def _build_decisive_log_reply(reason: str, rows: list[str]) -> str:
    extras: list[str] = []
    for raw in rows:
        text = str(raw or "").strip()
        if not text:
            continue
        ref_sub = _REF_SUB_REASON_PATTERN.search(text.replace('\\"', '"'))
        if ref_sub:
            detail = str(ref_sub.group(1) or "").strip(" ,，:：'\"")
            if detail and detail not in extras:
                extras.append(detail)
        child_count = _CHILD_COUNT_PATTERN.search(text)
        if child_count:
            detail = f"childCount={str(child_count.group(1) or '').strip()}"
            if detail not in extras:
                extras.append(detail)
        location = _STACK_LOCATION_PATTERN.search(text)
        if location:
            detail = f"堆栈位置={str(location.group(1) or '').strip()}"
            if detail not in extras:
                extras.append(detail)
        if len(extras) >= 3:
            break
    suffix = f" 关键信息：{'；'.join(extras[:3])}。" if extras else ""
    return f"日志已明确给出失败原因：{reason}。{suffix}".strip()


def _score_log_row(text: str) -> int:
    lowered = str(text or "").lower()
    score = 0
    if any(token in lowered for token in _FAILURE_HINTS):
        score += 4
    if any(token in lowered for token in _DETAIL_HINTS):
        score += 3
    if " at " in lowered or lowered.startswith("at "):
        score += 1
    return score


def _select_log_rows_for_llm(question: str, rows: list[str]) -> list[str]:
    _ = question
    if not rows:
        return []

    scored_rows: list[tuple[int, int, str]] = []
    for idx, row in enumerate(rows):
        text = str(row or "").strip()
        if not text:
            continue
        score = _score_log_row(text)
        if score > 0:
            scored_rows.append((score, idx, text))

    if scored_rows:
        top_items = sorted(scored_rows, key=lambda item: (-item[0], item[1]))[:_MAX_LOG_ROWS_FOR_LLM]
        selected = [item[2] for item in sorted(top_items, key=lambda item: item[1])]
    else:
        selected = [str(row or "").strip() for row in rows[:_MAX_LOG_ROWS_FOR_LLM] if str(row or "").strip()]

    evidence_rows: list[str] = []
    total_chars = 0
    for row in selected:
        clipped = _clip_log_row(row)
        next_total = total_chars + len(clipped) + 1
        if evidence_rows and next_total > _MAX_LOG_EVIDENCE_CHARS:
            break
        evidence_rows.append(clipped)
        total_chars = next_total
    return evidence_rows


def _analyze_log_evidence(question: str, rows: list[str]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "root_cause": "", "reply": "", "confidence": 0.0}
    decisive_reason = _extract_decisive_log_reason(rows)
    if decisive_reason:
        return {
            "available": True,
            "root_cause": decisive_reason,
            "reply": _build_decisive_log_reply(decisive_reason, rows),
            "confidence": 0.95,
        }

    selected_rows = _select_log_rows_for_llm(question=question, rows=rows)
    analysis = analyze_with_llm(question=question, evidence="\n".join(selected_rows))
    confidence = _coerce_confidence(analysis.get("confidence"))
    root_cause = str(analysis.get("root_cause") or "").strip()
    reply = str(analysis.get("reply") or "").strip()
    available = bool(root_cause or reply)
    return {
        "available": available,
        "root_cause": root_cause,
        "reply": reply,
        "confidence": confidence if available else 0.0,
    }


def _extract_code_locations(rows: list[str]) -> list[str]:
    locations: list[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text.startswith("code_file:"):
            continue
        location = text.split("code_file:", 1)[1].strip()
        if location and location not in locations:
            locations.append(location)
    return locations


def _extract_stack_locations_from_logs(rows: list[str]) -> list[str]:
    locations: list[str] = []
    for row in rows:
        for matched in _STACK_LOCATION_PATTERN.findall(str(row or "").strip()):
            location = str(matched or "").strip()
            if location and location not in locations:
                locations.append(location)
    return locations


def _extract_code_summaries(rows: list[str]) -> list[str]:
    summaries: list[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text:
            continue
        if text.startswith("[summary]"):
            text = text.split("[summary]", 1)[1].strip()
        if text.startswith("code_file:") or text.startswith("code_pull success:") or text.startswith("code_clone success:"):
            continue
        if text and text not in summaries:
            summaries.append(text)
    return summaries


def _analyze_code_evidence(rows: list[str], *, log_rows: list[str]) -> dict[str, Any]:
    locations = _extract_code_locations(rows)
    summaries = _extract_code_summaries(rows)
    available = bool(locations or summaries)
    if not available:
        log_locations = _extract_stack_locations_from_logs(log_rows)
        if not log_locations:
            return {"available": False, "root_cause": "", "reply": "", "confidence": 0.0, "locations": []}
        location_text = log_locations[0]
        return {
            "available": True,
            "root_cause": f"日志中的代码位置线索：{log_locations[0]}",
            "reply": f"未读取到本地代码；日志线索指向：{location_text}。",
            "confidence": 0.35,
            "locations": log_locations[:1],
            "derived_from_logs": True,
        }

    parts: list[str] = []
    if locations:
        parts.append(f"定位到相关代码位置：{locations[0]}")
    if summaries:
        parts.append(f"代码分析：{summaries[0]}")
    elif locations:
        parts.append("代码分析：已拉取并命中相关代码文件，可据此继续核对触发条件。")

    root_cause = summaries[0] if summaries else (f"相关代码位置：{locations[0]}" if locations else "")
    confidence = 0.75 if locations else 0.6
    return {
        "available": True,
        "root_cause": root_cause,
        "reply": "。".join(part.rstrip("。") for part in parts if part).strip(),
        "confidence": confidence,
        "locations": locations[:1],
    }


def _compose_reply(log_analysis: dict[str, Any], code_analysis: dict[str, Any]) -> str:
    rows: list[str] = []
    if bool(log_analysis.get("available")):
        rows.append(f"日志结论：{str(log_analysis.get('reply') or '').strip()}")
    else:
        rows.append("日志结论：本次未拿到足够日志证据。")
    if bool(code_analysis.get("available")):
        rows.append(f"代码分析：{str(code_analysis.get('reply') or '').strip()}")
    else:
        rows.append("代码分析：本次未定位到具体代码位置。")
    return "\n".join(row for row in rows if row).strip()


def _normalize_business_text(text: str) -> str:
    normalized = str(text or "").replace("\t", " ").replace("\r", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" -;；，,。")


def _extract_system_business_pairs(rows: list[str]) -> list[tuple[str, str]]:
    mapping: dict[str, str] = {}
    for row in list(rows or []):
        line = str(row or "").strip()
        if not line:
            continue
        candidates = [line]
        candidates.extend(segment.strip() for segment in line.splitlines() if segment.strip())
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            matched = _SYSTEM_BUSINESS_COLON_PATTERN.match(text)
            if matched:
                system_name = str(matched.group(1) or "").strip().lower()
                business_text = _normalize_business_text(str(matched.group(2) or ""))
                if system_name and business_text and system_name not in mapping:
                    mapping[system_name] = business_text
                continue
            inline = _SYSTEM_INLINE_PATTERN.search(text)
            if inline:
                system_name = str(inline.group(1) or "").strip().lower()
                business_text = _normalize_business_text(text)
                if system_name and business_text and system_name not in mapping:
                    mapping[system_name] = business_text
        if len(mapping) >= _MAX_SYSTEM_BUSINESS_ROWS:
            break
    return [(name, mapping[name]) for name in mapping]


def _compose_system_business_reply(pairs: list[tuple[str, str]]) -> str:
    rows = [f"{system_name}: {business}" for system_name, business in list(pairs or []) if system_name and business]
    if not rows:
        return "未提取到系统英文名称与对应业务。"
    return "\n".join(rows[:_MAX_SYSTEM_BUSINESS_ROWS]).strip()


def _is_binary_business_question(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return any(token in lowered for token in _BINARY_QUESTION_HINTS)


def _is_business_consult(intent_type: Any) -> bool:
    return str(intent_type or "").strip() == _BUSINESS_CONSULT_INTENT


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行分析步骤。

    入参：
    - payload: AgentState，需包含 question/merged_evidence 等字段。

    返参：
    - AgentState: 写入 analysis/root_cause/solution/confidence，并路由到 result_validate。
    """
    state: AgentState = dict(payload)
    context = dict(state.get("structured_context") or {})
    merged_evidence = dict(state.get("merged_evidence") or {})

    # 证据优先级：先用上游已拼好的 evidence_context，避免重复拼接。
    if context.get("evidence_context"):
        evidence = str(context.get("evidence_context") or "")
    else:
        # 回退逻辑：从 merged_evidence 的 logs/knowledge 中拼接证据。
        logs = [str(item.get("text") or "") for item in list(merged_evidence.get("logs") or [])]
        knowledge = [str(item.get("text") or "") for item in list(merged_evidence.get("knowledge") or [])]
        code = [str(item.get("text") or "") for item in list(merged_evidence.get("code") or [])]
        evidence = "\n".join(row for row in [*logs, *knowledge, *code] if row)

    question = str(state.get("question") or context.get("question") or "")
    intent_type = str(state.get("intent_type") or context.get("intent_type") or "").strip()
    log_rows = _normalize_rows(list(merged_evidence.get("logs") or []))
    code_rows = _normalize_rows(list(merged_evidence.get("code") or []))

    if _is_business_consult(intent_type):
        knowledge_rows = _normalize_rows(list(merged_evidence.get("knowledge") or []))
        evidence_rows = knowledge_rows or log_rows or code_rows
        pairs = _extract_system_business_pairs(evidence_rows)
        full_evidence = str(evidence or "").strip()
        if full_evidence:
            llm_evidence = _clip_log_row(full_evidence, max_chars=_MAX_BUSINESS_EVIDENCE_CHARS)
        else:
            compact_rows = [_clip_log_row(item, max_chars=600) for item in evidence_rows[:_MAX_LOG_ROWS_FOR_LLM]]
            llm_evidence = "\n".join(compact_rows).strip()
        llm_result: dict[str, Any] = (
            analyze_with_llm(
                question=question,
                evidence=llm_evidence,
                system_prompt_file=_BUSINESS_ANALYSIS_SYSTEM_PROMPT,
                user_prompt_file=_BUSINESS_ANALYSIS_USER_PROMPT,
            )
            if str(llm_evidence or "").strip()
            else {"root_cause": "", "confidence": 0.0, "reply": ""}
        )
        llm_reply = str(llm_result.get("reply") or "").strip()
        if not pairs:
            pairs = _extract_system_business_pairs(
                [
                    llm_reply,
                    str(llm_result.get("root_cause") or "").strip(),
                ]
            )
        if _is_binary_business_question(question) and llm_reply:
            solution = llm_reply
        else:
            solution = _compose_system_business_reply(pairs)
            if (not pairs) and llm_reply:
                solution = llm_reply
        confidence = _coerce_confidence(llm_result.get("confidence"))
        root_cause = str(llm_result.get("root_cause") or "").strip()
        if confidence <= 0:
            confidence = 0.8 if pairs else 0.3
        analysis = {
            "root_cause": root_cause,
            "confidence": _confidence_label(confidence),
            "reply": solution,
            "analysis_mode": "business_consult",
            "business_analysis": {
                "available": bool(pairs),
                "pair_count": len(pairs),
                "pairs_preview": [f"{name}: {business}" for name, business in pairs[:6]],
            },
            "log_analysis": {"available": False, "root_cause": "", "reply": "", "confidence": 0.0},
            "code_analysis": {"available": False, "root_cause": "", "reply": "", "confidence": 0.0},
        }

        state["analysis"] = analysis
        state["log_analysis"] = dict(analysis.get("log_analysis") or {})
        state["code_analysis"] = dict(analysis.get("code_analysis") or {})
        state["root_cause"] = root_cause
        state["solution"] = solution
        state["confidence"] = confidence
        state["route"] = "result_validate"
        return dict(state)

    log_analysis = _analyze_log_evidence(question=question, rows=log_rows)
    code_analysis = _analyze_code_evidence(rows=code_rows, log_rows=log_rows)

    root_cause = str(log_analysis.get("root_cause") or code_analysis.get("root_cause") or "").strip()
    confidence = max(_coerce_confidence(log_analysis.get("confidence")), _coerce_confidence(code_analysis.get("confidence")))
    solution = _compose_reply(log_analysis, code_analysis)
    if not root_cause and evidence.strip():
        fallback = analyze_with_llm(question=question, evidence=evidence)
        root_cause = str(fallback.get("root_cause") or "").strip()
        confidence = max(confidence, _coerce_confidence(fallback.get("confidence")))
        if not log_analysis.get("available") and str(fallback.get("reply") or "").strip():
            log_analysis = {
                "available": True,
                "root_cause": root_cause,
                "reply": str(fallback.get("reply") or "").strip(),
                "confidence": _coerce_confidence(fallback.get("confidence")),
            }
            solution = _compose_reply(log_analysis, code_analysis)

    analysis = {
        "root_cause": root_cause,
        "confidence": _confidence_label(confidence),
        "reply": solution,
        "log_analysis": log_analysis,
        "code_analysis": code_analysis,
    }

    state["analysis"] = analysis
    state["log_analysis"] = log_analysis
    state["code_analysis"] = code_analysis
    state["root_cause"] = root_cause
    state["solution"] = solution
    state["confidence"] = confidence
    state["route"] = "result_validate"
    return dict(state)
