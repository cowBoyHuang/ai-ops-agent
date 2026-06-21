"""Root cause analysis 节点。"""

from __future__ import annotations

import json
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_FAILURE_HINTS = (
    "失败",
    "异常",
    "error",
    "timeout",
    "超时",
    "拒绝",
    "not found",
    "校验不通过",
    "blocked",
    "500",
    "404",
)
_SUCCESS_VALUES = {"0", "200", "ok", "success", "true", "none", "null", ""}
_ERROR_CODE_REGEX = re.compile(
    r"(?:suberrorcode|refsuberrorcode|errorcode|bizerrorcode|errno|status|code)[\"'\s:=]+([A-Za-z0-9_-]{1,32})",
    re.IGNORECASE,
)


def _build_evidence_chain(evidence_rows: list[dict[str, Any]]) -> list[str]:
    chain: list[str] = []
    for idx, row in enumerate(list(evidence_rows or []), start=1):
        objective = str(row.get("objective") or "").strip() or f"目标{idx}"
        summary = str(row.get("summary") or row.get("observation") or "").strip()
        conclusion = str(row.get("conclusion") or "neutral").strip()
        chain.append(f"{idx}. [{objective}] {summary} (结论: {conclusion})")
    return chain


def _row_text(row: dict[str, Any]) -> str:
    raw_result = dict(row.get("raw_result") or {})
    effective_info = dict(raw_result.get("effective_info") or {})
    evidence_lines = [str(item).strip() for item in list(raw_result.get("evidence") or []) if str(item).strip()]
    return "\n".join(
        part
        for part in [
            str(row.get("summary") or "").strip(),
            str(row.get("observation") or "").strip(),
            str(effective_info.get("summary") or "").strip(),
            "\n".join(evidence_lines),
            json.dumps(effective_info.get("facts") or {}, ensure_ascii=False),
        ]
        if part
    )


def _is_failure_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if not text:
        return False
    return text not in _SUCCESS_VALUES


def _row_has_failure_signal(row: dict[str, Any]) -> bool:
    raw_result = dict(row.get("raw_result") or {})
    if raw_result.get("ok") is False:
        return True
    effective_info = dict(raw_result.get("effective_info") or {})
    facts = dict(effective_info.get("facts") or {})
    for key in ("status", "code", "errorCode", "subErrorCode", "refSubErrorCode", "bizErrorCode", "errno"):
        if key in facts and _is_failure_value(facts.get(key)):
            return True

    text = _row_text(row)
    if any(token in text.lower() for token in _FAILURE_HINTS):
        return True
    for match in _ERROR_CODE_REGEX.finditer(text):
        if _is_failure_value(match.group(1)):
            return True
    return False


def _extract_error_codes(text: str) -> list[str]:
    codes: list[str] = []
    for match in _ERROR_CODE_REGEX.finditer(text):
        value = str(match.group(1) or "").strip()
        lowered = value.lower()
        if value and lowered not in _SUCCESS_VALUES and value not in codes:
            codes.append(value)
    return codes


def _derive_root_cause(hypothesis: str, evidence_rows: list[dict[str, Any]]) -> str:
    rows = [dict(item or {}) for item in list(evidence_rows or [])]
    support_rows = [row for row in rows if str(row.get("conclusion") or "").strip() == "supports"]
    failure_rows = [row for row in rows if _row_has_failure_signal(row)]
    candidate = (support_rows or failure_rows or rows)[-1] if (support_rows or failure_rows or rows) else {}

    if candidate:
        objective = str(candidate.get("objective") or "关键排查目标").strip()
        summary = str(candidate.get("summary") or candidate.get("observation") or "").strip()
        row_text = _row_text(candidate)
        codes = _extract_error_codes(row_text)
        code_text = f"（关键码: {', '.join(codes[:4])}）" if codes else ""
        if summary:
            return f"{objective}发现异常：{summary}{code_text}"
        if row_text:
            return f"{objective}发现异常证据{code_text}"
        return f"{objective}存在异常"

    if hypothesis:
        return f"当前证据不足，暂无法确认“{hypothesis}”是否成立"
    return "当前证据未形成明确根因"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    plan = dict(state.get("plan") or {})
    execution = dict(state.get("execution") or {})
    evidence_graph = dict(execution.get("evidence_graph") or {})
    evidence_rows = [dict(item or {}) for item in list(evidence_graph.get("evidence") or [])]

    hypothesis = str(plan.get("hypothesis") or evidence_graph.get("hypothesis") or "").strip()
    evidence_chain = _build_evidence_chain(evidence_rows)
    root_cause = _derive_root_cause(hypothesis, evidence_rows)

    troubleshooting_process = [
        "基于假设生成调查目标",
        "按目标动态选择技能执行",
        "将每轮观察结果写入 Evidence Graph",
        "Evaluator 判定证据是否支持当前假设",
    ]

    analysis = dict(state.get("analysis") or {})
    final_reply = (
        f"根因判断：{root_cause}\n"
        f"证据链：{'；'.join(evidence_chain) if evidence_chain else '暂无'}"
    )
    analysis.update(
        {
            "root_cause": root_cause,
            "evidence_chain": evidence_chain,
            "troubleshooting_process": troubleshooting_process,
            "reply": final_reply,
        }
    )

    state["analysis"] = analysis
    state["root_cause"] = root_cause
    state["solution"] = "按证据链定位到异常模块后，补充对应修复和验证。"
    state["analysis_status"] = "SUCCESS"
    state["route"] = "finish"
    return dict(state)
