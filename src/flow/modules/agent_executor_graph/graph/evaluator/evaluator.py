"""Evidence-based evaluator。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_LOGGER = logging.getLogger(__name__)
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
_ERROR_FACT_KEYS = (
    "status",
    "code",
    "errorCode",
    "subErrorCode",
    "refSubErrorCode",
    "bizErrorCode",
    "errno",
)
_ERROR_CODE_REGEX = re.compile(
    r"(?:suberrorcode|refsuberrorcode|errorcode|bizerrorcode|errno|status|code)[\"'\s:=]+([A-Za-z0-9_-]{1,32})",
    re.IGNORECASE,
)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_by_conclusion(evidence: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"supports": 0, "refutes": 0, "neutral": 0, "insufficient": 0}
    for row in list(evidence or []):
        conclusion = str(dict(row or {}).get("conclusion") or "neutral").strip().lower()
        if conclusion not in stats:
            conclusion = "neutral"
        stats[conclusion] += 1
    return stats


def _is_failure_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if not text:
        return False
    return text not in _SUCCESS_VALUES


def _has_failure_hint(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in _FAILURE_HINTS)


def _row_has_concrete_failure_evidence(row: dict[str, Any]) -> bool:
    raw_result = dict(row.get("raw_result") or {})
    if raw_result.get("ok") is False:
        return True

    effective_info = dict(raw_result.get("effective_info") or {})
    facts = dict(effective_info.get("facts") or {})
    for key in _ERROR_FACT_KEYS:
        if key in facts and _is_failure_value(facts.get(key)):
            return True

    evidence_lines = [str(item).strip() for item in list(raw_result.get("evidence") or []) if str(item).strip()]
    text = "\n".join(
        part
        for part in [
            str(row.get("summary") or "").strip(),
            str(row.get("observation") or "").strip(),
            str(effective_info.get("summary") or "").strip(),
            "\n".join(evidence_lines),
            json.dumps(raw_result, ensure_ascii=False),
        ]
        if part
    )
    if _has_failure_hint(text):
        return True

    for match in _ERROR_CODE_REGEX.finditer(text):
        if _is_failure_value(match.group(1)):
            return True
    return False


def _has_concrete_failure_evidence(evidence: list[dict[str, Any]]) -> bool:
    return any(_row_has_concrete_failure_evidence(dict(row or {})) for row in list(evidence or []))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    plan = dict(state.get("plan") or {})
    execution = dict(state.get("execution") or {})
    evidence_graph = dict(execution.get("evidence_graph") or {})
    evidence_rows = [dict(item or {}) for item in list(evidence_graph.get("evidence") or [])]
    insufficient_round_count = _as_int(execution.get("insufficient_round_count"), 0)
    max_insufficient_rounds = max(
        1,
        _as_int(
            execution.get("max_insufficient_rounds"),
            _as_int(state.get("max_insufficient_rounds"), 2),
        ),
    )

    goals = [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()]
    covered_goals = {str(item.get("objective") or "").strip() for item in evidence_rows if str(item.get("objective") or "").strip()}
    has_failure_evidence = _has_concrete_failure_evidence(evidence_rows)

    if evidence_graph.get("supported") is True:
        if has_failure_evidence:
            status = "supported"
            reason = "evidence_graph.supported=true with concrete failure evidence"
        else:
            status = "insufficient"
            reason = "supported flag is set but concrete failure evidence is missing"
    elif evidence_graph.get("supported") is False:
        status = "unsupported"
        reason = "evidence_graph.supported=false"
    else:
        stats = _count_by_conclusion(evidence_rows)
        if stats["refutes"] > 0 and stats["supports"] == 0:
            status = "unsupported"
            reason = "all key evidence refutes current hypothesis"
        elif goals and len(covered_goals) >= len(goals) and stats["supports"] > 0 and stats["refutes"] == 0 and has_failure_evidence:
            status = "supported"
            reason = "all investigation goals are covered with supportive failure evidence"
        elif stats["supports"] > 0 and stats["refutes"] == 0 and not goals:
            if has_failure_evidence:
                status = "supported"
                reason = "supportive failure evidence exists"
            else:
                status = "insufficient"
                reason = "supportive conclusion exists but concrete failure evidence is missing"
        else:
            status = "insufficient"
            reason = "current evidence is not enough to prove or falsify hypothesis"

    if status == "insufficient":
        insufficient_round_count += 1
        if insufficient_round_count >= max_insufficient_rounds:
            status = "unsupported"
            reason = f"insufficient evidence rounds reached limit={max_insufficient_rounds}"
            insufficient_round_count = 0
    else:
        insufficient_round_count = 0

    execution["insufficient_round_count"] = insufficient_round_count
    execution["max_insufficient_rounds"] = max_insufficient_rounds
    state["execution"] = execution
    state["evaluation"] = {
        "status": status,
        "reason": reason,
        "covered_goals": sorted(item for item in covered_goals if item),
        "total_goals": len(goals),
        "evidence_count": len(evidence_rows),
        "has_concrete_failure_evidence": has_failure_evidence,
        "insufficient_round_count": insufficient_round_count,
        "max_insufficient_rounds": max_insufficient_rounds,
    }
    _LOGGER.info(
        "evaluator decided status=%s reason=%s total_goals=%d covered_goals=%d evidence_count=%d has_failure_evidence=%s stats=%s insufficient_round_count=%d/%d",
        status,
        reason,
        len(goals),
        len(covered_goals),
        len(evidence_rows),
        has_failure_evidence,
        _count_by_conclusion(evidence_rows),
        insufficient_round_count,
        max_insufficient_rounds,
    )
    state["route"] = status
    return dict(state)
