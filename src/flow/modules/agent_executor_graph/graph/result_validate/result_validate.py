"""结果验证节点。

业务职责：
- 判断分析结果是否可直接返回（SUCCESS）。
- 识别是否需要重试工具（NEED_RETRY）。
- 识别是否需要重规划（NEED_REPLAN）。
- 超预算或明显失败时标记 FAIL。
"""

from __future__ import annotations

import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_FAILURE_HINTS = (
    "校验不通过",
    "生单失败",
    "order_failed",
    "errormsg",
    "success\":false",
    "resultok\":false",
    "block_reason",
)
_UNCERTAIN_HINTS = ("无法确定", "不确定", "未知", "unknown", "need more", "更多信息")
_BIZ_CODE_PATTERN = re.compile(r"\b\d{2}_[0-9A-Z]{3,}_[0-9A-Z]{3,}_[0-9]{4}\b")
_REASON_PATTERNS = (
    re.compile(r'"(?:errorMsg|errMsg|failRes|block_reason|reason|msg)"\s*:\s*"([^"]{4,220})"'),
    re.compile(r"(?:校验不通过|失败)[^:：]{0,24}[:：]\s*(?:\([^,，]*[,，])?([^)\n]{4,220})"),
)


def _has_retryable_error(error_text: str) -> bool:
    """是否为可重试错误（网络抖动、超时、限流等）。"""
    lowered = str(error_text or "").lower()
    retry_hints = ("timeout", "network", "connection", "temporarily", "503", "429")
    return any(token in lowered for token in retry_hints)


def _has_uncertain_answer(root_cause: str, solution: str) -> bool:
    """分析文案是否表达了“不确定/信息不足”。"""
    text = f"{root_cause} {solution}".lower()
    return any(token in text for token in _UNCERTAIN_HINTS)


def _to_confidence(value: Any) -> float:
    """统一把置信度转换为 0~1 的浮点值。"""
    text = str(value or "").strip().lower()
    if text == "high":
        return 0.9
    if text == "medium":
        return 0.65
    if text == "low":
        return 0.35
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_analysis_content(section: dict[str, Any]) -> bool:
    if bool(section.get("available")):
        return True
    if str(section.get("root_cause") or "").strip():
        return True
    if str(section.get("reply") or "").strip():
        return True
    if list(section.get("locations") or []):
        return True
    return False


def _extract_decisive_log_reason(merged_evidence: dict[str, Any]) -> str:
    """从日志证据中提取“可直接给结论”的失败原因。"""
    logs = list(merged_evidence.get("logs") or [])
    knowledge = list(merged_evidence.get("knowledge") or [])
    rows = [*logs, *knowledge]
    for item in rows:
        raw = str(dict(item or {}).get("text") or "").strip()
        if not raw:
            continue
        normalized = raw.replace('\\"', '"')
        lowered = normalized.lower()
        if not any(token in lowered for token in _FAILURE_HINTS):
            continue

        for pattern in _REASON_PATTERNS:
            matched = pattern.search(normalized)
            if not matched:
                continue
            reason = str(matched.group(1) or "").strip(" ,，:：'\"")
            # bizCode 只作为辅助证据，不能单独作为“关键定位证据”。
            if _BIZ_CODE_PATTERN.fullmatch(reason):
                continue
            if reason and not any(hint in reason.lower() for hint in _UNCERTAIN_HINTS):
                return reason

    return ""


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """验证本轮分析结果并输出 analysis_status。

    入参：
    - payload: AgentState，需包含 analysis/tool_result/merged_evidence 等字段。

    返参：
    - AgentState: 写入 analysis_status，并统一路由到 retry_router。
    """
    state: AgentState = dict(payload)
    analysis = dict(state.get("analysis") or {})
    tool_result = dict(state.get("tool_result") or {})
    merged_evidence = dict(state.get("merged_evidence") or {})

    root_cause = str(state.get("root_cause") or analysis.get("root_cause") or "").strip()
    solution = str(state.get("solution") or analysis.get("reply") or "").strip()
    confidence = _to_confidence(state.get("confidence") or analysis.get("confidence"))
    log_analysis = dict(state.get("log_analysis") or analysis.get("log_analysis") or {})
    code_analysis = dict(state.get("code_analysis") or analysis.get("code_analysis") or {})
    tool_ok = bool(tool_result.get("ok", True))
    tool_error = str(tool_result.get("error") or "")

    tool_call_count = int(state.get("tool_call_count") or 0)
    max_tool_calls = max(1, int(state.get("max_tool_calls") or 8))
    current_step_index = int(state.get("current_step_index") or 0)
    plan_steps = list(state.get("current_plan") or state.get("plan_steps") or [])
    has_more_plan_steps = current_step_index < len(plan_steps)
    logs = list(merged_evidence.get("logs") or [])
    knowledge = list(merged_evidence.get("knowledge") or [])
    code = list(merged_evidence.get("code") or [])
    has_evidence = bool(logs or knowledge or code)
    decisive_reason = _extract_decisive_log_reason(merged_evidence)
    if decisive_reason:
        root_cause = root_cause or decisive_reason
        solution = solution or f"日志已明确给出失败原因：{decisive_reason}。请按该原因调整请求参数或业务规则后重试。"
        confidence = max(confidence, 0.9)
        if not _has_analysis_content(log_analysis):
            log_analysis = {
                "available": True,
                "root_cause": decisive_reason,
                "reply": f"日志已明确给出失败原因：{decisive_reason}。",
                "confidence": 0.9,
            }
        state["root_cause"] = root_cause
        state["solution"] = solution
        state["confidence"] = confidence
        state["log_analysis"] = log_analysis
        state["analysis"] = {
            **analysis,
            "root_cause": root_cause,
            "reply": solution,
            "confidence": "high",
            "decision_source": "decisive_log_evidence",
            "log_analysis": log_analysis,
            "code_analysis": code_analysis,
        }

    has_log_analysis = _has_analysis_content(log_analysis)
    has_code_analysis = _has_analysis_content(code_analysis)
    has_any_analysis = has_log_analysis or has_code_analysis or bool(root_cause)

    # 判定顺序遵循“先硬限制，再可重试，再成功，再重规划”。
    if tool_call_count >= max_tool_calls:
        status = "SUCCESS" if has_any_analysis else "FAIL"
    elif not tool_ok and _has_retryable_error(tool_error):
        status = "NEED_RETRY"
    elif has_more_plan_steps and not (has_log_analysis and has_code_analysis):
        status = "NEED_RETRY"
    elif has_any_analysis:
        status = "SUCCESS"
    elif confidence > 0.7 and bool(root_cause):
        status = "SUCCESS"
    elif not has_evidence:
        status = "NEED_REPLAN"
    elif _has_uncertain_answer(root_cause, solution):
        status = "NEED_REPLAN"
    elif has_more_plan_steps:
        status = "NEED_RETRY"
    elif not root_cause:
        status = "NEED_REPLAN"
    else:
        status = "NEED_REPLAN"

    state["analysis_status"] = status
    state["route"] = "retry_router"
    return dict(state)
