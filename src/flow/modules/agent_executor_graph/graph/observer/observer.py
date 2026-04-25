"""Observer 节点：分析单步结果并决策是否调整计划。"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState

_RETRYABLE_TOKENS = ("timeout", "network", "connection", "temporarily", "503", "429")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_retryable_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return any(token in lowered for token in _RETRYABLE_TOKENS)


def _count_recent_failures(execution_history: dict[str, Any], window: int = 2) -> int:
    keys = sorted(execution_history.keys(), key=lambda item: _as_int(str(item).split("_")[-1], 0))
    selected = keys[-window:]
    count = 0
    for key in selected:
        raw_result = dict(dict(execution_history.get(key) or {}).get("raw_result") or {})
        if not bool(raw_result.get("ok", True)):
            count += 1
    return count


def _extract_effective_info(raw_result: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    info = dict(raw_result.get("effective_info") or {})
    keywords = [str(item).strip() for item in list(info.get("keywords") or []) if str(item).strip()]
    facts = dict(info.get("facts") or {})
    return keywords, facts


def _build_merged_evidence(state: dict[str, Any], execution_history: dict[str, Any]) -> tuple[dict[str, Any], str]:
    structured_context = dict(state.get("structured_context") or {})
    docs = [dict(item) for item in list(state.get("rag_docs") or [])]
    docs.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

    merged = {
        "logs": [],
        "knowledge": [],
        "context": {
            "order_id": structured_context.get("order_id") or "",
            "request_id": structured_context.get("request_id") or "",
            "intent_type": state.get("intent_type") or "UNKNOWN",
        },
    }

    for item in docs[:6]:
        merged["knowledge"].append(
            {
                "id": item.get("id"),
                "source": item.get("source") or "rag",
                "score": _safe_float(item.get("score"), 0.0),
                "text": str(item.get("text") or ""),
            }
        )

    keys = sorted(execution_history.keys(), key=lambda item: _as_int(str(item).split("_")[-1], 0))
    for key in keys:
        item = dict(execution_history.get(key) or {})
        step = dict(item.get("step") or {})
        raw_result = dict(item.get("raw_result") or {})
        processed = dict(item.get("processed") or {})
        tool_name = str(raw_result.get("tool") or step.get("tool_name") or "")
        evidence_rows = [str(row) for row in list(raw_result.get("evidence") or []) if str(row).strip()]
        for idx, text in enumerate(evidence_rows, start=1):
            record = {"id": f"{key}-tool-{idx}", "source": tool_name or "tool", "score": 1.0, "text": text}
            if "log" in tool_name:
                merged["logs"].append(record)
            else:
                merged["knowledge"].append(record)
        summary = str(processed.get("summary") or "").strip()
        if summary:
            merged["knowledge"].append(
                {
                    "id": f"{key}-summary",
                    "source": "observer_step_summary",
                    "score": 0.7,
                    "text": summary,
                }
            )

    evidence_rows = [str(item.get("text") or "") for item in merged["logs"] + merged["knowledge"]]
    evidence_text = "\n".join(row for row in evidence_rows if row).strip()
    return merged, evidence_text


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    current_plan = list(state.get("current_plan") or state.get("plan_steps") or [])
    current_step_index = _as_int(state.get("current_step_index"), 0)
    execution_history = dict(state.get("execution_history") or {})
    current_step_result = dict(state.get("current_step_result") or {})
    raw_result = dict(current_step_result.get("raw_result") or {})
    step = dict(current_step_result.get("step") or {})
    clues = [str(item).strip() for item in list(state.get("newly_discovered_clues") or []) if str(item).strip()]
    model_keywords, effective_facts = _extract_effective_info(raw_result)
    observed_keywords = model_keywords or clues
    plan_complete = current_step_index >= len(current_plan)

    needs_adjustment = False
    adjustment_type = ""
    proposed_changes: dict[str, Any] = {}
    pending_insertions: list[dict[str, Any]] = []

    recent_failures = _count_recent_failures(execution_history, window=2)
    replan_scope = str(effective_facts.get("replan_scope") or "").strip().lower()
    if replan_scope == "global":
        needs_adjustment = True
        adjustment_type = "global"
        proposed_changes = {
            "reason": "effective_info_global_replan",
            "details": dict(effective_facts),
        }
    elif recent_failures >= 2:
        needs_adjustment = True
        adjustment_type = "global"
        proposed_changes = {"reason": "consecutive_step_failures", "recent_failures": recent_failures}
    elif not bool(raw_result.get("ok", True)) and _is_retryable_error(str(raw_result.get("error") or "")):
        needs_adjustment = True
        adjustment_type = "local"
        retry_count = _as_int(state.get("retry_count"), 0)
        max_retries = max(0, _as_int(state.get("max_retries"), 2))
        if retry_count < max_retries:
            state["retry_count"] = retry_count + 1
        params = dict(step.get("params") or {})
        pending_insertions = [
            {
                "action_type": "tool_call",
                "tool_name": str(step.get("tool_name") or "log_query"),
                "params": params,
            }
        ]
        proposed_changes = {"reason": "retryable_step_failure", "insert_after_index": current_step_index - 1}
    elif not plan_complete and observed_keywords and str(step.get("tool_name") or "") == "log_query":
        next_step = current_plan[current_step_index] if current_step_index < len(current_plan) else {}
        next_tool = str(dict(next_step or {}).get("tool_name") or "")
        if next_tool != "dependency_log_query":
            needs_adjustment = True
            adjustment_type = "local"
            pending_insertions = [
                {
                    "action_type": "tool_call",
                    "tool_name": "dependency_log_query",
                    "params": {"keywords": observed_keywords[:5]},
                }
            ]
            proposed_changes = {
                "reason": "new_clues_discovered",
                "insert_after_index": current_step_index - 1,
                "keyword_source": "effective_info" if model_keywords else "step_processed",
            }

    state["needs_adjustment"] = needs_adjustment
    state["adjustment_type"] = adjustment_type
    state["proposed_changes"] = proposed_changes
    state["pending_insertions"] = pending_insertions

    if needs_adjustment:
        state["route"] = "planner" if adjustment_type == "global" else "reactor"
        return dict(state)

    if current_step_index < len(current_plan):
        state["route"] = "executor"
        return dict(state)

    merged_evidence, evidence_context = _build_merged_evidence(state=state, execution_history=execution_history)
    structured_context = dict(state.get("structured_context") or {})
    state["merged_evidence"] = merged_evidence
    state["evidence"] = merged_evidence
    state["structured_context"] = {
        **structured_context,
        "evidence_context": evidence_context,
        "merged_evidence_summary": {
            "logs_count": len(list(merged_evidence.get("logs") or [])),
            "knowledge_count": len(list(merged_evidence.get("knowledge") or [])),
            "keywords_count": len(list(state.get("extracted_keywords") or [])),
        },
    }
    state["route"] = "analysis_execute"
    return dict(state)

