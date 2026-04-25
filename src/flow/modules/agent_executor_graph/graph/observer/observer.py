"""Observer 节点：分析单步结果并决策是否调整计划。"""

from __future__ import annotations

import json
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm

_RETRYABLE_TOKENS = ("timeout", "network", "connection", "temporarily", "503", "429")
_FALLBACK_MESSAGE = "暂未能自动定位问题，请联系人工排查。"
_ALLOWED_TOOL_NAMES = {"log_query", "dependency_log_query", "knowledge_lookup", "code_clone", "code_pull"}


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


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clear_adjustment_fields(state: dict[str, Any]) -> None:
    state["needs_adjustment"] = False
    state["adjustment_type"] = ""
    state["proposed_changes"] = {}
    state["pending_insertions"] = []


def _route_replan_with_budget(state: dict[str, Any], reason: str) -> dict[str, Any]:
    replan_count = _as_int(state.get("replan_count"), 0)
    max_replan = max(0, _as_int(state.get("max_replan"), 2))
    if replan_count >= max_replan:
        analysis = dict(state.get("analysis") or {})
        state["analysis"] = {**analysis, "reply": str(analysis.get("reply") or _FALLBACK_MESSAGE)}
        state["final_answer"] = _FALLBACK_MESSAGE
        state["status"] = "degraded"
        _clear_adjustment_fields(state)
        state["route"] = "fallback"
        return dict(state)

    state["replan_count"] = replan_count + 1
    state["current_step_index"] = 0
    state["needs_adjustment"] = True
    state["adjustment_type"] = "global"
    state["proposed_changes"] = {"reason": reason}
    state["pending_insertions"] = []
    state["route"] = "planner"
    return dict(state)


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


def _is_new_problem(raw_result: dict[str, Any], effective_facts: dict[str, Any]) -> bool:
    for key in ("new_problem", "new_issue", "new_situation", "problem_mismatch"):
        if bool(effective_facts.get(key)):
            return True
    text_rows = [
        str(effective_facts.get("reason") or ""),
        str(raw_result.get("error") or ""),
        str(effective_facts.get("status") or ""),
    ]
    merged = " ".join(text_rows).lower()
    return any(token in merged for token in ("new problem", "new issue", "mismatch", "完全对不上", "新问题"))


def _build_local_adjustment_step(
    *,
    decision: dict[str, Any],
    step: dict[str, Any],
    observed_keywords: list[str],
) -> dict[str, Any] | None:
    tool_name = str(decision.get("tool_name") or "").strip()
    if tool_name and tool_name not in _ALLOWED_TOOL_NAMES:
        tool_name = ""
    params = decision.get("params")
    params_dict = dict(params) if isinstance(params, dict) else {}

    if not tool_name:
        if str(step.get("tool_name") or "") == "log_query":
            tool_name = "dependency_log_query"
            if observed_keywords:
                params_dict.setdefault("keywords", observed_keywords[:5])
        else:
            tool_name = str(step.get("tool_name") or "")

    if tool_name == "dependency_log_query" and observed_keywords and not list(params_dict.get("keywords") or []):
        params_dict["keywords"] = observed_keywords[:5]

    if not tool_name:
        return None
    return {"action_type": "tool_call", "tool_name": tool_name, "params": params_dict}


def _decide_adjust_or_replan_with_llm(
    *,
    state: dict[str, Any],
    step: dict[str, Any],
    raw_result: dict[str, Any],
    observed_keywords: list[str],
    plan_complete: bool,
) -> dict[str, Any]:
    fallback = {"decision": "replan", "reason": "llm_decision_parse_failed", "tool_name": "", "params": {}}
    user_prompt = (
        "你是排障流程路由器。请基于当前步骤执行结果判断下一步是局部调整(adjust)还是重规划(replan)。\n"
        "返回 JSON：decision(adjust|replan), reason, tool_name, params。\n"
        "规则：\n"
        "1) 如果日志无命中、结果与当前计划明显不匹配、出现新问题，请返回 replan。\n"
        "2) 如果仅需调整查询参数或替换下一步查询字段（例如按新错误码继续查），返回 adjust 并给 tool_name/params。\n"
        "3) 如果已无后续步骤且证据不足，返回 replan。\n\n"
        f"plan_complete={plan_complete}\n"
        f"question={str(state.get('question') or '')}\n"
        f"current_step={json.dumps(step, ensure_ascii=False)}\n"
        f"raw_result={json.dumps(raw_result, ensure_ascii=False)}\n"
        f"observed_keywords={json.dumps(observed_keywords, ensure_ascii=False)}"
    )
    parsed = _parse_json_object(chat_with_llm(question=user_prompt, system_prompt=""))
    if not parsed:
        return fallback

    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"adjust", "replan"}:
        return fallback
    return {
        "decision": decision,
        "reason": str(parsed.get("reason") or ""),
        "tool_name": str(parsed.get("tool_name") or ""),
        "params": dict(parsed.get("params") or {}),
    }


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

    _clear_adjustment_fields(state)
    tool_ok = bool(raw_result.get("ok", True))
    tool_error = str(raw_result.get("error") or "")
    evidence_rows = [str(item).strip() for item in list(raw_result.get("evidence") or []) if str(item).strip()]
    recent_failures = _count_recent_failures(execution_history, window=2)
    replan_scope = str(effective_facts.get("replan_scope") or "").strip().lower()
    if replan_scope == "global":
        return _route_replan_with_budget(state, "effective_info_global_replan")

    if _is_new_problem(raw_result, effective_facts):
        return _route_replan_with_budget(state, "new_problem_detected")

    if not tool_ok and _is_retryable_error(tool_error):
        retry_count = _as_int(state.get("retry_count"), 0)
        max_retries = max(0, _as_int(state.get("max_retries"), 2))
        if retry_count < max_retries:
            state["retry_count"] = retry_count + 1
            state["current_step_index"] = max(0, current_step_index - 1)
            state["route"] = "executor"
            return dict(state)
        return _route_replan_with_budget(state, "retryable_error_retry_exhausted")

    if recent_failures >= 2:
        return _route_replan_with_budget(state, "consecutive_step_failures")

    if not tool_ok and not evidence_rows:
        return _route_replan_with_budget(state, "no_logs_or_evidence")

    next_step = current_plan[current_step_index] if current_step_index < len(current_plan) else {}
    next_tool = str(dict(next_step or {}).get("tool_name") or "")
    followup_mismatch = (
        not plan_complete
        and bool(observed_keywords)
        and str(step.get("tool_name") or "") == "log_query"
        and next_tool != "dependency_log_query"
    )
    should_call_llm = (not tool_ok) or followup_mismatch
    if should_call_llm:
        decision = _decide_adjust_or_replan_with_llm(
            state=state,
            step=step,
            raw_result=raw_result,
            observed_keywords=observed_keywords,
            plan_complete=plan_complete,
        )
        if str(decision.get("decision") or "") == "replan":
            return _route_replan_with_budget(state, str(decision.get("reason") or "llm_decision_replan"))
        local_step = _build_local_adjustment_step(decision=decision, step=step, observed_keywords=observed_keywords)
        if local_step and not plan_complete:
            state["needs_adjustment"] = True
            state["adjustment_type"] = "local"
            state["proposed_changes"] = {
                "reason": str(decision.get("reason") or "llm_decision_adjust"),
                "insert_after_index": current_step_index - 1,
            }
            state["pending_insertions"] = [local_step]
            state["route"] = "reactor"
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
