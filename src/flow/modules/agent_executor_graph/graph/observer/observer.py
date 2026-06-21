"""Observer 节点：根据 reactor_report 决定 next-goal / replan / finish。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm

_LOGGER = logging.getLogger(__name__)
_REQUIRED_UNKNOWN_VALUES = {"", "none", "null", "unknown", "n/a", "-"}
_LOG_QUERY_TOOLS = {"log_query", "dependency_log_query", "query_log"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: Any, max_len: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


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


def _build_action_chain_summary(action_chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in list(action_chain or []):
        row = dict(item or {})
        summary.append(
            {
                "seq": _as_int(row.get("seq"), 0),
                "tool_name": str(row.get("tool_name") or "").strip(),
                "skill": str(row.get("skill") or "").strip(),
                "params_summary": dict(row.get("params_summary") or {}),
                "result_summary": str(row.get("result_summary") or "").strip(),
                "ok": bool(row.get("ok")),
                "error": str(row.get("error") or "").strip(),
                "conclusion": str(row.get("conclusion") or "").strip(),
            }
        )
    return summary


def _has_successful_log_action(goal_reports: list[dict[str, Any]]) -> bool:
    for report in list(goal_reports or []):
        action_chain = [dict(item or {}) for item in list(dict(report or {}).get("action_chain") or [])]
        for action in action_chain:
            tool_name = str(action.get("tool_name") or "").strip().lower()
            if tool_name not in _LOG_QUERY_TOOLS:
                continue
            if bool(action.get("ok")):
                return True
    return False


def _build_replan_context(*, state: dict[str, Any], report: dict[str, Any], reason: str) -> dict[str, Any]:
    action_chain = _build_action_chain_summary(list(report.get("action_chain") or []))
    tool_failures = [
        {
            "seq": row.get("seq"),
            "tool_name": row.get("tool_name"),
            "error": row.get("error"),
            "conclusion": row.get("conclusion"),
        }
        for row in action_chain
        if not bool(row.get("ok"))
    ]
    hypothesis = str(dict(state.get("plan") or {}).get("hypothesis") or "").strip()
    rejected = [str(item).strip() for item in list(state.get("rejected_hypothesis") or []) if str(item).strip()]
    return {
        "failed_goal": str(report.get("goal_objective") or "").strip(),
        "failure_reason": reason,
        "action_chain_summary": action_chain,
        "tool_failures": tool_failures,
        "previous_hypothesis": hypothesis,
        "rejected_hypothesis": rejected,
    }


def _append_goal_report(execution: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    goal_reports = [dict(item or {}) for item in list(execution.get("goal_reports") or [])]
    normalized = {
        "goal_index": _as_int(report.get("goal_index"), 0),
        "goal_objective": str(report.get("goal_objective") or "").strip(),
        "goal_status": str(report.get("goal_status") or "").strip(),
        "act_times": _as_int(report.get("act_times"), 0),
        "retry_count": _as_int(report.get("retry_count"), 0),
        "goal_conclusion": str(report.get("goal_conclusion") or "").strip(),
        "failure_reason": str(report.get("failure_reason") or "").strip(),
        "action_chain": _build_action_chain_summary(list(report.get("action_chain") or [])),
    }
    goal_reports.append(normalized)
    execution["goal_reports"] = goal_reports
    return goal_reports


def _normalize_required_answers(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("required_answers")
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        field = ""
        question = ""
        required = True
        if isinstance(item, str):
            field = str(item).strip()
            question = f"给出字段 {field} 的取值" if field else ""
        elif isinstance(item, dict):
            field = str(item.get("field") or item.get("name") or item.get("key") or "").strip()
            question = str(item.get("question") or item.get("objective") or "").strip()
            required = bool(item.get("required", True))
        if not field:
            continue
        lowered = field.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(
            {
                "field": field,
                "question": question or f"给出字段 {field} 的取值",
                "required": required,
            }
        )
    return normalized


def _collect_evidence_texts(state: dict[str, Any], goal_reports: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    execution = dict(state.get("execution") or {})
    evidence_graph = dict(execution.get("evidence_graph") or {})
    for row in list(evidence_graph.get("evidence") or []):
        item = dict(row or {})
        for key in ("summary", "observation"):
            value = str(item.get(key) or "").strip()
            if value:
                texts.append(value)
        raw_result = dict(item.get("raw_result") or {})
        effective_info = dict(raw_result.get("effective_info") or {})
        summary = str(effective_info.get("summary") or "").strip()
        if summary:
            texts.append(summary)
        facts = effective_info.get("facts")
        if isinstance(facts, dict) and facts:
            texts.append(json.dumps(facts, ensure_ascii=False))
        for value in list(raw_result.get("evidence") or []):
            text = str(value or "").strip()
            if text:
                texts.append(text)

    for report in goal_reports:
        action_chain = [dict(item or {}) for item in list(dict(report or {}).get("action_chain") or [])]
        for action in action_chain:
            for key in ("result_summary", "error"):
                value = str(action.get(key) or "").strip()
                if value:
                    texts.append(value)
    return texts


def _find_required_field_value(*, field: str, evidence_texts: list[str]) -> tuple[str, str]:
    escaped = re.escape(str(field or "").strip())
    if not escaped:
        return "", "none"
    pattern = re.compile(rf'(?i)["\']?{escaped}["\']?\s*[:=]\s*["\']?([A-Za-z0-9_.-]+)')
    for text in evidence_texts:
        for match in pattern.finditer(str(text or "")):
            value = str(match.group(1) or "").strip().strip('",')
            if value and value.lower() not in _REQUIRED_UNKNOWN_VALUES:
                return value, "exact_text"

    semantic_value = _infer_required_field_value_semantic(field=field, evidence_texts=evidence_texts)
    if semantic_value:
        return semantic_value, "semantic_fallback"
    return "", "none"


def _infer_required_field_value_semantic(*, field: str, evidence_texts: list[str]) -> str:
    target = str(field or "").strip()
    if not target:
        return ""
    samples = [_clip(item, 1200) for item in evidence_texts[:12] if str(item or "").strip()]
    if not samples:
        return ""
    system_prompt = (
        "你是日志字段抽取助手。当前目标字段在日志中没有精确字段名命中时，"
        "可根据相似字段或自然语言语义给出最可能字段值；若无法判断则返回空字符串。"
        "仅返回 JSON: {\"value\":\"...\",\"matched_by\":\"similar_field|natural_language|none\",\"confidence\":\"high|medium|low\"}."
    )
    user_prompt = json.dumps({"target_field": target, "evidence_texts": samples}, ensure_ascii=False)
    try:
        raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    except Exception:  # noqa: BLE001
        return ""
    parsed = _parse_json_object(raw) or {}
    matched_by = str(parsed.get("matched_by") or "").strip().lower()
    value = str(parsed.get("value") or "").strip()
    if matched_by not in {"similar_field", "natural_language"}:
        return ""
    if not value or value.lower() in _REQUIRED_UNKNOWN_VALUES:
        return ""
    return value


def _resolve_required_answer_results(state: dict[str, Any], goal_reports: list[dict[str, Any]]) -> dict[str, Any]:
    plan = dict(state.get("plan") or {})
    required_answers = _normalize_required_answers(plan)
    if not required_answers:
        return {
            "required_answers": [],
            "resolved": {},
            "missing": [],
            "evidence_text_count": 0,
        }
    evidence_texts = _collect_evidence_texts(state, goal_reports)
    resolved: dict[str, str] = {}
    resolved_source: dict[str, str] = {}
    missing: list[dict[str, Any]] = []
    for item in required_answers:
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        value, source = _find_required_field_value(field=field, evidence_texts=evidence_texts)
        if value:
            resolved[field] = value
            resolved_source[field] = source
        elif bool(item.get("required", True)):
            missing.append(dict(item))
    return {
        "required_answers": required_answers,
        "resolved": resolved,
        "resolved_source": resolved_source,
        "missing": missing,
        "evidence_text_count": len(evidence_texts),
    }


def _decide_post_plan_route(*, state: dict[str, Any], goal_reports: list[dict[str, Any]]) -> tuple[str, str]:
    question = str(state.get("question") or "").strip()
    plan = dict(state.get("plan") or {})
    summary_input = {
        "user_question": question,
        "hypothesis": str(plan.get("hypothesis") or "").strip(),
        "investigation_goals": [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()],
        "goal_reports": goal_reports,
        "replan_count": _as_int(state.get("replan_count"), 0),
        "max_replan": _as_int(state.get("max_replan"), 0),
    }

    system_prompt = (
        "你是排障流程决策器。必须基于完整 goal 报告决定是否需要 replan。"
        "只返回 JSON: {\"decision\":\"finish|replan\",\"reason\":\"...\"}."
    )
    user_prompt = json.dumps(summary_input, ensure_ascii=False)
    try:
        raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    except Exception as exc:  # noqa: BLE001
        return "finish", f"post_plan_llm_error:{exc}"

    parsed = _parse_json_object(raw) or {}
    decision = str(parsed.get("decision") or "finish").strip().lower()
    reason = str(parsed.get("reason") or "post_plan_default_finish").strip()
    if decision not in {"finish", "replan"}:
        return "finish", "post_plan_invalid_decision"
    return decision, reason


def _build_summary_input(
    *,
    state: dict[str, Any],
    goal_reports: list[dict[str, Any]],
    decision: str,
    reason: str,
) -> dict[str, Any]:
    plan = dict(state.get("plan") or {})
    execution = dict(state.get("execution") or {})
    evidence_graph = dict(execution.get("evidence_graph") or {})
    return {
        "user_question": str(state.get("question") or "").strip(),
        "hypothesis": str(plan.get("hypothesis") or "").strip(),
        "investigation_goals": [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()],
        "required_answers": _normalize_required_answers(plan),
        "goal_reports": goal_reports,
        "evidence_graph": evidence_graph,
        "required_answer_results": dict(execution.get("required_answer_results") or {}),
        "final_decision": decision,
        "decision_reason": reason,
    }


def _build_fallback_final_analysis(summary_input: dict[str, Any]) -> dict[str, Any]:
    goal_reports = [dict(item or {}) for item in list(summary_input.get("goal_reports") or [])]
    evidence_chain: list[str] = []
    last_conclusion = ""
    for idx, report in enumerate(goal_reports, start=1):
        objective = str(report.get("goal_objective") or f"目标{idx}").strip()
        goal_status = str(report.get("goal_status") or "").strip()
        conclusion = str(report.get("goal_conclusion") or report.get("failure_reason") or "").strip()
        action_chain = [dict(item or {}) for item in list(report.get("action_chain") or [])]
        action_summary = ""
        if action_chain:
            action_summary = str(action_chain[-1].get("result_summary") or action_chain[-1].get("error") or "").strip()
        line_parts = [f"[{objective}] status={goal_status}"]
        if conclusion:
            line_parts.append(conclusion)
            last_conclusion = conclusion
        if action_summary and action_summary not in line_parts:
            line_parts.append(action_summary)
            last_conclusion = action_summary
        evidence_chain.append("；".join(part for part in line_parts if part))
        if len(evidence_chain) >= 6:
            break

    root_cause = last_conclusion or str(summary_input.get("decision_reason") or "当前证据不足以形成明确根因").strip()
    solution = "建议按证据链中的异常模块与错误码继续验证，并补充对应修复与回归。"
    reply = f"根因判断：{root_cause}\n证据链：{'；'.join(evidence_chain) if evidence_chain else '暂无明确证据链'}"
    return {
        "root_cause": root_cause,
        "solution": solution,
        "evidence_chain": evidence_chain,
        "reply": reply,
        "summary_input": summary_input,
    }


def _synthesize_final_analysis(
    *,
    state: dict[str, Any],
    goal_reports: list[dict[str, Any]],
    decision: str,
    reason: str,
) -> dict[str, Any]:
    summary_input = _build_summary_input(
        state=state,
        goal_reports=goal_reports,
        decision=decision,
        reason=reason,
    )
    system_prompt = (
        "你是排障 Root Cause Analysis 生成器。"
        "请基于用户问题+执行证据输出最终结论，返回 JSON："
        "{\"root_cause\":\"...\",\"solution\":\"...\",\"evidence_chain\":[\"...\"],\"reply\":\"...\"}。"
        "要求：结论必须引用 evidence 中的信息，不得编造。"
        "你只能把日志中完整、明确出现的字段当作已确认事实。"
        "若字段只是类似名称（例如 subErrorCode 与 bizErrorCode）或仅可推断，必须使用“可能/疑似/推测”等不确定性措辞，"
        "禁止把类似字段直接当作目标字段的确定值。"
        "若 required_answers 中某字段未在日志原文中明确出现，必须在 reply 中明确说明“该字段未在日志中直接出现”。"
    )
    try:
        raw = chat_with_llm(question=json.dumps(summary_input, ensure_ascii=False), system_prompt=system_prompt)
    except Exception:  # noqa: BLE001
        return _build_fallback_final_analysis(summary_input)
    parsed = _parse_json_object(raw) or {}

    root_cause = str(parsed.get("root_cause") or "").strip()
    solution = str(parsed.get("solution") or "").strip()
    evidence_chain = [
        str(item).strip()
        for item in list(parsed.get("evidence_chain") or [])
        if str(item).strip()
    ]
    reply = str(parsed.get("reply") or "").strip()
    if not root_cause:
        return _build_fallback_final_analysis(summary_input)
    if not solution:
        solution = "建议根据异常模块补充保护与回归验证。"
    if not reply:
        reply = f"根因判断：{root_cause}\n证据链：{'；'.join(evidence_chain) if evidence_chain else '详见执行记录'}"
    return {
        "root_cause": root_cause,
        "solution": solution,
        "evidence_chain": evidence_chain,
        "reply": reply,
        "summary_input": summary_input,
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    plan = dict(state.get("plan") or {})
    goals = [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()]

    execution = dict(state.get("execution") or {})
    goal_index = _as_int(execution.get("goal_index"), 0)
    report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})
    goal_status = str(report.get("goal_status") or "").strip().lower()

    max_replan = max(0, _as_int(state.get("max_replan"), 0))
    replan_count = max(0, _as_int(state.get("replan_count"), 0))

    _LOGGER.info(
        "observer.route.decide goal_index=%d/%d goal_status=%s replan_count=%d/%d",
        goal_index + 1,
        len(goals),
        goal_status,
        replan_count,
        max_replan,
    )

    if report:
        goal_reports = _append_goal_report(execution, report)
    else:
        goal_reports = [dict(item or {}) for item in list(execution.get("goal_reports") or [])]

    if goal_status == "success":
        next_goal_index = goal_index + 1
        execution["goal_index"] = next_goal_index
        execution["reactor_runtime"] = {}

        if next_goal_index < len(goals):
            state["execution"] = execution
            state["route"] = "reactor"
            return dict(state)

        required_answer_results = _resolve_required_answer_results(state, goal_reports)
        execution["required_answer_results"] = required_answer_results
        state["execution"] = execution
        missing_required = [dict(item or {}) for item in list(required_answer_results.get("missing") or [])]
        if missing_required:
            missing_fields = [str(item.get("field") or "").strip() for item in missing_required if str(item.get("field") or "").strip()]
            _LOGGER.info(
                "observer.required_answers.missing fields=%s resolved=%s strategy=best_effort_finish_or_llm_decision replan_count=%d/%d",
                missing_fields,
                dict(required_answer_results.get("resolved") or {}),
                replan_count,
                max_replan,
            )
            # 当必答字段仍缺失且尚无成功日志查询时，至少回 Reactor 再补一次，避免直接 finish。
            if (
                not bool(execution.get("missing_required_reactor_retry_done"))
                and not _has_successful_log_action(goal_reports)
                and goals
            ):
                execution["goal_index"] = max(0, min(len(goals) - 1, next_goal_index - 1))
                execution["reactor_runtime"] = {}
                execution["missing_required_reactor_retry_done"] = True
                state["execution"] = execution
                _LOGGER.info(
                    "observer.route.decide missing_required_retry route=reactor fields=%s goal_index=%d",
                    missing_fields,
                    execution["goal_index"],
                )
                state["route"] = "reactor"
                return dict(state)
        else:
            execution.pop("missing_required_reactor_retry_done", None)

        decision, reason = _decide_post_plan_route(state=state, goal_reports=goal_reports)
        execution["final_decision"] = {"decision": decision, "reason": reason}
        state["execution"] = execution

        if decision == "replan" and replan_count < max_replan:
            replan_context = _build_replan_context(state=state, report=report, reason=reason)
            state["replan_context"] = replan_context
            state["replan_reason"] = reason
            _LOGGER.info("observer.route.decide post_plan decision=replan reason=%s", _clip(reason, 160))
            state["route"] = "replan"
            return dict(state)

        if decision == "replan" and replan_count >= max_replan:
            reason = f"{reason}; max_replan_reached={max_replan}"
        final_analysis = _synthesize_final_analysis(
            state=state,
            goal_reports=goal_reports,
            decision=decision,
            reason=reason,
        )
        analysis = dict(state.get("analysis") or {})
        state["analysis"] = {
            **analysis,
            "reply": str(final_analysis.get("reply") or ""),
            "root_cause": str(final_analysis.get("root_cause") or ""),
            "evidence_chain": list(final_analysis.get("evidence_chain") or []),
        }
        state["root_cause"] = str(final_analysis.get("root_cause") or "")
        state["solution"] = str(final_analysis.get("solution") or "")
        state["final_summary_input"] = dict(final_analysis.get("summary_input") or {})
        state["replan_reason"] = reason
        state["route"] = "finish"
        return dict(state)

    if goal_status in {"failed", "unexecutable"}:
        failure_reason = str(report.get("failure_reason") or f"goal_status={goal_status}").strip()
        replan_context = _build_replan_context(state=state, report=report, reason=failure_reason)
        state["replan_context"] = replan_context
        state["replan_reason"] = failure_reason
        _LOGGER.info(
            "observer.replan.context failed_goal=%s reason=%s tool_failures=%d",
            _clip(replan_context.get("failed_goal"), 120),
            _clip(failure_reason, 160),
            len(list(replan_context.get("tool_failures") or [])),
        )

        state["execution"] = execution
        if replan_count < max_replan:
            state["route"] = "replan"
            return dict(state)

        final_analysis = _synthesize_final_analysis(
            state=state,
            goal_reports=goal_reports,
            decision="finish",
            reason=f"{failure_reason}; max_replan_reached={max_replan}",
        )
        analysis = dict(state.get("analysis") or {})
        state["analysis"] = {
            **analysis,
            "reply": str(final_analysis.get("reply") or ""),
            "root_cause": str(final_analysis.get("root_cause") or ""),
            "evidence_chain": list(final_analysis.get("evidence_chain") or []),
        }
        state["root_cause"] = str(final_analysis.get("root_cause") or "")
        state["solution"] = str(final_analysis.get("solution") or "")
        state["final_summary_input"] = dict(final_analysis.get("summary_input") or {})
        state["route"] = "finish"
        return dict(state)

    if goal_index >= len(goals):
        final_analysis = _synthesize_final_analysis(
            state=state,
            goal_reports=goal_reports,
            decision="finish",
            reason="goals_exhausted",
        )
        analysis = dict(state.get("analysis") or {})
        state["analysis"] = {
            **analysis,
            "reply": str(final_analysis.get("reply") or ""),
            "root_cause": str(final_analysis.get("root_cause") or ""),
            "evidence_chain": list(final_analysis.get("evidence_chain") or []),
        }
        state["root_cause"] = str(final_analysis.get("root_cause") or "")
        state["solution"] = str(final_analysis.get("solution") or "")
        state["final_summary_input"] = dict(final_analysis.get("summary_input") or {})
        state["execution"] = execution
        state["route"] = "finish"
        return dict(state)

    # reactor_report 缺失或状态异常，进入 fallback 保护。
    analysis = dict(state.get("analysis") or {})
    state["analysis"] = {
        **analysis,
        "reply": str(analysis.get("reply") or "observer 无法识别 reactor_report，流程降级。"),
    }
    state["route"] = "fallback"
    return dict(state)
