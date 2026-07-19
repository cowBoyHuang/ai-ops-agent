"""Reactor 节点：在单个 goal 内执行动作循环，输出 reactor_report 给 observer。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.executor import executor as executor_mod
from llm.llm import chat_with_llm

_LOGGER = logging.getLogger(__name__)
_MAX_RETRY_PER_GOAL = 3
_REQUIRED_UNKNOWN_VALUES = {"", "none", "null", "unknown", "n/a", "-", "未命中", "未检索到日志命中"}
_TRACE_ID_PATTERN = re.compile(r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)", re.IGNORECASE)
_ORDER_ID_PATTERN = re.compile(r"\b(?:xep|sid|hpv|zvp)[A-Za-z0-9]{6,}\b", re.IGNORECASE)
_ASCII_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{6,128}$")
_QUERYLOG_METHOD_ALIASES = {"querylog", "query_log", "log_query", ""}
_LOG_QUERY_TOOLS = {"querylog", "dependency_log_query"}
_LEGACY_TOOL_ALIASES = {
    "querylog": "queryLog",
    "query_log": "queryLog",
    "log_query": "queryLog",
    "getcreateorderresult": "getCreateOrderResult",
    "get_create_order_result": "getCreateOrderResult",
    "getflightcreateorderresult": "getFlightCreateOrderResult",
    "get_flight_create_order_result": "getFlightCreateOrderResult",
    "dependency_log_query": "dependency_log_query",
    "query_dependency_log": "dependency_log_query",
}


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


def _summarize_tool_params(params: dict[str, Any]) -> dict[str, Any]:
    summary = executor_mod._summarize_tool_params(params)
    if "query" in params and "query" not in summary:
        summary["query"] = _clip(params.get("query"), 120)
    return summary


def _normalize_legacy_tool_decision(tool_name: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cleaned = dict(params or {})
    legacy_method = str(cleaned.pop("log_method", "") or "").strip()
    candidate = str(tool_name or "").strip()
    if legacy_method and candidate.lower() in {"", "log_query", "query_log"}:
        candidate = legacy_method
    normalized = _LEGACY_TOOL_ALIASES.get(candidate.lower(), candidate)
    return normalized or "queryLog", cleaned


def _build_goal_history_preview(*, objective: str, action_chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in action_chain:
        row = dict(item or {})
        rows.append(
            {
                "idx": row.get("seq"),
                "tool_name": str(row.get("tool_name") or "").strip(),
                "tool_params": dict(row.get("params_summary") or {}),
                "ok": bool(row.get("ok")),
                "objective": objective,
                "conclusion": str(row.get("conclusion") or "").strip(),
                "skill": str(row.get("skill") or "").strip(),
            }
        )
    return rows


def _normalize_required_answers(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("required_answers")
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        field = ""
        question = ""
        if isinstance(item, str):
            field = str(item).strip()
            question = f"给出字段 {field} 的取值" if field else ""
        elif isinstance(item, dict):
            field = str(item.get("field") or item.get("name") or item.get("key") or "").strip()
            question = str(item.get("question") or item.get("objective") or "").strip()
        if not field:
            continue
        lowered = field.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append({"field": field, "question": question})
    return normalized


def _extract_target_field_from_objective(objective: str) -> str:
    text = str(objective or "").strip()
    if not text:
        return ""
    match = re.search(r"字段\s*([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.IGNORECASE)
    if match:
        return str(match.group(1) or "").strip()
    return ""


def _resolve_goal_required_field(*, plan: dict[str, Any], objective: str) -> str:
    required_answers = _normalize_required_answers(plan)
    if not required_answers:
        return _extract_target_field_from_objective(objective)
    if len(required_answers) == 1:
        # 单目标场景直接锁定唯一必答字段，避免 goal 文本未显式提字段时丢失 required_field。
        single_field = str(required_answers[0].get("field") or "").strip()
        if single_field:
            return single_field
    objective_text = str(objective or "")
    objective_lower = objective_text.lower()
    for item in required_answers:
        field = str(item.get("field") or "").strip()
        question = str(item.get("question") or "").strip()
        if not field:
            continue
        if field.lower() in objective_lower:
            return field
        if question and question in objective_text:
            return field
    inferred = _extract_target_field_from_objective(objective_text)
    if not inferred:
        return ""
    for item in required_answers:
        field = str(item.get("field") or "").strip()
        if field.lower() == inferred.lower():
            return field
    return inferred


def _extract_required_field_value(
    *,
    field: str,
    raw_result: dict[str, Any],
    result_summary: str,
    tool_params: dict[str, Any],
) -> tuple[str, str]:
    target = str(field or "").strip()
    if not target:
        return "", "none"
    lowered_target = target.lower()

    # 1) 先从结构化 facts 读取（仅接受精确字段名）
    effective_info = dict(raw_result.get("effective_info") or {})
    facts = effective_info.get("facts")
    if isinstance(facts, dict):
        for key, value in facts.items():
            if str(key or "").strip().lower() != lowered_target:
                continue
            resolved = str(value or "").strip()
            if resolved and resolved.lower() not in _REQUIRED_UNKNOWN_VALUES:
                return resolved, "exact_field"

    # 2) 再从文本中匹配 "field: value" / "field=value"
    texts: list[str] = []
    for value in [result_summary, effective_info.get("summary"), raw_result.get("error")]:
        text = str(value or "").strip()
        if text:
            texts.append(text)
    for item in list(raw_result.get("evidence") or []):
        text = str(item or "").strip()
        if text:
            texts.append(text)
    if isinstance(facts, dict) and facts:
        texts.append(json.dumps(facts, ensure_ascii=False))

    escaped = re.escape(target)
    pattern = re.compile(rf'(?i)["\']?{escaped}["\']?\s*[:=：]\s*["\']?([A-Za-z0-9_.-]+)')
    for text in texts:
        for match in pattern.finditer(str(text or "")):
            candidate = str(match.group(1) or "").strip().strip('",')
            if candidate and candidate.lower() not in _REQUIRED_UNKNOWN_VALUES:
                return candidate, "exact_text"

    # 3) requestId 的兜底：可用已确认 trace_id（用户已给定请求唯一标识）
    if lowered_target in {"requestid", "request_id"}:
        for key in ("trace_id", "traceId", "request_id", "requestId"):
            candidate = str(tool_params.get(key) or "").strip()
            if candidate and candidate.lower() not in _REQUIRED_UNKNOWN_VALUES:
                return candidate, "request_id_fallback"

    # 4) 语义兜底：精确字段不存在时，允许相似字段 / 自然语言含义映射。
    semantic_value = _infer_required_field_value_semantic(field=target, texts=texts, facts=facts)
    if semantic_value:
        return semantic_value, "semantic_fallback"

    return "", "none"


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


def _infer_required_field_value_semantic(*, field: str, texts: list[str], facts: Any) -> str:
    target = str(field or "").strip()
    if not target:
        return ""
    evidence_parts: list[str] = []
    for item in texts[:10]:
        text = str(item or "").strip()
        if text:
            evidence_parts.append(_clip(text, 1200))
    if isinstance(facts, dict) and facts:
        evidence_parts.append(_clip(json.dumps(facts, ensure_ascii=False), 1200))
    if not evidence_parts:
        return ""

    system_prompt = (
        "你是日志字段抽取助手。任务：在“精确字段缺失”时，从相似字段或自然语言描述中提取目标字段值。"
        "若无法合理映射，返回空字符串。"
        "只返回 JSON: {\"value\":\"...\",\"matched_by\":\"similar_field|natural_language|none\",\"confidence\":\"high|medium|low\"}."
    )
    user_prompt = json.dumps(
        {
            "target_field": target,
            "evidence_texts": evidence_parts,
        },
        ensure_ascii=False,
    )
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


def _ensure_reactor_runtime(execution: dict[str, Any], *, goal_index: int, max_act_times: int) -> dict[str, Any]:
    runtime = dict(execution.get("reactor_runtime") or {})
    if _as_int(runtime.get("goal_index"), -1) != goal_index:
        runtime = {
            "goal_index": goal_index,
            "act_times": 0,
            "retry_count": 0,
            "max_retry": _MAX_RETRY_PER_GOAL,
            "max_act_times": max_act_times,
            "action_chain": [],
        }
        return runtime

    runtime["goal_index"] = goal_index
    runtime.setdefault("act_times", 0)
    runtime.setdefault("retry_count", 0)
    runtime.setdefault("max_retry", _MAX_RETRY_PER_GOAL)
    runtime.setdefault("max_act_times", max_act_times)
    runtime.setdefault("action_chain", [])
    return runtime


def _build_goal_report(
    *,
    goal_index: int,
    objective: str,
    hypothesis: str,
    goal_status: str,
    act_times: int,
    max_act_times: int,
    retry_count: int,
    action_chain: list[dict[str, Any]],
    goal_conclusion: str,
    failure_reason: str,
    all_goals_completed: bool = False,
) -> dict[str, Any]:
    return {
        "goal_index": goal_index,
        "goal_objective": objective,
        "hypothesis": hypothesis,
        "goal_status": goal_status,
        "act_times": act_times,
        "max_act_times": max_act_times,
        "retry_count": retry_count,
        "max_retry": _MAX_RETRY_PER_GOAL,
        "action_chain": action_chain,
        "goal_conclusion": goal_conclusion,
        "plan_unexecutable": goal_status == "unexecutable",
        "failure_reason": failure_reason,
        "all_goals_completed": all_goals_completed,
    }


def _extract_action_summary(raw_result: dict[str, Any]) -> str:
    effective_info = dict(raw_result.get("effective_info") or {})
    summary = str(effective_info.get("summary") or "").strip()
    if summary:
        return summary
    evidence = [str(item).strip() for item in list(raw_result.get("evidence") or []) if str(item).strip()]
    if evidence:
        return evidence[0]
    return str(raw_result.get("error") or "").strip()


def _pick_best_effort_action(action_chain: list[dict[str, Any]], *, required_field: str = "") -> dict[str, Any]:
    if required_field:
        for item in reversed(list(action_chain or [])):
            row = dict(item or {})
            if not bool(row.get("ok")):
                continue
            if not str(row.get("result_summary") or "").strip():
                continue
            if bool(row.get("required_field_ok")):
                return row
    for item in reversed(list(action_chain or [])):
        row = dict(item or {})
        if bool(row.get("ok")) and str(row.get("result_summary") or "").strip():
            return row
    return {}


def _extract_identifier_tokens(values: list[Any]) -> list[str]:
    results: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if _TRACE_ID_PATTERN.search(text):
            if text not in results:
                results.append(text)
            continue
        if _ORDER_ID_PATTERN.search(text):
            if text not in results:
                results.append(text)
            continue
        if _ASCII_ID_PATTERN.fullmatch(text) and lowered.startswith(("xep", "sid", "hpv", "trace", "order")):
            if text not in results:
                results.append(text)
            continue
        if _ASCII_ID_PATTERN.fullmatch(text) and any(ch in text for ch in ("_", "-", ".")) and any(
            ch.isdigit() for ch in text
        ):
            if text not in results:
                results.append(text)
    return results


def _has_meaningful_log_hit(action: dict[str, Any]) -> bool:
    row = dict(action or {})
    tool_name = str(row.get("tool_name") or "").strip().lower()
    if tool_name not in _LOG_QUERY_TOOLS:
        return False
    raw_result = dict(row.get("raw_result") or {})
    if not bool(raw_result.get("ok")):
        return False
    try:
        if int(raw_result.get("log_hit_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    summary = str(row.get("result_summary") or "").strip().lower()
    if not summary:
        return False
    if "未检索到日志命中" in summary or "missing" in summary or "timeout" in summary:
        return False
    return True


def _should_force_querylog_fallback(
    *,
    act_times: int,
    max_act_times: int,
    action_chain: list[dict[str, Any]],
    required_field_unresolved: bool = False,
    force_final_querylog: bool = False,
) -> bool:
    # 语义兜底命中后，允许在 n+1 的最后一轮无条件强制 queryLog。
    if force_final_querylog:
        return act_times == max_act_times

    # 仅在最后一轮触发：若必答字段仍未命中，直接强制兜底 queryLog。
    if act_times != max_act_times:
        return False
    if required_field_unresolved:
        return True
    # 前几轮均未拿到有效日志时，也强制兜底 queryLog。
    if len(action_chain) < max(0, max_act_times - 1):
        return False
    return not any(_has_meaningful_log_hit(item) for item in action_chain)


def _is_querylog_call(*, tool_name: str, tool_params: dict[str, Any]) -> bool:
    normalized_tool = str(tool_name or "").strip().lower()
    if normalized_tool == "querylog":
        return True
    if normalized_tool in {"log_query", "query_log"}:
        normalized_method = str(tool_params.get("method") or tool_params.get("skill") or "").strip().lower()
        return normalized_method in _QUERYLOG_METHOD_ALIASES
    return False


def _validate_querylog_params(*, tool_name: str, tool_params: dict[str, Any]) -> str:
    if not _is_querylog_call(tool_name=tool_name, tool_params=tool_params):
        return ""
    app_code = str(tool_params.get("app_code") or "").strip()
    logname = str(tool_params.get("logname") or "").strip()
    if not app_code or not logname:
        return "queryLog missing required params: app_code/logname"
    phrase_tokens = _extract_identifier_tokens(list(tool_params.get("match_phrase_list") or []))
    if not phrase_tokens:
        return "queryLog requires match_phrase_list to include traceId/orderNo"
    match_list = [str(item or "").strip() for item in list(tool_params.get("match_list") or []) if str(item or "").strip()]
    if match_list:
        return "queryLog requires match_list=[]"
    return ""


def _append_new_evidence_items(
    *,
    evidence_graph: dict[str, Any],
    objective: str,
    actions: list[dict[str, Any]],
) -> None:
    rows = [dict(item or {}) for item in list(evidence_graph.get("evidence") or [])]
    for item in actions:
        raw_result = dict(item.get("raw_result") or {})
        rows.append(
            {
                "objective": objective,
                "skill": str(item.get("skill") or item.get("tool_name") or "").strip(),
                "observation": str(item.get("result_summary") or "").strip(),
                "summary": str(item.get("result_summary") or "").strip(),
                "conclusion": str(item.get("conclusion") or "insufficient").strip(),
                "raw_result": raw_result,
            }
        )
    evidence_graph["evidence"] = rows


def _required_field_already_resolved(*, required_field: str, action_chain: list[dict[str, Any]]) -> bool:
    target = str(required_field or "").strip().lower()
    if not target:
        return False
    for item in reversed(list(action_chain or [])):
        row = dict(item or {})
        field = str(row.get("required_field") or "").strip().lower()
        if field and field != target:
            continue
        if bool(row.get("required_field_ok")):
            return True
    return False


def _last_resolved_required_field(*, required_field: str, action_chain: list[dict[str, Any]]) -> tuple[str, str]:
    target = str(required_field or "").strip().lower()
    if not target:
        return "", ""
    for item in reversed(list(action_chain or [])):
        row = dict(item or {})
        field = str(row.get("required_field") or "").strip().lower()
        if field and field != target:
            continue
        if not bool(row.get("required_field_ok")):
            continue
        value = str(row.get("required_field_value") or "").strip()
        if not value or value.lower() in _REQUIRED_UNKNOWN_VALUES:
            continue
        source = str(row.get("required_field_source") or "").strip()
        return value, source
    return "", ""


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    plan = dict(state.get("plan") or {})
    hypothesis = str(plan.get("hypothesis") or "").strip()
    goals = [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()]

    execution = dict(state.get("execution") or {})
    goal_index = _as_int(execution.get("goal_index"), 0)
    configured_max_act_times = max(
        1,
        min(
            _MAX_RETRY_PER_GOAL,
            _as_int(execution.get("max_act_times"), _MAX_RETRY_PER_GOAL),
        ),
    )
    max_act_times = configured_max_act_times

    if not goals:
        report = _build_goal_report(
            goal_index=goal_index,
            objective="",
            hypothesis=hypothesis,
            goal_status="unexecutable",
            act_times=0,
            max_act_times=max_act_times,
            retry_count=0,
            action_chain=[],
            goal_conclusion="",
            failure_reason="planner returned empty investigation_goals",
            all_goals_completed=True,
        )
        state["current_step_result"] = {"reactor_report": report}
        state["execution"] = execution
        state["route"] = "observer"
        _LOGGER.info("reactor.goal.end goal_index=%d status=unexecutable reason=no_goals", goal_index)
        return dict(state)

    if goal_index >= len(goals):
        report = _build_goal_report(
            goal_index=goal_index,
            objective="",
            hypothesis=hypothesis,
            goal_status="success",
            act_times=0,
            max_act_times=max_act_times,
            retry_count=0,
            action_chain=[],
            goal_conclusion="all goals completed",
            failure_reason="",
            all_goals_completed=True,
        )
        state["current_step_result"] = {"reactor_report": report}
        state["execution"] = execution
        state["route"] = "observer"
        _LOGGER.info(
            "reactor.goal.end goal_index=%d status=success all_goals_completed=true total_goals=%d",
            goal_index,
            len(goals),
        )
        return dict(state)

    objective = goals[goal_index]
    required_field = _resolve_goal_required_field(plan=plan, objective=objective)
    runtime = _ensure_reactor_runtime(execution, goal_index=goal_index, max_act_times=max_act_times)
    act_times = _as_int(runtime.get("act_times"), 0)
    retry_count = _as_int(runtime.get("retry_count"), 0)
    action_chain = [dict(item or {}) for item in list(runtime.get("action_chain") or [])]
    force_querylog_after_semantic = bool(runtime.get("force_querylog_after_semantic", False))
    semantic_fallback_force_executed = bool(runtime.get("semantic_fallback_force_executed", False))
    if force_querylog_after_semantic and not semantic_fallback_force_executed:
        max_act_times = configured_max_act_times + 1
    start_chain_len = len(action_chain)

    goal_status = "in_progress"
    goal_conclusion = ""
    failure_reason = ""

    _LOGGER.info(
        "reactor.goal.start goal_index=%d/%d objective=%s act_times=%d/%d retry_count=%d/%d",
        goal_index + 1,
        len(goals),
        _clip(objective),
        act_times,
        max_act_times,
        retry_count,
        _MAX_RETRY_PER_GOAL,
    )

    while act_times < max_act_times:
        act_times += 1
        required_field_unresolved = bool(required_field) and not _required_field_already_resolved(
            required_field=required_field,
            action_chain=action_chain,
        )
        force_querylog_round = _should_force_querylog_fallback(
            act_times=act_times,
            max_act_times=max_act_times,
            action_chain=action_chain,
            required_field_unresolved=required_field_unresolved,
            force_final_querylog=force_querylog_after_semantic and not semantic_fallback_force_executed,
        )
        force_querylog_reason = (
            "当前为最后一轮兜底执行：必须由大模型依据 skill 自主选择 queryLog，并补齐 app_code/logname。"
            if force_querylog_round
            else ""
        )
        goal_history_preview = _build_goal_history_preview(objective=objective, action_chain=action_chain)
        decision_state = dict(state)
        decision_state["tool_history"] = [
            *[dict(item or {}) for item in list(state.get("tool_history") or [])],
            *goal_history_preview,
        ]
        if action_chain:
            last_action = dict(action_chain[-1] or {})
            _LOGGER.info(
                "reactor.feedback goal_index=%d next_act=%d objective=%s last_tool=%s last_conclusion=%s last_summary=%s",
                goal_index,
                act_times,
                _clip(objective, 120),
                str(last_action.get("tool_name") or ""),
                str(last_action.get("conclusion") or ""),
                _clip(last_action.get("result_summary"), 120),
            )
        current_evidence = {
            "conclusions": [str(item.get("conclusion") or "") for item in action_chain if str(item.get("conclusion") or "").strip()],
            "keywords": [str(item.get("result_summary") or "") for item in action_chain[-3:] if str(item.get("result_summary") or "").strip()],
            "attempts": [
                {
                    "seq": item.get("seq"),
                    "objective": objective,
                    "tool_name": item.get("tool_name"),
                    "skill": item.get("skill"),
                    "params_summary": item.get("params_summary"),
                    "result_summary": item.get("result_summary"),
                    "conclusion": item.get("conclusion"),
                    "error": item.get("error"),
                }
                for item in action_chain[-3:]
            ],
        }
        skill_decision = executor_mod._decide_skill_with_llm(
            state=decision_state,
            hypothesis=hypothesis,
            objective=objective,
            current_evidence=current_evidence,
            evidence_rows=[
                {
                    "objective": objective,
                    "tool_name": item.get("tool_name"),
                    "skill": item.get("skill"),
                    "params_summary": item.get("params_summary"),
                    "summary": item.get("result_summary"),
                    "conclusion": item.get("conclusion"),
                    "error": item.get("error"),
                }
                for item in action_chain
            ],
            retry_count=retry_count,
            force_querylog=force_querylog_round,
            force_querylog_reason=force_querylog_reason,
        )

        llm_raw = str(skill_decision.get("llm_raw") or "")
        llm_decision = executor_mod._parse_json_object(llm_raw) or {}
        if bool(llm_decision.get("cannot_execute") or llm_decision.get("plan_unexecutable")):
            goal_status = "unexecutable"
            failure_reason = str(llm_decision.get("reason") or "reactor llm decided cannot_execute").strip()
            _LOGGER.info(
                "reactor.goal.end goal_index=%d status=unexecutable reason=%s",
                goal_index,
                _clip(failure_reason),
            )
            break

        raw_tool_name = str(skill_decision.get("tool_name") or "queryLog")
        tool_name, tool_params = _normalize_legacy_tool_decision(raw_tool_name, dict(skill_decision.get("params") or {}))
        tool_params.setdefault("query", str(state.get("question") or ""))
        required_tool_params = dict(executor_mod._build_required_tool_params(state) or {})
        context_locked_keys = {"trace_id", "traceId", "order_id", "orderNo", "begin_time", "end_time"}
        for key, value in required_tool_params.items():
            if value:
                if key in context_locked_keys:
                    tool_params[key] = value
                else:
                    tool_params.setdefault(key, value)

        if force_querylog_round:
            if force_querylog_after_semantic:
                semantic_fallback_force_executed = True
            _LOGGER.info(
                "reactor.action.force_fallback goal_index=%d act=%d mode=llm_guided_querylog",
                goal_index,
                act_times,
            )

        _LOGGER.info(
            "reactor.action.dispatch goal_index=%d act=%d tool=%s params=%s",
            goal_index,
            act_times,
            tool_name,
            _summarize_tool_params(tool_params),
        )

        validation_error = ""
        if force_querylog_round and not _is_querylog_call(tool_name=tool_name, tool_params=tool_params):
            validation_error = "forced fallback round requires LLM to choose queryLog and generate full params from skills"
        if not validation_error:
            validation_error = _validate_querylog_params(tool_name=tool_name, tool_params=tool_params)

        if validation_error:
            raw_result = {
                "tool": tool_name,
                "ok": False,
                "error": validation_error,
                "evidence": [],
                "effective_info": {"summary": validation_error, "facts": {}},
                "log_hit_count": 0,
            }
            _LOGGER.warning(
                "reactor.action.invalid_params goal_index=%d act=%d tool=%s error=%s",
                goal_index,
                act_times,
                tool_name,
                _clip(validation_error, 220),
            )
        else:
            raw_result = executor_mod._execute_tool_call(tool_name=tool_name, params=tool_params, state=state)
        conclusion = executor_mod._infer_conclusion(objective=objective, hypothesis=hypothesis, raw_result=raw_result)
        result_summary = _extract_action_summary(raw_result)
        required_field_value, required_field_source = _extract_required_field_value(
            field=required_field,
            raw_result=raw_result,
            result_summary=result_summary,
            tool_params=tool_params,
        )
        if required_field and not required_field_value:
            locked_value, locked_source = _last_resolved_required_field(
                required_field=required_field,
                action_chain=action_chain,
            )
            if locked_value:
                required_field_value = locked_value
                required_field_source = f"locked_{locked_source}" if locked_source else "locked_previous"
        required_field_ok = (not required_field) or bool(required_field_value)
        if not required_field_ok:
            missing_hint = f"required field {required_field} not resolved"
            if result_summary:
                result_summary = f"{result_summary} ({missing_hint})"
            else:
                result_summary = missing_hint
        elif required_field and required_field_value:
            if required_field_source == "semantic_fallback":
                suffix = f"{required_field}={required_field_value} (source=semantic_fallback)"
            elif required_field_source.startswith("locked_"):
                suffix = f"{required_field}={required_field_value} (source={required_field_source})"
            else:
                suffix = f"{required_field}={required_field_value}"
            result_summary = f"{result_summary} | {suffix}" if result_summary else suffix

        action_item = {
            "seq": len(action_chain) + 1,
            "tool_name": tool_name,
            "skill": str(skill_decision.get("skill") or tool_name),
            "params_summary": _summarize_tool_params(tool_params),
            "result_summary": result_summary,
            "ok": bool(raw_result.get("ok")),
            "error": str(raw_result.get("error") or "").strip(),
            "conclusion": conclusion,
            "required_field": required_field,
            "required_field_value": required_field_value,
            "required_field_source": required_field_source,
            "required_field_ok": required_field_ok,
            "raw_result": raw_result,
        }
        action_chain.append(action_item)

        _LOGGER.info(
            "reactor.action.result goal_index=%d act=%d tool=%s ok=%s conclusion=%s required_field=%s required_field_ok=%s required_field_source=%s required_field_value=%s summary=%s error=%s",
            goal_index,
            act_times,
            tool_name,
            bool(raw_result.get("ok")),
            conclusion,
            required_field,
            required_field_ok,
            required_field_source,
            _clip(required_field_value, 80),
            _clip(result_summary),
            _clip(raw_result.get("error"), 120),
        )

        if bool(raw_result.get("ok")) and conclusion in {"supports", "neutral", "refutes"}:
            if required_field and not required_field_ok:
                # 必答字段未命中时，当前 goal 不允许提前成功。
                retry_count += 1
                if act_times < max_act_times:
                    continue
                goal_status = "failed"
                failure_reason = f"required field {required_field} unresolved after {act_times} acts"
                break
            if required_field and required_field_ok and required_field_source == "semantic_fallback":
                if not force_querylog_after_semantic:
                    # 语义命中后继续跑到最后一轮兜底 queryLog：总轮次可突破到 n+1。
                    force_querylog_after_semantic = True
                    semantic_fallback_force_executed = False
                    max_act_times = configured_max_act_times + 1
                    _LOGGER.info(
                        "reactor.required_field.semantic_fallback goal_index=%d act=%d field=%s extend_max_act_times=%d",
                        goal_index,
                        act_times,
                        required_field,
                        max_act_times,
                    )
            if force_querylog_after_semantic and not semantic_fallback_force_executed:
                # 一旦出现 semantic_fallback，必须继续跑到最后一轮 queryLog 兜底后再收敛。
                continue
            goal_status = "success"
            goal_conclusion = result_summary or conclusion
            retry_count = 0
            break

        retry_count += 1
        if retry_count >= _MAX_RETRY_PER_GOAL:
            best_effort = _pick_best_effort_action(action_chain, required_field=required_field)
            if best_effort:
                best_summary = str(best_effort.get("result_summary") or "").strip()
                if required_field and not bool(best_effort.get("required_field_ok")):
                    goal_status = "failed"
                    failure_reason = (
                        f"required field {required_field} unresolved after retries; "
                        f"best_effort={best_summary}"
                    )
                else:
                    goal_status = "success"
                    goal_conclusion = best_summary
                retry_count = 0
                break

            goal_status = "failed"
            if required_field and not required_field_ok:
                failure_reason = (
                    f"required field {required_field} unresolved after {retry_count} attempts and no usable best-effort evidence"
                )
            else:
                failure_reason = str(raw_result.get("error") or f"goal retry exhausted with conclusion={conclusion}").strip()
            break

    if goal_status == "in_progress":
        if required_field and _required_field_already_resolved(required_field=required_field, action_chain=action_chain):
            best_effort = _pick_best_effort_action(action_chain, required_field=required_field)
            if best_effort:
                goal_status = "success"
                goal_conclusion = str(best_effort.get("result_summary") or "").strip()
            else:
                goal_status = "success"
                goal_conclusion = f"required field {required_field} resolved in previous actions"
        else:
            goal_status = "unexecutable"
            failure_reason = "act_times_exhausted"

    new_actions = action_chain[start_chain_len:]
    evidence_graph = dict(execution.get("evidence_graph") or {})
    evidence_graph["hypothesis"] = hypothesis
    _append_new_evidence_items(evidence_graph=evidence_graph, objective=objective, actions=new_actions)
    if goal_status in {"failed", "unexecutable"}:
        evidence_graph["supported"] = False
    elif goal_status == "success" and goal_index >= len(goals) - 1:
        evidence_graph["supported"] = True

    report = _build_goal_report(
        goal_index=goal_index,
        objective=objective,
        hypothesis=hypothesis,
        goal_status=goal_status,
        act_times=act_times,
        max_act_times=max_act_times,
        retry_count=retry_count,
        action_chain=action_chain,
        goal_conclusion=goal_conclusion,
        failure_reason=failure_reason,
    )

    runtime.update(
        {
            "goal_index": goal_index,
            "act_times": act_times,
            "retry_count": retry_count,
            "max_retry": _MAX_RETRY_PER_GOAL,
            "max_act_times": max_act_times,
            "configured_max_act_times": configured_max_act_times,
            "force_querylog_after_semantic": force_querylog_after_semantic,
            "semantic_fallback_force_executed": semantic_fallback_force_executed,
            "action_chain": action_chain,
        }
    )
    execution["reactor_runtime"] = runtime
    execution["evidence_graph"] = evidence_graph

    state["execution"] = execution
    state["current_step_result"] = {"reactor_report": report}
    state["route"] = "observer"

    _LOGGER.info(
        "reactor.goal.end goal_index=%d status=%s act_times=%d retry_count=%d actions=%d failure_reason=%s",
        goal_index,
        goal_status,
        act_times,
        retry_count,
        len(action_chain),
        _clip(failure_reason),
    )
    return dict(state)
