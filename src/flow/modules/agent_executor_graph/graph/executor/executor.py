"""ReAct Executor：按 investigation goal 由 LLM 读取 skills 文档决策并沉淀 Evidence Graph。"""

from __future__ import annotations

import json
import logging
import re
import datetime as dt
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.executor.sub_executor import run_log_sub_executor
from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import (
    resolve_intent_label_for_rag,
)
from llm.llm import chat_with_llm, load_prompt, render_prompt
from tool.registry import build_tool_schemas_for_prompt, get_all_tools, invoke_tool

_LOGGER = logging.getLogger(__name__)
_ALLOWED_TOOLS = {tool.name for tool in get_all_tools()}
_TOOL_ALIASES = {
    "querylog": "queryLog",
    "query_log": "queryLog",
    "log_query": "queryLog",
    "getcreateorderresult": "getCreateOrderResult",
    "get_create_order_result": "getCreateOrderResult",
    "getflightcreateorderresult": "getFlightCreateOrderResult",
    "get_flight_create_order_result": "getFlightCreateOrderResult",
    "query_dependency_log": "dependency_log_query",
    "dependency_log_query": "dependency_log_query",
    "knowledge_lookup": "knowledge_lookup",
    "knowledge_search": "knowledge_lookup",
    "rag_parent_chunk_query": "rag_parent_chunk_query",
    "query_rag_parent_chunk": "rag_parent_chunk_query",
    "rag_parent_doc_query": "rag_parent_chunk_query",
}
_TRACE_ID_PATTERN = re.compile(r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)", re.IGNORECASE)
_TRACE_KEY_PATTERN = re.compile(r"\btrace[_-]?id\b\s*[:=：]?\s*([A-Za-z0-9_.:\-]{4,128})", re.IGNORECASE)
_ORDER_TOKEN_PATTERN = re.compile(r"\b(?:xep|sid|hpv|fod)\d{8,}\b", re.IGNORECASE)
_ORDER_KEY_PATTERN = re.compile(
    r"(?:\border[_-]?(?:id|no)\b|订单号|订单id|订单ID|子单号)\s*[:：=]?\s*([A-Za-z0-9_.:\-]{4,128})",
    re.IGNORECASE,
)
_ASCII_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{6,128}$")
_YYMMDD_HHMMSS_PATTERN = re.compile(r"(?<!\d)(\d{6})[\s_.-]+(\d{6})(?!\d)")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: Any, max_len: int = 200) -> str:
    raw = str(value or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


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


def _normalize_tool_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    mapped = _TOOL_ALIASES.get(text, text)
    return mapped if mapped in _ALLOWED_TOOLS else ""


def _build_tool_schemas() -> list[dict[str, Any]]:
    return build_tool_schemas_for_prompt()


def _summarize_tool_params(params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "trace_id",
        "order_id",
        "begin_time",
        "end_time",
        "app_code",
        "logname",
        "query",
        "match_phrase_list",
        "match_list",
    ):
        if key not in params:
            continue
        value = params.get(key)
        if isinstance(value, list):
            summary[key] = [_clip(item, 80) for item in value[:6]]
            if len(value) > 6:
                summary[f"{key}_truncated"] = len(value) - 6
        else:
            summary[key] = _clip(value, 120)
    return summary


def _build_history_preview(history: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in history[-4:]:
        row = dict(item or {})
        rows.append(
            {
                "idx": row.get("idx"),
                "skill": str(row.get("skill") or "").strip(),
                "tool_name": str(row.get("tool_name") or "").strip(),
                "ok": bool(row.get("ok")),
                "conclusion": str(row.get("conclusion") or "").strip(),
                "tool_params": _summarize_tool_params(dict(row.get("tool_params") or {})),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


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


def _collect_planner_knowledge_for_reactor(state: dict[str, Any]) -> dict[str, Any]:
    structured = dict(state.get("structured_context") or {})
    knowledge_context = dict(structured.get("knowledge_context") or state.get("knowledge_context") or {})

    def _as_docs(rows: Any) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        docs: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row or {})
            docs.append(
                {
                    "path": str(item.get("path") or "").strip(),
                    "knowledge_type": str(item.get("knowledge_type") or "").strip(),
                    "score": item.get("score"),
                    "content": str(item.get("content") or item.get("text") or "").strip(),
                }
            )
        return docs

    domain_docs = _as_docs(knowledge_context.get("domain_docs"))
    case_docs = _as_docs(knowledge_context.get("case_docs"))
    code_docs = _as_docs(knowledge_context.get("code_docs"))
    return {
        "domain_docs": domain_docs,
        "case_docs": case_docs,
        "code_docs": code_docs,
        "counts": {
            "domain_docs": len(domain_docs),
            "case_docs": len(case_docs),
            "code_docs": len(code_docs),
        },
    }


def _decide_skill_with_llm(
    *,
    state: dict[str, Any],
    hypothesis: str,
    objective: str,
    current_evidence: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    retry_count: int,
    force_querylog: bool = False,
    force_querylog_reason: str = "",
) -> dict[str, Any]:
    previous_observation = dict(evidence_rows[-1] or {}) if evidence_rows else {}
    plan = dict(state.get("plan") or {})
    required_answers = _normalize_required_answers(plan)
    current_subtask = {
        "hypothesis": hypothesis,
        "objective": objective,
        "retry_count": retry_count,
        "current_evidence": current_evidence,
        "required_answers": required_answers,
    }
    planner_knowledge = _collect_planner_knowledge_for_reactor(state)
    plan_payload = {
        "hypothesis": str(plan.get("hypothesis") or "").strip(),
        "investigation_goals": [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()],
    }
    user_prompt = render_prompt(
        "executor_react_user_prompt.txt",
        question=str(state.get("question") or ""),
        plan_json=json.dumps(plan_payload, ensure_ascii=False),
        required_answers_json=json.dumps(required_answers, ensure_ascii=False),
        planner_knowledge_json=json.dumps(planner_knowledge, ensure_ascii=False),
        current_subtask_json=json.dumps(current_subtask, ensure_ascii=False),
        previous_observation_json=json.dumps(previous_observation, ensure_ascii=False),
        tool_schemas_json=json.dumps(_build_tool_schemas(), ensure_ascii=False),
        history_preview=_build_history_preview([dict(item) for item in list(state.get("tool_history") or [])]),
        force_querylog="true" if force_querylog else "false",
        force_querylog_reason=str(force_querylog_reason or "").strip(),
    )
    system_prompt = load_prompt("executor_react_system_prompt.txt", default="")
    llm_raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    parsed = _parse_json_object(llm_raw) or {}

    action = parsed.get("action")
    action_dict = dict(action) if isinstance(action, dict) else {}
    params = action_dict.get("params")
    params_dict = dict(params) if isinstance(params, dict) else dict(parsed.get("params") or {})
    legacy_method = str(params_dict.pop("log_method", "") or "").strip()
    candidate_tool = action_dict.get("tool_name") or parsed.get("tool_name")
    if legacy_method and str(candidate_tool or "").strip().lower() in {"", "log_query", "query_log"}:
        candidate_tool = legacy_method
    tool_name = _normalize_tool_name(candidate_tool)
    skill = str(parsed.get("skill") or tool_name or "").strip()

    if not tool_name:
        tool_name = "queryLog"
    if not skill:
        skill = tool_name

    return {
        "skill": skill,
        "tool_name": tool_name,
        "params": params_dict,
        "layer": "LLM",
        "llm_parse_ok": bool(parsed),
        "llm_raw": llm_raw,
    }


def _collect_current_evidence(evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    conclusions = [str(item.get("conclusion") or "").strip() for item in evidence_rows if str(item.get("conclusion") or "").strip()]
    keywords: list[str] = []
    for item in evidence_rows:
        raw = dict(item.get("raw_result") or {})
        effective_info = dict(raw.get("effective_info") or {})
        for keyword in list(effective_info.get("keywords") or []):
            value = str(keyword).strip()
            if value and value not in keywords:
                keywords.append(value)
    return {
        "conclusions": conclusions,
        "keywords": keywords,
    }


def _extract_trace_and_order_from_text(text: Any) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    trace_id = ""
    order_id = ""
    match = _TRACE_ID_PATTERN.search(raw)
    if match:
        trace_id = str(match.group(0) or "").strip()
    if not trace_id:
        key_match = _TRACE_KEY_PATTERN.search(raw)
        if key_match:
            trace_id = str(key_match.group(1) or "").strip()

    order_match = _ORDER_TOKEN_PATTERN.search(raw)
    if order_match:
        order_id = str(order_match.group(0) or "").strip()
    if not order_id:
        key_match = _ORDER_KEY_PATTERN.search(raw)
        if key_match:
            order_id = str(key_match.group(1) or "").strip()
    return trace_id, order_id


def _to_iso_time(yymmdd: str, hhmmss: str) -> str:
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone(dt.timedelta(hours=8))).isoformat()


def _infer_time_window_from_text(text: Any) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    matched = _YYMMDD_HHMMSS_PATTERN.search(raw)
    if not matched:
        return "", ""
    try:
        event_time = dt.datetime.fromisoformat(_to_iso_time(str(matched.group(1)), str(matched.group(2))))
    except ValueError:
        return "", ""
    return (
        (event_time - dt.timedelta(hours=1)).isoformat(),
        (event_time + dt.timedelta(hours=1)).isoformat(),
    )


def _infer_time_window_from_candidates(values: list[Any]) -> tuple[str, str]:
    for value in values:
        begin_time, end_time = _infer_time_window_from_text(value)
        if begin_time and end_time:
            return begin_time, end_time
    return "", ""


def _extract_ids_from_phrase_terms(value: Any) -> tuple[str, str]:
    trace_id = ""
    order_id = ""
    for item in list(value or []):
        text = str(item or "").strip()
        if not text:
            continue
        if not trace_id and (_TRACE_ID_PATTERN.search(text) or ("trace" in text.lower() and _ASCII_ID_PATTERN.fullmatch(text))):
            trace_id = text
        if not order_id and (_ORDER_TOKEN_PATTERN.search(text) or (text.lower().startswith("xep") and _ASCII_ID_PATTERN.fullmatch(text))):
            order_id = text
        if trace_id and order_id:
            break
    return trace_id, order_id


def _extract_recent_ids_from_history(state: dict[str, Any]) -> tuple[str, str]:
    # 1) 优先从最近工具调用参数拿，可靠性最高。
    trace_id = ""
    order_id = ""
    for row in reversed(list(state.get("tool_history") or [])):
        item = dict(row or {})
        params = dict(item.get("tool_params") or {})
        for key in ("trace_id", "traceId", "request_id", "requestId"):
            value = str(params.get(key) or "").strip()
            if value:
                trace_id = trace_id or value
        for key in ("orderNo", "order_no", "order_id", "orderId"):
            value = str(params.get(key) or "").strip()
            if value:
                order_id = order_id or value
        phrase_trace, phrase_order = _extract_ids_from_phrase_terms(params.get("match_phrase_list"))
        trace_id = trace_id or phrase_trace
        order_id = order_id or phrase_order
        if trace_id and order_id:
            return trace_id, order_id

    # 2) 再从 message_context 最近轮次（用户追问场景）提取。
    context = dict(state.get("context") or {})
    raw_message_context = context.get("message_context")
    rounds: list[Any] = []
    if hasattr(raw_message_context, "rounds"):
        rounds = list(getattr(raw_message_context, "rounds") or [])
    elif isinstance(raw_message_context, dict):
        rounds = list(raw_message_context.get("rounds") or [])
    for round_row in reversed(rounds):
        row = dict(round_row or {}) if isinstance(round_row, dict) else {
            "message": getattr(round_row, "message", ""),
            "aiResponse": getattr(round_row, "aiResponse", ""),
            "toolsContext": getattr(round_row, "toolsContext", {}),
        }
        for source in (
            row.get("message"),
            row.get("aiResponse"),
            json.dumps(dict(row.get("toolsContext") or {}), ensure_ascii=False, default=str),
        ):
            trace_hit, order_hit = _extract_trace_and_order_from_text(source)
            trace_id = trace_id or trace_hit
            order_id = order_id or order_hit
            if trace_id and order_id:
                return trace_id, order_id

    # 3) 最后从 conversation_context 文本兜底。
    for line in reversed(list(state.get("conversation_context") or [])):
        trace_hit, order_hit = _extract_trace_and_order_from_text(line)
        trace_id = trace_id or trace_hit
        order_id = order_id or order_hit
        if trace_id and order_id:
            return trace_id, order_id
    return trace_id, order_id


def _build_required_tool_params(state: dict[str, Any]) -> dict[str, Any]:
    structured_context = dict(state.get("structured_context") or {})
    query_rewrite = dict(state.get("query_rewrite") or structured_context.get("query_rewrite") or {})
    time_window = dict(query_rewrite.get("time_window") or {})
    history_trace_id, history_order_id = _extract_recent_ids_from_history(state)
    trace_id = str(query_rewrite.get("trace_id") or structured_context.get("trace_id") or history_trace_id or "").strip()
    order_id = str(query_rewrite.get("order_id") or structured_context.get("order_id") or history_order_id or "").strip()
    begin_time = str(structured_context.get("begin_time") or time_window.get("begin_time") or "").strip()
    end_time = str(structured_context.get("end_time") or time_window.get("end_time") or "").strip()
    if not begin_time or not end_time:
        inferred_begin, inferred_end = _infer_time_window_from_candidates(
            [
                trace_id,
                order_id,
                str(query_rewrite.get("normalized_query") or ""),
                str(state.get("question") or ""),
                *list(state.get("conversation_context") or [])[-3:],
            ]
        )
        begin_time = begin_time or inferred_begin
        end_time = end_time or inferred_end

    return {
        "trace_id": trace_id,
        "traceId": trace_id,
        "order_id": order_id,
        "orderNo": order_id,
        "begin_time": begin_time,
        "end_time": end_time,
        "app_code": str(structured_context.get("app_code") or "").strip(),
        "logname": str(structured_context.get("logname") or "").strip(),
    }


def _execute_tool_call(*, tool_name: str, params: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    step = {
        "action_type": "tool_call",
        "tool_name": tool_name,
        "params": params,
    }
    structured_context = dict(state.get("structured_context") or {})
    if tool_name in {"queryLog", "dependency_log_query", "getCreateOrderResult", "getFlightCreateOrderResult"}:
        return run_log_sub_executor(step=step, state=state, structured_context=structured_context)
    if tool_name == "rag_parent_chunk_query":
        question = str(params.get("query") or state.get("question") or "").strip()
        intent_zh = resolve_intent_label_for_rag(state)
        raw = invoke_tool(
            "rag_parent_chunk_query",
            {
                "query": question,
                "intent_zh": intent_zh,
                "sub_chunk_top_k": params.get("sub_chunk_top_k"),
                "parent_top_k": params.get("parent_top_k"),
            },
        )
        if isinstance(raw, dict) and raw.get("ok") is False:
            return raw
        raw_dict = dict(raw or {}) if isinstance(raw, dict) else {}
        sub_chunks = list(raw_dict.get("sub_chunks") or [])
        parent_chunks = list(raw_dict.get("parent_chunks") or [])
        parent_docs = list(raw_dict.get("parent_docs") or [])
        evidence = [
            f"[sub_chunk#{idx + 1}] score={item.get('score')} path={dict(item.get('payload') or {}).get('path')}"
            for idx, item in enumerate(sub_chunks[:3])
        ]
        return {
            "tool": "rag_parent_chunk_query",
            "ok": True,
            "error": "",
            "evidence": evidence,
            "sub_chunk_count": len(sub_chunks),
            "parent_chunk_count": len(parent_chunks),
            "parent_doc_count": len(parent_docs),
            "topk_docs": parent_docs,
        }
    if tool_name == "knowledge_lookup":
        docs = list(dict(state.get("knowledge_context") or {}).get("domain_docs") or [])
        raw = invoke_tool("knowledge_lookup", {"docs": docs})
        return dict(raw or {}) if isinstance(raw, dict) else {"tool": "knowledge_lookup", "ok": False, "error": "invalid tool result", "evidence": []}
    raw = invoke_tool(tool_name, params)
    if isinstance(raw, dict):
        return raw
    return {"tool": tool_name, "ok": True, "error": "", "evidence": [str(raw)]}


def _infer_conclusion(*, objective: str, hypothesis: str, raw_result: dict[str, Any]) -> str:
    if not bool(raw_result.get("ok")):
        return "insufficient"
    evidence_text = "\n".join(str(item) for item in list(raw_result.get("evidence") or []))
    lowered = evidence_text.lower()

    has_failure = any(token in lowered for token in ("fail", "失败", "异常", "error", "timeout", "超时"))
    has_success = any(token in lowered for token in ("success", "成功", "ok"))

    objective_text = f"{objective} {hypothesis}".lower()
    if has_failure:
        return "supports"
    if has_success and any(token in objective_text for token in ("失败", "异常", "mq", "生单")):
        return "refutes"
    if evidence_text.strip():
        return "neutral"
    return "insufficient"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    plan = dict(state.get("plan") or {})
    hypothesis = str(plan.get("hypothesis") or "").strip()
    goals = [str(item).strip() for item in list(plan.get("investigation_goals") or []) if str(item).strip()]

    execution = dict(state.get("execution") or {})
    goal_index = _as_int(execution.get("goal_index"), 0)
    retry_count = _as_int(execution.get("objective_retry_count"), 0)

    evidence_graph = dict(execution.get("evidence_graph") or {})
    evidence_graph["hypothesis"] = hypothesis
    evidence_rows = [dict(item or {}) for item in list(evidence_graph.get("evidence") or [])]
    _LOGGER.info(
        "executor start hypothesis=%s goal_index=%d total_goals=%d evidence_count=%d",
        _clip(hypothesis, 180),
        goal_index,
        len(goals),
        len(evidence_rows),
    )

    if not goals:
        evidence_graph["supported"] = False
        execution["evidence_graph"] = evidence_graph
        state["execution"] = execution
        state["evaluation"] = {"status": "unsupported", "reason": "planner returned empty investigation_goals"}
        state["route"] = "evaluator"
        _LOGGER.info("executor skip no investigation goals: route=evaluator status=unsupported")
        return dict(state)

    if goal_index >= len(goals):
        execution["evidence_graph"] = evidence_graph
        state["execution"] = execution
        state["route"] = "evaluator"
        _LOGGER.info(
            "executor skip goals exhausted: goal_index=%d total_goals=%d route=evaluator",
            goal_index,
            len(goals),
        )
        return dict(state)

    objective = goals[goal_index]
    current_evidence = _collect_current_evidence(evidence_rows)
    _LOGGER.info(
        "executor step.begin idx=%d/%d objective=%s prior_conclusions=%s prior_keywords=%s",
        goal_index + 1,
        len(goals),
        _clip(objective, 220),
        current_evidence.get("conclusions") or [],
        current_evidence.get("keywords") or [],
    )
    skill_decision = _decide_skill_with_llm(
        state=state,
        hypothesis=hypothesis,
        objective=objective,
        current_evidence=current_evidence,
        evidence_rows=evidence_rows,
        retry_count=retry_count,
    )
    _LOGGER.info(
        "executor llm_decision objective=%s parse_ok=%s raw=%s",
        _clip(objective, 220),
        bool(skill_decision.get("llm_parse_ok")),
        _clip(skill_decision.get("llm_raw"), 320),
    )

    tool_name = str(skill_decision.get("tool_name") or "queryLog")
    tool_params = dict(skill_decision.get("params") or {})
    tool_params.setdefault("query", str(state.get("question") or ""))
    for key, value in _build_required_tool_params(state).items():
        if value:
            tool_params.setdefault(key, value)
    _LOGGER.info(
        "executor dispatch objective=%s skill=%s tool=%s retry=%d tool_params=%s",
        _clip(objective, 220),
        str(skill_decision.get("skill") or tool_name),
        tool_name,
        retry_count,
        _summarize_tool_params(tool_params),
    )

    raw_result = _execute_tool_call(tool_name=tool_name, params=tool_params, state=state)
    conclusion = _infer_conclusion(objective=objective, hypothesis=hypothesis, raw_result=raw_result)
    summary = str(dict(raw_result.get("effective_info") or {}).get("summary") or "").strip()
    if not summary:
        evidence_lines = [str(item).strip() for item in list(raw_result.get("evidence") or []) if str(item).strip()]
        summary = evidence_lines[0] if evidence_lines else str(raw_result.get("error") or "")

    evidence_item = {
        "objective": objective,
        "skill": str(skill_decision.get("skill") or tool_name),
        "observation": summary,
        "summary": summary,
        "conclusion": conclusion,
        "raw_result": raw_result,
    }
    evidence_rows.append(evidence_item)
    evidence_graph["evidence"] = evidence_rows
    if conclusion == "refutes":
        evidence_graph["supported"] = False
    elif goal_index >= len(goals) - 1 and conclusion in {"supports", "neutral"}:
        evidence_graph["supported"] = True

    history = [dict(item) for item in list(state.get("tool_history") or [])]
    history.append(
        {
            "idx": len(history) + 1,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "ok": bool(raw_result.get("ok")),
            "objective": objective,
            "conclusion": conclusion,
            "skill": str(skill_decision.get("skill") or tool_name),
        }
    )

    # 若当前目标证据不足，先切换技能再试一次（最多一次），避免过早 RePlan。
    if conclusion == "insufficient" and retry_count < 1:
        execution["objective_retry_count"] = retry_count + 1
        _LOGGER.info(
            "executor retry scheduled objective=%s next_retry=%d reason=insufficient",
            _clip(objective, 220),
            int(execution.get("objective_retry_count") or 0),
        )
    else:
        execution["objective_retry_count"] = 0
        if conclusion in {"supports", "neutral", "refutes"}:
            execution["goal_index"] = goal_index + 1
        else:
            execution["goal_index"] = goal_index
        _LOGGER.info(
            "executor progression objective=%s conclusion=%s new_goal_index=%d",
            _clip(objective, 220),
            conclusion,
            int(execution.get("goal_index") or 0),
        )

    execution["last_objective"] = objective
    execution["last_skill"] = str(skill_decision.get("skill") or tool_name)
    execution["evidence_graph"] = evidence_graph
    _LOGGER.info(
        "executor step.end objective=%s tool=%s ok=%s conclusion=%s summary=%s log_hit_count=%s degraded=%s next_goal_index=%d objective_retry_count=%d evidence_count=%d supported=%s error=%s",
        _clip(objective, 220),
        tool_name,
        bool(raw_result.get("ok")),
        conclusion,
        _clip(summary, 220),
        str(raw_result.get("log_hit_count") if raw_result.get("log_hit_count") is not None else ""),
        bool(raw_result.get("degraded")),
        int(execution.get("goal_index") or 0),
        int(execution.get("objective_retry_count") or 0),
        len(evidence_rows),
        str(evidence_graph.get("supported")),
        str(raw_result.get("error") or ""),
    )

    state["execution"] = execution
    state["tool_history"] = history
    state["tool_name"] = tool_name
    state["tool_params"] = tool_params
    state["tool_result"] = raw_result
    state["current_step_result"] = {
        "objective": objective,
        "skill_decision": skill_decision,
        "raw_result": raw_result,
        "conclusion": conclusion,
    }
    state["route"] = "evaluator"
    return dict(state)
