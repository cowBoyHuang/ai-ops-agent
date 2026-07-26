"""Shared helpers for domain executors."""

from __future__ import annotations

import json
import re
from typing import Any

from llm.llm import chat_with_llm, load_prompt, render_prompt
from tool.registry import build_tool_schemas_for_prompt, invoke_tool

_FLIGHT_DETAIL_GOAL_PATTERN = re.compile(
    r"(intercepted_passenger|passenger|travell?er|乘机人|旅客|姓名|证件|年龄|儿童|特殊产品|被拦截|不满足规则|年龄限制)",
    re.IGNORECASE,
)


def find_goal(investigation: dict[str, Any]) -> dict[str, Any]:
    goal_id = str(investigation.get("current_goal_id") or "").strip()
    plan = dict(investigation.get("plan") or {})
    for item in list(plan.get("goals") or []):
        row = dict(item or {})
        if str(row.get("id") or "").strip() == goal_id:
            return row
    return {}


def filtered_tool_schemas(allowed_tools: list[str]) -> list[dict[str, Any]]:
    allowed = {str(item).strip() for item in list(allowed_tools or []) if str(item).strip()}
    return [row for row in build_tool_schemas_for_prompt() if str(row.get("tool_name") or "") in allowed]


def _requires_tool_evidence(*, executor: str, current_goal: dict[str, Any]) -> bool:
    capability = str(current_goal.get("required_capability") or "").strip()
    return executor == "LogExecutor" or capability == "runtime_evidence"


def _has_useful_tool_result(raw_result: Any) -> bool:
    if raw_result is None:
        return False
    if isinstance(raw_result, list):
        return bool(raw_result)
    if isinstance(raw_result, dict):
        if raw_result.get("ok") is False:
            return False
        for key in ("evidence", "logs", "items", "rows", "results", "data", "content", "result"):
            value = raw_result.get(key)
            if isinstance(value, list | tuple | dict | str) and bool(value):
                return True
        return any(bool(value) for key, value in raw_result.items() if key not in {"ok", "error", "message"})
    if isinstance(raw_result, str):
        return bool(raw_result.strip())
    return bool(raw_result)


def _structured_value(structured_context: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = structured_context.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    query_rewrite = dict(structured_context.get("query_rewrite") or {})
    for key in keys:
        value = query_rewrite.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _complete_log_tool_params(tool_name: str, params: dict[str, Any], structured_context: dict[str, Any]) -> dict[str, Any]:
    completed = dict(params or {})
    trace_id = _structured_value(structured_context, "trace_id", "traceId")
    order_id = _structured_value(structured_context, "order_id", "orderId", "orderNo")
    begin_time = _structured_value(structured_context, "begin_time", "beginTime", "start_time", "startTime")
    end_time = _structured_value(structured_context, "end_time", "endTime", "finish_time", "finishTime")

    if tool_name in {"getCreateOrderResult", "getFlightCreateOrderResult"}:
        completed.setdefault("trace_id", trace_id)
        completed.setdefault("begin_time", begin_time)
        completed.setdefault("end_time", end_time)
    elif tool_name in {"queryLog", "dependency_log_query"}:
        completed.setdefault("begin_time", begin_time)
        completed.setdefault("end_time", end_time)
        if not list(completed.get("match_phrase_list") or []):
            phrases = [item for item in [trace_id, order_id] if item]
            if phrases:
                completed["match_phrase_list"] = phrases
        completed.setdefault("match_list", [])
    return {key: value for key, value in completed.items() if value not in ("", None)}


def _fallback_log_action(
    allowed_tools: list[str],
    structured_context: dict[str, Any],
    current_goal: dict[str, Any] | None = None,
    question: str = "",
) -> tuple[str, dict[str, Any]]:
    allowed = [str(item).strip() for item in list(allowed_tools or []) if str(item).strip()]
    trace_id = _structured_value(structured_context, "trace_id", "traceId")
    if trace_id:
        goal_text = json.dumps(current_goal or {}, ensure_ascii=False, default=str)
        preferred = (
            ("getFlightCreateOrderResult", "getCreateOrderResult")
            if _FLIGHT_DETAIL_GOAL_PATTERN.search(f"{goal_text}\n{question}")
            else ("getCreateOrderResult", "getFlightCreateOrderResult")
        )
        for tool_name in preferred:
            if tool_name in allowed:
                return tool_name, _complete_log_tool_params(tool_name, {}, structured_context)
    return "", {}


def normalize_result(
    *,
    executor: str,
    goal_id: str,
    result_id: str,
    goal_complete: bool,
    summary: str,
    facts: dict[str, Any] | None = None,
    evidence: list[Any] | None = None,
    artifacts: list[Any] | None = None,
    confidence: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    status = "succeeded" if goal_complete else "failed"
    return {
        "executor": executor,
        "result_id": result_id,
        "goal_id": goal_id,
        "goal_complete": bool(goal_complete),
        "status": status,
        "summary": str(summary or ""),
        "facts": dict(facts or {}),
        "evidence": list(evidence or []),
        "artifacts": list(artifacts or []),
        "confidence": confidence,
        "error": "" if goal_complete else str(error or "empty_evidence"),
    }


def execute_domain_goal(
    *,
    executor: str,
    prompt_name: str,
    question: str,
    current_goal: dict[str, Any],
    allowed_tools: list[str],
    existing_evidence: list[dict[str, Any]],
    structured_context: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    goal_id = str(current_goal.get("id") or "").strip()
    result_id = f"exec_{goal_id}_{max(1, int(attempt or 1))}"
    tool_schemas = filtered_tool_schemas(allowed_tools)
    system_prompt = load_prompt(prompt_name, default="")
    user_prompt = render_prompt(
        "domain_executor_react_user_prompt.txt",
        question=question,
        current_goal_json=json.dumps(current_goal, ensure_ascii=False),
        allowed_tools_json=json.dumps(allowed_tools, ensure_ascii=False),
        tool_schemas_json=json.dumps(tool_schemas, ensure_ascii=False),
        existing_evidence_json=json.dumps(existing_evidence, ensure_ascii=False, default=str),
        structured_context_json=json.dumps(structured_context, ensure_ascii=False, default=str),
    )
    parsed: dict[str, Any] = {}
    if system_prompt and user_prompt and tool_schemas:
        try:
            raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}

    action = dict(parsed.get("action") or {})
    tool_name = str(action.get("tool_name") or "").strip()
    tool_params = dict(action.get("params") or {})
    if _requires_tool_evidence(executor=executor, current_goal=current_goal) and tool_name not in set(allowed_tools):
        tool_name, tool_params = _fallback_log_action(allowed_tools, structured_context, current_goal, question)
    if _requires_tool_evidence(executor=executor, current_goal=current_goal):
        tool_params = _complete_log_tool_params(tool_name, tool_params, structured_context)
    evidence_rows: list[Any] = []
    facts: dict[str, Any] = {}
    has_tool_evidence = False
    if tool_name and tool_name in set(allowed_tools):
        raw_result = invoke_tool(tool_name, tool_params)
        if isinstance(raw_result, dict) and raw_result.get("ok") is False:
            return normalize_result(
                executor=executor,
                goal_id=goal_id,
                result_id=result_id,
                goal_complete=False,
                summary="",
                error=str(raw_result.get("error") or "tool_error"),
            )
        evidence_rows.append({"type": "tool_result", "source": tool_name, "content": raw_result, "confidence": 0.6})
        has_tool_evidence = _has_useful_tool_result(raw_result)
    final_evidence = dict(parsed.get("final_evidence") or {})
    if final_evidence:
        facts.update(dict(final_evidence.get("facts") or {}))
        evidence_rows.extend(list(final_evidence.get("evidence") or []))
    summary = str(final_evidence.get("summary") or parsed.get("final_answer") or "").strip()
    goal_complete = bool(parsed.get("goal_complete")) or bool(summary or evidence_rows)
    if _requires_tool_evidence(executor=executor, current_goal=current_goal) and not has_tool_evidence:
        goal_complete = False
    return normalize_result(
        executor=executor,
        goal_id=goal_id,
        result_id=result_id,
        goal_complete=goal_complete,
        summary=summary or ("执行完成" if goal_complete else ""),
        facts=facts,
        evidence=evidence_rows,
        confidence=0.7 if goal_complete else 0.0,
        error="" if goal_complete else "empty_evidence",
    )
