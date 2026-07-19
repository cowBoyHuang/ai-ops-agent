"""Shared helpers for domain executors."""

from __future__ import annotations

import json
from typing import Any

from llm.llm import chat_with_llm, load_prompt, render_prompt
from tool.registry import build_tool_schemas_for_prompt, invoke_tool


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
    evidence_rows: list[Any] = []
    facts: dict[str, Any] = {}
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
    final_evidence = dict(parsed.get("final_evidence") or {})
    if final_evidence:
        facts.update(dict(final_evidence.get("facts") or {}))
        evidence_rows.extend(list(final_evidence.get("evidence") or []))
    summary = str(final_evidence.get("summary") or parsed.get("final_answer") or "").strip()
    goal_complete = bool(parsed.get("goal_complete")) or bool(summary or evidence_rows)
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
