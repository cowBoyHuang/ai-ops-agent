"""Executor 节点：单步执行当前计划步骤。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.executor.sub_executor import (
    run_code_sub_executor,
    run_log_sub_executor,
)
from llm.llm import chat_with_llm, load_prompt, render_prompt

_LOGGER = logging.getLogger(__name__)
_ALLOWED_TOOL_NAMES = {"log_query", "dependency_log_query", "knowledge_lookup", "code_clone", "code_pull", "none"}
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{2,64}")
_MAX_LLM_HISTORY_ROWS = 4
_MAX_SUMMARY_LEN = 300


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_text(text: Any, max_len: int = _MAX_SUMMARY_LEN) -> str:
    raw = str(text or "").strip()
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


def _tool_success(tool: str, evidence: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"tool": tool, "ok": True, "error": "", "evidence": evidence}
    if extra:
        payload.update(extra)
    return payload


def _tool_failed(tool: str, error: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"tool": tool, "ok": False, "error": str(error or "unknown_error"), "evidence": []}
    if extra:
        payload.update(extra)
    return payload


def _execute_tool_call(
    tool_name: str,
    tool_params: dict[str, Any],
    state: dict[str, Any],
    structured_context: dict[str, Any],
) -> dict[str, Any]:
    normalized_tool = str(tool_name or "none").strip()
    if normalized_tool not in _ALLOWED_TOOL_NAMES:
        return _tool_failed(normalized_tool or "none", f"unsupported tool: {normalized_tool}")

    tool_call_count = _as_int(state.get("tool_call_count"), 0)
    max_tool_calls = max(1, _as_int(state.get("max_tool_calls"), 6))
    question = str(state.get("question") or "")
    if tool_call_count >= max_tool_calls:
        return _tool_failed(normalized_tool, "max_tool_calls_exceeded", extra={"budget_exhausted": True})

    if bool(structured_context.get("simulate_tool_timeout_once")) and not bool(
        structured_context.get("_simulate_tool_timeout_used")
    ):
        structured_context["_simulate_tool_timeout_used"] = True
        return _tool_failed(normalized_tool, "network timeout")

    if normalized_tool in {"log_query", "dependency_log_query"}:
        return run_log_sub_executor(
            step={"tool_name": normalized_tool, "params": tool_params},
            state=state,
            structured_context=structured_context,
        )

    if normalized_tool == "knowledge_lookup":
        return _tool_success("knowledge_lookup", [f"知识库证据：{question[:64]}"])

    if normalized_tool in {"code_clone", "code_pull"}:
        return run_code_sub_executor(
            step={"tool_name": normalized_tool, "params": tool_params},
            state=state,
            structured_context=structured_context,
        )

    return _tool_success("none", [])


def _build_step_history_preview(execution_history: dict[str, Any]) -> str:
    rows: list[str] = []
    keys = sorted(execution_history.keys(), key=lambda item: _as_int(str(item).split("_")[-1], 0))
    for key in keys[-_MAX_LLM_HISTORY_ROWS:]:
        item = dict(execution_history.get(key) or {})
        step = dict(item.get("step") or {})
        raw_result = dict(item.get("raw_result") or {})
        processed = dict(item.get("processed") or {})
        rows.append(
            " | ".join(
                [
                    key,
                    f"tool={step.get('tool_name') or 'none'}",
                    f"ok={bool(raw_result.get('ok'))}",
                    f"summary={_clip_text(processed.get('summary'), 120)}",
                ]
            )
        )
    return "\n".join(rows).strip() or "无历史步骤"


def _fallback_keywords(step: dict[str, Any], raw_result: dict[str, Any]) -> list[str]:
    rows = [
        str(step.get("tool_name") or ""),
        str(raw_result.get("error") or ""),
        " ".join(str(item) for item in list(raw_result.get("evidence") or [])[:3]),
    ]
    merged = " ".join(rows)
    tokens = {token for token in _KEYWORD_PATTERN.findall(merged) if len(token) >= 3}
    return sorted(tokens)[:20]


def _process_step_result_with_llm(
    step: dict[str, Any],
    raw_result: dict[str, Any],
    execution_history: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = load_prompt("plan_execute_step_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "plan_execute_step_user_prompt.txt",
        step_json=json.dumps(step, ensure_ascii=False),
        result_json=json.dumps(raw_result, ensure_ascii=False),
        history_preview=_build_step_history_preview(execution_history),
    )
    fallback = {
        "summary": _clip_text(raw_result.get("error") or "步骤执行完成"),
        "extracted_keywords": _fallback_keywords(step, raw_result),
        "structured_facts": {},
        "retry_current_step": False,
        "retry_params": {},
        "continue_execution": bool(raw_result.get("ok")),
        "next_step_guidance": "",
    }
    if not user_prompt:
        return fallback
    llm_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    parsed = _parse_json_object(llm_output)
    if not parsed:
        return fallback
    keywords = [str(item).strip() for item in list(parsed.get("extracted_keywords") or []) if str(item).strip()]
    return {
        "summary": _clip_text(parsed.get("summary") or fallback["summary"]),
        "extracted_keywords": sorted(set(keywords))[:30],
        "structured_facts": dict(parsed.get("structured_facts") or {}),
        "retry_current_step": bool(parsed.get("retry_current_step")),
        "retry_params": dict(parsed.get("retry_params") or {}),
        "continue_execution": bool(parsed.get("continue_execution", bool(raw_result.get("ok")))),
        "next_step_guidance": _clip_text(parsed.get("next_step_guidance"), 200),
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    current_plan = list(state.get("current_plan") or state.get("plan_steps") or [])
    current_step_index = _as_int(state.get("current_step_index"), 0)
    execution_history = dict(state.get("execution_history") or {})
    intermediate_results = dict(state.get("intermediate_results") or {})
    extracted_keywords = {str(item).strip() for item in list(state.get("extracted_keywords") or []) if str(item).strip()}
    structured_context = dict(state.get("structured_context") or {})

    if current_step_index >= len(current_plan):
        state["current_plan"] = current_plan
        state["plan_steps"] = current_plan
        state["current_step_result"] = {
            "step_index": current_step_index,
            "step": {},
            "raw_result": _tool_success("none", []),
            "processed": {"summary": "no step to execute", "extracted_keywords": [], "structured_facts": {}},
        }
        state["newly_discovered_clues"] = []
        state["route"] = "observer"
        return dict(state)

    raw_step = current_plan[current_step_index]
    step = dict(raw_step) if isinstance(raw_step, dict) else {}
    step.setdefault("action_type", "tool_call")
    action_type = str(step.get("action_type") or "tool_call")

    if action_type == "merge_evidence":
        raw_result = _tool_success("none", [], extra={"action_type": "merge_evidence"})
        processed = {
            "summary": "merge_evidence marker",
            "extracted_keywords": [],
            "structured_facts": {},
            "retry_current_step": False,
            "retry_params": {},
            "continue_execution": True,
            "next_step_guidance": "",
        }
    else:
        tool_name = str(step.get("tool_name") or "knowledge_lookup")
        tool_params = dict(step.get("params") or {})
        tool_params.setdefault("query", state.get("question") or "")
        tool_params.setdefault("order_id", structured_context.get("order_id") or "")
        tool_params.setdefault("request_id", structured_context.get("request_id") or "")
        raw_result = _execute_tool_call(tool_name=tool_name, tool_params=tool_params, state=state, structured_context=structured_context)
        processed = _process_step_result_with_llm(step=step, raw_result=raw_result, execution_history=execution_history)
        if tool_name != "none":
            state["tool_call_count"] = _as_int(state.get("tool_call_count"), 0) + 1
        state["tool_name"] = tool_name
        state["tool_params"] = tool_params

        history = [dict(item) for item in list(state.get("tool_history") or [])]
        history.append(
            {
                "idx": len(history) + 1,
                "tool_name": tool_name,
                "tool_params": tool_params,
                "ok": bool(raw_result.get("ok")),
                "error": str(raw_result.get("error") or ""),
            }
        )
        state["tool_history"] = history

    step_key = f"step_{current_step_index}"
    execution_history[step_key] = {
        "index": current_step_index,
        "step": step,
        "raw_result": raw_result,
        "processed": processed,
    }
    intermediate_results[step_key] = {
        "summary": str(processed.get("summary") or ""),
        "structured_facts": dict(processed.get("structured_facts") or {}),
        "tool_ok": bool(raw_result.get("ok")),
    }
    clues = [str(item).strip() for item in list(processed.get("extracted_keywords") or []) if str(item).strip()]
    extracted_keywords.update(clues)

    state["execution_history"] = execution_history
    state["intermediate_results"] = intermediate_results
    state["extracted_keywords"] = sorted(extracted_keywords)
    state["current_plan"] = current_plan
    state["plan_steps"] = current_plan
    state["current_step_result"] = {
        "step_index": current_step_index,
        "step": step,
        "raw_result": raw_result,
        "processed": processed,
    }
    state["newly_discovered_clues"] = clues
    state["tool_result"] = raw_result
    state["structured_context"] = structured_context
    state["current_step_index"] = current_step_index + 1
    state["route"] = "observer"
    _LOGGER.info("executor 单步执行完成: step=%d tool=%s ok=%s", current_step_index, str(step.get("tool_name") or "none"), bool(raw_result.get("ok")))
    return dict(state)

