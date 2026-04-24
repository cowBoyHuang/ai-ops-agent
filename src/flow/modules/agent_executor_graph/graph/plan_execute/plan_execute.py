"""计划执行复合节点（Plan Execute）。

业务职责：
- 接收 planner 输出的完整 plan_steps，并在一个节点内循环执行。
- 每步执行后调用大模型解析结果，提取关键字与结构化信息。
- 统一维护 execution_history / intermediate_results / extracted_keywords。
- 在节点内完成证据融合，输出给 analysis_execute。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm, load_prompt, render_prompt
from tool.code_tool import clone_repo, pull_repo

_LOGGER = logging.getLogger(__name__)
_ALLOWED_TOOL_NAMES = {"log_query", "dependency_log_query", "knowledge_lookup", "code_clone", "code_pull", "none"}
_MAX_STEP_RETRY = 1
_MAX_LLM_HISTORY_ROWS = 4
_MAX_SUMMARY_LEN = 300
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{2,64}")


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_success(tool: str, evidence: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "tool": tool,
        "ok": True,
        "error": "",
        "evidence": evidence,
    }
    if extra:
        payload.update(extra)
    return payload


def _tool_failed(tool: str, error: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "tool": tool,
        "ok": False,
        "error": str(error or "unknown_error"),
        "evidence": [],
    }
    if extra:
        payload.update(extra)
    return payload


def _extract_git_url(tool_params: dict[str, Any], structured_context: dict[str, Any]) -> str:
    value = (
        tool_params.get("git_url")
        or tool_params.get("repo_url")
        or structured_context.get("git_url")
        or dict(structured_context.get("code_repo") or {}).get("git_url")
        or ""
    )
    return str(value).strip()


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
    """调用大模型解析单步结果并提取关键字。"""
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


def _execute_tool_call(
    tool_name: str,
    tool_params: dict[str, Any],
    state: dict[str, Any],
    structured_context: dict[str, Any],
) -> dict[str, Any]:
    """执行单个工具调用，保持与原 tool_execute 的行为兼容。"""
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

    if normalized_tool == "log_query":
        order_id = str(tool_params.get("order_id") or structured_context.get("order_id") or "")
        evidence_rows = [
            f"log_query命中：{question[:64]}",
            f"order_id={order_id or 'N/A'}",
        ]
        return _tool_success("log_query", evidence_rows)

    if normalized_tool == "dependency_log_query":
        request_id = str(tool_params.get("request_id") or structured_context.get("request_id") or "")
        return _tool_success(
            "dependency_log_query",
            [f"依赖调用日志：{question[:64]}", f"request_id={request_id or 'N/A'}"],
        )

    if normalized_tool == "knowledge_lookup":
        return _tool_success("knowledge_lookup", [f"知识库证据：{question[:64]}"])

    if normalized_tool == "code_clone":
        git_url = _extract_git_url(tool_params, structured_context)
        if not git_url:
            return _tool_failed("code_clone", "missing git_url")
        result = clone_repo(git_url=git_url)
        if bool(result.get("ok")):
            return _tool_success(
                "code_clone",
                [
                    f"clone success: {str(result.get('target_dir') or '')}",
                    f"status={str(result.get('status') or '')}",
                ],
                extra={"tool_payload": result},
            )
        return _tool_failed("code_clone", str(result.get("message") or "clone failed"), extra={"tool_payload": result})

    if normalized_tool == "code_pull":
        git_url = _extract_git_url(tool_params, structured_context)
        if not git_url:
            return _tool_failed("code_pull", "missing git_url")
        result = pull_repo(git_url=git_url)
        if bool(result.get("ok")):
            return _tool_success(
                "code_pull",
                [
                    f"pull success: {str(result.get('target_dir') or '')}",
                    f"status={str(result.get('status') or '')}",
                ],
                extra={"tool_payload": result},
            )
        return _tool_failed("code_pull", str(result.get("message") or "pull failed"), extra={"tool_payload": result})

    return _tool_success("none", [])


def _execute_single_step(step: dict[str, Any], state: dict[str, Any], structured_context: dict[str, Any]) -> dict[str, Any]:
    """执行单个计划步骤（支持 tool_call / merge_evidence）。"""
    action_type = str(step.get("action_type") or "tool_call")
    if action_type == "merge_evidence":
        return _tool_success("none", [], extra={"action_type": "merge_evidence"})

    tool_name = str(step.get("tool_name") or "knowledge_lookup")
    tool_params = dict(step.get("params") or {})
    tool_params.setdefault("query", state.get("question") or "")
    tool_params.setdefault("order_id", structured_context.get("order_id") or "")
    tool_params.setdefault("request_id", structured_context.get("request_id") or "")

    result = _execute_tool_call(tool_name=tool_name, tool_params=tool_params, state=state, structured_context=structured_context)

    # 仅真实工具调用推进预算计数。
    if tool_name != "none":
        state["tool_call_count"] = _as_int(state.get("tool_call_count"), 0) + 1

    state["tool_name"] = tool_name
    state["tool_params"] = tool_params
    state["tool_result"] = result

    history = [dict(item) for item in list(state.get("tool_history") or [])]
    history.append(
        {
            "idx": len(history) + 1,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "ok": bool(result.get("ok")),
            "error": str(result.get("error") or ""),
        }
    )
    state["tool_history"] = history
    return result


def _merge_all_evidence(state: dict[str, Any], execution_history: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """合并 RAG 与步骤执行证据，生成 merged_evidence + evidence_context。"""
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
            record = {
                "id": f"{key}-tool-{idx}",
                "source": tool_name or "tool",
                "score": 1.0,
                "text": text,
            }
            if "log" in tool_name:
                merged["logs"].append(record)
            else:
                merged["knowledge"].append(record)

        summary = str(processed.get("summary") or "").strip()
        if summary:
            merged["knowledge"].append(
                {
                    "id": f"{key}-summary",
                    "source": "llm_step_summary",
                    "score": 0.7,
                    "text": summary,
                }
            )

    evidence_rows = [str(item.get("text") or "") for item in merged["logs"] + merged["knowledge"]]
    evidence_text = "\n".join(row for row in evidence_rows if row).strip()
    return merged, evidence_text


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """复合执行节点主入口。"""
    state: AgentState = dict(payload)
    plan_steps = list(state.get("plan_steps") or [])
    current_step_index = _as_int(state.get("current_step_index"), 0)
    execution_history = dict(state.get("execution_history") or {})
    intermediate_results = dict(state.get("intermediate_results") or {})
    extracted_keywords = {str(item).strip() for item in list(state.get("extracted_keywords") or []) if str(item).strip()}
    structured_context = dict(state.get("structured_context") or {})

    _LOGGER.info(
        "plan_execute 开始执行: total_steps=%d start_index=%d tool_calls=%d",
        len(plan_steps),
        current_step_index,
        _as_int(state.get("tool_call_count"), 0),
    )

    last_tool_result = dict(state.get("tool_result") or {})
    stop_reason = ""

    while current_step_index < len(plan_steps):
        raw_step = plan_steps[current_step_index]
        step = dict(raw_step) if isinstance(raw_step, dict) else {}
        step.setdefault("action_type", "tool_call")
        step_key = f"step_{current_step_index}"
        step_attempt = 0
        step_completed = True

        while step_attempt <= _MAX_STEP_RETRY:
            raw_result = _execute_single_step(step=step, state=state, structured_context=structured_context)
            processed = _process_step_result_with_llm(step=step, raw_result=raw_result, execution_history=execution_history)

            execution_history[step_key] = {
                "index": current_step_index,
                "attempt": step_attempt,
                "step": step,
                "raw_result": raw_result,
                "processed": processed,
            }
            intermediate_results[step_key] = {
                "summary": str(processed.get("summary") or ""),
                "structured_facts": dict(processed.get("structured_facts") or {}),
                "tool_ok": bool(raw_result.get("ok")),
            }
            extracted_keywords.update(str(item).strip() for item in list(processed.get("extracted_keywords") or []) if str(item).strip())
            last_tool_result = raw_result

            if bool(raw_result.get("ok")):
                break

            # 错误处理：LLM 决策可触发当前步骤自动重试一次。
            retry_current_step = bool(processed.get("retry_current_step"))
            retry_params = dict(processed.get("retry_params") or {})
            if retry_current_step and step_attempt < _MAX_STEP_RETRY:
                step = {
                    **step,
                    "params": {
                        **dict(step.get("params") or {}),
                        **retry_params,
                    },
                }
                step_attempt += 1
                _LOGGER.info("plan_execute 步骤重试: step=%s attempt=%d", step_key, step_attempt)
                continue

            continue_execution = bool(processed.get("continue_execution"))
            if not continue_execution:
                stop_reason = str(raw_result.get("error") or "step_failed")
                step_completed = False
                _LOGGER.warning("plan_execute 终止: step=%s reason=%s", step_key, stop_reason)
            break

        if step_completed:
            current_step_index += 1
            state["current_step_index"] = current_step_index

        if stop_reason:
            break
        if str(last_tool_result.get("error") or "") == "max_tool_calls_exceeded":
            _LOGGER.warning("plan_execute 工具预算耗尽，提前结束循环")
            break

    merged_evidence, evidence_context = _merge_all_evidence(state=state, execution_history=execution_history)

    structured_context = {
        **structured_context,
        "evidence_context": evidence_context,
        "merged_evidence_summary": {
            "logs_count": len(list(merged_evidence.get("logs") or [])),
            "knowledge_count": len(list(merged_evidence.get("knowledge") or [])),
            "keywords_count": len(extracted_keywords),
        },
        "execution_summary": {
            "executed_steps": len(execution_history),
            "current_step_index": current_step_index,
            "total_steps": len(plan_steps),
            "stop_reason": stop_reason,
        },
    }

    if not last_tool_result:
        last_tool_result = _tool_success("none", [])

    state["execution_history"] = execution_history
    state["intermediate_results"] = intermediate_results
    state["extracted_keywords"] = sorted(extracted_keywords)
    state["merged_evidence"] = merged_evidence
    state["evidence"] = merged_evidence
    state["structured_context"] = structured_context
    state["tool_result"] = last_tool_result
    state["route"] = "analysis_execute"

    _LOGGER.info(
        "plan_execute 执行完成: executed=%d current_index=%d logs=%d knowledge=%d",
        len(execution_history),
        current_step_index,
        len(list(merged_evidence.get("logs") or [])),
        len(list(merged_evidence.get("knowledge") or [])),
    )
    return dict(state)
