"""Planner 节点：一次性生成完整 InvestigationPlanV2。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import chat_with_llm, load_prompt, render_prompt

_LOGGER = logging.getLogger(__name__)
_MAX_DOCS = 4
_MAX_DOC_TEXT = 400
_REQUIRED_FIELD_PATTERNS = {
    "bizErrorCode": re.compile(r"biz[_-]?error[_-]?code", re.IGNORECASE),
    "subErrorCode": re.compile(r"sub[_-]?error[_-]?code", re.IGNORECASE),
    "refSubErrorCode": re.compile(r"ref[_-]?sub[_-]?error[_-]?code", re.IGNORECASE),
}
_GENERIC_FIELD_SUFFIXES = ("code", "id", "status", "time", "timestamp", "errno", "result")
_GENERIC_FIELD_HINTS = ("字段", "参数", "返回", "结果", "值", "编码", "错误码", "code", "id")
_IDENTITY_HINT_PATTERN = re.compile(
    r"(被拦截|哪位|是谁|乘机人|旅客|姓名|证件|证件号|idno|certno|identityno|passenger|traveller|traveler|name)",
    re.IGNORECASE,
)
_CODE_HINT_PATTERN = re.compile(r"(biz[_-]?error[_-]?code|sub[_-]?error[_-]?code|ref[_-]?sub[_-]?error[_-]?code|error[_-]?code|错误码|错误编码)", re.IGNORECASE)
_STATUS_HINT_PATTERN = re.compile(r"(status|状态|结果状态|result[_-]?status|是否成功|success)", re.IGNORECASE)
_TIME_HINT_PATTERN = re.compile(r"(time|timestamp|时间|时间点|何时|什么时候|失败时间)", re.IGNORECASE)
_REASON_HINT_PATTERN = re.compile(r"(原因|失败原因|why|reason|errormsg|errmsg|失败信息)", re.IGNORECASE)
_TRACE_ID_PATTERN = re.compile(
    r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)(?=$|[^A-Za-z0-9_\.\-])",
    re.IGNORECASE,
)
_TRACE_FAILURE_REASON_PATTERN = re.compile(r"(失败原因|失败.*原因|原因.*失败|为什么.*失败|生单失败|创建失败|下单失败)", re.IGNORECASE)
_PLACEHOLDER_PLAN_PATTERN = re.compile(r"(占位符|确认实际用户问题|用户问题文本|实际查询文本|缺少.*查询文本|缺少.*问题内容)")


def _clip(text: Any, max_len: int = _MAX_DOC_TEXT) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


def _json_clip(value: Any, max_len: int = 3000) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        raw = str(value or "")
    return _clip(raw, max_len)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_doc_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for item in rows[:_MAX_DOCS]:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def _summarize_docs(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"【{title}】\n无"
    lines = [f"【{title}】"]
    for idx, item in enumerate(rows, start=1):
        path = str(item.get("path") or "N/A").strip()
        content = str(item.get("content") or item.get("text") or "").strip()
        lines.append(f"{idx}. path={path} content={content}")
    return "\n".join(lines)


def _build_prompt_context(state: dict[str, Any]) -> dict[str, Any]:
    structured = dict(state.get("structured_context") or {})
    knowledge_context = dict(structured.get("knowledge_context") or state.get("knowledge_context") or {})
    query_rewrite = dict(state.get("query_rewrite") or structured.get("query_rewrite") or {})
    rejected = [str(item).strip() for item in list(state.get("rejected_hypothesis") or []) if str(item).strip()]
    investigation = dict(state.get("investigation") or {})
    previous_plan = dict(investigation.get("plan") or {})
    goal_status = dict(investigation.get("goal_status") or {})
    evidence = [dict(item or {}) for item in list(investigation.get("evidence") or [])]
    failed_goal_id = str(investigation.get("current_goal_id") or "").strip()
    failed_goal = {}
    for item in list(previous_plan.get("goals") or []):
        row = dict(item or {})
        if str(row.get("id") or "").strip() == failed_goal_id:
            failed_goal = row
            break

    domain_docs = _as_doc_rows(knowledge_context.get("domain_docs"))
    case_docs = _as_doc_rows(knowledge_context.get("case_docs"))
    code_docs = _as_doc_rows(knowledge_context.get("code_docs"))

    return {
        "user_query": str(state.get("question") or "").strip(),
        "normalized_query": str(query_rewrite.get("normalized_query") or "").strip(),
        "keywords": ", ".join(str(item).strip() for item in list(query_rewrite.get("keywords") or []) if str(item).strip()),
        "domain_knowledge": _summarize_docs("业务知识", domain_docs),
        "case_knowledge": _summarize_docs("Case知识", case_docs),
        "code_knowledge": _summarize_docs("代码知识", code_docs),
        "rejected_hypothesis": "\n".join(f"- {item}" for item in rejected) if rejected else "- 无",
        "replan_reason": str(state.get("replan_reason") or "").strip() or "无",
        "previous_plan_json": _json_clip(previous_plan),
        "goal_status_json": _json_clip(goal_status),
        "evidence_json": _json_clip(evidence),
        "failed_goal_json": _json_clip(failed_goal),
    }


def _parse_required_answers(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    answers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        field = ""
        question = ""
        required = True
        if isinstance(item, str):
            field = str(item).strip()
            question = f"必须获取字段：{field}" if field else ""
        elif isinstance(item, dict):
            field = str(item.get("field") or item.get("name") or item.get("key") or "").strip()
            question = str(item.get("question") or item.get("objective") or item.get("description") or "").strip()
            required = bool(item.get("required", True))
        if not field:
            continue
        lowered = field.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        answers.append(
            {
                "field": field,
                "question": question or f"必须获取字段：{field}",
                "required": required,
            }
        )
    return answers


def _derive_required_answers_from_query(state: dict[str, Any]) -> list[dict[str, Any]]:
    query_rewrite = dict(state.get("query_rewrite") or {})
    text = " ".join(
        [
            str(state.get("question") or ""),
            str(query_rewrite.get("normalized_query") or ""),
            " ".join(str(item) for item in list(query_rewrite.get("keywords") or [])),
        ]
    )
    lowered_text = text.lower()
    derived: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_field(field: str, question: str) -> None:
        normalized = str(field or "").strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            return
        seen.add(lowered)
        derived.append(
            {
                "field": normalized,
                "question": question or f"给出 {normalized} 的明确取值",
                "required": True,
            }
        )

    for field, pattern in _REQUIRED_FIELD_PATTERNS.items():
        if pattern.search(text):
            _append_field(field, f"给出 {field} 的明确取值")

    if any(hint in text for hint in _GENERIC_FIELD_HINTS) or any(hint in lowered_text for hint in _GENERIC_FIELD_HINTS):
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
        for token in tokens:
            lowered = token.lower()
            if lowered.endswith(_GENERIC_FIELD_SUFFIXES) or "errorcode" in lowered:
                _append_field(token, f"给出 {token} 的明确取值")

    if ("错误码" in text or "error code" in lowered_text or "errorcode" in lowered_text) and not any(
        str(item.get("field") or "").lower().endswith("code") for item in derived
    ):
        _append_field("errorCode", "给出失败返回中的错误码字段与取值")
    if _IDENTITY_HINT_PATTERN.search(text):
        _append_field("intercepted_passenger", "被拦截的乘机人是谁")
        if re.search(r"(证件|证件号|idno|certno|identityno)", text, re.IGNORECASE):
            _append_field("intercepted_passenger_id_no", "被拦截的乘机人证件号是什么")
        if re.search(r"(姓名|name|哪位|是谁)", text, re.IGNORECASE):
            _append_field("intercepted_passenger_name", "被拦截的乘机人姓名是谁")
    return derived


def _question_focus_categories(state: dict[str, Any]) -> set[str]:
    query_rewrite = dict(state.get("query_rewrite") or {})
    text = " ".join(
        [
            str(state.get("question") or ""),
            str(query_rewrite.get("normalized_query") or ""),
            " ".join(str(item) for item in list(query_rewrite.get("keywords") or [])),
        ]
    )
    categories: set[str] = set()
    if _IDENTITY_HINT_PATTERN.search(text):
        categories.add("identity")
    if _CODE_HINT_PATTERN.search(text):
        categories.add("code")
    if _STATUS_HINT_PATTERN.search(text):
        categories.add("status")
    if _TIME_HINT_PATTERN.search(text):
        categories.add("time")
    if _REASON_HINT_PATTERN.search(text):
        categories.add("reason")
    return categories


def _required_answer_categories(item: dict[str, Any]) -> set[str]:
    field = str(item.get("field") or "").strip()
    question = str(item.get("question") or "").strip()
    text = f"{field} {question}"
    categories: set[str] = set()
    if _IDENTITY_HINT_PATTERN.search(text):
        categories.add("identity")
    if _CODE_HINT_PATTERN.search(text):
        categories.add("code")
    if _STATUS_HINT_PATTERN.search(text):
        categories.add("status")
    if _TIME_HINT_PATTERN.search(text):
        categories.add("time")
    if _REASON_HINT_PATTERN.search(text):
        categories.add("reason")
    return categories


def _is_direct_field_requested(state: dict[str, Any], field: str) -> bool:
    if not field:
        return False
    query_rewrite = dict(state.get("query_rewrite") or {})
    text = " ".join(
        [
            str(state.get("question") or ""),
            str(query_rewrite.get("normalized_query") or ""),
            " ".join(str(item) for item in list(query_rewrite.get("keywords") or [])),
        ]
    ).lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(field).lower())
    if not normalized:
        return False
    return normalized in re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def _filter_required_answers_by_question(required_answers: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    if not required_answers:
        return []
    focus = _question_focus_categories(state)
    if not focus:
        return required_answers

    filtered: list[dict[str, Any]] = []
    for item in required_answers:
        row = dict(item or {})
        field = str(row.get("field") or "").strip()
        if not field:
            continue
        if _is_direct_field_requested(state, field):
            filtered.append(row)
            continue
        categories = _required_answer_categories(row)
        if categories & focus:
            filtered.append(row)

    if filtered:
        return filtered
    return []


_DEFAULT_REPLAN_TRIGGERS = [
    "goal_unexecutable",
    "evidence_conflict",
    "missing_required_context",
    "capability_not_supported",
]


def _parse_plan(raw_output: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return "", [], []
    if not isinstance(parsed, dict):
        return "", [], []

    hypothesis = str(parsed.get("hypothesis") or "").strip()
    goals_raw = parsed.get("investigation_goals")
    goals = [str(item).strip() for item in list(goals_raw or []) if str(item).strip()] if isinstance(goals_raw, list) else []
    required_answers = _parse_required_answers(parsed.get("required_answers"))
    return hypothesis, goals, required_answers


def _normalize_v2_goal(raw: Any, idx: int) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {"goal": str(raw or "").strip()}
    goal_id = str(item.get("id") or f"g{idx}").strip()
    goal_text = str(item.get("goal") or item.get("objective") or "").strip()
    capability = str(item.get("required_capability") or "runtime_evidence").strip()
    try:
        priority = int(item.get("priority") or idx)
    except (TypeError, ValueError):
        priority = idx
    success_criteria = [str(row).strip() for row in list(item.get("success_criteria") or []) if str(row).strip()]
    expected_evidence = [str(row).strip() for row in list(item.get("expected_evidence") or []) if str(row).strip()]
    depends_on = [str(row).strip() for row in list(item.get("depends_on") or []) if str(row).strip()]
    return {
        "id": goal_id or f"g{idx}",
        "goal": goal_text or f"调查目标{idx}",
        "required_capability": capability or "runtime_evidence",
        "priority": priority,
        "required": bool(item.get("required", True)),
        "success_criteria": success_criteria or ["获得可验证证据"],
        "expected_evidence": expected_evidence or ["evidence"],
        "depends_on": depends_on,
    }


def _normalize_investigation_plan_v2(parsed: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    raw_goals = parsed.get("goals")
    if not isinstance(raw_goals, list) or not raw_goals:
        legacy_goals = [str(item).strip() for item in list(parsed.get("investigation_goals") or []) if str(item).strip()]
        raw_goals = [
            {
                "id": f"g{idx}",
                "goal": goal,
                "required_capability": "runtime_evidence",
                "priority": idx,
                "required": True,
                "success_criteria": ["获得可验证运行时证据"],
                "expected_evidence": ["runtime_evidence"],
                "depends_on": [],
            }
            for idx, goal in enumerate(legacy_goals, start=1)
        ]
    goals = [_normalize_v2_goal(item, idx) for idx, item in enumerate(list(raw_goals or []), start=1)]
    goals = sorted(goals, key=lambda row: int(row.get("priority") or 0))
    if not goals:
        _, fallback_goals, _ = _fallback_plan(state)
        goals = [_normalize_v2_goal(goal, idx) for idx, goal in enumerate(fallback_goals, start=1)]

    hypothesis = str(parsed.get("hypothesis") or "").strip()
    if not hypothesis:
        hypothesis = f"{_clip(state.get('question'), 40)}相关链路存在异常"
    finish_criteria = [str(row).strip() for row in list(parsed.get("finish_criteria") or []) if str(row).strip()]
    replan_triggers = [str(row).strip() for row in list(parsed.get("replan_triggers") or []) if str(row).strip()]
    return {
        "plan_id": str(parsed.get("plan_id") or "plan_001").strip() or "plan_001",
        "hypothesis": hypothesis,
        "goals": goals,
        "finish_criteria": finish_criteria or ["根因明确", "证据链闭合", "用户问题已被直接回答"],
        "replan_triggers": replan_triggers or list(_DEFAULT_REPLAN_TRIGGERS),
    }


def _needs_trace_runtime_evidence(state: dict[str, Any]) -> bool:
    query_rewrite = dict(state.get("query_rewrite") or {})
    structured = dict(state.get("structured_context") or {})
    text = " ".join(
        [
            str(state.get("question") or ""),
            str(query_rewrite.get("normalized_query") or ""),
            " ".join(str(item) for item in list(query_rewrite.get("keywords") or [])),
        ]
    )
    trace_id = str(query_rewrite.get("trace_id") or structured.get("trace_id") or "").strip()
    return bool((trace_id or _TRACE_ID_PATTERN.search(text)) and _TRACE_FAILURE_REASON_PATTERN.search(text))


def _has_placeholder_planning(plan: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(plan.get("hypothesis") or ""),
            " ".join(str(dict(item or {}).get("goal") or "") for item in list(plan.get("goals") or [])),
        ]
    )
    return bool(_PLACEHOLDER_PLAN_PATTERN.search(text))


def _force_trace_runtime_evidence_plan(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    query_rewrite = dict(state.get("query_rewrite") or {})
    trace_id = str(query_rewrite.get("trace_id") or dict(state.get("structured_context") or {}).get("trace_id") or "").strip()
    goal_text = "查询 trace 运行时日志定位生单失败原因"
    if not trace_id and _TRACE_ID_PATTERN.search(str(state.get("question") or "")):
        goal_text = "查询用户问题中的 trace 运行时日志定位生单失败原因"
    updated = dict(plan)
    updated["hypothesis"] = "生单失败原因需要通过运行时日志证据定位"
    updated["goals"] = [
        {
            "id": "g1",
            "goal": goal_text,
            "required_capability": "runtime_evidence",
            "priority": 1,
            "required": True,
            "success_criteria": ["拿到 trace 相关日志证据", "明确生单失败阶段或失败原因"],
            "expected_evidence": ["log_event", "failure_reason"],
            "depends_on": [],
        }
    ]
    updated["finish_criteria"] = list(plan.get("finish_criteria") or ["根因明确", "证据链闭合", "用户问题已被直接回答"])
    updated["replan_triggers"] = list(plan.get("replan_triggers") or _DEFAULT_REPLAN_TRIGGERS)
    return updated


def _sync_v2_goals_with_goal_texts(investigation_plan: dict[str, Any], goal_texts: list[str]) -> dict[str, Any]:
    """保持 V2 plan.goals 与兼容字段 investigation_goals 同步。"""
    normalized_texts = [str(item).strip() for item in list(goal_texts or []) if str(item).strip()]
    if not normalized_texts:
        return investigation_plan

    original_goals = [dict(item or {}) for item in list(investigation_plan.get("goals") or [])]
    original_texts = [str(item.get("goal") or "").strip() for item in original_goals]
    if original_texts == normalized_texts:
        return investigation_plan

    by_text = {str(item.get("goal") or "").strip(): item for item in original_goals if str(item.get("goal") or "").strip()}
    synced_goals: list[dict[str, Any]] = []
    for idx, text in enumerate(normalized_texts, start=1):
        if text in by_text:
            row = dict(by_text[text])
            row["priority"] = idx
            row["id"] = str(row.get("id") or f"g{idx}")
        else:
            row = {
                "id": f"g{idx}",
                "goal": text,
                "required_capability": "runtime_evidence",
                "priority": idx,
                "required": True,
                "success_criteria": ["获得可验证证据"],
                "expected_evidence": ["evidence"],
                "depends_on": [],
            }
        synced_goals.append(_normalize_v2_goal(row, idx))

    updated = dict(investigation_plan)
    updated["goals"] = synced_goals
    return updated


def _parse_investigation_plan_v2(raw_output: str, state: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_output)
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if not parsed:
        hypothesis, goals, _ = _fallback_plan(state)
        parsed = {"hypothesis": hypothesis, "investigation_goals": goals}
    return _normalize_investigation_plan_v2(parsed, state)


def _build_minimal_goals_from_required_answers(required_answers: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in required_answers:
        if not bool(item.get("required", True)):
            continue
        field = str(item.get("field") or "").strip()
        question = str(item.get("question") or "").strip()
        if not field:
            continue
        if question:
            rows.append(f"获取并确认字段 {field}：{question}")
        else:
            rows.append(f"获取并确认字段 {field} 的明确取值")
    if rows:
        return rows

    # 兜底：当 required 标记为空时，仍尽量按字段拆分。
    for item in required_answers:
        field = str(item.get("field") or "").strip()
        if field:
            rows.append(f"获取并确认字段 {field} 的明确取值")
    return rows


def _normalize_goals_with_required_answers(goals: list[str], required_answers: list[dict[str, Any]]) -> list[str]:
    normalized_goals = [str(item).strip() for item in list(goals or []) if str(item).strip()]
    required_fields = [
        str(item.get("field") or "").strip()
        for item in required_answers
        if bool(item.get("required", True)) and str(item.get("field") or "").strip()
    ]

    if len(required_fields) < 2:
        return normalized_goals

    lowered_fields = [field.lower() for field in required_fields]

    def _goal_field_count(goal: str) -> int:
        lowered_goal = str(goal or "").lower()
        return sum(1 for field in lowered_fields if field in lowered_goal)

    has_mixed_goal = any(_goal_field_count(goal) > 1 for goal in normalized_goals)
    covered_fields = {
        field
        for field, lowered in zip(required_fields, lowered_fields, strict=False)
        if any(lowered in str(goal or "").lower() for goal in normalized_goals)
    }
    missing_fields = [field for field in required_fields if field not in covered_fields]

    # 多字段场景下，目标数不足、出现混合目标、或字段未覆盖时，强制拆成最小执行单元。
    if len(normalized_goals) < len(required_fields) or has_mixed_goal or missing_fields:
        split_goals = _build_minimal_goals_from_required_answers(required_answers)
        if split_goals:
            return split_goals
    return normalized_goals


def _align_goals_with_required_answers(goals: list[str], required_answers: list[dict[str, Any]]) -> list[str]:
    normalized_goals = [str(item).strip() for item in list(goals or []) if str(item).strip()]
    required_fields = [
        str(item.get("field") or "").strip()
        for item in list(required_answers or [])
        if bool(item.get("required", True)) and str(item.get("field") or "").strip()
    ]
    if not normalized_goals or not required_fields:
        return normalized_goals

    lowered_fields = [field.lower() for field in required_fields]
    matched_goals = [
        goal
        for goal in normalized_goals
        if any(field in str(goal).lower() for field in lowered_fields)
    ]
    if len(required_fields) == 1 and matched_goals:
        return matched_goals[:1]
    if matched_goals and len(matched_goals) >= len(required_fields):
        return matched_goals[: len(required_fields)]
    if len(normalized_goals) > len(required_fields):
        return normalized_goals[: len(required_fields)]
    return normalized_goals


def _fallback_plan(state: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    question = str(state.get("question") or "问题定位").strip()
    hypothesis = f"{_clip(question, 40)}相关链路存在异常"
    goals = [
        "匹配历史排障案例",
        "确认异常发生在哪个服务",
        "定位触发异常的代码逻辑",
    ]
    required_answers = _derive_required_answers_from_query(state)
    return hypothesis, goals, required_answers


def _avoid_rejected(hypothesis: str, rejected_hypothesis: list[str]) -> str:
    rejected_set = {item.strip() for item in rejected_hypothesis if item.strip()}
    if not hypothesis:
        return hypothesis
    if hypothesis not in rejected_set:
        return hypothesis
    return f"排除已证伪方向后，重新定位：{hypothesis}"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    previous_investigation = dict(state.get("investigation") or {})
    previous_events = [dict(item or {}) for item in list(previous_investigation.get("events") or [])]
    context = _build_prompt_context(state)
    knowledge_context = dict(state.get("knowledge_context") or {})
    rejected = [str(item).strip() for item in list(state.get("rejected_hypothesis") or []) if str(item).strip()]
    _LOGGER.info(
        "planner start query=%s rejected_count=%d replan_reason=%s domain_docs=%d case_docs=%d code_docs=%d",
        _clip(context.get("user_query"), 120),
        len(rejected),
        _clip(context.get("replan_reason"), 120),
        len(list(knowledge_context.get("domain_docs") or [])),
        len(list(knowledge_context.get("case_docs") or [])),
        len(list(knowledge_context.get("code_docs") or [])),
    )

    system_prompt = load_prompt("planner_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "planner_user_prompt.txt",
        user_query=context["user_query"],
        normalized_query=context["normalized_query"],
        keywords=context["keywords"],
        domain_knowledge=context["domain_knowledge"],
        case_knowledge=context["case_knowledge"],
        code_knowledge=context["code_knowledge"],
        rejected_hypothesis=context["rejected_hypothesis"],
        replan_reason=context["replan_reason"],
        previous_plan_json=context["previous_plan_json"],
        goal_status_json=context["goal_status_json"],
        evidence_json=context["evidence_json"],
        failed_goal_json=context["failed_goal_json"],
    )
    raw_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt)

    investigation_plan = _parse_investigation_plan_v2(raw_output, state)
    hypothesis = str(investigation_plan.get("hypothesis") or "").strip()
    goals = [str(dict(item or {}).get("goal") or "").strip() for item in list(investigation_plan.get("goals") or [])]
    _, _, required_answers = _parse_plan(raw_output)
    if not hypothesis or not goals:
        fallback_hypothesis, fallback_goals, required_answers = _fallback_plan(state)
        investigation_plan = _normalize_investigation_plan_v2(
            {"hypothesis": fallback_hypothesis, "investigation_goals": fallback_goals},
            state,
        )
        hypothesis = str(investigation_plan.get("hypothesis") or "").strip()
        goals = [str(dict(item or {}).get("goal") or "").strip() for item in list(investigation_plan.get("goals") or [])]
    if not required_answers:
        required_answers = _derive_required_answers_from_query(state)
    required_answers = _filter_required_answers_by_question(required_answers, state)
    if not required_answers:
        required_answers = _derive_required_answers_from_query(state)
    goals = _normalize_goals_with_required_answers(goals, required_answers)
    goals = _align_goals_with_required_answers(goals, required_answers)
    if not goals:
        goals = _build_minimal_goals_from_required_answers(required_answers)
    if not goals:
        hypothesis, goals, required_answers = _fallback_plan(state)

    if _needs_trace_runtime_evidence(state) and (
        _has_placeholder_planning(investigation_plan)
        or not any(str(dict(item or {}).get("required_capability") or "") == "runtime_evidence" for item in list(investigation_plan.get("goals") or []))
    ):
        investigation_plan = _force_trace_runtime_evidence_plan(investigation_plan, state)
        hypothesis = str(investigation_plan.get("hypothesis") or "").strip()
        goals = [str(dict(item or {}).get("goal") or "").strip() for item in list(investigation_plan.get("goals") or [])]

    hypothesis = _avoid_rejected(hypothesis, rejected)
    investigation_plan = _sync_v2_goals_with_goal_texts(investigation_plan, goals)
    goals = [str(dict(item or {}).get("goal") or "").strip() for item in list(investigation_plan.get("goals") or [])]

    plan = {
        "hypothesis": hypothesis,
        "investigation_goals": goals,
        "required_answers": required_answers,
        "goals": list(investigation_plan.get("goals") or []),
        "finish_criteria": list(investigation_plan.get("finish_criteria") or []),
        "replan_triggers": list(investigation_plan.get("replan_triggers") or []),
    }

    execution = dict(state.get("execution") or {})
    execution["goal_index"] = 0
    execution["objective_retry_count"] = 0
    execution["insufficient_round_count"] = 0
    execution["evidence_graph"] = {
        "hypothesis": hypothesis,
        "evidence": [],
        "supported": None,
    }

    structured = dict(state.get("structured_context") or {})
    structured["planner_plan"] = dict(plan)
    structured["planner_llm_trace"] = {
        "llm_request": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "llm_response": {
            "raw_output": raw_output,
            "parse_ok": bool(hypothesis and goals),
        },
    }

    state["plan"] = plan
    planner_event = {
        "type": "planner",
        "message": "created investigation plan",
        "payload": {
            "plan_id": investigation_plan.get("plan_id"),
            "replan_count": _as_int(previous_investigation.get("replan_count"), 0),
        },
    }
    max_replans = previous_investigation.get("max_replans")
    if max_replans is None:
        max_replans = state.get("max_replan")
    state["investigation"] = {
        "plan": investigation_plan,
        "current_goal_id": str(dict(list(investigation_plan.get("goals") or [{}])[0]).get("id") or ""),
        "goal_status": {
            str(dict(item or {}).get("id") or f"g{idx}"): "pending"
            for idx, item in enumerate(list(investigation_plan.get("goals") or []), start=1)
        },
        "evidence": [],
        "events": [*previous_events, planner_event],
        "pending_execution": {},
        "last_route_result": {},
        "last_executor_result": {},
        "consumed_result_ids": [],
        "retry_counts_by_goal": {},
        "max_retries_per_goal": 2,
        "replan_count": _as_int(dict(state.get("investigation") or {}).get("replan_count"), 0),
        "max_replans": _as_int(max_replans, 1),
        "failure_reason": "",
    }
    state["execution"] = execution
    state["structured_context"] = structured
    state["route"] = "plan_controller"
    _LOGGER.info(
        "planner finished hypothesis=%s goals=%d required_answers=%d rejected_count=%d",
        _clip(hypothesis, 200),
        len(goals),
        len(required_answers),
        len(rejected),
    )
    for idx, goal in enumerate(goals, start=1):
        _LOGGER.info("planner goal[%d]=%s", idx, _clip(goal, 240))
    for idx, item in enumerate(required_answers, start=1):
        _LOGGER.info("planner required_answer[%d]=field:%s required:%s question:%s", idx, str(item.get("field") or ""), bool(item.get("required")), _clip(item.get("question"), 200))
    return dict(state)
