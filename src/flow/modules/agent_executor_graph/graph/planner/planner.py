"""规划节点（Planner）。

业务职责：
- 聚合 RAG 检索结果（日志诊断/chunk/父文档）构建 LLM 规划上下文。
- 调用大模型输出结构化排障计划，并映射为可执行 plan_steps。
- LLM 不可用或输出异常时，回退到稳定的规则计划模板。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.plan_step import PlanStep
from llm.llm import chat_with_llm, load_prompt, render_prompt

_LOGGER = logging.getLogger(__name__)
_MAX_LOG_DOCS_FOR_PROMPT = 3
_MAX_RAG_DOCS_FOR_PROMPT = 5
_MAX_LOG_TEXT_LEN = 200
_MAX_RAG_TEXT_LEN = 300
_MAX_QUERY_LEN = 500
_MAX_PLAN_STEPS = 6
_ALLOWED_TOOL_NAMES = {"log_query", "dependency_log_query", "knowledge_lookup", "code_clone", "code_pull"}
_SKILLS_DIR = Path(__file__).resolve().parents[5] / "skills"
_MAX_SKILL_TEXT_LEN = 1200
_MAX_SKILL_TOTAL_LEN = 12000


def _as_int(value: Any, default: int) -> int:
    """把输入转换成 int，异常时使用默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_text(text: Any, max_len: int) -> str:
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


def _read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""


# 方法注释（业务）:
# - 入参：无。
# - 出参：`list[dict[str, str]]`=skills 文件清单（path/content）。
# - 方法逻辑：每次规划都强制扫描并读取 `src/skills` 下全部 Markdown 技能文件，供大模型制定 planStep。
def _load_all_skills() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not _SKILLS_DIR.is_dir():
        _LOGGER.warning("planner skills 目录不存在: %s", _SKILLS_DIR)
        return rows

    for file_path in sorted(_SKILLS_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() != ".md":
            continue
        content = _read_text_safely(file_path)
        rel_path = str(file_path.relative_to(_SKILLS_DIR.parent))
        rows.append({"path": rel_path, "content": content})

    _LOGGER.info("planner 强制读取 skills 完成: count=%d", len(rows))
    return rows


def _build_skills_context(skills: list[dict[str, str]]) -> str:
    sections: list[str] = []
    total_len = 0
    for idx, item in enumerate(skills, start=1):
        path = str(item.get("path") or "").strip()
        content = _clip_text(item.get("content"), _MAX_SKILL_TEXT_LEN)
        section = f"[skill {idx}] path={path}\n{content}".strip()
        if not section:
            continue
        total_len += len(section)
        if total_len > _MAX_SKILL_TOTAL_LEN:
            break
        sections.append(section)
    return "\n\n".join(sections).strip() or "无可用 skills 定义"


# 方法注释（业务）:
# - 入参：`state`(dict[str, Any])=当前 AgentState。
# - 出参：`str`=用户原始问题文本。
# - 方法逻辑：优先 messages[-1].content，其次 structured_context.user_query/question，最后回退 state.question。
def _pick_user_query(state: dict[str, Any]) -> str:
    message_rows = state.get("messages")
    if not isinstance(message_rows, list):
        message_rows = dict(state.get("context") or {}).get("messages")
    if isinstance(message_rows, list) and message_rows:
        for row in reversed(message_rows):
            if isinstance(row, dict):
                content = row.get("content")
            else:
                content = getattr(row, "content", "")
            text = str(content or "").strip()
            if text:
                return text

    structured_context = dict(state.get("structured_context") or {})
    for key in ("user_query", "question"):
        value = structured_context.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(state.get("question") or "").strip()


def _normalize_bm25_log_docs(rag_diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(rag_diagnosis.get("bm25_log_docs") or [])
    result: list[dict[str, Any]] = []
    for item in rows[:_MAX_LOG_DOCS_FOR_PROMPT]:
        if not isinstance(item, dict):
            continue
        text = _clip_text(item.get("text"), _MAX_LOG_TEXT_LEN)
        if not text:
            continue
        result.append(
            {
                "path": str(item.get("path") or ""),
                "text": text,
                "score": _to_float(item.get("score"), 0.0),
            }
        )

    # 兼容 rag_retrieve 当前输出结构：bm25_top_logs / stage1_top_docs（字符串列表）。
    if result:
        return result
    top_logs = list(rag_diagnosis.get("bm25_top_logs") or rag_diagnosis.get("stage1_top_docs") or [])
    for item in top_logs[:_MAX_LOG_DOCS_FOR_PROMPT]:
        text = _clip_text(item, _MAX_LOG_TEXT_LEN)
        if not text:
            continue
        result.append({"path": "", "text": text, "score": 0.0})
    return result


# 方法注释（业务）:
# - 入参：`state`(dict[str, Any])=当前 AgentState；`skills`=run 阶段已强制读取的 skill 文件列表。
# - 出参：`dict[str, Any]`=供大模型规划的摘要上下文（用户问题/日志摘要/RAG 摘要/完整文档引用）。
# - 方法逻辑：从 structured_context 与顶层字段提取并裁剪检索结果，按优先级组织成低 token 成本摘要。
def _prepare_context_for_llm(state: dict[str, Any], skills: list[dict[str, str]]) -> dict[str, Any]:
    structured_context = dict(state.get("structured_context") or {})
    rag_diagnosis = dict(structured_context.get("rag_diagnosis") or {})
    user_query = _clip_text(_pick_user_query(state), _MAX_QUERY_LEN)
    if not user_query:
        user_query = _clip_text(state.get("question"), _MAX_QUERY_LEN)

    bm25_docs = _normalize_bm25_log_docs(rag_diagnosis)
    error_info = dict(rag_diagnosis.get("error_info") or {})
    error_code = str(error_info.get("error_code") or rag_diagnosis.get("error_code") or "").strip()
    exception_type = str(error_info.get("exception") or rag_diagnosis.get("exception_type") or "").strip()
    rag_query = str(rag_diagnosis.get("rag_query") or "").strip()

    log_rows: list[str] = []
    if error_code or exception_type:
        log_rows.append(f"错误摘要: error_code={error_code or 'N/A'}, exception={exception_type or 'N/A'}")
    if rag_query:
        log_rows.append(f"增强检索词: {rag_query}")
    for idx, item in enumerate(bm25_docs, start=1):
        log_rows.append(
            f"[日志{idx}] path={item['path'] or 'N/A'} score={item['score']:.4f} text={item['text']}"
        )
    log_diagnosis_summary = "\n".join(log_rows).strip() or "无日志诊断信息"

    rag_docs = list(structured_context.get("rag_docs") or state.get("rag_docs") or [])
    rag_rows: list[str] = []
    for idx, item in enumerate(rag_docs[:_MAX_RAG_DOCS_FOR_PROMPT], start=1):
        if not isinstance(item, dict):
            continue
        payload = dict(item.get("payload") or {})
        path = str(payload.get("path") or item.get("path") or "").strip()
        file_name = str(payload.get("file_name") or Path(path).name or "unknown")
        text = _clip_text(item.get("text"), _MAX_RAG_TEXT_LEN)
        if not text:
            continue
        score = _to_float(item.get("score"), 0.0)
        rag_rows.append(f"[文档{idx}] 来源={file_name} path={path or 'N/A'} score={score:.4f} 内容={text}")
    rag_solutions_summary = "\n".join(rag_rows).strip() or "无 RAG 文档摘要"

    parent_docs = list(structured_context.get("rag_parent_docs") or state.get("rag_parent_docs") or [])
    ref_rows: list[str] = []
    for idx, item in enumerate(parent_docs, start=1):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        score = _to_float(item.get("score"), 0.0)
        file_name = Path(path).name or path
        ref_rows.append(f"- [{idx}] {file_name} | path={path} | score={score:.4f}")
    full_docs_references = "\n".join(ref_rows).strip() or "无完整文档引用"
    skills_context = _build_skills_context(skills)

    return {
        "user_query": user_query,
        "log_diagnosis_summary": log_diagnosis_summary,
        "rag_solutions_summary": rag_solutions_summary,
        "full_docs_references": full_docs_references,
        "skills_context": skills_context,
        "skills_count": len(skills),
        "skills_catalog": [str(item.get("path") or "") for item in skills],
        "error_code": error_code,
        "exception_type": exception_type,
        "rag_query": rag_query,
    }


def _build_planner_user_prompt(context: dict[str, Any]) -> str:
    return render_prompt(
        "planner_user_prompt.txt",
        user_query=str(context.get("user_query") or ""),
        skills_context=str(context.get("skills_context") or ""),
        log_diagnosis_summary=str(context.get("log_diagnosis_summary") or ""),
        rag_solutions_summary=str(context.get("rag_solutions_summary") or ""),
        full_docs_references=str(context.get("full_docs_references") or ""),
    )


def _map_tool_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if name in _ALLOWED_TOOL_NAMES:
        return name
    lower = name.lower()
    if "code_pull" in lower or ("pull" in lower and "git" in lower):
        return "code_pull"
    if "code_clone" in lower or "clone" in lower:
        return "code_clone"
    if "depend" in lower or "dependency" in lower:
        return "dependency_log_query"
    if "log" in lower:
        return "log_query"
    if any(token in lower for token in ("knowledge", "doc", "rag", "wiki")):
        return "knowledge_lookup"
    return ""


def _normalize_plan_steps(raw_steps: Any) -> list[PlanStep]:
    rows = list(raw_steps or []) if isinstance(raw_steps, list) else []
    result: list[PlanStep] = []
    for item in rows[:_MAX_PLAN_STEPS]:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("action_type") or "").strip()
        if action_type == "merge_evidence":
            result.append({"action_type": "merge_evidence", "tool_name": None, "params": {}})
            continue
        tool_name = _map_tool_name(item.get("tool_name"))
        if not tool_name:
            continue
        params = dict(item.get("params") or {})
        result.append(
            {
                "action_type": "tool_call",
                "tool_name": tool_name,
                "params": params,
            }
        )

    has_tool_call = any(str(step.get("action_type")) == "tool_call" for step in result)
    if not has_tool_call:
        return []
    if not result or str(result[-1].get("action_type")) != "merge_evidence":
        result.append({"action_type": "merge_evidence", "tool_name": None, "params": {}})
    return result


def _fallback_plan_steps(intent_type: str, replan_count: int) -> list[PlanStep]:
    if replan_count > 0:
        return [
            {"action_type": "tool_call", "tool_name": "log_query", "params": {}},
            {"action_type": "tool_call", "tool_name": "dependency_log_query", "params": {}},
            {"action_type": "merge_evidence", "tool_name": None, "params": {}},
        ]
    if intent_type == "OPS_ANALYSIS":
        return [
            {"action_type": "tool_call", "tool_name": "log_query", "params": {}},
            {"action_type": "tool_call", "tool_name": "dependency_log_query", "params": {}},
            {"action_type": "merge_evidence", "tool_name": None, "params": {}},
        ]
    if intent_type in {"GENERAL_QA", "ORDER_INFO_QUERY"}:
        return [
            {"action_type": "tool_call", "tool_name": "knowledge_lookup", "params": {}},
            {"action_type": "merge_evidence", "tool_name": None, "params": {}},
        ]
    return [
        {"action_type": "tool_call", "tool_name": "knowledge_lookup", "params": {}},
        {"action_type": "tool_call", "tool_name": "log_query", "params": {}},
        {"action_type": "merge_evidence", "tool_name": None, "params": {}},
    ]


def _plan_with_llm(context: dict[str, Any]) -> tuple[list[PlanStep], dict[str, Any]]:
    system_prompt = load_prompt("planner_system_prompt.txt", default="")
    user_prompt = _build_planner_user_prompt(context)
    if not user_prompt:
        _LOGGER.warning("planner prompt 缺失: planner_user_prompt.txt")
        return [], {"raw_output": "", "parse_ok": False}
    raw_output = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    parsed = _parse_json_object(raw_output)
    if not parsed:
        return [], {"raw_output": raw_output, "parse_ok": False}
    steps = _normalize_plan_steps(parsed.get("plan_steps"))
    plan_meta = {
        "raw_output": raw_output,
        "parse_ok": True,
        "problem_analysis": str(parsed.get("problem_analysis") or "").strip(),
        "solution_steps": list(parsed.get("solution_steps") or []),
        "validation": str(parsed.get("validation") or "").strip(),
        "references": list(parsed.get("references") or []),
    }
    return steps, plan_meta


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """生成或重生成排障计划。

    入参：
    - payload: AgentState，至少包含 question、intent_type、replan_count，并携带 rag_retrieve 结果。

    返参：
    - AgentState: 写入 plan_steps/current_step_index，并路由到 plan_execute。
    """
    state: AgentState = dict(payload)
    question = str(state.get("question") or "").strip()
    intent_type = str(state.get("intent_type") or "UNKNOWN")
    replan_count = _as_int(state.get("replan_count"), 0)
    _LOGGER.info(
        "planner 开始制定计划: intent=%s replan=%d question=%s",
        intent_type,
        replan_count,
        _clip_text(question, 120),
    )
    # 强制步骤：每次规划前先读取 src/skills 下全部 skill 文件。
    skills = _load_all_skills()
    context_for_llm = _prepare_context_for_llm(state, skills)
    llm_plan_steps, llm_plan_meta = _plan_with_llm(context_for_llm)

    if llm_plan_steps:
        plan_steps = llm_plan_steps
        _LOGGER.info("planner LLM 规划成功: steps=%d", len(plan_steps))
    else:
        plan_steps = _fallback_plan_steps(intent_type=intent_type, replan_count=replan_count)
        _LOGGER.info("planner 使用规则兜底计划: steps=%d intent=%s replan=%d", len(plan_steps), intent_type, replan_count)

    # 首次规划或 replan 都从第 0 步开始执行新计划。
    if not state.get("plan_steps") or replan_count > 0:
        state["current_step_index"] = 0

    structured_context = dict(state.get("structured_context") or {})
    state["structured_context"] = {
        **structured_context,
        "user_query": str(context_for_llm.get("user_query") or ""),
        "skills_catalog": list(context_for_llm.get("skills_catalog") or []),
        "skills_count": _as_int(context_for_llm.get("skills_count"), 0),
        "planner_context_for_llm": {
            "skills_context": str(context_for_llm.get("skills_context") or ""),
            "log_diagnosis_summary": str(context_for_llm.get("log_diagnosis_summary") or ""),
            "rag_solutions_summary": str(context_for_llm.get("rag_solutions_summary") or ""),
            "full_docs_references": str(context_for_llm.get("full_docs_references") or ""),
        },
        "planner_plan": {
            "problem_analysis": str(llm_plan_meta.get("problem_analysis") or ""),
            "solution_steps": list(llm_plan_meta.get("solution_steps") or []),
            "validation": str(llm_plan_meta.get("validation") or ""),
            "references": list(llm_plan_meta.get("references") or []),
            "parse_ok": bool(llm_plan_meta.get("parse_ok")),
        },
    }
    _LOGGER.info(
        "planner 上下文摘要: skills=%d rag_docs=%d rag_parent_docs=%d parse_ok=%s",
        _as_int(context_for_llm.get("skills_count"), 0),
        len(list(state.get("rag_docs") or [])),
        len(list(state.get("rag_parent_docs") or [])),
        bool(llm_plan_meta.get("parse_ok")),
    )

    state["plan_steps"] = plan_steps
    state["route"] = "plan_execute"
    return dict(state)
