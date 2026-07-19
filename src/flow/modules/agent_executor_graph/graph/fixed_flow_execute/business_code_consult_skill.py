"""Business consult skill: combine business docs with local code-index analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import query_parent_docs_from_rag
from tool.registry import invoke_tool

_MAX_DOC_ROWS = 5
_MAX_DOC_CHARS = 400
_LOGGER = logging.getLogger(__name__)
_SKILL_DOC_PATH = Path(__file__).resolve().parents[5] / "skills" / "business_code_consult" / "SKILL.md"


def _clip(text: Any, max_len: int = _MAX_DOC_CHARS) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


def _split_docs(parent_docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    domain_docs: list[dict[str, Any]] = []
    case_docs: list[dict[str, Any]] = []
    code_docs: list[dict[str, Any]] = []
    for item in list(parent_docs or []):
        row = dict(item or {})
        knowledge_type = str(row.get("knowledge_type") or "").strip().lower()
        if knowledge_type == "case":
            case_docs.append(row)
        elif knowledge_type == "code":
            code_docs.append(row)
        else:
            domain_docs.append(row)
    if not domain_docs and parent_docs:
        domain_docs = [dict(parent_docs[0])]
    return {
        "domain_docs": domain_docs,
        "case_docs": case_docs,
        "code_docs": code_docs,
    }


def _doc_rows_as_evidence(rows: list[dict[str, Any]], *, title: str) -> list[str]:
    evidence: list[str] = [title]
    if not rows:
        evidence.append("- 无")
        return evidence
    for idx, row in enumerate(rows[:_MAX_DOC_ROWS], start=1):
        path = str(row.get("path") or "").strip()
        score = row.get("score")
        content = _clip(row.get("content") or row.get("text") or "")
        evidence.append(f"- [{idx}] path={path} score={score} content={content}")
    return evidence


def _build_evidence_context(
    *,
    question: str,
    domain_docs: list[dict[str, Any]],
    case_docs: list[dict[str, Any]],
    code_analysis: dict[str, Any],
) -> str:
    rows: list[str] = [
        "================ 用户问题 ================",
        str(question or "").strip() or "无",
        "",
        "================ 业务文档证据 ================",
    ]
    rows.extend(_doc_rows_as_evidence(domain_docs, title="[domain_docs]"))
    rows.extend(_doc_rows_as_evidence(case_docs, title="[case_docs]"))
    rows.extend(
        [
            "",
            "================ 实际代码分析证据（Code Index） ================",
        ]
    )
    if bool(code_analysis.get("ok")):
        rows.append(f"- 状态: 命中 ({str(code_analysis.get('mode') or '')})")
        rows.append(f"- 摘要: {str(code_analysis.get('summary') or '').strip()}")
        method = dict(code_analysis.get("current_method") or {})
        if method:
            rows.append(f"- 当前方法: {method}")
        symbol = dict(code_analysis.get("current_symbol") or {})
        if symbol:
            rows.append(f"- 当前符号: {symbol}")
        caller = list(code_analysis.get("caller") or [])
        callee = list(code_analysis.get("callee") or [])
        if caller:
            rows.append(f"- 调用者: {caller[:5]}")
        if callee:
            rows.append(f"- 被调用者: {callee[:5]}")
        for item in list(code_analysis.get("evidence") or [])[:4]:
            rows.append(f"- {str(item)}")
    else:
        rows.append("- 状态: 未命中")
        rows.append(f"- 原因: {str(code_analysis.get('error') or code_analysis.get('summary') or '').strip()}")
    rows.extend(
        [
            "",
            "请必须同时参考“业务文档证据”和“实际代码分析证据”回答；如果任一侧证据不足，明确说明不足点。",
        ]
    )
    return "\n".join(rows).strip()


def run(
    *,
    question: str,
    structured_context: dict[str, Any],
) -> dict[str, Any]:
    skill_doc_loaded = False
    skill_doc_chars = 0
    try:
        skill_doc_text = _SKILL_DOC_PATH.read_text(encoding="utf-8")
        skill_doc_loaded = True
        skill_doc_chars = len(skill_doc_text)
    except Exception:  # noqa: BLE001
        pass
    _LOGGER.info(
        "business_code_consult.skill_doc path=%s loaded=%s chars=%d",
        str(_SKILL_DOC_PATH),
        skill_doc_loaded,
        skill_doc_chars,
    )

    sub_chunks, parent_chunks, parent_docs = query_parent_docs_from_rag(
        question=str(question or "").strip(),
        intent_zh="业务咨询",
        sub_chunk_top_k=None,
        parent_top_k=None,
    )
    split_docs = _split_docs(parent_docs)
    domain_docs = list(split_docs.get("domain_docs") or [])
    case_docs = list(split_docs.get("case_docs") or [])
    code_docs = list(split_docs.get("code_docs") or [])

    code_seed_rows = [str(question or "").strip()]
    code_seed_rows.extend(str(item.get("content") or "") for item in code_docs[:2])
    code_analysis = invoke_tool(
        "analyzeCodeForBusinessConsult",
        {
            "question": str(question or "").strip(),
            "structured_context": dict(structured_context or {}),
            "evidence_rows": code_seed_rows,
        },
    )
    _LOGGER.info(
        "business_code_consult.code_index ok=%s mode=%s error=%s",
        bool(dict(code_analysis or {}).get("ok")),
        str(dict(code_analysis or {}).get("mode") or ""),
        str(dict(code_analysis or {}).get("error") or ""),
    )

    evidence_context = _build_evidence_context(
        question=question,
        domain_docs=domain_docs,
        case_docs=case_docs,
        code_analysis=code_analysis,
    )
    merged_evidence = {
        "logs": [],
        "knowledge": [{"text": str(row)} for row in _doc_rows_as_evidence(domain_docs, title="[domain_docs]")]
        + [{"text": str(row)} for row in _doc_rows_as_evidence(case_docs, title="[case_docs]")],
        "code": [{"text": str(row)} for row in list(code_analysis.get("evidence") or [])],
    }
    return {
        "skill_name": "business_code_consult",
        "skill_doc": {
            "path": str(_SKILL_DOC_PATH),
            "loaded": skill_doc_loaded,
            "chars": skill_doc_chars,
        },
        "knowledge_context": {
            "domain_docs": domain_docs,
            "case_docs": case_docs,
            "code_docs": code_docs,
            "rag_sub_chunk_docs": sub_chunks,
            "rag_docs": parent_chunks,
            "rag_parent_docs": parent_docs,
        },
        "code_analysis": code_analysis,
        "merged_evidence": merged_evidence,
        "evidence_context": evidence_context,
    }
