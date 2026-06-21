"""Dynamic knowledge retrieve 节点。"""

from __future__ import annotations

import logging
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import (
    query_parent_docs_from_rag,
    resolve_intent_label_for_rag,
)

_LOGGER = logging.getLogger(__name__)


def _build_retrieve_query(state: dict[str, Any], *, objective: str) -> str:
    query_rewrite = dict(state.get("query_rewrite") or {})
    normalized_query = str(query_rewrite.get("normalized_query") or state.get("question") or "").strip()
    hypothesis = str(dict(state.get("plan") or {}).get("hypothesis") or "").strip()
    keywords = [str(item).strip() for item in list(query_rewrite.get("keywords") or []) if str(item).strip()]
    parts = [normalized_query, hypothesis, objective, " ".join(keywords)]
    return "\n".join(part for part in parts if part)


def _resolve_current_objective(state: dict[str, Any]) -> tuple[str, int]:
    execution = dict(state.get("execution") or {})
    goal_index = int(execution.get("goal_index") or 0)
    goals = [str(item).strip() for item in list(dict(state.get("plan") or {}).get("investigation_goals") or []) if str(item).strip()]
    if not goals:
        return "", 0
    if goal_index < 0:
        goal_index = 0
    if goal_index >= len(goals):
        goal_index = len(goals) - 1
    return goals[goal_index], goal_index


def _split_knowledge_docs(parent_docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    domain_docs: list[dict[str, Any]] = []
    case_docs: list[dict[str, Any]] = []
    code_docs: list[dict[str, Any]] = []

    for item in list(parent_docs or []):
        row = dict(item or {})
        knowledge_type = str(row.get("knowledge_type") or "").strip().lower()
        if knowledge_type == "case":
            case_docs.append(row)
            continue
        if knowledge_type == "domain":
            domain_docs.append(row)
            continue
        if knowledge_type == "code":
            code_docs.append(row)
            continue

        text = f"{row.get('path') or ''} {row.get('content') or ''}".lower()
        if any(token in text for token in ("case", "案例", "复盘", "故障")):
            case_docs.append(row)
        elif any(token in text for token in ("java", "代码", "class", "method", "repo", "git")):
            code_docs.append(row)
        else:
            domain_docs.append(row)

    # 兜底：至少给 domain 一份，避免 planner 上下文为空。
    if not domain_docs and parent_docs:
        domain_docs = [dict(parent_docs[0])]

    return {
        "domain_docs": domain_docs,
        "case_docs": case_docs,
        "code_docs": code_docs,
    }


def retrieve_domain_docs(basis: dict[str, Any], parent_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ = basis
    return _split_knowledge_docs(parent_docs)["domain_docs"]


def retrieve_case_docs(basis: dict[str, Any], parent_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ = basis
    return _split_knowledge_docs(parent_docs)["case_docs"]


def retrieve_code_docs(basis: dict[str, Any], parent_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ = basis
    return _split_knowledge_docs(parent_docs)["code_docs"]


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    hypothesis = str(dict(state.get("plan") or {}).get("hypothesis") or "").strip()
    objective, goal_index = _resolve_current_objective(state)
    query_text = _build_retrieve_query(state, objective=objective)
    intent_zh = resolve_intent_label_for_rag(state)

    sub_chunks, parent_chunks, parent_docs = query_parent_docs_from_rag(
        question=query_text,
        intent_zh=intent_zh,
        sub_chunk_top_k=None,
        parent_top_k=None,
    )

    basis = {
        "hypothesis": hypothesis,
        "objective": objective,
        "query_rewrite": dict(state.get("query_rewrite") or {}),
    }
    split_docs = _split_knowledge_docs(parent_docs)
    state["knowledge_context"] = {
        "query_basis": basis,
        "domain_docs": split_docs["domain_docs"],
        "case_docs": split_docs["case_docs"],
        "code_docs": split_docs["code_docs"],
        "rag_sub_chunk_docs": sub_chunks,
        "rag_docs": parent_chunks,
        "rag_parent_docs": parent_docs,
    }

    structured = dict(state.get("structured_context") or {})
    state["structured_context"] = {
        **structured,
        "knowledge_context": state["knowledge_context"],
    }

    execution = dict(state.get("execution") or {})
    execution.setdefault("goal_index", goal_index)
    state["execution"] = execution

    # 初次检索（尚无 hypothesis）先进入 Planner；补证场景直接回 Reactor。
    state["route"] = "planner" if not hypothesis else "reactor"
    _LOGGER.info(
        "knowledge_retrieve done route=%s goal_index=%d objective=%s hypothesis_present=%s sub_chunks=%d parent_chunks=%d parent_docs=%d domain_docs=%d case_docs=%d code_docs=%d",
        state["route"],
        goal_index,
        objective,
        bool(hypothesis),
        len(sub_chunks),
        len(parent_chunks),
        len(parent_docs),
        len(split_docs["domain_docs"]),
        len(split_docs["case_docs"]),
        len(split_docs["code_docs"]),
    )
    return dict(state)
