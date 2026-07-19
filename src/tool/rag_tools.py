"""Annotated RAG tools used by the agent runtime."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import query_parent_docs_from_rag


@tool(
    "rag_parent_chunk_query",
    description=(
        "RAG 父文档检索工具。根据问题查询子 chunk TopK、父 chunk TopK，并加载完整父文档内容，"
        "形成可复查业务文档证据；如果父文档无法读取，应明确只基于片段结论。"
    ),
)
def rag_parent_chunk_query_tool(
    query: str,
    intent_zh: str = "业务咨询",
    sub_chunk_top_k: int | None = None,
    parent_top_k: int | None = None,
) -> dict[str, Any]:
    """Retrieve RAG sub chunks, parent chunks, and full parent docs."""
    sub_chunks, parent_chunks, parent_docs = query_parent_docs_from_rag(
        question=query,
        intent_zh=intent_zh,
        sub_chunk_top_k=sub_chunk_top_k,
        parent_top_k=parent_top_k,
    )
    return {
        "tool": "rag_parent_chunk_query",
        "ok": True,
        "error": "",
        "sub_chunks": sub_chunks,
        "parent_chunks": parent_chunks,
        "parent_docs": parent_docs,
    }


@tool(
    "knowledge_lookup",
    description="知识上下文取证工具。从已检索的 domain_docs 中抽取证据片段；仅用于补充背景，不能替代具体字段值查询。",
)
def knowledge_lookup_tool(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Extract evidence snippets from already retrieved docs."""
    snippets = [
        str(dict(item).get("content") or dict(item).get("text") or "").strip()
        for item in list(docs or [])[:2]
        if str(dict(item).get("content") or dict(item).get("text") or "").strip()
    ]
    return {
        "tool": "knowledge_lookup",
        "ok": bool(snippets),
        "error": "" if snippets else "knowledge empty",
        "evidence": snippets,
        "effective_info": {"summary": snippets[0] if snippets else "", "keywords": []},
    }


RAG_TOOLS = [
    rag_parent_chunk_query_tool,
    knowledge_lookup_tool,
]
