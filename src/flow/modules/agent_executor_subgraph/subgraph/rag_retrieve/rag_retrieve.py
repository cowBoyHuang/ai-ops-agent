"""Dual retrieval module (BM25 + vector, with local fallback)."""

from __future__ import annotations

from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload)
    question = str((state.get("structured_context") or {}).get("question") or "")

    # 占位：模拟 BM25 + 向量双路召回结果
    docs = [
        {"id": "bm25-1", "score": 0.82, "source": "bm25", "text": f"日志关键字命中: {question[:32]}"},
        {"id": "vec-1", "score": 0.79, "source": "vector", "text": f"语义相似片段: {question[:32]}"},
    ]
    state["rag_docs"] = docs
    state["route"] = "evidence_merge"
    return state

