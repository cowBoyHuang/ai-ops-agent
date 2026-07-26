"""RAG 检索节点。

业务职责：
- 根据原始问题执行向量检索（Qdrant）。
- 提供两个能力方法：子 chunk TopK 查询、父 chunk TopK + 父文档全文回查。
- 输出 rag_sub_chunk_docs/rag_docs/rag_parent_docs/rag_scores，供 planner 与执行节点使用。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from flow.modules.agent_executor_graph.agent_state import AgentState
from qdrant import QdrantStore

_LOGGER = logging.getLogger(__name__)
_MAX_RAG_DOCS = 30
_MAX_RAG_PARENT_DOCS = 12
_DEFAULT_PARENT_DOC_TOP_N = 6
_DEFAULT_SUB_CHUNK_TOP_K = 12
_MAX_LOG_QUESTION_LEN = 120
_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

_INTENT_TO_CN = {
    "SYSTEM_LOGIC_CONSULT": "业务咨询",
    "OPS_ANALYSIS": "线上问题排查",
}

_INTENT_CN_NORMALIZE = {
    "线上问题咨询": "线上问题排查",
    "订单信息查询": "业务咨询",
    "未知意图": "业务咨询",
}


def _positive_int_env(env_key: str, default: int) -> int:
    raw = str(os.getenv(env_key, default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def _sub_chunk_top_k() -> int:
    return min(_positive_int_env("RAG_RETRIEVE_SUB_CHUNK_TOP_K", _DEFAULT_SUB_CHUNK_TOP_K), _MAX_RAG_DOCS)


def _parent_doc_top_n() -> int:
    """读取父文档回查 TopN，环境变量无效时回退默认值。"""
    return min(_positive_int_env("RAG_RETRIEVE_PARENT_DOC_TOP_N", _DEFAULT_PARENT_DOC_TOP_N), _MAX_RAG_PARENT_DOCS)


# 方法注释（业务）:
# - 入参：`text`(str)=待输出到日志的文本；`max_len`(int)=最大保留长度。
# - 出参：`str`=裁剪后的安全日志文本。
# - 方法逻辑：避免把超长问题全文打到日志，降低噪声并减少敏感内容暴露面。
def _clip_for_log(text: str, max_len: int = _MAX_LOG_QUESTION_LEN) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len]}..."


# 方法注释（业务）:
# - 入参：`path`(str)=本地文档绝对路径。
# - 出参：`str`=文件文本内容；读取失败或二进制文件返回空字符串。
# - 方法逻辑：优先 utf-8，回退 gb18030；对异常与二进制内容统一降级为空文本。
def _read_local_doc(path: str) -> str:
    text_path = Path(str(path or "").strip())
    if not text_path.is_file():
        return ""
    if text_path.suffix.lower() == ".docx":
        return _read_docx_text_safely(text_path)
    try:
        raw = text_path.read_bytes()
    except Exception:  # pragma: no cover - 文件系统异常统一降级
        _LOGGER.warning("rag_retrieve 读取父文档失败: path=%s", text_path)
        return ""
    if b"\x00" in raw:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").strip()


def _read_docx_text_safely(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            root = ET.fromstring(zf.read("word/document.xml"))
    except Exception:
        return ""

    rows: list[str] = []
    for para in root.findall(".//w:p", _DOCX_NS):
        text = "".join(node.text or "" for node in para.findall(".//w:t", _DOCX_NS)).strip()
        if text:
            rows.append(" ".join(text.split()))
    return "\n".join(rows).strip()


# 方法注释（业务）:
# - 入参：`state`(dict[str, Any])=当前 AgentState 字典。
# - 出参：`str`=检索使用的问题文本；无有效问题时返回空字符串。
# - 方法逻辑：优先读取 `state.question`，再回退 `context/structured_context` 的问题相关字段。
def _pick_question(state: dict[str, Any]) -> str:
    question = str(state.get("question") or "").strip()
    if question:
        return question
    context = dict(state.get("context") or {})
    structured_context = dict(state.get("structured_context") or {})
    for key in ("question", "message", "query", "content"):
        value = context.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = structured_context.get("question")
    if value is not None and str(value).strip():
        return str(value).strip()
    return ""


# 方法注释（业务）:
# - 入参：`state`(dict[str, Any])=当前 AgentState 字典，可能包含 intent_type/intent_recognition。
# - 出参：`str`=中文意图标签（用于拼接 RAG 查询词）。
# - 方法逻辑：优先使用 `intent_recognition.best_intent`，否则由 `intent_type` 映射为中文标签并兜底“业务咨询”。
def resolve_intent_label_for_rag(state: dict[str, Any]) -> str:
    recognition = dict(state.get("intent_recognition") or {})
    best_intent = str(recognition.get("best_intent") or "").strip()
    if best_intent:
        return _INTENT_CN_NORMALIZE.get(best_intent, best_intent)
    intent_type = str(state.get("intent_type") or "SYSTEM_LOGIC_CONSULT").strip() or "SYSTEM_LOGIC_CONSULT"
    return _INTENT_TO_CN.get(intent_type, "业务咨询")


# 方法注释（业务）:
# - 入参：`query`(str)=RAG 检索查询词；`intent_zh`(str)=中文意图标签。
# - 出参：`list[dict[str, Any]]`=RAG chunk 候选列表（含 payload.parent_id）。
# - 方法逻辑：将“查询词+意图”拼接后调用 Qdrant，抽取文本与 payload 元信息，统一输出结构。
def _search_qdrant_rag(query: str, intent_zh: str, *, limit: int = _MAX_RAG_DOCS) -> list[dict[str, Any]]:
    if not query:
        _LOGGER.info("rag_retrieve 跳过 Qdrant: query 为空")
        return []
    # 将意图拼入查询词，提升向量检索在多意图场景下的区分度。
    query_text = f"{query}\n意图：{intent_zh}".strip()
    _LOGGER.info(
        "rag_retrieve 开始 Qdrant 检索: intent=%s query=%s",
        intent_zh,
        _clip_for_log(query_text),
    )
    store = QdrantStore()
    capped_limit = max(1, min(limit, _MAX_RAG_DOCS))
    use_hybrid = hasattr(store, "search_hybrid")
    try:
        if use_hybrid:
            domain_rows = store.search_hybrid(
                query=query_text,
                limit=capped_limit,
                collection_name=store.config.domain_collection_name,
            )
            case_rows = store.search_hybrid(
                query=query_text,
                limit=capped_limit,
                collection_name=store.config.case_collection_name,
            )
        else:
            domain_rows = store.search(
                query=query_text,
                limit=capped_limit,
                collection_name=store.config.domain_collection_name,
            )
            case_rows = store.search(
                query=query_text,
                limit=capped_limit,
                collection_name=store.config.case_collection_name,
            )
        rows: list[dict[str, Any]] = []
        for row in list(domain_rows or []):
            payload = dict(row.get("payload") or {})
            payload.setdefault("knowledge_type", "domain")
            rows.append({**row, "payload": payload})
        for row in list(case_rows or []):
            payload = dict(row.get("payload") or {})
            payload.setdefault("knowledge_type", "case")
            rows.append({**row, "payload": payload})
        # 兼容老数据：分库都没命中时，回退 legacy collection。
        if not rows:
            if use_hybrid:
                legacy_rows = store.search_hybrid(
                    query=query_text,
                    limit=capped_limit,
                    collection_name=store.config.collection_name,
                )
            else:
                legacy_rows = store.search(
                    query=query_text,
                    limit=capped_limit,
                    collection_name=store.config.collection_name,
                )
            for row in list(legacy_rows or []):
                payload = dict(row.get("payload") or {})
                payload.setdefault("knowledge_type", "legacy")
                rows.append({**row, "payload": payload})
        rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        rows = rows[:capped_limit]
    except Exception as err:  # pragma: no cover - 外部依赖异常统一降级
        _LOGGER.warning("rag_retrieve Qdrant 查询失败: %s", err)
        return []

    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(list(rows or [])[: max(1, min(limit, _MAX_RAG_DOCS))], start=1):
        payload = dict(row.get("payload") or {}) if isinstance(row, dict) else {}
        # 兼容不同 payload 字段命名，优先 text，其次 content。
        text = str(
            payload.get("text")
            or payload.get("content")
            or (row.get("text") if isinstance(row, dict) else "")
            or ""
        ).strip()
        if not text:
            continue
        raw_id = row.get("id") if isinstance(row, dict) else None
        doc_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else f"rag-{idx}"
        try:
            score = float((row.get("score") if isinstance(row, dict) else 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        docs.append(
            {
                "id": doc_id,
                "score": score,
                "source": "rag",
                "text": text,
                "payload": payload,
            }
        )
    _LOGGER.info("rag_retrieve Qdrant 检索完成: chunk 命中=%d", len(docs))
    return docs


# 方法注释（业务）:
# - 入参：`rag_chunk_docs`(list[dict[str, Any]])=RAG chunk 命中列表（含 payload.parent_id）。
# - 出参：`list[dict[str, Any]]`=按父文档去重后的文档列表（每个 parent 仅保留最高分 chunk）。
# - 方法逻辑：以 payload.parent_id 为主键聚合并保留最高分子块，最终按分数降序输出。
def _dedup_rag_by_parent_id(rag_chunk_docs: list[dict[str, Any]], *, top_k: int = _MAX_RAG_PARENT_DOCS) -> list[dict[str, Any]]:
    _LOGGER.info("rag_retrieve 开始按 parent_id 去重: 输入 chunk=%d", len(rag_chunk_docs))
    parent_best: dict[str, dict[str, Any]] = {}
    for row in rag_chunk_docs:
        payload = dict(row.get("payload") or {})
        parent_id = str(payload.get("parent_id") or "").strip()
        if not parent_id:
            parent_id = str(row.get("id") or "").strip()
        if not parent_id:
            continue

        score = float(row.get("score") or 0.0)
        previous = parent_best.get(parent_id)
        if previous is not None and float(previous.get("score") or 0.0) >= score:
            continue

        parent_best[parent_id] = {
            "id": parent_id,
            "score": score,
            "source": "rag_parent",
            "text": str(row.get("text") or ""),
            "parent_id": parent_id,
            "path": str(payload.get("path") or ""),
            "chunk_id": str(row.get("id") or ""),
            "chunk_score": score,
            "knowledge_type": str(payload.get("knowledge_type") or ""),
        }

    docs = list(parent_best.values())
    docs.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    result = docs[: max(1, min(top_k, _MAX_RAG_PARENT_DOCS))]
    _LOGGER.info("rag_retrieve parent 去重完成: 输出 parent=%d", len(result))
    return result


# 方法注释（业务）:
# - 入参：`rag_docs`(list[dict[str, Any]])=按 parent_id 去重后的 RAG 文档。
# - 出参：`list[dict[str, Any]]`=父文档完整内容列表（parent_id/path/content）。
# - 方法逻辑：使用 chunk payload 中的 path 直接读取父文档全文并挂上检索分数透传下游。
def _load_parent_documents(rag_docs: list[dict[str, Any]], *, top_n: int | None = None) -> list[dict[str, Any]]:
    if not rag_docs:
        _LOGGER.info("rag_retrieve 跳过父文档回查: rag_docs 为空")
        return []

    top_n = max(1, min(int(top_n or _parent_doc_top_n()), _MAX_RAG_PARENT_DOCS))
    _LOGGER.info("rag_retrieve 开始加载父文档: 候选=%d top_n=%d", len(rag_docs), top_n)
    rows: list[dict[str, Any]] = []
    for item in rag_docs[:top_n]:
        parent_id = str(item.get("parent_id") or item.get("id") or "").strip()
        if not parent_id:
            continue
        doc_path = str(item.get("path") or "").strip()
        if not doc_path:
            continue
        full_content = _read_local_doc(doc_path)
        if not full_content:
            continue
        rows.append(
            {
                "parent_id": parent_id,
                "path": doc_path,
                "content": full_content,
                "score": float(item.get("score") or 0.0),
                "chunk_id": str(item.get("chunk_id") or ""),
                "knowledge_type": str(item.get("knowledge_type") or ""),
            }
        )
    _LOGGER.info("rag_retrieve 父文档加载完成: full_docs=%d", len(rows))
    return rows


def _query_sub_chunks_from_rag(*, question: str, intent_zh: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """查询子 chunk TopK（内部方法）。"""
    chunk_top_k = max(1, min(int(top_k or _sub_chunk_top_k()), _MAX_RAG_DOCS))
    return _search_qdrant_rag(query=question, intent_zh=intent_zh, limit=chunk_top_k)


def _query_parent_chunks_from_sub_chunks(
    *,
    sub_chunk_docs: list[dict[str, Any]],
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """根据子 chunk 排序结果回查父 chunk TopK，并读取父文档完整内容（内部方法）。"""
    parent_top_k = max(1, min(int(top_k or _parent_doc_top_n()), _MAX_RAG_PARENT_DOCS))
    parent_chunk_docs = _dedup_rag_by_parent_id(sub_chunk_docs, top_k=parent_top_k)
    parent_full_docs = _load_parent_documents(parent_chunk_docs, top_n=parent_top_k)
    return parent_chunk_docs, parent_full_docs


def query_parent_docs_from_rag(
    *,
    question: str,
    intent_zh: str,
    sub_chunk_top_k: int | None = None,
    parent_top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """对外方法：按问题直接查询父文档链路（内部自动完成子 chunk 检索与排序）。"""
    sub_chunks = _query_sub_chunks_from_rag(question=question, intent_zh=intent_zh, top_k=sub_chunk_top_k)
    parent_chunks, parent_docs = _query_parent_chunks_from_sub_chunks(sub_chunk_docs=sub_chunks, top_k=parent_top_k)
    return sub_chunks, parent_chunks, parent_docs


# 方法注释（业务）:
# - 入参：`payload`(dict[str, Any])=AgentState，至少包含 question/intent 等上下文字段。
# - 出参：`dict[str, Any]`=写回 `rag_sub_chunk_docs/rag_docs/rag_parent_docs/rag_scores/route` 后的状态。
# - 方法逻辑：先查询子 chunk TopK，再查询父 chunk TopK 并回查父文档全文。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行检索步骤。

    入参：
    - payload: AgentState，需包含 question。

    返参：
    - AgentState: 写入 rag_docs/rag_parent_docs/rag_scores，并路由到 planner。
    """
    state: AgentState = dict(payload)
    run_started = time.perf_counter()
    question = _pick_question(state)
    intent_zh = resolve_intent_label_for_rag(state)
    _LOGGER.info(
        "rag_retrieve 开始执行: chat_id=%s intent=%s question=%s",
        str(state.get("chat_id") or ""),
        intent_zh,
        _clip_for_log(question),
    )

    sub_chunk_top_k = _sub_chunk_top_k()
    parent_chunk_top_k = _parent_doc_top_n()
    # 双阶段：先查子 chunk TopK，再按子 chunk 排序结果回查父文档 TopK 与完整文档。
    rag_started = time.perf_counter()
    rag_sub_chunk_docs, rag_docs, parent_docs = query_parent_docs_from_rag(
        question=question,
        intent_zh=intent_zh,
        sub_chunk_top_k=sub_chunk_top_k,
        parent_top_k=parent_chunk_top_k,
    )
    rag_cost_ms = (time.perf_counter() - rag_started) * 1000
    _LOGGER.info(
        "rag_retrieve Qdrant 阶段完成: sub_chunk_docs=%d parent_chunk_docs=%d full_docs=%d cost_ms=%.2f intent=%s",
        len(rag_sub_chunk_docs),
        len(rag_docs),
        len(parent_docs),
        rag_cost_ms,
        intent_zh,
    )

    # 输出给下游：子 chunk、父 chunk（最佳子块代表）与父文档全文。
    state["rag_sub_chunk_docs"] = rag_sub_chunk_docs
    state["rag_docs"] = rag_docs
    state["rag_parent_docs"] = parent_docs
    state["rag_scores"] = [float(item.get("score") or 0.0) for item in rag_docs]
    state["route"] = "planner"
    total_cost_ms = (time.perf_counter() - run_started) * 1000
    _LOGGER.info(
        "rag_retrieve 执行完成: output_docs=%d total_cost_ms=%.2f question_len=%d",
        len(rag_docs),
        total_cost_ms,
        len(question),
    )
    return dict(state)
