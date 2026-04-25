"""RAG 检索节点。

业务职责：
- 根据原始问题执行向量检索（Qdrant）。
- 基于首轮 RAG 结果提取错误信息，构建二阶段增强查询并再次检索。
- 输出 rag_docs/rag_parent_docs/rag_scores，供 planner 与分析节点使用。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from db import ChatDBStore
from flow.modules.agent_executor_graph.agent_state import AgentState
from qdrant import QdrantStore

_LOGGER = logging.getLogger(__name__)
_MAX_DIAGNOSE_DOCS = 5
_MAX_RAG_DOCS = 30
_MAX_RAG_PARENT_DOCS = 12
_DEFAULT_PARENT_DOC_TOP_N = 6
_MAX_LOG_QUESTION_LEN = 120
_ERROR_CODE_PATTERN = re.compile(r"(?:error[_\s-]?code|错误码)\s*[:=]\s*([A-Za-z0-9_-]{2,64})", re.IGNORECASE)
_EXCEPTION_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error))\b")
_RAG_DOC_DB_STORE: ChatDBStore | None = None

_INTENT_TO_CN = {
    "SYSTEM_LOGIC_CONSULT": "系统逻辑咨询",
    "OPS_ANALYSIS": "线上问题咨询",
    "ORDER_INFO_QUERY": "订单信息查询",
    "UNKNOWN_INTENT": "未知意图",
    "UNKNOWN": "未知意图",
}


def _parent_doc_top_n() -> int:
    """读取父文档回查 TopN，环境变量无效时回退默认值。"""
    raw = str(os.getenv("RAG_RETRIEVE_PARENT_DOC_TOP_N", _DEFAULT_PARENT_DOC_TOP_N)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PARENT_DOC_TOP_N
    if value <= 0:
        return _DEFAULT_PARENT_DOC_TOP_N
    return value


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
# - 入参：无。
# - 出参：`ChatDBStore`=用于 `rag_document` 映射回查的 DB 访问实例。
# - 方法逻辑：延迟初始化并复用单例，避免每次检索重复创建连接。
def _get_rag_doc_db_store() -> ChatDBStore:
    global _RAG_DOC_DB_STORE
    if _RAG_DOC_DB_STORE is None:
        _RAG_DOC_DB_STORE = ChatDBStore()
        _LOGGER.info("rag_retrieve 初始化 rag_document DB 访问实例")
    return _RAG_DOC_DB_STORE


# 方法注释（业务）:
# - 入参：`path`(str)=本地文档绝对路径。
# - 出参：`str`=文件文本内容；读取失败或二进制文件返回空字符串。
# - 方法逻辑：优先 utf-8，回退 gb18030；对异常与二进制内容统一降级为空文本。
def _read_local_doc(path: str) -> str:
    text_path = Path(str(path or "").strip())
    if not text_path.is_file():
        return ""
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
# - 方法逻辑：优先使用 `intent_recognition.best_intent`，否则由 `intent_type` 映射为中文标签并兜底“未知意图”。
def _pick_intent_zh(state: dict[str, Any]) -> str:
    recognition = dict(state.get("intent_recognition") or {})
    best_intent = str(recognition.get("best_intent") or "").strip()
    if best_intent:
        return best_intent
    intent_type = str(state.get("intent_type") or "UNKNOWN").strip() or "UNKNOWN"
    return _INTENT_TO_CN.get(intent_type, "未知意图")


# 方法注释（业务）:
# - 入参：`query`(str)=RAG 检索查询词；`intent_zh`(str)=中文意图标签。
# - 出参：`list[dict[str, Any]]`=RAG chunk 候选列表（含 payload.parent_id）。
# - 方法逻辑：将“查询词+意图”拼接后调用 Qdrant，抽取文本与 payload 元信息，统一输出结构。
def _search_qdrant_rag(query: str, intent_zh: str) -> list[dict[str, Any]]:
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
    try:
        rows = QdrantStore().search(query=query_text, limit=_MAX_RAG_DOCS)
    except Exception as err:  # pragma: no cover - 外部依赖异常统一降级
        _LOGGER.warning("rag_retrieve Qdrant 查询失败: %s", err)
        return []

    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(list(rows or [])[:_MAX_RAG_DOCS], start=1):
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
# - 入参：`docs`(list[dict[str, Any]])=首轮 RAG chunk 命中结果。
# - 出参：`dict[str, Any]`=错误信息摘要（error_code/exception/keywords）。
# - 方法逻辑：从高分 RAG 文档中提取错误码与异常类名，并生成增强检索关键词。
def _extract_error_info(docs: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(item.get("text") or "") for item in docs[:_MAX_DIAGNOSE_DOCS]]
    merged = "\n".join(texts)

    error_code = ""
    match_code = _ERROR_CODE_PATTERN.search(merged)
    if match_code:
        error_code = str(match_code.group(1) or "").strip()

    exception = ""
    match_exception = _EXCEPTION_PATTERN.search(merged)
    if match_exception:
        exception = str(match_exception.group(1) or "").strip()

    keywords = " ".join(part for part in [error_code, exception] if part).strip()
    result = {
        "error_code": error_code,
        "exception": exception,
        "keywords": keywords,
    }
    _LOGGER.info(
        "rag_retrieve 文档诊断提取完成: error_code=%s exception=%s",
        error_code,
        exception,
    )
    return result


# 方法注释（业务）:
# - 入参：`question`(str)=原始问题；`error_info`(dict[str, Any])=错误信息摘要。
# - 出参：`str`=用于第二阶段文档检索的查询词。
# - 方法逻辑：若日志已提取到错误码/异常，则将关键词拼接到问题后进行增强检索；否则回退原问题。
def _build_stage2_rag_query(question: str, error_info: dict[str, Any]) -> str:
    keywords = str(error_info.get("keywords") or "").strip()
    if keywords:
        _LOGGER.info("rag_retrieve 二阶段检索词增强: keywords=%s", keywords)
        return f"{question} {keywords}".strip()
    _LOGGER.info("rag_retrieve 二阶段检索词未增强: 使用原问题")
    return str(question or "").strip()


# 方法注释（业务）:
# - 入参：`rag_chunk_docs`(list[dict[str, Any]])=RAG chunk 命中列表（含 payload.parent_id）。
# - 出参：`list[dict[str, Any]]`=按父文档去重后的文档列表（每个 parent 仅保留最高分 chunk）。
# - 方法逻辑：以 payload.parent_id 为主键聚合并保留最高分子块，最终按分数降序输出。
def _dedup_rag_by_parent_id(rag_chunk_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        }

    docs = list(parent_best.values())
    docs.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    result = docs[:_MAX_RAG_PARENT_DOCS]
    _LOGGER.info("rag_retrieve parent 去重完成: 输出 parent=%d", len(result))
    return result


# 方法注释（业务）:
# - 入参：`rag_docs`(list[dict[str, Any]])=按 parent_id 去重后的 RAG 文档。
# - 出参：`list[dict[str, Any]]`=父文档完整内容列表（parent_id/path/content）。
# - 方法逻辑：取 TopN parent_id 回查 rag_document 映射，读取本地文件全文并挂上检索分数透传下游。
def _load_parent_documents(rag_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rag_docs:
        _LOGGER.info("rag_retrieve 跳过父文档回查: rag_docs 为空")
        return []
    db_store = _get_rag_doc_db_store()
    if not db_store.enabled:
        _LOGGER.info("rag_retrieve 跳过父文档回查: rag_document DB 未启用")
        return []

    top_n = _parent_doc_top_n()
    _LOGGER.info("rag_retrieve 开始回查父文档: 候选=%d top_n=%d", len(rag_docs), top_n)
    rows: list[dict[str, Any]] = []
    for item in rag_docs[:top_n]:
        parent_id = str(item.get("parent_id") or item.get("id") or "").strip()
        if not parent_id:
            continue
        try:
            mapping = db_store.get_rag_document(parent_id=parent_id)
        except Exception as err:  # pragma: no cover - 外部依赖异常统一降级
            _LOGGER.warning("rag_retrieve 父文档映射查询失败: parent_id=%s err=%s", parent_id, err)
            continue

        doc_path = str((mapping or {}).get("path") or "").strip()
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
            }
        )
    _LOGGER.info("rag_retrieve 父文档回查完成: full_docs=%d", len(rows))
    return rows


# 方法注释（业务）:
# - 入参：`payload`(dict[str, Any])=AgentState，至少包含 question/intent 等上下文字段。
# - 出参：`dict[str, Any]`=写回 `rag_docs/rag_parent_docs/rag_scores/route` 后的状态。
# - 方法逻辑：执行单阶段 RAG 检索并按 parent_id 去重，然后回查父文档全文透传下游。
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
    intent_zh = _pick_intent_zh(state)
    _LOGGER.info(
        "rag_retrieve 开始执行: chat_id=%s intent=%s question=%s",
        str(state.get("chat_id") or ""),
        intent_zh,
        _clip_for_log(question),
    )

    # 单阶段：仅做文档检索，不做错误码/异常提取与二阶段增强查询。
    rag_started = time.perf_counter()
    rag_chunk_docs = _search_qdrant_rag(question, intent_zh)
    rag_docs = _dedup_rag_by_parent_id(rag_chunk_docs)
    parent_docs = _load_parent_documents(rag_docs)
    rag_cost_ms = (time.perf_counter() - rag_started) * 1000
    _LOGGER.info(
        "rag_retrieve Qdrant 阶段完成: chunk_docs=%d parent_docs=%d full_docs=%d cost_ms=%.2f intent=%s",
        len(rag_chunk_docs),
        len(rag_docs),
        len(parent_docs),
        rag_cost_ms,
        intent_zh,
    )

    # 输出以“完整文档召回”为主，不注入额外诊断提取信息。
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
