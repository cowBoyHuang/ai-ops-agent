"""RAG 检索节点。

业务职责：
- 根据原始问题执行 BM25（ES）与向量检索（Qdrant）。
- 使用 RRF 对多路召回结果融合排序，并做去重筛选。
- 输出 rag_docs/rag_scores，供 planner 与分析节点使用。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from log import QueryType, query_external_logs
from qdrant import QdrantStore

_LOGGER = logging.getLogger(__name__)
_DEFAULT_ES_LOOKBACK_MINUTES = 15
_MAX_BM25_DOCS = 30
_MAX_BM25_DIAGNOSE_DOCS = 5
_MAX_RAG_DOCS = 30
_MAX_RAG_PARENT_DOCS = 12
_ERROR_CODE_PATTERN = re.compile(r"(?:error[_\s-]?code|错误码)\s*[:=]\s*([A-Za-z0-9_-]{2,64})", re.IGNORECASE)
_EXCEPTION_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error))\b")

_INTENT_TO_CN = {
    "SYSTEM_LOGIC_CONSULT": "系统逻辑咨询",
    "OPS_ANALYSIS": "线上问题咨询",
    "ORDER_INFO_QUERY": "订单信息查询",
    "UNKNOWN_INTENT": "未知意图",
    "UNKNOWN": "未知意图",
}


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
# - 入参：`dicts`(list[dict[str, Any]])=候选字典列表；`keys`(tuple[str, ...])=按优先级查找的键名。
# - 出参：`Any`=命中的第一个非空值；未命中返回 `None`。
# - 方法逻辑：按“字典顺序 + 键顺序”扫描，过滤 `None` 与空白字符串。
def _pick_value_from_dicts(dicts: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    for row in dicts:
        for key in keys:
            if key not in row:
                continue
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


# 方法注释（业务）:
# - 入参：`value`(Any)=时间字段原始值（datetime/字符串/None）。
# - 出参：`dt.datetime | str | None`=标准化时间值。
# - 方法逻辑：`datetime` 原样返回；字符串尝试按 ISO 解析，失败则保留原字符串；空值返回 `None`。
def _normalize_datetime(value: Any) -> dt.datetime | str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(iso_candidate)
    except ValueError:
        return text


# 方法注释（业务）:
# - 入参：`state`(dict[str, Any])=当前 AgentState 字典。
# - 出参：`dict[str, Any] | None`=ES 检索参数（app_code/logname/begin_time/end_time）；关键参数缺失时返回 `None`。
# - 方法逻辑：从 `structured_context -> context -> state` 提取参数，并对时间字段做标准化及默认窗口补齐。
def _extract_es_params(state: dict[str, Any]) -> dict[str, Any] | None:
    structured_context = dict(state.get("structured_context") or {})
    raw_context = dict(state.get("context") or {})
    # 参数读取优先级：结构化上下文 > 原始上下文 > 顶层 state。
    candidates = [structured_context, raw_context, state]

    app_code = str(
        _pick_value_from_dicts(candidates, ("app_code", "appCode", "app_core", "appCore")) or ""
    ).strip()
    logname = str(_pick_value_from_dicts(candidates, ("logname", "log_name")) or "").strip()
    if not app_code or not logname:
        return None

    begin_raw = _pick_value_from_dicts(candidates, ("begin_time", "beginTime", "start_time", "startTime"))
    end_raw = _pick_value_from_dicts(candidates, ("end_time", "endTime"))
    begin_time = _normalize_datetime(begin_raw)
    end_time = _normalize_datetime(end_raw)

    # 未传时间范围时，兜底为“当前时间往前 15 分钟”窗口。
    if end_time is None:
        end_time = dt.datetime.now()
    if begin_time is None:
        begin_time = dt.datetime.now() - dt.timedelta(minutes=_DEFAULT_ES_LOOKBACK_MINUTES)

    return {
        "app_code": app_code,
        "logname": logname,
        "begin_time": begin_time,
        "end_time": end_time,
    }


# 方法注释（业务）:
# - 入参：`question`(str)=原始问题；`state`(dict[str, Any])=当前 AgentState。
# - 出参：`list[dict[str, Any]]`=BM25 候选文档列表（统一成 id/score/source/text 结构）。
# - 方法逻辑：提取 ES 参数后调用现有日志查询接口，按分数降序整理并限制最大返回条数；异常降级为空列表。
def _search_es_bm25(question: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    if not question:
        return []
    params = _extract_es_params(state)
    if params is None:
        _LOGGER.info("rag_retrieve bm25 skipped: missing app_code/logname")
        return []

    try:
        rows = query_external_logs(
            app_code=str(params["app_code"]),
            logname=str(params["logname"]),
            begin_time=params["begin_time"],
            end_time=params["end_time"],
            content=question,
            type=QueryType.MATCH.value,
        )
    except Exception as err:  # pragma: no cover - 外部依赖异常统一降级
        _LOGGER.warning("bm25 log query failed: %s", err)
        return []

    # BM25 原分数由 ES 返回，先按分数降序再做统一结构映射。
    ranked = sorted(
        list(rows or []),
        key=lambda item: float(getattr(item, "score", 0.0) or 0.0),
        reverse=True,
    )
    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked[:_MAX_BM25_DOCS], start=1):
        text = str(getattr(row, "content", "") or "").strip()
        if not text:
            continue
        try:
            score = float(getattr(row, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        docs.append(
            {
                "id": f"bm25-{idx}",
                "score": score,
                "source": "bm25",
                "text": text,
            }
        )
    return docs


# 方法注释（业务）:
# - 入参：`query`(str)=RAG 检索查询词；`intent_zh`(str)=中文意图标签。
# - 出参：`list[dict[str, Any]]`=RAG chunk 候选列表（含 payload.parent_id）。
# - 方法逻辑：将“查询词+意图”拼接后调用 Qdrant，抽取文本与 payload 元信息，统一输出结构。
def _search_qdrant_rag(query: str, intent_zh: str) -> list[dict[str, Any]]:
    if not query:
        return []
    # 将意图拼入查询词，提升向量检索在多意图场景下的区分度。
    query_text = f"{query}\n意图：{intent_zh}".strip()
    try:
        rows = QdrantStore().search(query=query_text, limit=_MAX_RAG_DOCS)
    except Exception as err:  # pragma: no cover - 外部依赖异常统一降级
        _LOGGER.warning("qdrant search failed: %s", err)
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
    return docs


# 方法注释（业务）:
# - 入参：`bm25_docs`(list[dict[str, Any]])=日志 BM25 命中结果。
# - 出参：`dict[str, Any]`=错误信息摘要（error_code/exception/keywords）。
# - 方法逻辑：从高分日志中提取错误码与异常类名，并生成增强检索关键词。
def _extract_error_info(bm25_docs: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(item.get("text") or "") for item in bm25_docs[:_MAX_BM25_DIAGNOSE_DOCS]]
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
    return {
        "error_code": error_code,
        "exception": exception,
        "keywords": keywords,
    }


# 方法注释（业务）:
# - 入参：`question`(str)=原始问题；`error_info`(dict[str, Any])=错误信息摘要。
# - 出参：`str`=用于第二阶段文档检索的查询词。
# - 方法逻辑：若日志已提取到错误码/异常，则将关键词拼接到问题后进行增强检索；否则回退原问题。
def _build_stage2_rag_query(question: str, error_info: dict[str, Any]) -> str:
    keywords = str(error_info.get("keywords") or "").strip()
    if keywords:
        return f"{question} {keywords}".strip()
    return str(question or "").strip()


# 方法注释（业务）:
# - 入参：`rag_chunk_docs`(list[dict[str, Any]])=RAG chunk 命中列表（含 payload.parent_id）。
# - 出参：`list[dict[str, Any]]`=按父文档去重后的文档列表（每个 parent 仅保留最高分 chunk）。
# - 方法逻辑：以 payload.parent_id 为主键聚合并保留最高分子块，最终按分数降序输出。
def _dedup_rag_by_parent_id(rag_chunk_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return docs[:_MAX_RAG_PARENT_DOCS]


# 方法注释（业务）:
# - 入参：`payload`(dict[str, Any])=AgentState，至少包含 question/intent 等上下文字段。
# - 出参：`dict[str, Any]`=写回 `rag_docs/rag_scores/route` 后的状态。
# - 方法逻辑：先做 BM25 日志诊断并提取错误信息，再基于增强查询词做 RAG 检索，最后按 parent_id 去重输出。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    """执行检索步骤。

    入参：
    - payload: AgentState，需包含 question。

    返参：
    - AgentState: 写入 rag_docs/rag_scores，并路由到 planner。
    """
    state: AgentState = dict(payload)
    run_started = time.perf_counter()
    question = _pick_question(state)
    intent_zh = _pick_intent_zh(state)

    # 分阶段计时：便于区分 BM25 诊断与 RAG 解决两个环节的耗时瓶颈。
    bm25_started = time.perf_counter()
    bm25_docs = _search_es_bm25(question, state)
    bm25_cost_ms = (time.perf_counter() - bm25_started) * 1000
    _LOGGER.info("rag_retrieve bm25 done: docs=%d cost_ms=%.2f", len(bm25_docs), bm25_cost_ms)

    diagnose_started = time.perf_counter()
    error_info = _extract_error_info(bm25_docs)
    rag_query = _build_stage2_rag_query(question, error_info)
    diagnose_cost_ms = (time.perf_counter() - diagnose_started) * 1000
    _LOGGER.info(
        "rag_retrieve diagnose done: error_code=%s exception=%s cost_ms=%.2f",
        str(error_info.get("error_code") or ""),
        str(error_info.get("exception") or ""),
        diagnose_cost_ms,
    )

    rag_started = time.perf_counter()
    rag_chunk_docs = _search_qdrant_rag(rag_query, intent_zh)
    rag_docs = _dedup_rag_by_parent_id(rag_chunk_docs)
    rag_cost_ms = (time.perf_counter() - rag_started) * 1000
    _LOGGER.info(
        "rag_retrieve qdrant done: chunk_docs=%d parent_docs=%d cost_ms=%.2f intent=%s",
        len(rag_chunk_docs),
        len(rag_docs),
        rag_cost_ms,
        intent_zh,
    )

    # 输出以“解决文档”为主；诊断阶段结果通过 structured_context 透传便于调试。
    state["rag_docs"] = rag_docs
    state["rag_scores"] = [float(item.get("score") or 0.0) for item in rag_docs]
    state["route"] = "planner"
    state["structured_context"] = {
        **dict(state.get("structured_context") or {}),
        "rag_diagnosis": {
            "bm25_top_logs": [str(item.get("text") or "") for item in bm25_docs[:_MAX_BM25_DIAGNOSE_DOCS]],
            "error_info": error_info,
            "rag_query": rag_query,
        },
    }
    total_cost_ms = (time.perf_counter() - run_started) * 1000
    _LOGGER.info(
        "rag_retrieve completed: output_docs=%d total_cost_ms=%.2f question_len=%d",
        len(rag_docs),
        total_cost_ms,
        len(question),
    )
    return dict(state)
