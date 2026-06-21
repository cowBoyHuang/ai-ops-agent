"""Flow 级 query rewrite 节点。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm.llm import chat_with_llm

_LOGGER = logging.getLogger(__name__)

_TRACE_ID_PATTERN = re.compile(
    r"(?:[a-z]+[_-]slugger[_a-z0-9\.\-]+|flight_supply_open_api_[a-z0-9_.\-]+)(?=$|[^A-Za-z0-9_\.\-])",
    re.IGNORECASE,
)
_TRACE_KEY_PATTERN = re.compile(r"\btrace[_-]?id\b\s*[:=：]?\s*([A-Za-z0-9_.:\-]{4,128})", re.IGNORECASE)
_ORDER_TOKEN_PATTERN = re.compile(r"\b(?:xep|sid|fod|hpv)[A-Za-z0-9]{6,}\b", re.IGNORECASE)
_ORDER_KEY_PATTERN = re.compile(
    r"(?:\border[_-]?(?:id|no)\b|订单号|订单id|订单ID|子单号)\s*[:：=]?\s*([A-Za-z0-9_.:\-]{4,128})",
    re.IGNORECASE,
)
_API_NAME_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*_api_[A-Za-z0-9._-]+)\b")

_PROMPT_TEMPLATE = """# Role
你是一个日志与链路追踪系统的查询改写专家。你的任务是将用户的自然语言问题转换为适合 RAG 检索的结构化查询。

# Input
用户原始问题: {{user_query}}
最近历史对话（可为空）:
{{history_context}}

# Rules
1. 【实体提取】必须完整保留所有 TraceID、错误码、IP、时间戳、API名称等技术标识符，禁止截断或修改。
2. 【历史继承】若当前问题缺少 traceId/orderNo，必须先从“最近历史对话”提取并继承；不可凭空编造。
3. 【意图标准化】将口语化表述映射为标准排查动作。
   示例：
   - "为什么失败" → "生单失败原因分析"
   - "bizErrorCode是什么" → "bizErrorCode字段值查询"
4. 【上下文补全】若问题中缺少关键维度（如环境、服务名），根据命名规范自动推断并补充。
   示例：
   - "flight_supply_open_api" → "机票供给开放服务-生产环境"
5. 【多意图拆分】若包含多个独立问题，用 ||| 分隔输出多个子查询。
6. 【禁止回答】你只负责改写查询，不要尝试回答问题本身。

# Output Format (Strict JSON)
{
  "rewritten_queries": ["子查询1", "子查询2"],
  "extracted_entities": {
    "trace_id": "",
    "order_id": "",
    "api_name": "",
    "error_code_field": ""
  },
  "inferred_context": ""
}"""


def _pick_original_question(context: dict[str, Any]) -> str:
    for key in ("message", "query", "content"):
        value = context.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _clip(value: Any, max_len: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _round_tools_context_lines(tools_context: dict[str, Any]) -> list[str]:
    if not isinstance(tools_context, dict):
        return []
    rows: list[str] = []
    raw_text = json.dumps(tools_context, ensure_ascii=False, default=str)
    trace_id = _extract_trace_id(raw_text)
    order_id = _extract_order_id(raw_text)
    if trace_id or order_id:
        rows.append(f"历史继承上下文：trace_id={trace_id} order_id={order_id}".strip())

    resolved = dict(tools_context.get("required_answer_resolved") or {})
    if resolved:
        rows.append(f"历史字段结果：{_clip(json.dumps(resolved, ensure_ascii=False), 500)}")

    last_tool = dict(tools_context.get("last_tool") or {})
    if last_tool:
        tool_name = str(last_tool.get("tool_name") or "").strip()
        summary = _clip(last_tool.get("result_summary"), 320)
        conclusion = str(last_tool.get("conclusion") or "").strip()
        if tool_name or summary or conclusion:
            rows.append(
                f"历史工具结果：tool={tool_name} summary={summary} conclusion={_clip(conclusion, 180)}".strip()
            )

    conclusion = str(tools_context.get("conclusion") or "").strip()
    if conclusion:
        rows.append(f"历史结论：{_clip(conclusion, 400)}")
    return rows


def _collect_recent_history_lines(context: dict[str, Any], *, limit: int = 3) -> list[str]:
    rows: list[str] = []
    conversation_rows = [str(item).strip() for item in list(context.get("conversation_context") or []) if str(item).strip()]
    if conversation_rows:
        rows.extend(conversation_rows[-max(1, limit) :])

    message_context = context.get("message_context")
    rounds: list[Any] = []
    if hasattr(message_context, "rounds"):
        rounds = list(getattr(message_context, "rounds") or [])
    elif isinstance(message_context, dict):
        rounds = list(message_context.get("rounds") or [])

    for item in rounds[-max(1, limit) :]:
        row = dict(item or {}) if isinstance(item, dict) else {
            "message": getattr(item, "message", ""),
            "aiResponse": getattr(item, "aiResponse", ""),
            "toolsContext": getattr(item, "toolsContext", {}),
        }
        user_text = str(row.get("message") or "").strip()
        ai_text = str(row.get("aiResponse") or "").strip()
        tools_context = dict(row.get("toolsContext") or {})
        if user_text:
            rows.append(f"用户：{user_text}")
        if ai_text:
            rows.append(f"助手：{ai_text}")
        rows.extend(_round_tools_context_lines(tools_context))

    compact: list[str] = []
    seen: set[str] = set()
    for text in rows[-max(1, limit * 2) :]:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        compact.append(key)
    return compact


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


def _extract_trace_id(text: str) -> str:
    match = _TRACE_ID_PATTERN.search(text)
    if match:
        return str(match.group(0) or "").strip()
    key_match = _TRACE_KEY_PATTERN.search(text)
    if key_match:
        return str(key_match.group(1) or "").strip()
    return ""


def _extract_order_id(text: str) -> str:
    match = _ORDER_TOKEN_PATTERN.search(text)
    if match:
        return str(match.group(0) or "").strip()
    key_match = _ORDER_KEY_PATTERN.search(text)
    if key_match:
        return str(key_match.group(1) or "").strip()
    return ""


def _extract_ids(question: str, history_lines: list[str]) -> tuple[str, str]:
    trace_id = _extract_trace_id(question)
    order_id = _extract_order_id(question)
    if trace_id and order_id:
        return trace_id, order_id
    for text in history_lines:
        if not trace_id:
            trace_id = _extract_trace_id(text)
        if not order_id:
            order_id = _extract_order_id(text)
        if trace_id and order_id:
            break
    return trace_id, order_id


def _extract_error_code_field(question: str) -> str:
    lowered = str(question or "").lower()
    if "bizerrorcode" in lowered:
        return "bizErrorCode"
    if "suberrorcode" in lowered:
        return "subErrorCode"
    if "refsuberrorcode" in lowered:
        return "refSubErrorCode"
    if "errorcode" in lowered or "错误码" in question:
        return "errorCode"
    return ""


def _infer_context(question: str, api_name: str) -> str:
    lowered = f"{question} {api_name}".lower()
    if "flight_supply_open_api" in lowered:
        return "机票供给开放服务-生产环境"
    return ""


def _fallback_rewrite(question: str) -> dict[str, Any]:
    trace_id = _extract_trace_id(question)
    order_id = _extract_order_id(question)
    api_match = _API_NAME_PATTERN.search(question)
    rewritten = question if question else ""
    return {
        "rewritten_queries": [rewritten] if rewritten else [],
        "extracted_entities": {
            "trace_id": trace_id,
            "order_id": order_id,
            "api_name": str(api_match.group(1) if api_match else "").strip(),
            "error_code_field": _extract_error_code_field(question),
        },
        "inferred_context": _infer_context(question, str(api_match.group(1) if api_match else "").strip()),
        "source": "fallback",
    }


def _normalize_rewritten_queries(raw: Any, question: str) -> list[str]:
    if isinstance(raw, list):
        rows = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        rows = [part.strip() for part in raw.split("|||") if part.strip()]
    else:
        rows = []
    return rows or ([question] if question else [])


def _normalize_result(parsed: dict[str, Any], question: str, history_lines: list[str]) -> dict[str, Any]:
    extracted = dict(parsed.get("extracted_entities") or {})
    fallback = _fallback_rewrite(question)
    fallback_trace_id, fallback_order_id = _extract_ids(question, history_lines)
    rewritten_queries = _normalize_rewritten_queries(parsed.get("rewritten_queries"), question)
    api_name = str(extracted.get("api_name") or fallback["extracted_entities"]["api_name"]).strip()
    trace_id = str(extracted.get("trace_id") or parsed.get("trace_id") or fallback_trace_id or fallback["extracted_entities"]["trace_id"]).strip()
    order_id = str(extracted.get("order_id") or parsed.get("order_id") or fallback_order_id or fallback["extracted_entities"].get("order_id") or "").strip()
    error_code_field = str(extracted.get("error_code_field") or fallback["extracted_entities"]["error_code_field"]).strip()
    inferred_context = str(parsed.get("inferred_context") or "").strip() or _infer_context(question, api_name)
    normalized_query = " ||| ".join(rewritten_queries).strip()
    return {
        "rewritten_queries": rewritten_queries,
        "normalized_query": normalized_query,
        "keywords": [item for item in rewritten_queries[:6]],
        "trace_id": trace_id,
        "order_id": order_id,
        "time_window": {},
        "extracted_entities": {
            "trace_id": trace_id,
            "order_id": order_id,
            "api_name": api_name,
            "error_code_field": error_code_field,
        },
        "inferred_context": inferred_context,
        "source": "remote_llm",
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload)
    question = _pick_original_question(context)
    if not question:
        return context

    history_lines = _collect_recent_history_lines(context)
    history_context = "\n".join(history_lines) if history_lines else "无"
    prompt = (
        _PROMPT_TEMPLATE.replace("{{user_query}}", question).replace("{{history_context}}", history_context)
    )
    raw = chat_with_llm(question=prompt, system_prompt="")
    parsed = _parse_json_object(raw)
    if isinstance(parsed, dict):
        query_rewrite = _normalize_result(parsed, question, history_lines)
    else:
        query_rewrite = _fallback_rewrite(question)
        fallback_trace_id, fallback_order_id = _extract_ids(question, history_lines)
        if fallback_trace_id:
            query_rewrite["extracted_entities"]["trace_id"] = fallback_trace_id
        if fallback_order_id:
            query_rewrite["extracted_entities"]["order_id"] = fallback_order_id
        query_rewrite["normalized_query"] = " ||| ".join(list(query_rewrite.get("rewritten_queries") or []))
        query_rewrite["keywords"] = list(query_rewrite.get("rewritten_queries") or [])[:6]
        query_rewrite["trace_id"] = str(dict(query_rewrite.get("extracted_entities") or {}).get("trace_id") or "")
        query_rewrite["order_id"] = str(dict(query_rewrite.get("extracted_entities") or {}).get("order_id") or "")
        query_rewrite["time_window"] = {}
        _LOGGER.info("flow.query_rewrite.parse_failed raw_len=%d", len(str(raw or "")))

    context["query_rewrite"] = query_rewrite
    context["query_rewrite_trace"] = {
        "model": "remote_llm",
        "input_question": question,
        "history_preview": history_lines,
        "prompt": prompt,
        "raw_output": raw,
        "parse_ok": isinstance(parsed, dict),
    }
    _LOGGER.info(
        "flow.query_rewrite.done source=%s rewritten_count=%d trace_id=%s api_name=%s",
        str(query_rewrite.get("source") or ""),
        len(list(query_rewrite.get("rewritten_queries") or [])),
        str(dict(query_rewrite.get("extracted_entities") or {}).get("trace_id") or ""),
        str(dict(query_rewrite.get("extracted_entities") or {}).get("api_name") or ""),
    )
    return context
