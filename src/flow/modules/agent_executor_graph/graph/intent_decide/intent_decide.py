"""意图识别节点。

业务职责：
- 基于上游构建的 question 完成单次意图识别。
- 识别失败时构建历史提示词并交由图条件边触发节点级重试。
- 不在方法内部执行循环重试。
"""

from __future__ import annotations

from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import recognize_intent

_INTENT_MAP = {
    "系统逻辑咨询": "SYSTEM_LOGIC_CONSULT",
    "线上问题咨询": "OPS_ANALYSIS",
    "订单信息查询": "ORDER_INFO_QUERY",
    "未知意图": "UNKNOWN_INTENT",
}
_SCORE_INTENT_LABELS = ("系统逻辑咨询", "线上问题咨询", "订单信息查询")
_UNKNOWN_SCORE_THRESHOLD = 0.5


# 方法注释（业务）:
# - 业务：将任意输入安全转换为整数。
# - 入参：`value`(Any)=待转换值；`default`(int)=转换失败时默认值。
# - 出参：`int`=转换结果或默认值。
# - 逻辑：尝试 `int(value)`，异常时回退默认值。
def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# 方法注释（业务）:
# - 业务：将任意输入安全转换为浮点数。
# - 入参：`value`(Any)=待转换值；`default`(float)=转换失败时默认值。
# - 出参：`float`=转换结果或默认值。
# - 逻辑：尝试 `float(value)`，异常时回退默认值。
def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 方法注释（业务）:
# - 业务：把大模型返回的中文意图标签映射为内部统一枚举值。
# - 入参：`best_intent`(str)=大模型识别出的最佳意图标签。
# - 出参：`str`=内部意图类型；未命中时返回 `UNKNOWN`。
# - 逻辑：使用 `_INTENT_MAP` 做字典映射并兜底默认值。
def _to_internal_intent(best_intent: str) -> str:
    return _INTENT_MAP.get(str(best_intent or "").strip(), "UNKNOWN")


# 方法注释（业务）:
# - 业务：从意图打分结果中选择 `final_score` 最高的意图。
# - 入参：`intent_result`(dict[str, Any])=大模型意图识别原始结果。
# - 出参：`tuple[str, float]`=(最高分意图标签, 最高分)；异常结构返回 `("", 0.0)`。
# - 逻辑：遍历 `_SCORE_INTENT_LABELS`，读取 `scores[label].final_score` 并比较最大值。
def _pick_best_intent_by_score(intent_result: dict[str, Any]) -> tuple[str, float]:
    scores = intent_result.get("scores")
    if not isinstance(scores, dict):
        return "", 0.0

    best_label = ""
    best_score = -1.0
    for label in _SCORE_INTENT_LABELS:
        row = scores.get(label)
        if not isinstance(row, dict):
            continue
        score = _to_float(row.get("final_score"), 0.0)
        if score > best_score:
            best_score = score
            best_label = label
    if best_score < 0:
        return "", 0.0
    return best_label, best_score


# 方法注释（业务）:
# - 业务：获取当前轮用于意图识别的问题文本。
# - 入参：`state`(dict[str, Any])=子图状态字典。
# - 出参：`str`=最终用于识别的用户问题文本。
# - 逻辑：优先读取 `question`，未命中时回退到 `structured_context.question`。
def _pick_message_for_intent(state: dict[str, Any]) -> str:
    question = str(state.get("question") or "").strip()
    if question:
        return question
    return str(dict(state.get("structured_context") or {}).get("question") or "").strip()


# 方法注释（业务）:
# - 业务：构建“识别失败后下一次重试”的提示词文本。
# - 入参：`question`(str)=用户问题；`history_rows`(list[dict])=历史失败结果。
# - 出参：`str`=可直接传给大模型的用户提示词。
# - 逻辑：拼接最近最多两次失败记录，并附加重新分析指导语。
def _build_intent_history_prompt(question: str, history_rows: list[dict[str, Any]]) -> str:
    rows = list(history_rows)[-2:]
    lines: list[str] = [
        "请分析用户问题并为每个意图打分。",
        f'用户问题: "{question}"',
        "",
    ]
    if rows:
        lines.append("注意：之前的识别尝试失败了，以下是之前的判断结果：")
        for idx, row in enumerate(rows, 1):
            confidence = _to_float(row.get("confidence"), 0.0)
            lines.append(f"第{idx}次尝试: {str(row.get('best_intent') or '未知')} (置信度: {confidence:.2f})")
            lines.append(f"理由: {str(row.get('reasoning') or '').strip()}")
        lines.extend(
            [
                "",
                "请重新分析，考虑以下可能性：",
                "1. 问题可能表述不清，尝试从不同角度理解",
                "2. 可能是多意图问题，选择最主要的一个",
                "3. 如果实在无法判断，选择最保守的意图（订单信息查询）",
            ]
        )
    return "\n".join(lines)


# 方法注释（业务）:
# - 业务：执行单次意图识别并确定子图下一跳。
# - 入参：`payload`(dict[str, Any])=子图输入状态，包含 question/structured_context 等字段。
# - 出参：`dict[str, Any]`=写回 `question`、`intent_type`、`intent_recognition`、`route` 等字段后的状态。
# - 逻辑：
#   1) 调用大模型做单次识别（优先使用 state.intent_history_prompt）；
#   2) 识别失败且未超最大重试时，仅写入重试信息并将 route 指向 intent_decide；
#   3) 识别成功或重试耗尽时按原路由规则继续流转。
def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    structured_context = dict(state.get("structured_context") or {})
    question = _pick_message_for_intent(state)
    max_retry = _to_int(state.get("max_retries"), 2)
    retry_count = _to_int(state.get("intent_retry_count"), 0)
    history = state.get("conversation_context") or structured_context.get("recent_messages") or []
    conversation_context = [str(item) for item in history if str(item).strip()]
    intent_history_prompt = str(state.get("intent_history_prompt") or "").strip()

    # 调用大模型
    intent_result = recognize_intent(
        question,
        intent_history_prompt=intent_history_prompt if intent_history_prompt else None,
    )
    best_intent, best_score = _pick_best_intent_by_score(intent_result)
    if not best_intent:
        best_intent = str(intent_result.get("best_intent") or "")
        best_score = _to_float(intent_result.get("confidence"), 0.0)

    state["conversation_context"] = conversation_context
    state["intent_recognition"] = {
        **intent_result,
        "best_intent": best_intent,
        "confidence": best_score,
        "intent_retry_count": retry_count,
    }

    failed = best_score <= _UNKNOWN_SCORE_THRESHOLD
    if failed and retry_count < max_retry:
        retry_rows = list(state.get("intent_retry_results") or [])
        retry_rows.append(
            {
                "best_intent": best_intent or "未知",
                "confidence": best_score,
                "reasoning": str(intent_result.get("reasoning") or ""),
            }
        )
        retry_count += 1
        state["intent_retry_count"] = retry_count
        state["intent_retry_results"] = retry_rows
        state["intent_history_prompt"] = _build_intent_history_prompt(question, retry_rows)
        state["intent_type"] = "UNKNOWN_INTENT"
        state["route"] = "intent_decide"
        state["structured_context"] = {
            **structured_context,
            "question": str(state.get("question") or structured_context.get("question") or ""),
            "conversation_context": conversation_context,
            "order_id": str(structured_context.get("order_id") or ""),
            "request_id": str(structured_context.get("request_id") or ""),
            "intent_recognition": state["intent_recognition"],
        }
        return dict(state)

    if failed:
        best_intent = "未知意图"
        state["intent_recognition"] = {
            **state["intent_recognition"],
            "best_intent": best_intent,
            "confidence": best_score,
            "reasoning": "最高意图分未超过 0.5，判定为未知意图",
            "intent_retry_count": retry_count,
        }

    intent_type = _to_internal_intent(best_intent)
    if not question:
        intent_type = "UNKNOWN"

    state["intent_type"] = intent_type
    state["intent_history_prompt"] = ""
    state["intent_retry_results"] = []
    state["intent_retry_count"] = retry_count
    state["structured_context"] = {
        **structured_context,
        "question": str(state.get("question") or structured_context.get("question") or ""),
        "conversation_context": conversation_context,
        "order_id": str(structured_context.get("order_id") or ""),
        "request_id": str(structured_context.get("request_id") or ""),
        "intent_recognition": state["intent_recognition"],
    }

    if intent_type in {"OPS_ANALYSIS", "SYSTEM_LOGIC_CONSULT"}:
        state["route"] = "rag_retrieve"
    elif intent_type == "UNKNOWN_INTENT":
        state["route"] = "fallback"
    else:
        state["route"] = "fixed_flow_execute"
    return dict(state)
