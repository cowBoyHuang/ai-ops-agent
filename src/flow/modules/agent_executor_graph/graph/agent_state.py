"""AgentState 统一状态结构定义。

该对象是整个排障图的共享上下文：
- 每个节点读取自己关心的字段。
- 每个节点只更新自己负责的字段。
- 下游节点基于这些字段做路由和结果输出。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

IntentType = Literal["OPS_ANALYSIS", "GENERAL_QA", "UNKNOWN"]
AnalysisStatus = Literal["SUCCESS", "NEED_RETRY", "NEED_REPLAN", "FAIL"]


class AgentState(TypedDict, total=False):
    # ===== 上游通用字段（来自主 flow）=====
    # 用户本轮输入文本（已经过 request_ingest 标准化）。
    message: str
    # 兼容旧接口的 query 字段。
    query: str
    # 会话 ID（snake_case）。
    chat_id: str
    # 会话 ID（camelCase，兼容外部接口）。
    chatId: str
    # 用户 ID（snake_case）。
    user_id: str
    # 用户 ID（camelCase，兼容外部接口）。
    userId: str
    # 上游短路标记：True 表示不再进入执行图。
    pipeline_stop: bool
    # 当前错误文案。
    error: str
    # 当前错误码。
    error_code: str
    # 节点执行有效性标记，主 flow 用于统一校验。
    result: bool

    # ===== 用户输入与上下文 =====
    # 用户原始问题（可能来自 message/query）。
    question: str
    # 归一化后的问题（去除多余空白等）。
    normalized_question: str
    # 对话历史摘要（最近几轮文本）。
    conversation_context: list[str]

    # ===== 意图识别 =====
    # 当前问题类型：运维排障、通用问答、未知。
    intent_type: IntentType

    # ===== 结构化上下文 =====
    # 节点共享上下文容器，放提取结果和中间信息。
    structured_context: dict[str, Any]
    # 从问题中提取的订单号。
    order_id: str
    # 从问题中提取的 traceId。
    trace_id: str
    # 从问题中提取的 requestId。
    request_id: str

    # ===== RAG 检索结果 =====
    # 检索文档列表（BM25/向量/知识库）。
    rag_docs: list[dict[str, Any]]
    # 对应文档分数列表。
    rag_scores: list[float]

    # ===== 规划结果 =====
    # 执行计划步骤列表。
    plan_steps: list[str]
    # 当前正在执行的步骤下标。
    current_step_index: int

    # ===== 工具执行 =====
    # 本轮选择的工具名。
    tool_name: str
    # 本轮工具参数。
    tool_params: dict[str, Any]
    # 本轮工具返回结果。
    tool_result: dict[str, Any]
    # 工具执行历史（用于审计和重试判断）。
    tool_history: list[dict[str, Any]]

    # ===== 工具调用控制 =====
    # 已调用次数。
    tool_call_count: int
    # 允许最大调用次数，防止工具死循环。
    max_tool_calls: int

    # ===== 证据融合 =====
    # 合并后的证据结构，包含日志、知识、上下文。
    merged_evidence: dict[str, Any]

    # ===== 分析结果 =====
    # LLM 原始结构化输出。
    analysis: dict[str, Any]
    # 根因文本。
    root_cause: str
    # 置信度（0~1）。
    confidence: float
    # 处理建议文本。
    solution: str

    # ===== 结果状态 =====
    # 验证节点输出：SUCCESS / NEED_RETRY / NEED_REPLAN / FAIL。
    analysis_status: AnalysisStatus

    # ===== Retry 控制 =====
    # 当前重试次数。
    retry_count: int
    # 最大重试次数（新字段）。
    max_retry: int
    # 最大重试次数（兼容旧字段）。
    max_retries: int

    # ===== Replan 控制 =====
    # 当前重规划次数。
    replan_count: int
    # 最大重规划次数。
    max_replan: int

    # ===== 最终输出 =====
    # 最终回复文本。
    final_answer: str

    # ===== 运行路由/调试字段 =====
    # 当前路由目标节点名。
    route: str
    # 任务状态：running/finished/degraded 等。
    status: str
    # 对外返回的响应体。
    response: dict[str, Any]
    # True 表示 planner 需要重置步骤下标。
    planner_reset: bool
    # 测试开关：首轮工具调用模拟超时。
    simulate_tool_timeout_once: bool
    # 测试内部标记：是否已经触发过一次模拟超时。
    _simulate_tool_timeout_used: bool
