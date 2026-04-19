"""Agent 执行图状态定义。"""

from __future__ import annotations

from typing import TypedDict, Dict, Any, List

from flow.modules.agent_executor_graph.plan_step import PlanStep


class AgentState(TypedDict):
    # ==============================
    # 1. 输入与意图理解
    # ==============================
    question: str                          # 用户原始问题
    intent_type: str                       # 意图分类结果，如 "ORDER_INQUIRY", "LOGISTICS_QUERY"
    structured_context: Dict[str, Any]     # 从问题中提取的结构化实体（如 order_id, user_id）

    # ==============================
    # 2. 规划与执行控制
    # ==============================
    current_plan_id: int                   # 当前轮次编号，如 1、2、3
    plan_steps: List[PlanStep]             # 当前执行计划步骤
    status: str                            # 整体流程状态： "pending", "running", "completed", "failed"

    # ==============================
    # 3. 工具调用历史（按轮次组织）
    # ==============================
    tool_history: Dict[
        str,                               # 轮次编号，如 "1", "2", ...
        Dict[
            str,                           # 工具名称，如 "query_order_detail"
            Dict[str, Any]                 # 内容：{"params": {...}, "result": {...}, "status": "success/failed"}
        ]
    ]

    # ==============================
    # 4. 分析与证据融合
    # ==============================
    merged_evidence: Dict[str, Any]        # 所有工具返回结果的聚合视图（可选，便于分析器使用）
    analysis: str                          # LLM 基于 evidence 生成的自然语言分析结论
    analysis_status: str                   # 分析状态："not_started", "in_progress", "completed", "failed"

    # ==============================
    # 5. 输出与重试控制
    # ==============================
    final_answer: str                      # 最终返回给用户的答案
    retry_count: int                       # 当前工具/步骤的重试次数
    replan_count: int                      # 因计划失败触发的重新规划次数
