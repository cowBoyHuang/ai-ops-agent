"""Agent 执行图状态定义。"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from flow.modules.agent_executor_graph.plan_step import PlanStep


class AgentState(TypedDict, total=False):
    # ==============================
    # 1. 输入与意图理解
    # ==============================
    question: str
    intent_type: str
    structured_context: Dict[str, Any]
    conversation_context: List[str]
    intent_recognition: Dict[str, Any]
    intent_history_prompt: str
    intent_retry_results: List[Dict[str, Any]]
    intent_retry_count: int
    context: Dict[str, Any]

    chat_id: str
    user_id: str
    error: str
    error_code: str
    result: str

    # ==============================
    # 2. 规划与执行控制
    # ==============================
    current_plan_id: int
    plan_steps: List[PlanStep]
    current_plan: List[PlanStep]
    original_plan: List[PlanStep]
    status: str
    route: str
    current_step_index: int
    needs_adjustment: bool
    adjustment_type: str
    proposed_changes: Dict[str, Any]
    pending_insertions: List[PlanStep]
    adjustment_history: List[Dict[str, Any]]
    adjustment_applied: Dict[str, Any]

    # ==============================
    # 3. 工具调用历史（按轮次组织）
    # ==============================
    tool_history: Any
    tool_name: str
    tool_params: Dict[str, Any]
    tool_result: Dict[str, Any]
    tool_call_count: int
    max_tool_calls: int

    # ==============================
    # 4. 分析与证据融合
    # ==============================
    merged_evidence: Dict[str, Any]
    evidence: Dict[str, Any]
    execution_history: Dict[str, Any]
    current_step_result: Dict[str, Any]
    newly_discovered_clues: List[str]
    intermediate_results: Dict[str, Any]
    extracted_keywords: List[str]
    analysis: Any
    analysis_status: str
    root_cause: str
    confidence: float
    solution: str
    rag_docs: List[Dict[str, Any]]
    rag_parent_docs: List[Dict[str, Any]]
    rag_scores: List[float]

    # ==============================
    # 5. 输出与重试控制
    # ==============================
    final_answer: str
    retry_count: int
    replan_count: int
    max_retries: int
    max_replan: int
    response: Dict[str, Any]
