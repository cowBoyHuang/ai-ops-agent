"""Plan step 类型定义。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class PlanStep(TypedDict, total=False):
    action_type: Literal["tool_call", "merge_evidence", "react_subtask"]
    tool_name: Optional[str]          # 仅当 action_type=="tool_call" 时有效
    params: Dict[str, Any]            # 仅当 action_type=="tool_call" 时有效
    subtask: str
    hypothesis: str
    success_criteria: str
    suggested_tools: List[str]
