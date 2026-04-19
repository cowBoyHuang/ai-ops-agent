"""Plan step 类型定义。"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict


class PlanStep(TypedDict):
    action_type: Literal["tool_call", "merge_evidence"]
    tool_name: Optional[str]          # 仅当 action_type=="tool_call" 时有效
    params: Dict[str, Any]            # 仅当 action_type=="tool_call" 时有效
