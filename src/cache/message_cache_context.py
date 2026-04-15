"""Message cache context object used by Redis message_cache_context_{chat_id}."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageCacheContext:
    """Cache payload for one chat context snapshot."""

    UserQuestion: str = ""
    agentAnswer: str = ""
    toolsContext: dict[str, Any] = field(default_factory=dict)
    UserQuestionEmbedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "UserQuestion": str(self.UserQuestion or ""),
            "agentAnswer": str(self.agentAnswer or ""),
            "toolsContext": dict(self.toolsContext or {}),
            "UserQuestionEmbedding": [float(item) for item in list(self.UserQuestionEmbedding or [])],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageCacheContext":
        return cls(
            UserQuestion=str(data.get("UserQuestion") or ""),
            agentAnswer=str(data.get("agentAnswer") or ""),
            toolsContext=dict(data.get("toolsContext") or {}),
            UserQuestionEmbedding=[float(item) for item in list(data.get("UserQuestionEmbedding") or [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "MessageCacheContext | None":
        try:
            payload = json.loads(str(text or ""))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return cls.from_dict(payload)
