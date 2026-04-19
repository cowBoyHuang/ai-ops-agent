"""Message cache object used by Redis key message_cache_context_{chat_id}."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundMessageContext:
    """One conversation round cache row."""

    message: str = ""
    userMessageEmbedding: list[float] = field(default_factory=list)
    aiResponse: str = ""
    toolsContext: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self.message or ""),
            "userMessageEmbedding": [float(item) for item in list(self.userMessageEmbedding or [])],
            "aiResponse": str(self.aiResponse or ""),
            "toolsContext": dict(self.toolsContext or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundMessageContext":
        # Backward compatibility: map old flat payload to one round.
        message = str(data.get("message") or data.get("UserQuestion") or "")
        ai_response = str(data.get("aiResponse") or data.get("agentAnswer") or "")
        tools_context = dict(data.get("toolsContext") or {})
        embedding = list(data.get("userMessageEmbedding") or data.get("UserQuestionEmbedding") or [])
        return cls(
            message=message,
            userMessageEmbedding=[float(item) for item in embedding],
            aiResponse=ai_response,
            toolsContext=tools_context,
        )


@dataclass
class MessageCacheContext:
    """Top-level message_cache object."""

    summary: str = ""
    rounds: list[RoundMessageContext] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": str(self.summary or ""),
            "rounds": [item.to_dict() for item in list(self.rounds or [])],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageCacheContext":
        summary = str(data.get("summary") or "")
        rounds_raw = data.get("rounds")
        if isinstance(rounds_raw, list):
            rounds: list[RoundMessageContext] = []
            for item in rounds_raw:
                if isinstance(item, dict):
                    rounds.append(RoundMessageContext.from_dict(item))
            return cls(summary=summary, rounds=rounds)

        # Backward compatibility: legacy single-round schema.
        legacy_round = RoundMessageContext.from_dict(data)
        if legacy_round.message or legacy_round.aiResponse or legacy_round.userMessageEmbedding or legacy_round.toolsContext:
            return cls(summary=summary, rounds=[legacy_round])
        return cls(summary=summary, rounds=[])

    @classmethod
    def from_json(cls, text: str) -> "MessageCacheContext | None":
        try:
            payload = json.loads(str(text or ""))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return cls.from_dict(payload)
