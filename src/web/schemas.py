from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class AnalyzeRequest(BaseModel):
    question: str = Field(
        default="",
        description="User question text.",
        validation_alias=AliasChoices("question", "query", "message", "content"),
    )
    chat_id: str = Field(
        default="",
        description="Optional chat id from caller.",
        validation_alias=AliasChoices("chat_id", "chatId"),
    )
    user_id: str = Field(
        default="",
        description="Optional user id from caller.",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Pass-through fields.")


class AnalyzeResponse(BaseModel):
    chatId: str
    status: str
    message: str
