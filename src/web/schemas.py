from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(default="", description="User question text.")
    chatId: str = Field(default="", description="Optional chat id from caller.")
    userId: str = Field(default="", description="Optional user id from caller.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Pass-through fields.")


class AnalyzeResponse(BaseModel):
    chatId: str
    status: str
    message: str
