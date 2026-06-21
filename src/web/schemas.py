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
    begin_time: str = Field(
        default="",
        description="Optional log begin time from caller.",
        validation_alias=AliasChoices("begin_time", "beginTime", "start_time", "startTime"),
    )
    end_time: str = Field(
        default="",
        description="Optional log end time from caller.",
        validation_alias=AliasChoices("end_time", "endTime", "finish_time", "finishTime"),
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Pass-through fields.")


class AnalyzeResponse(BaseModel):
    chatId: str
    status: str
    message: str


class ClearStorageResponse(BaseModel):
    status: str
    db_enabled: bool
    db_deleted: dict[str, int]
    redis_enabled: bool
    redis_deleted: int
    repeat_chat_fallback_cleared: int
    message_cache_fallback_cleared: int
    trace_rows_cleared: int
