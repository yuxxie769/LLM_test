from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="allow")


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody
