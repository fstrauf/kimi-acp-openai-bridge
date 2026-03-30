"""Pydantic models for OpenAI API compatibility."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """OpenAI chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolFunction(BaseModel):
    """OpenAI tool function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class Tool(BaseModel):
    """OpenAI tool definition."""

    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCall(BaseModel):
    """OpenAI tool call from assistant."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolCallFunction(BaseModel):
    """Function call details."""

    name: str
    arguments: str  # JSON string


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[Message]
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] = "auto"
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=1)  # We only support n=1
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    user: str | None = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Completion choice."""

    index: int
    message: Message
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None
    logprobs: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None


class ChoiceDelta(BaseModel):
    """Delta content in streaming response."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamingChoice(BaseModel):
    """Streaming completion choice."""

    index: int
    delta: ChoiceDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI chat completion streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoice]
    system_fingerprint: str | None = None


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI error response format."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    kimi_available: bool
    version: str
