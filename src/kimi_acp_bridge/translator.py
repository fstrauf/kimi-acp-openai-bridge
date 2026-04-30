"""Protocol translation between OpenAI and ACP formats."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kimi_acp_bridge.models import ResponseFormat

from kimi_acp_bridge.models import (
    ChatCompletionChunk,
    ChoiceDelta,
    Message,
    StreamingChoice,
    Tool,
    ToolCall,
    ToolCallFunction,
)

# OpenAI to ACP role mapping
OPENAI_TO_ACP_ROLES = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool_result",
}

# ACP to OpenAI role mapping
ACP_TO_OPENAI_ROLES = {
    "assistant": "assistant",
    "user": "user",
    "system": "system",
    "tool_call": "assistant",
    "tool_result": "tool",
}


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


def inject_response_format_to_prompt(
    preamble: str | None,
    response_format: "ResponseFormat | None",
) -> str | None:
    """Inject JSON schema constraints into the system prompt.

    This enables structured output (Extractor<T>) support for Rig and similar frameworks.
    Kimi ACP doesn't natively support response_format, so we inject constraints via prompt.
    """
    if response_format is None or response_format.type == "text":
        return preamble

    constraints_parts: list[str] = []

    if response_format.type == "json_object":
        constraints_parts.append(
            "You must respond with valid JSON only. "
            "Do not include any prose, explanations, markdown formatting, or code fences. "
            "Return only the raw JSON object."
        )

    elif response_format.type == "json_schema" and response_format.json_schema:
        schema = response_format.json_schema
        schema_json = json.dumps(schema, indent=2)

        constraints_parts.append(
            "You must respond with valid JSON that conforms to the following schema:\n"
            f"```json\n{schema_json}\n```\n\n"
            "Requirements:\n"
            "- Return ONLY the JSON object, no markdown, no code fences, no explanations\n"
            "- Ensure all required fields are present\n"
            "- Use the exact field names and types specified in the schema\n"
            "- Do not include any additional fields not in the schema"
        )

        # If schema has a description, include it
        if schema.get("description"):
            constraints_parts.insert(
                0,
                f"Response description: {schema['description']}"
            )

    constraints = "\n\n".join(constraints_parts)

    if preamble:
        return f"{preamble}\n\n{constraints}"
    else:
        return constraints


def openai_to_acp_messages(
    messages: list[Message],
    response_format: "ResponseFormat | None" = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI messages to ACP format.

    Args:
        messages: OpenAI format messages
        response_format: Optional response format for structured output

    Returns:
        Tuple of (system_preamble, acp_messages)
    """
    preamble_parts: list[str] = []
    acp_messages = []

    for msg in messages:
        if msg.role == "system":
            # Collect all system messages as preamble
            if msg.content:
                preamble_parts.append(msg.content)
        elif msg.role == "tool":
            # Tool result message
            acp_messages.append(
                {
                    "role": "tool_result",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )
        elif msg.role == "assistant" and msg.tool_calls:
            # Assistant message with tool calls
            acp_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [  # type: ignore[dict-item]
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
        else:
            # Regular user or assistant message
            acp_messages.append(
                {
                    "role": OPENAI_TO_ACP_ROLES.get(msg.role, msg.role),
                    "content": msg.content or "",
                }
            )

    preamble = "\n\n".join(preamble_parts) if preamble_parts else None

    # Inject response format constraints into preamble
    if response_format is not None:
        preamble = inject_response_format_to_prompt(preamble, response_format)

    return preamble, acp_messages


def openai_to_acp_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tools to ACP format."""
    if not tools:
        return None

    acp_tools = []
    for tool in tools:
        acp_tools.append(
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "inputSchema": tool.function.parameters,
            }
        )

    return acp_tools


def acp_to_openai_chunk(
    event: dict[str, Any],
    model: str,
    completion_id: str,
    created: int,
) -> ChatCompletionChunk | None:
    """Convert an ACP event to an OpenAI streaming chunk.

    Returns None if the event should be skipped (e.g., internal ACP messages).
    """
    event_type = event.get("type", "")

    if event_type == "message.delta":
        # Text content delta
        delta = ChoiceDelta(content=event.get("delta", ""))
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

    elif event_type == "message.start":
        # Start of assistant message
        delta = ChoiceDelta(role="assistant")
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

    elif event_type == "tool_call.start":
        # Start of tool call
        tool_call_id = event.get("tool_call", {}).get("id", generate_tool_call_id())
        tool_name = event.get("tool_call", {}).get("name", "")
        delta = ChoiceDelta(
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    function=ToolCallFunction(
                        name=tool_name,
                        arguments="",
                    ),
                )
            ]
        )
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

    elif event_type == "tool_call.delta":
        # Tool call arguments delta (streaming JSON)
        tool_call_id = event.get("tool_call", {}).get("id", "")
        arguments_delta = event.get("delta", "")
        delta = ChoiceDelta(
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    function=ToolCallFunction(
                        name="",
                        arguments=arguments_delta,
                    ),
                )
            ]
        )
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

    elif event_type == "message.complete":
        # End of message
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
        )

    elif event_type == "tool_call.complete":
        # End of tool call
        return None  # Don't send separate chunk for tool call completion

    elif event_type == "error":
        # ACP error - we'll handle this separately
        return None

    elif event_type in ("done", "session.created", "session.updated"):
        # Control events - skip in stream
        return None

    # Unknown event type - skip
    return None


def create_final_chunk(
    model: str,
    completion_id: str,
    created: int,
    finish_reason: str = "stop",
) -> ChatCompletionChunk:
    """Create the final chunk to signal completion."""
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            StreamingChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ],
    )


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count.

    This is a simple approximation. For more accuracy, a proper tokenizer
    would be needed, but for local usage this is sufficient.
    """
    # Rough estimate: ~4 characters per token on average
    return max(1, len(text) // 4)
