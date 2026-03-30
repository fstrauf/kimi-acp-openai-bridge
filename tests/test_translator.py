"""Tests for protocol translation."""

import pytest

from kimi_acp_bridge.models import Message, Tool, ToolFunction
from kimi_acp_bridge.translator import (
    generate_completion_id,
    generate_tool_call_id,
    openai_to_acp_messages,
    openai_to_acp_tools,
    estimate_token_count,
)


class TestGenerateIds:
    """Test ID generation functions."""

    def test_generate_completion_id_format(self):
        """Test completion ID format."""
        id1 = generate_completion_id()
        assert id1.startswith("chatcmpl-")
        assert len(id1) == 33  # "chatcmpl-" + 24 hex chars

    def test_generate_completion_id_unique(self):
        """Test that IDs are unique."""
        ids = {generate_completion_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_tool_call_id_format(self):
        """Test tool call ID format."""
        id1 = generate_tool_call_id()
        assert id1.startswith("call_")
        assert len(id1) == 29  # "call_" + 24 hex chars


class TestOpenAIToACP:
    """Test OpenAI to ACP translation."""

    def test_simple_message_conversion(self):
        """Test converting simple user message."""
        messages = [
            Message(role="user", content="Hello!"),
        ]

        preamble, acp_messages = openai_to_acp_messages(messages)

        assert preamble is None
        assert len(acp_messages) == 1
        assert acp_messages[0]["role"] == "user"
        assert acp_messages[0]["content"] == "Hello!"

    def test_system_message_as_preamble(self):
        """Test that system message becomes preamble."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ]

        preamble, acp_messages = openai_to_acp_messages(messages)

        assert preamble == "You are a helpful assistant."
        assert len(acp_messages) == 1
        assert acp_messages[0]["role"] == "user"

    def test_assistant_message_conversion(self):
        """Test converting assistant message."""
        messages = [
            Message(role="assistant", content="Hello! How can I help?"),
        ]

        preamble, acp_messages = openai_to_acp_messages(messages)

        assert acp_messages[0]["role"] == "assistant"
        assert acp_messages[0]["content"] == "Hello! How can I help?"

    def test_tool_result_conversion(self):
        """Test converting tool result message."""
        messages = [
            Message(role="tool", content="File contents here", tool_call_id="call_123"),
        ]

        preamble, acp_messages = openai_to_acp_messages(messages)

        assert acp_messages[0]["role"] == "tool_result"
        assert acp_messages[0]["content"] == "File contents here"
        assert acp_messages[0]["tool_call_id"] == "call_123"

    def test_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        from kimi_acp_bridge.models import ToolCall, ToolCallFunction

        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        function=ToolCallFunction(
                            name="read_file",
                            arguments='{"path": "/tmp/test.txt"}',
                        ),
                    )
                ],
            ),
        ]

        preamble, acp_messages = openai_to_acp_messages(messages)

        assert acp_messages[0]["role"] == "assistant"
        assert acp_messages[0]["tool_calls"][0]["id"] == "call_123"
        assert acp_messages[0]["tool_calls"][0]["function"]["name"] == "read_file"


class TestToolConversion:
    """Test tool conversion."""

    def test_openai_to_acp_tools(self):
        """Test converting OpenAI tools to ACP format."""
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="read_file",
                    description="Read a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                ),
            ),
        ]

        acp_tools = openai_to_acp_tools(tools)

        assert len(acp_tools) == 1
        assert acp_tools[0]["name"] == "read_file"
        assert acp_tools[0]["description"] == "Read a file"
        assert "inputSchema" in acp_tools[0]

    def test_empty_tools(self):
        """Test that empty tools returns None."""
        assert openai_to_acp_tools([]) is None
        assert openai_to_acp_tools(None) is None


class TestTokenEstimation:
    """Test token estimation."""

    def test_estimate_token_count(self):
        """Test rough token estimation."""
        # Very rough estimate: ~4 chars per token
        text = "a" * 100
        tokens = estimate_token_count(text)
        assert tokens == 25  # 100 // 4

    def test_empty_text(self):
        """Test that empty text returns at least 1."""
        tokens = estimate_token_count("")
        assert tokens == 1

    def test_short_text(self):
        """Test short text."""
        tokens = estimate_token_count("hi")
        assert tokens == 1


class TestACPToOpenAI:
    """Test ACP to OpenAI translation."""

    def test_message_delta_conversion(self):
        """Test converting message delta event."""
        from kimi_acp_bridge.translator import acp_to_openai_chunk

        event = {
            "type": "message.delta",
            "delta": "Hello",
        }

        chunk = acp_to_openai_chunk(event, "kimi-k2.5", "chatcmpl-123", 1234567890)

        assert chunk is not None
        assert chunk.model == "kimi-k2.5"
        assert chunk.id == "chatcmpl-123"
        assert chunk.choices[0].delta.content == "Hello"

    def test_message_start_conversion(self):
        """Test converting message start event."""
        from kimi_acp_bridge.translator import acp_to_openai_chunk

        event = {
            "type": "message.start",
        }

        chunk = acp_to_openai_chunk(event, "kimi-k2.5", "chatcmpl-123", 1234567890)

        assert chunk is not None
        assert chunk.choices[0].delta.role == "assistant"

    def test_skip_control_events(self):
        """Test that control events return None."""
        from kimi_acp_bridge.translator import acp_to_openai_chunk

        control_events = [
            {"type": "session.created"},
            {"type": "session.updated"},
            {"type": "done"},
        ]

        for event in control_events:
            chunk = acp_to_openai_chunk(event, "kimi-k2.5", "chatcmpl-123", 1234567890)
            assert chunk is None
