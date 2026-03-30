"""Kimi ACP ↔ OpenAI Bridge.

A local HTTP server that translates between the OpenAI-compatible API format
and the Agent Client Protocol (ACP) used by Kimi Code CLI.
"""

__version__ = "0.1.0"

from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Message,
    Tool,
    ToolCall,
)

__all__ = [
    "__version__",
    "BridgeConfig",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "Message",
    "Tool",
    "ToolCall",
]
