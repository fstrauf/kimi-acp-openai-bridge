"""Configuration for Kimi ACP Bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


def _get_config_dir() -> Path:
    """Get the configuration directory."""
    if os.name == "nt":  # Windows
        base_dir = Path(os.environ.get("APPDATA", "~"))
    else:  # Unix/Linux/macOS
        base_dir = Path.home() / ".config"
    return base_dir / "kimi-acp-bridge"


def _get_config_file() -> Path | None:
    """Get the configuration file path if it exists."""
    config_file = _get_config_dir() / "config.yaml"
    if config_file.exists():
        return config_file
    return None


@dataclass
class BridgeConfig:
    """Configuration for the Kimi ACP Bridge.

    Attributes:
        kimi_binary: Path to the kimi CLI binary
        kimi_args: Arguments to pass to kimi CLI
        host: Server bind address
        port: Server port
        session_mode: Session persistence mode (ephemeral)
        session_timeout: Session timeout in seconds
        enable_tools: Whether to enable tool calling
        enable_streaming: Whether to enable streaming responses
        auto_approve_tools: Whether to auto-approve tool calls
        log_level: Logging level
        log_acp_messages: Whether to log all ACP messages
    """

    # Kimi CLI settings
    kimi_binary: str = "kimi"
    kimi_args: list[str] = field(default_factory=lambda: ["acp", "--stdio"])

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    # Session settings
    session_mode: Literal["ephemeral", "persistent"] = "ephemeral"
    session_timeout: int = 300  # seconds

    # Feature flags
    enable_tools: bool = True
    enable_streaming: bool = True
    auto_approve_tools: bool = True

    # Logging
    log_level: str = "INFO"
    log_acp_messages: bool = False

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Create configuration from environment variables."""
        return cls(
            kimi_binary=os.getenv("KIMI_BINARY", "kimi"),
            kimi_args=os.getenv("KIMI_ARGS", "acp --stdio").split(),
            host=os.getenv("KIMI_BRIDGE_HOST", "127.0.0.1"),
            port=int(os.getenv("KIMI_BRIDGE_PORT", "8080")),
            session_mode=os.getenv("KIMI_BRIDGE_SESSION_MODE", "ephemeral"),  # type: ignore[arg-type]
            session_timeout=int(os.getenv("KIMI_BRIDGE_SESSION_TIMEOUT", "300")),
            enable_tools=os.getenv("KIMI_BRIDGE_ENABLE_TOOLS", "true").lower() == "true",
            enable_streaming=os.getenv("KIMI_BRIDGE_ENABLE_STREAMING", "true").lower() == "true",
            auto_approve_tools=os.getenv("KIMI_BRIDGE_AUTO_APPROVE", "true").lower() == "true",
            log_level=os.getenv("KIMI_BRIDGE_LOG_LEVEL", "INFO"),
            log_acp_messages=os.getenv("KIMI_BRIDGE_LOG_ACP", "false").lower() == "true",
        )

    @classmethod
    def from_file(cls, path: Path) -> BridgeConfig:
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "server" in data:
            config.host = data["server"].get("host", config.host)
            config.port = data["server"].get("port", config.port)

        if "kimi" in data:
            config.kimi_binary = data["kimi"].get("binary", config.kimi_binary)
            config.kimi_args = data["kimi"].get("args", config.kimi_args)

        if "session" in data:
            config.session_mode = data["session"].get("mode", config.session_mode)
            config.session_timeout = data["session"].get("timeout", config.session_timeout)

        if "features" in data:
            config.enable_tools = data["features"].get("enable_tools", config.enable_tools)
            config.enable_streaming = data["features"].get(
                "enable_streaming", config.enable_streaming
            )
            config.auto_approve_tools = data["features"].get(
                "auto_approve_tools", config.auto_approve_tools
            )

        if "logging" in data:
            config.log_level = data["logging"].get("level", config.log_level)
            config.log_acp_messages = data["logging"].get(
                "log_acp_messages", config.log_acp_messages
            )

        return config

    @classmethod
    def load(cls, config_path: Path | None = None) -> BridgeConfig:
        """Load configuration from file and/or environment.

        Priority: config_path arg > default config file > env vars > defaults
        """
        # Start with defaults
        config = cls()

        # Load from config file if exists
        if config_path:
            config = cls.from_file(config_path)
        elif default_config := _get_config_file():
            config = cls.from_file(default_config)

        # Override with environment variables
        env_config = cls.from_env()

        # Only override if env vars are explicitly set
        if os.getenv("KIMI_BINARY"):
            config.kimi_binary = env_config.kimi_binary
        if os.getenv("KIMI_ARGS"):
            config.kimi_args = env_config.kimi_args
        if os.getenv("KIMI_BRIDGE_HOST"):
            config.host = env_config.host
        if os.getenv("KIMI_BRIDGE_PORT"):
            config.port = env_config.port
        if os.getenv("KIMI_BRIDGE_SESSION_MODE"):
            config.session_mode = env_config.session_mode
        if os.getenv("KIMI_BRIDGE_SESSION_TIMEOUT"):
            config.session_timeout = env_config.session_timeout
        if os.getenv("KIMI_BRIDGE_ENABLE_TOOLS"):
            config.enable_tools = env_config.enable_tools
        if os.getenv("KIMI_BRIDGE_ENABLE_STREAMING"):
            config.enable_streaming = env_config.enable_streaming
        if os.getenv("KIMI_BRIDGE_AUTO_APPROVE"):
            config.auto_approve_tools = env_config.auto_approve_tools
        if os.getenv("KIMI_BRIDGE_LOG_LEVEL"):
            config.log_level = env_config.log_level
        if os.getenv("KIMI_BRIDGE_LOG_ACP"):
            config.log_acp_messages = env_config.log_acp_messages

        return config
