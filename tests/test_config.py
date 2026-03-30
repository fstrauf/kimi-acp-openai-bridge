"""Tests for configuration."""

from kimi_acp_bridge.config import BridgeConfig


class TestBridgeConfig:
    """Test BridgeConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BridgeConfig()

        assert config.kimi_binary == "kimi"
        assert config.kimi_args == ["acp", "--stdio"]
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.session_mode == "ephemeral"
        assert config.session_timeout == 300
        assert config.enable_tools is True
        assert config.enable_streaming is True
        assert config.auto_approve_tools is True
        assert config.log_level == "INFO"
        assert config.log_acp_messages is False

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("KIMI_BINARY", "/usr/local/bin/kimi")
        monkeypatch.setenv("KIMI_BRIDGE_HOST", "0.0.0.0")
        monkeypatch.setenv("KIMI_BRIDGE_PORT", "9000")
        monkeypatch.setenv("KIMI_BRIDGE_LOG_LEVEL", "DEBUG")

        config = BridgeConfig.from_env()

        assert config.kimi_binary == "/usr/local/bin/kimi"
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.log_level == "DEBUG"

    def test_from_file(self, tmp_path):
        """Test loading from configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
server:
  host: 0.0.0.0
  port: 9000

kimi:
  binary: /usr/bin/kimi
  args: ["acp", "--stdio", "--verbose"]

session:
  mode: ephemeral
  timeout: 600

features:
  enable_tools: false
  auto_approve_tools: false

logging:
  level: DEBUG
  log_acp_messages: true
""")

        config = BridgeConfig.from_file(config_file)

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.kimi_binary == "/usr/bin/kimi"
        assert config.kimi_args == ["acp", "--stdio", "--verbose"]
        assert config.session_timeout == 600
        assert config.enable_tools is False
        assert config.auto_approve_tools is False
        assert config.log_level == "DEBUG"
        assert config.log_acp_messages is True

    def test_load_priority(self, tmp_path, monkeypatch):
        """Test that env vars override file config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
server:
  host: 0.0.0.0
  port: 9000
""")

        monkeypatch.setenv("KIMI_BRIDGE_PORT", "7000")

        config = BridgeConfig.load(config_file)

        # From file
        assert config.host == "0.0.0.0"
        # From env (overrides file)
        assert config.port == 7000
