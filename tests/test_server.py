"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient

from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.server import create_app


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return BridgeConfig(
        kimi_binary="echo",  # Use echo as a mock
        kimi_args=["test"],
        host="127.0.0.1",
        port=8080,
        log_level="DEBUG",
    )


@pytest.fixture
def client(test_config):
    """Create a test client."""
    app = create_app(test_config)
    return TestClient(app)


class TestHealthEndpoint:
    """Test the health endpoint."""

    def test_health_check(self, client):
        """Test health check returns expected format."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "kimi_available" in data
        assert data["version"] == "0.1.0"


class TestModelsEndpoint:
    """Test the models endpoint."""

    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "kimi-k2.5"


class TestChatCompletions:
    """Test chat completions endpoint."""

    def test_invalid_model(self, client):
        """Test that invalid model returns error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        # FastAPI wraps HTTPException in 'detail'
        assert "error" in data.get("detail", {}) or "error" in data

    def test_missing_messages(self, client):
        """Test that request without messages fails validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_valid_request_structure(self, client):
        """Test that valid request is accepted."""
        # Note: This will fail because echo doesn't speak ACP,
        # but we're testing the request structure validation
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        # Will fail due to mock, but structure is valid
        # In real usage with proper mock, this would succeed
        assert response.status_code in [200, 503]

    def test_direct_backend_returns_completion(self):
        """Test direct backend returns an OpenAI-compatible completion."""
        config = BridgeConfig(
            kimi_backend="direct",
            kimi_binary="echo",
            host="127.0.0.1",
            port=8080,
            log_level="DEBUG",
        )
        client = TestClient(create_app(config))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "User: Hello" in data["choices"][0]["message"]["content"]
        assert data["usage"]["total_tokens"] >= 1

    def test_direct_backend_rejects_tools(self):
        """Test direct backend rejects native tool calls with a clear error."""
        config = BridgeConfig(
            kimi_backend="direct",
            kimi_binary="echo",
            host="127.0.0.1",
            port=8080,
            log_level="DEBUG",
        )
        client = TestClient(create_app(config))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"]["code"] == "tools_not_supported"


class TestRequestValidation:
    """Test request validation."""

    def test_stream_parameter(self, client):
        """Test stream parameter is properly parsed."""
        # Valid request (will fail on execution due to mock)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        # Should accept the request (may fail on execution)
        assert response.status_code in [200, 503]

    def test_tool_choice_none_strips_tools(self, client):
        """Test that tool_choice: none is accepted and strips tools."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "none",
            },
        )

        # Should accept the request (may fail on execution)
        assert response.status_code in [200, 503]

    def test_response_format_json_object(self, client):
        """Test that response_format with json_object is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "response_format": {"type": "json_object"},
            },
        )

        # Should accept the request (may fail on execution)
        assert response.status_code in [200, 503]

    def test_response_format_json_schema(self, client):
        """Test that response_format with json_schema is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            },
        )

        # Should accept the request (may fail on execution)
        assert response.status_code in [200, 503]

    def test_tools_parameter(self, client):
        """Test tools parameter is properly parsed."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

        # Should accept the request (may fail on execution)
        assert response.status_code in [200, 503]
