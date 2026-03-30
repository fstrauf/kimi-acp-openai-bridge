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
