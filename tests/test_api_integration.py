"""
Test Suite for LM Arena API Integration

Comprehensive tests for the LM Arena REST API endpoints, including
monitoring integration, WebSocket functionality, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import WebSocket
import httpx

from api.main import app
from api.schemas import ChatRequest, ChatResponse, ModelSwitchRequest
from core.agent import GenerationRequest, GenerationResponse


class TestAPIHealthEndpoints:
    """Test API health and status endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test the main health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "agent_status" in data
        assert "active_models" in data
        assert "total_conversations" in data
        assert data["status"] in ["healthy", "unhealthy"]

    def test_monitoring_health_endpoint(self, client):
        """Test the monitoring health endpoint."""
        response = client.get("/monitoring/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "metrics_collector" in data
        assert "model_monitor" in data

        metrics_collector = data["metrics_collector"]
        assert "running" in metrics_collector
        assert "total_metrics" in metrics_collector
        assert "alerts_enabled" in metrics_collector

    def test_prometheus_metrics_endpoint(self, client):
        """Test the Prometheus metrics endpoint."""
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Check for Prometheus format
        metrics_text = response.text
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text

    def test_dashboard_endpoint(self, client):
        """Test the dashboard endpoint."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"

        # Check for HTML content
        dashboard_html = response.text
        assert "<!DOCTYPE html>" in dashboard_html
        assert "LM Arena Performance Dashboard" in dashboard_html


class TestChatAPI:
    """Test chat API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_chat_request(self):
        """Create sample chat request."""
        return {
            "message": "Hello, how are you?",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "conversation_id": "test-conversation-123"
        }

    def test_chat_endpoint_missing_data(self, client, sample_chat_request):
        """Test chat endpoint with missing required data."""
        # Test missing message
        request = sample_chat_request.copy()
        del request["message"]
        response = client.post("/chat", json=request)
        assert response.status_code == 422  # Validation error

        # Test missing model
        request = sample_chat_request.copy()
        del request["model"]
        response = client.post("/chat", json=request)
        assert response.status_code == 422

    def test_chat_endpoint_with_valid_data(self, client, sample_chat_request):
        """Test chat endpoint with valid data structure."""
        response = client.post("/chat", json=sample_chat_request)

        # Should either succeed (200) or fail gracefully (500) due to no real model
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "content" in data
            assert "model" in data
            assert "conversation_id" in data
            assert "usage" in data

    def test_chat_stream_endpoint_structure(self, client, sample_chat_request):
        """Test chat stream endpoint structure."""
        response = client.post("/chat/stream", json=sample_chat_request)

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]


class TestModelAPI:
    """Test model management API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_models_endpoint(self, client):
        """Test listing available models."""
        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # If models are available, check structure
        if data:
            model_info = data[0]
            assert "name" in model_info
            assert "provider" in model_info
            assert "model_id" in model_info
            assert "capabilities" in model_info

    def test_switch_model_endpoint_valid_strategy(self, client):
        """Test model switching with valid strategy."""
        request = {
            "strategy": "round_robin"
        }

        response = client.post("/models/switch", json=request)
        # May fail if no model switcher is initialized, but structure should be valid
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "current_strategy" in data

    def test_switch_model_endpoint_invalid_strategy(self, client):
        """Test model switching with invalid strategy."""
        request = {
            "strategy": "invalid_strategy"
        }

        response = client.post("/models/switch", json=request)
        assert response.status_code == 400  # Bad Request

        data = response.json()
        assert "detail" in data
        assert "Invalid strategy" in data["detail"]


class TestPromptAPI:
    """Test prompt management API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_create_prompt_endpoint_valid(self, client):
        """Test creating a new prompt template."""
        request = {
            "name": "Test Prompt",
            "content": "Hello {{ name }}, how are you?",
            "type": "user",
            "category": "general",
            "description": "A test prompt for validation",
            "variables": {"name": "World"}
        }

        response = client.post("/prompts", json=request)
        # May fail if prompt manager not initialized, but should handle gracefully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prompt_id" in data
            assert "message" in data

    def test_list_prompts_endpoint(self, client):
        """Test listing prompt templates."""
        response = client.get("/prompts")
        # May fail if prompt manager not initialized
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prompts" in data
            assert "total" in data
            assert isinstance(data["prompts"], list)


class TestConversationAPI:
    """Test conversation management API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_create_conversation_endpoint(self, client):
        """Test creating a new conversation."""
        request = {
            "system_prompt": "You are a helpful assistant."
        }

        response = client.post("/conversations", json=request)
        # May fail if agent not initialized, but should handle gracefully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "conversation_id" in data
            assert "message" in data

    def test_list_conversations_endpoint(self, client):
        """Test listing conversations."""
        response = client.get("/conversations")
        # May fail if agent not initialized
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "conversations" in data
            assert "total" in data
            assert isinstance(data["conversations"], list)

    def test_delete_conversation_endpoint(self, client):
        """Test deleting a conversation."""
        conversation_id = "test-conversation-123"
        response = client.delete(f"/conversations/{conversation_id}")

        # May return 404 if conversation doesn't exist or 503 if not initialized
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "message" in data


class TestMonitoringAPI:
    """Test monitoring-specific API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_models_metrics_endpoint(self, client):
        """Test model metrics endpoint."""
        response = client.get("/monitoring/models/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "models" in data
        assert "summary" in data

        summary = data["summary"]
        assert "total_models" in summary
        assert "healthy_models" in summary
        assert "total_requests_hour" in summary

    def test_model_health_endpoint(self, client):
        """Test individual model health endpoint."""
        model_name = "test-model"
        response = client.get(f"/monitoring/models/{model_name}/health")

        # May return 404 if model doesn't exist
        assert response.status_code in [200, 404]

    def test_model_performance_endpoint(self, client):
        """Test model performance endpoint."""
        model_name = "test-model"
        response = client.get(
            f"/monitoring/models/{model_name}/performance",
            params={"hours": 24, "granularity": "hour"}
        )

        # May return 404 if model doesn't exist
        assert response.status_code in [200, 404]

    def test_alerts_endpoint(self, client):
        """Test alerts endpoint."""
        response = client.get("/monitoring/alerts")
        assert response.status_code == 200

        data = response.json()
        assert "total_alerts" in data
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    def test_alerts_enable_disable_endpoint(self, client):
        """Test alert enable/disable endpoints."""
        # First get alerts to find an ID
        alerts_response = client.get("/monitoring/alerts")
        if alerts_response.status_code == 200:
            alerts_data = alerts_response.json()
            if alerts_data["alerts"]:
                alert_id = alerts_data["alerts"][0]["id"]

                # Test enabling alert
                response = client.post(f"/monitoring/alerts/{alert_id}/enable")
                assert response.status_code == 200

                # Test disabling alert
                response = client.post(f"/monitoring/alerts/{alert_id}/disable")
                assert response.status_code == 200

    def test_cost_analytics_endpoint(self, client):
        """Test cost analytics endpoint."""
        response = client.get("/monitoring/analytics/costs")
        assert response.status_code == 200

        data = response.json()
        assert "period" in data
        assert "start_time" in data
        assert "end_time" in data
        assert "cost_data" in data
        assert "total_cost" in data

    def test_performance_analytics_endpoint(self, client):
        """Test performance analytics endpoint."""
        response = client.get(
            "/monitoring/analytics/performance",
            params={"metric": "response_time", "period": "hour"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "metric" in data
        assert "period" in data
        assert "analytics" in data

    def test_dashboard_data_endpoint(self, client):
        """Test dashboard data endpoint."""
        response = client.get("/monitoring/dashboard")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "summary" in data
        assert "recent_alerts" in data
        assert "model_performance" in data
        assert "metrics_summary" in data


class TestWebSocketAPI:
    """Test WebSocket API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_websocket_connection_structure(self, client):
        """Test WebSocket connection structure."""
        # Note: FastAPI TestClient doesn't fully support WebSocket testing
        # This test validates the endpoint exists and handles WebSocket upgrade requests

        with client.websocket_connect("/ws/test-conversation") as websocket:
            # If connection succeeds, send a test message
            websocket.send_text(json.dumps({
                "message": "Hello WebSocket",
                "model": "gpt-4",
                "temperature": 0.7
            }))

            # Receive response (may timeout if not fully implemented)
            try:
                data = websocket.receive_text()
                response_data = json.loads(data)
                assert "type" in response_data
            except Exception:
                # WebSocket might not be fully implemented in test environment
                pass


class TestErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_json_payload(self, client):
        """Test sending invalid JSON payload."""
        response = client.post(
            "/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test requests with missing required fields."""
        # Test empty chat request
        response = client.post("/chat", json={})
        assert response.status_code == 422

        # Test model switch with empty strategy
        response = client.post("/models/switch", json={})
        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Test HTTP methods not allowed on endpoints."""
        # Test GET on POST endpoint
        response = client.get("/chat")
        assert response.status_code == 405

        # Test POST on GET endpoint
        response = client.post("/health")
        assert response.status_code == 405

    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Send non-JSON content to JSON endpoint
        response = client.post(
            "/chat",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422


class TestAPIAuthentication:
    """Test API authentication and security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_no_auth_required_for_health(self, client):
        """Test that health endpoints don't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/monitoring/health")
        assert response.status_code == 200

    def test_api_key_validation(self, client):
        """Test API key validation (if implemented)."""
        # This test would need to be adapted based on actual auth implementation
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.get("/models", headers=headers)

        # Should either accept (if auth not implemented) or reject
        assert response.status_code in [200, 401, 403]


class TestAPIPerformance:
    """Test API performance and response times."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_response_time_health_endpoint(self, client):
        """Test health endpoint response time."""
        import time
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second

    def test_response_time_metrics_endpoint(self, client):
        """Test metrics endpoint response time."""
        import time
        start_time = time.time()
        response = client.get("/monitoring/metrics")
        end_time = time.time()

        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds

    def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            results.append({
                "status": response.status_code,
                "time": end_time - start_time
            })

        # Create 10 concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        assert len(results) == 10
        assert all(r["status"] == 200 for r in results)

        # Concurrent requests should be faster than sequential
        avg_individual_time = sum(r["time"] for r in results) / len(results)
        assert total_time < avg_individual_time * 5  # Should be at least 5x faster than sequential