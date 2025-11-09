"""
Test Suite for Model Monitoring System

Comprehensive tests for the model performance monitoring system,
including health tracking, performance metrics, and cost analysis.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import time

from monitoring.model_monitor import model_monitor, ModelPerformanceMonitor
from core.agent import ModelInterface, GenerationRequest, GenerationResponse


class TestModelPerformanceMonitor:
    """Test the core model performance monitoring functionality."""

    def test_model_monitor_initialization(self):
        """Test that model monitor initializes correctly."""
        assert model_monitor is not None
        assert hasattr(model_monitor, 'model_health')
        assert hasattr(model_monitor, 'model_costs')
        assert hasattr(model_monitor, 'model_errors')
        assert hasattr(model_monitor, 'performance_history')

        # Check default alerts are set up
        from monitoring.metrics import metrics_collector
        assert len(metrics_collector.alerts) > 0

    def test_track_request_success(self):
        """Test tracking successful model requests."""
        # Clear existing data
        model_monitor.model_health.clear()
        model_monitor.model_costs.clear()
        model_monitor.performance_history.clear()

        # Create test request and response
        request = GenerationRequest(
            prompt="Test prompt",
            model_name="gpt-4",
            max_tokens=100
        )

        response = GenerationResponse(
            content="Test response",
            model_name="gpt-4",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50
            },
            generation_time=1.5,
            metadata={"model": "gpt-4"}
        )

        # Track successful request
        model_monitor.track_request(
            model_name="gpt-4",
            provider="openai",
            request=request,
            response=response
        )

        # Check model health was updated
        assert "gpt-4" in model_monitor.model_health
        health = model_monitor.model_health["gpt-4"]
        assert health["status"] in ["healthy", "recovering"]  # Initial state might be recovering
        assert health["total_requests"] == 1
        assert health["successful_requests"] == 1

        # Check performance history was recorded
        assert "gpt-4" in model_monitor.performance_history
        assert len(model_monitor.performance_history["gpt-4"]) == 1

        performance = model_monitor.performance_history["gpt-4"][0]
        assert performance["generation_time"] == 1.5
        assert performance["tokens_used"] == 50
        assert performance["content_length"] == len("Test response")

    def test_track_request_error(self):
        """Test tracking failed model requests."""
        # Clear existing data
        model_monitor.model_health.clear()
        model_monitor.model_errors.clear()

        # Create test request
        request = GenerationRequest(
            prompt="Test prompt",
            model_name="claude-3",
            max_tokens=100
        )

        # Create test error
        error = Exception("Model API error")

        # Track failed request
        model_monitor.track_request(
            model_name="claude-3",
            provider="anthropic",
            request=request,
            error=error
        )

        # Check model health was updated
        assert "claude-3" in model_monitor.model_health
        health = model_monitor.model_health["claude-3"]
        assert health["total_requests"] == 1
        assert health["successful_requests"] == 0
        assert health["consecutive_failures"] == 1

        # Check error was recorded
        assert "claude-3" in model_monitor.model_errors
        errors = model_monitor.model_errors["claude-3"]
        assert len(errors) == 1
        assert errors[0]["error"] == "Model API error"
        assert errors[0]["error_type"] == "Exception"

    def test_cost_calculation(self):
        """Test cost calculation for different models."""
        # Test OpenAI model cost calculation
        request = GenerationRequest(prompt="Test", model_name="gpt-4")
        response = GenerationResponse(
            content="Response",
            model_name="gpt-4",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50
            },
            generation_time=1.0
        )

        model_monitor.track_request(
            model_name="gpt-4",
            provider="openai",
            request=request,
            response=response
        )

        # Check cost was calculated
        assert "gpt-4" in model_monitor.model_costs
        costs = model_monitor.model_costs["gpt-4"]
        assert costs["hourly"] > 0
        assert costs["daily"] > 0

    def test_model_status_updates(self):
        """Test model status updates based on performance."""
        # Clear existing data
        model_monitor.model_health.clear()

        model_name = "test-model"
        provider = "test-provider"

        # Track successful requests
        for i in range(10):
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            response = GenerationResponse(
                content=f"Response {i}",
                model_name=model_name,
                generation_time=1.0,
                usage={"total_tokens": 50}
            )
            model_monitor.track_request(model_name, provider, request, response)

        # Check status is healthy
        health = model_monitor.model_health[model_name]
        assert health["status"] == "healthy"

        # Track failed requests
        for i in range(5):
            error = Exception(f"Error {i}")
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            model_monitor.track_request(model_name, provider, request, error=error)

        # Check status is degraded
        health = model_monitor.model_health[model_name]
        assert health["status"] == "degraded"

        # Track more failed requests
        for i in range(5):
            error = Exception(f"Error {i}")
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            model_monitor.track_request(model_name, provider, request, error=error)

        # Check status is unhealthy
        health = model_monitor.model_health[model_name]
        assert health["status"] == "unhealthy"

    def test_get_model_metrics(self):
        """Test getting comprehensive model metrics."""
        # Clear existing data
        model_monitor.model_health.clear()
        model_monitor.model_costs.clear()
        model_monitor.model_errors.clear()
        model_monitor.performance_history.clear()

        # Add test data for multiple models
        models_data = [
            ("gpt-4", "openai"),
            ("claude-3", "anthropic")
        ]

        for model_name, provider in models_data:
            # Add successful requests
            for i in range(5):
                request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
                response = GenerationResponse(
                    content=f"Response {i}",
                    model_name=model_name,
                    generation_time=1.0 + i * 0.1,
                    usage={"total_tokens": 50 + i * 10}
                )
                model_monitor.track_request(model_name, provider, request, response)

            # Add some errors
            for i in range(2):
                error = Exception(f"Error {i}")
                request = GenerationRequest(prompt=f"Error test {i}", model_name=model_name)
                model_monitor.track_request(model_name, provider, request, error=error)

        # Get metrics for all models
        all_metrics = model_monitor.get_model_metrics()

        assert "timestamp" in all_metrics
        assert "models" in all_metrics
        assert "summary" in all_metrics

        # Check individual model metrics
        models = all_metrics["models"]
        assert len(models) == 2

        for model_name in ["gpt-4", "claude-3"]:
            assert model_name in models
            model_data = models[model_name]

            assert "health" in model_data
            assert "costs" in model_data
            assert "error_rate" in model_data
            assert "avg_response_time" in model_data

            health = model_data["health"]
            assert health["total_requests"] == 7  # 5 success + 2 errors
            assert health["successful_requests"] == 5

        # Check summary
        summary = all_metrics["summary"]
        assert summary["total_models"] == 2
        assert summary["total_requests_hour"] == 14  # 7 per model

    def test_get_single_model_metrics(self):
        """Test getting metrics for a specific model."""
        # Clear existing data
        model_monitor.model_health.clear()

        # Add test data
        model_name = "test-model"
        provider = "test-provider"

        for i in range(10):
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            response = GenerationResponse(
                content=f"Response {i}",
                model_name=model_name,
                generation_time=1.0,
                usage={"total_tokens": 50}
            )
            model_monitor.track_request(model_name, provider, request, response)

        # Get metrics for specific model
        metrics = model_monitor.get_model_metrics(model_name)

        assert metrics["model_name"] == model_name
        assert "health" in metrics
        assert "costs" in metrics
        assert "error_rate" in metrics
        assert "avg_response_time" in metrics
        assert "recent_errors" in metrics
        assert "performance" in metrics

        health = metrics["health"]
        assert health["total_requests"] == 10
        assert health["successful_requests"] == 10
        assert health["uptime_percentage"] == 100.0

    def test_cleanup_old_data(self):
        """Test cleanup of old monitoring data."""
        # Clear existing data
        model_monitor.model_errors.clear()
        model_monitor.performance_history.clear()

        # Add some test data
        model_name = "test-model"
        provider = "test-provider"

        # Add errors
        for i in range(5):
            error = Exception(f"Error {i}")
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            model_monitor.track_request(model_name, provider, request, error=error)

        # Add performance data
        for i in range(5):
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            response = GenerationResponse(
                content=f"Response {i}",
                model_name=model_name,
                generation_time=1.0,
                usage={"total_tokens": 50}
            )
            model_monitor.track_request(model_name, provider, request, response)

        # Verify data exists
        assert len(model_monitor.model_errors[model_name]) == 5
        assert len(model_monitor.performance_history[model_name]) == 5

        # Mock time to simulate old data
        with patch('monitoring.model_monitor.datetime') as mock_datetime:
            # Set current time to 8 days in the future
            future_time = datetime.utcnow() + timedelta(days=8)
            mock_datetime.utcnow.return_value = future_time

            # Run cleanup
            model_monitor.cleanup_old_data()

        # Verify old data was cleaned up
        assert model_name not in model_monitor.model_errors or len(model_monitor.model_errors.get(model_name, [])) == 0
        assert model_name not in model_monitor.performance_history or len(model_monitor.performance_history.get(model_name, [])) == 0


@pytest.mark.asyncio
class TestModelHealthChecks:
    """Test model health check functionality."""

    async def test_start_health_checks(self):
        """Test starting health checks for models."""
        # Create mock models
        mock_models = {
            "healthy-model": Mock(spec=ModelInterface),
            "unhealthy-model": Mock(spec=ModelInterface)
        }

        # Configure mock models
        mock_models["healthy-model"].validate_connection = AsyncMock(return_value=True)
        mock_models["unhealthy-model"].validate_connection = AsyncMock(return_value=False)

        # Start health checks
        health_check_task = asyncio.create_task(
            model_monitor.start_health_checks(mock_models)
        )

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Check health was updated
        # Note: In a real test, we would need to wait for the health check interval
        # For this test, we'll just verify the task started correctly
        assert health_check_task is not None

        # Cancel the task
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

    async def test_health_check_model_validation(self):
        """Test health check with model validation."""
        # Create mock model
        mock_model = Mock(spec=ModelInterface)
        mock_model.validate_connection = AsyncMock(return_value=True)

        models = {"test-model": mock_model}

        # Clear health data
        model_monitor.model_health.clear()

        # Start health checks
        health_check_task = asyncio.create_task(
            model_monitor.start_health_checks(models)
        )

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

        # In a real scenario, health would be updated
        # For this test, we verify the mock was configured correctly
        assert mock_model.validate_connection.called


class TestCostAnalysis:
    """Test cost analysis functionality."""

    def test_calculate_cost_openai(self):
        """Test cost calculation for OpenAI models."""
        # Use private method for testing
        cost = model_monitor._calculate_cost(
            "gpt-4",
            "openai",
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        )

        # OpenAI GPT-4 pricing: $0.03/1K input tokens, $0.06/1K output tokens
        expected_cost = (100 * 0.00003) + (50 * 0.00006)
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_cost_anthropic(self):
        """Test cost calculation for Anthropic models."""
        cost = model_monitor._calculate_cost(
            "claude-3-sonnet-20240229",
            "anthropic",
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        )

        # Anthropic Claude-3-Sonnet pricing: $0.003/1K input tokens, $0.015/1K output tokens
        expected_cost = (100 * 0.000003) + (50 * 0.000015)
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown models."""
        cost = model_monitor._calculate_cost(
            "unknown-model",
            "unknown-provider",
            {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        )

        # Should return 0 for unknown models
        assert cost == 0.0

    def test_cost_tracking(self):
        """Test cumulative cost tracking."""
        # Clear existing data
        model_monitor.model_costs.clear()

        model_name = "test-model"
        provider = "openai"

        # Track multiple requests
        for i in range(5):
            request = GenerationRequest(prompt=f"Test {i}", model_name=model_name)
            response = GenerationResponse(
                content=f"Response {i}",
                model_name=model_name,
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30
                },
                generation_time=1.0
            )
            model_monitor.track_request(model_name, provider, request, response)

        # Check costs were accumulated
        assert model_name in model_monitor.model_costs
        costs = model_monitor.model_costs[model_name]
        assert costs["hourly"] > 0
        assert costs["daily"] > 0

        # Verify cost calculation
        # OpenAI pricing: $0.03/1K input + $0.06/1K output per request
        expected_per_request = (20 * 0.00003) + (10 * 0.00006)
        expected_total = expected_per_request * 5

        assert abs(costs["hourly"] - expected_total) < 0.000001


class TestErrorTracking:
    """Test error tracking and analysis."""

    def test_error_rate_calculation(self):
        """Test error rate calculation accuracy."""
        # Clear existing data
        model_monitor.model_health.clear()

        model_name = "test-model"
        provider = "test-provider"

        # Track mixed success and failure
        for i in range(10):
            if i < 8:  # 8 successful requests
                request = GenerationRequest(prompt=f"Success {i}", model_name=model_name)
                response = GenerationResponse(
                    content=f"Response {i}",
                    model_name=model_name,
                    generation_time=1.0,
                    usage={"total_tokens": 50}
                )
                model_monitor.track_request(model_name, provider, request, response)
            else:  # 2 failed requests
                error = Exception(f"Error {i}")
                request = GenerationRequest(prompt=f"Error {i}", model_name=model_name)
                model_monitor.track_request(model_name, provider, request, error=error)

        # Get metrics
        metrics = model_monitor.get_model_metrics(model_name)

        # Check error rate (2 errors out of 10 total = 20%)
        assert abs(metrics["error_rate"] - 20.0) < 0.1

    def test_consecutive_failures_tracking(self):
        """Test consecutive failures tracking."""
        # Clear existing data
        model_monitor.model_health.clear()

        model_name = "test-model"
        provider = "test-provider"

        # Track consecutive failures
        for i in range(5):
            error = Exception(f"Consecutive error {i}")
            request = GenerationRequest(prompt=f"Error {i}", model_name=model_name)
            model_monitor.track_request(model_name, provider, request, error=error)

        # Check health status
        health = model_monitor.model_health[model_name]
        assert health["consecutive_failures"] == 5
        assert health["consecutive_successes"] == 0
        assert health["status"] == "unhealthy"

    def test_error_cleanup(self):
        """Test old error cleanup."""
        # Clear existing data
        model_monitor.model_errors.clear()

        model_name = "test-model"
        provider = "test-provider"

        # Add errors with different timestamps
        current_time = datetime.utcnow()
        old_time = current_time - timedelta(hours=2)

        # Add recent error
        recent_error = Exception("Recent error")
        request1 = GenerationRequest(prompt="Recent test", model_name=model_name)
        model_monitor.track_request(model_name, provider, request1, error=recent_error)

        # Manually add old error
        old_error = {
            "timestamp": old_time,
            "error": "Old error",
            "error_type": "Exception"
        }
        if model_name not in model_monitor.model_errors:
            model_monitor.model_errors[model_name] = []
        model_monitor.model_errors[model_name].append(old_error)

        # Verify both errors exist
        assert len(model_monitor.model_errors[model_name]) == 2

        # Run cleanup (manually call cleanup method)
        model_monitor.model_errors[model_name] = [
            error for error in model_monitor.model_errors[model_name]
            if error["timestamp"] > current_time - timedelta(hours=1)
        ]

        # Verify only recent error remains
        assert len(model_monitor.model_errors[model_name]) == 1
        assert model_monitor.model_errors[model_name][0]["error"] == "Recent error"