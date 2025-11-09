"""
Test Suite for Model Switching System

Comprehensive tests for the enhanced model switching with monitoring,
including all switching strategies and performance validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import time

from core.model_switcher_monitored import (
    MonitoredModelSwitcher, ModelMetrics, ModelStatus, SwitchingStrategy,
    RoundRobinStrategy, LoadBalancedStrategy, CostOptimizedStrategy,
    PerformanceOptimizedStrategy, PriorityBasedStrategy, RandomStrategy,
    AdaptiveStrategy
)
from core.agent import GenerationRequest, GenerationResponse


class TestModelMetrics:
    """Test the ModelMetrics class functionality."""

    def test_model_metrics_initialization(self):
        """Test ModelMetrics initialization."""
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider"
        )

        assert metrics.model_name == "test-model"
        assert metrics.provider == "test-provider"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.status == ModelStatus.HEALTHY
        assert metrics.is_available is True

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider"
        )

        # Initially zero success rate
        assert metrics.success_rate == 0.0

        # Add some requests
        metrics.total_requests = 10
        metrics.successful_requests = 8

        assert metrics.success_rate == 80.0

    def test_is_available_check(self):
        """Test model availability checks."""
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider"
        )

        # Initially available
        assert metrics.is_available is True

        # Test unhealthy status
        metrics.status = ModelStatus.UNHEALTHY
        assert metrics.is_available is False

        # Test rate limited status
        metrics.status = ModelStatus.RATE_LIMITED
        metrics.max_requests_per_minute = 60
        metrics.current_minute_requests = 30
        assert metrics.is_available is True

        # Exceed rate limit
        metrics.current_minute_requests = 60
        assert metrics.is_available is False

    def test_update_request_metrics(self):
        """Test updating request metrics."""
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider",
            cost_per_1k_tokens=0.01
        )

        # Test successful request
        metrics.update_request_metrics(
            success=True,
            response_time=1.5,
            tokens_used=100,
            cost=0.001
        )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert metrics.total_tokens_used == 100
        assert metrics.total_cost == 0.001
        assert metrics.average_response_time == 1.5

        # Test failed request
        metrics.update_request_metrics(
            success=False,
            response_time=0,
            tokens_used=0,
            cost=0
        )

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.consecutive_successes == 0
        assert metrics.consecutive_failures == 1

    def test_status_updates(self):
        """Test automatic status updates."""
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider"
        )

        # Multiple failures should make it unhealthy
        for i in range(6):
            metrics.update_request_metrics(False, 0, 0, 0)

        assert metrics.status == ModelStatus.UNHEALTHY

        # Reset and test degradation
        metrics = ModelMetrics(
            model_name="test-model",
            provider="test-provider"
        )
        metrics.total_requests = 10
        metrics.successful_requests = 7  # 70% success rate

        metrics.update_request_metrics(False, 0, 0, 0)
        assert metrics.status == ModelStatus.DEGRADED  # Below 80% success rate


class TestSwitchingStrategies:
    """Test all model switching strategies."""

    @pytest.fixture
    def sample_models_metrics(self):
        """Create sample model metrics for testing."""
        return {
            "fast-model": ModelMetrics("fast-model", "provider1"),
            "balanced-model": ModelMetrics("balanced-model", "provider2"),
            "cheap-model": ModelMetrics("cheap-model", "provider3")
        }

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self, sample_models_metrics):
        """Test round-robin switching strategy."""
        strategy = RoundRobinStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # First selection
        model_name = await strategy.select_model(sample_models_metrics, request)
        assert model_name in sample_models_metrics

        # Second selection should be different
        model_name2 = await strategy.select_model(sample_models_metrics, request)
        assert model_name2 in sample_models_metrics
        assert model_name2 != model_name

    @pytest.mark.asyncio
    async def test_load_balanced_strategy(self, sample_models_metrics):
        """Test load-balanced switching strategy."""
        strategy = LoadBalancedStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # Add some load to models
        sample_models_metrics["fast-model"].current_minute_requests = 10
        sample_models_metrics["balanced-model"].current_minute_requests = 5
        sample_models_metrics["cheap-model"].current_minute_requests = 1

        # Should select the model with lowest load
        selected = await strategy.select_model(sample_models_metrics, request)
        assert selected == "cheap-model"

    @pytest.mark.asyncio
    async def test_cost_optimized_strategy(self, sample_models_metrics):
        """Test cost-optimized switching strategy."""
        strategy = CostOptimizedStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # Set different costs
        sample_models_metrics["fast-model"].cost_per_1k_tokens = 0.05
        sample_models_metrics["balanced-model"].cost_per_1k_tokens = 0.02
        sample_models_metrics["cheap-model"].cost_per_1k_tokens = 0.001

        # Should select cheapest model
        selected = await strategy.select_model(sample_models_metrics, request)
        assert selected == "cheap-model"

    @pytest.mark.asyncio
    async def test_priority_based_strategy(self, sample_models_metrics):
        """Test priority-based switching strategy."""
        strategy = PriorityBasedStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # Set different priorities (lower number = higher priority)
        sample_models_metrics["fast-model"].priority = 3
        sample_models_metrics["balanced-model"].priority = 1
        sample_models_metrics["cheap-model"].priority = 2

        # Should select highest priority model
        selected = await strategy.select_model(sample_models_metrics, request)
        assert selected == "balanced-model"

    @pytest.mark.asyncio
    async def test_random_strategy(self, sample_models_metrics):
        """Test random switching strategy."""
        strategy = RandomStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # Should return some valid model
        selected = await strategy.select_model(sample_models_metrics, request)
        assert selected in sample_models_metrics

    @pytest.mark.asyncio
    async def test_performance_optimized_strategy(self, sample_models_metrics):
        """Test performance-optimized switching strategy."""
        strategy = PerformanceOptimizedStrategy()
        request = GenerationRequest(prompt="Test", model_name="any")

        # Set performance data
        sample_models_metrics["fast-model"].success_rate = 95
        sample_models_metrics["fast-model"].average_response_time = 0.5

        sample_models_metrics["balanced-model"].success_rate = 90
        sample_models_metrics["balanced-model"].average_response_time = 1.0

        sample_models_metrics["cheap-model"].success_rate = 85
        sample_models_metrics["cheap-model"].average_response_time = 2.0

        # Should select best performing model
        selected = await strategy.select_model(sample_models_metrics, request)
        assert selected == "fast-model"


class TestMonitoredModelSwitcher:
    """Test the MonitoredModelSwitcher class."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        models = {}
        for name in ["gpt-4", "claude-3", "local-model"]:
            model = Mock()
            model.name = name
            model.provider = "test-provider"
            model.generate = AsyncMock()
            model.validate_connection = AsyncMock(return_value=True)
            models[name] = model
        return models

    def test_model_switcher_initialization(self, mock_models):
        """Test model switcher initialization."""
        switcher = MonitoredModelSwitcher(mock_models)

        assert len(switcher.models) == 3
        assert len(switcher.model_metrics) == 3
        assert isinstance(switcher.strategy, LoadBalancedStrategy)

        # Check metrics were initialized
        for model_name in mock_models:
            assert model_name in switcher.model_metrics
            metrics = switcher.model_metrics[model_name]
            assert metrics.model_name == model_name

    def test_set_strategy(self, mock_models):
        """Test changing switching strategy."""
        switcher = MonitoredModelSwitcher(mock_models)

        # Test strategy change
        switcher.set_strategy(SwitchingStrategy.ROUND_ROBIN)
        assert isinstance(switcher.strategy, RoundRobinStrategy)

        switcher.set_strategy(SwitchingStrategy.COST_OPTIMIZED)
        assert isinstance(switcher.strategy, CostOptimizedStrategy)

    @pytest.mark.asyncio
    async def test_select_model(self, mock_models):
        """Test model selection."""
        switcher = MonitoredModelSwitcher(mock_models)
        request = GenerationRequest(prompt="Test", model_name="any")

        # Test model selection
        model, model_name = await switcher.select_model(request)

        assert model is not None
        assert model_name in mock_models
        assert model == mock_models[model_name]

    @pytest.mark.asyncio
    async def test_execute_with_switching_success(self, mock_models):
        """Test successful execution with model switching."""
        switcher = MonitoredModelSwitcher(mock_models)
        request = GenerationRequest(prompt="Test prompt", model_name="gpt-4")

        # Mock successful response
        response = GenerationResponse(
            content="Test response",
            model_name="gpt-4",
            usage={"total_tokens": 50},
            generation_time=1.0
        )

        mock_models["gpt-4"].generate.return_value = response

        # Execute request
        result = await switcher.execute_with_switching(request, max_retries=1)

        assert result == response
        mock_models["gpt-4"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_switching_failure_retry(self, mock_models):
        """Test execution with model switching on failure."""
        switcher = MonitoredModelSwitcher(mock_models)
        request = GenerationRequest(prompt="Test prompt", model_name="gpt-4")

        # Mock first model failure, second success
        mock_models["gpt-4"].generate.side_effect = [
            Exception("Model failed"),
            GenerationResponse(
                content="Fallback response",
                model_name="claude-3",
                usage={"total_tokens": 50},
                generation_time=1.5
            )
        ]

        # Execute with retry
        result = await switcher.execute_with_switching(request, max_retries=1)

        assert result.content == "Fallback response"
        assert result.model_name == "claude-3"

    @pytest.mark.asyncio
    async def test_execute_with_switching_all_fail(self, mock_models):
        """Test execution when all models fail."""
        switcher = MonitoredModelSwitcher(mock_models)
        request = GenerationRequest(prompt="Test prompt", model_name="gpt-4")

        # Mock all models failing
        for model in mock_models.values():
            model.generate.side_effect = Exception("All models failed")

        # Should raise exception
        with pytest.raises(Exception) as exc_info:
            await switcher.execute_with_switching(request, max_retries=2)

        assert "All 3 model attempts failed" in str(exc_info.value)

    def test_get_model_metrics(self, mock_models):
        """Test getting model metrics."""
        switcher = MonitoredModelSwitcher(mock_models)

        # Get all metrics
        all_metrics = switcher.get_model_metrics()
        assert len(all_metrics) == 3

        for model_name in mock_models:
            assert model_name in all_metrics
            assert "model_name" in all_metrics[model_name]
            assert "provider" in all_metrics[model_name]

        # Get specific model metrics
        gpt_metrics = switcher.get_model_metrics("gpt-4")
        assert gpt_metrics["model_name"] == "gpt-4"
        assert gpt_metrics["provider"] == "test-provider"

    def test_get_all_metrics(self, mock_models):
        """Test getting comprehensive metrics."""
        switcher = MonitoredModelSwitcher(mock_models)

        # Add some request data
        for model_name, metrics in switcher.model_metrics.items():
            metrics.total_requests = 10
            metrics.successful_requests = 8
            metrics.total_cost = 0.05

        all_metrics = switcher.get_all_metrics()

        assert "strategy" in all_metrics
        assert "total_models" in all_metrics
        assert "available_models" in all_metrics
        assert "overall_stats" in all_metrics
        assert "model_metrics" in all_metrics

        overall_stats = all_metrics["overall_stats"]
        assert overall_stats["total_requests"] == 30  # 10 per model
        assert overall_stats["successful_requests"] == 24  # 8 per model
        assert overall_stats["success_rate"] == 80.0
        assert overall_stats["total_cost"] == 0.15  # 0.05 per model

    @pytest.mark.asyncio
    async def test_health_check(self, mock_models):
        """Test health check functionality."""
        switcher = MonitoredModelSwitcher(mock_models)

        # Mock different health states
        mock_models["gpt-4"].validate_connection.return_value = True
        mock_models["claude-3"].validate_connection.return_value = False
        mock_models["local-model"].validate_connection.side_effect = Exception("Connection error")

        # Run health check
        health_results = await switcher.health_check()

        assert "overall_health" in health_results
        assert "model_health" in health_results
        assert len(health_results["model_health"]) == 3

        # Check individual model health
        model_health = health_results["model_health"]
        assert model_health["gpt-4"]["healthy"] is True
        assert model_health["claude-3"]["healthy"] is False
        assert model_health["local-model"]["healthy"] is False
        assert "error" in model_health["local-model"]


class TestModelMetricsIntegration:
    """Test integration between model metrics and monitoring system."""

    @pytest.mark.asyncio
    async def test_metrics_push_to_monitoring(self, mock_models):
        """Test that metrics are pushed to monitoring system."""
        switcher = MonitoredModelSwitcher(mock_models)
        request = GenerationRequest(prompt="Test", model_name="gpt-4")

        response = GenerationResponse(
            content="Test response",
            model_name="gpt-4",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50
            },
            generation_time=1.5
        )

        mock_models["gpt-4"].generate.return_value = response

        # Execute request
        await switcher.execute_with_switching(request, max_retries=0)

        # Check metrics were updated
        metrics = switcher.model_metrics["gpt-4"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_tokens_used == 50
        assert metrics.total_cost > 0