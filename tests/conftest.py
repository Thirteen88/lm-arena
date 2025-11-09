"""
LM Arena Test Configuration

Comprehensive test configuration for pytest with fixtures,
mocks, and test utilities for the enhanced monitoring system.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import json
import time

from core.agent import LMArenaAgent, GenerationRequest, GenerationResponse
from core.model_switcher_monitored import MonitoredModelSwitcher, ModelMetrics, ModelStatus
from models.openai_model import OpenAIModel, OpenAIConfig
from models.anthropic_model import AnthropicModel, AnthropicConfig
from prompts.prompt_manager import PromptManager
from monitoring.metrics import metrics_collector, MetricType, AlertSeverity
from monitoring.model_monitor import model_monitor
from config.settings import load_config, get_config


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
async def test_config(temp_dir):
    """Create test configuration with temporary directories."""
    config = load_config()
    config.data_dir = temp_dir / "data"
    config.prompts_dir = temp_dir / "prompts"
    config.models_dir = temp_dir / "models"
    config.logs_dir = temp_dir / "logs"

    # Create directories
    for dir_path in [config.data_dir, config.prompts_dir, config.models_dir, config.logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return config


@pytest.fixture
def mock_openai_model():
    """Create a mock OpenAI model for testing."""
    mock_model = Mock(spec=OpenAIModel)
    mock_model.name = "test-gpt-4"
    mock_model.provider = "openai"
    mock_model.config = Mock(spec=OpenAIConfig)
    mock_model.config.model_id = "gpt-4"
    mock_model.config.api_key = "test-key"
    mock_model.capabilities = Mock()
    mock_model.capabilities.max_tokens = 4096
    mock_model.capabilities.supports_functions = True
    mock_model.capabilities.supports_streaming = True
    mock_model.capabilities.context_window = 8192
    mock_model.capabilities.cost_per_input_token = 0.00003
    mock_model.capabilities.cost_per_output_token = 0.00006

    # Mock async methods
    mock_model.generate = AsyncMock()
    mock_model.generate_stream = AsyncMock()
    mock_model.validate_connection = AsyncMock(return_value=True)
    mock_model.count_tokens = AsyncMock(return_value=100)

    return mock_model


@pytest.fixture
def mock_anthropic_model():
    """Create a mock Anthropic model for testing."""
    mock_model = Mock(spec=AnthropicModel)
    mock_model.name = "test-claude-3"
    mock_model.provider = "anthropic"
    mock_model.config = Mock(spec=AnthropicConfig)
    mock_model.config.model_id = "claude-3-sonnet-20240229"
    mock_model.config.api_key = "test-key"
    mock_model.capabilities = Mock()
    mock_model.capabilities.max_tokens = 4096
    mock_model.capabilities.supports_functions = False
    mock_model.capabilities.supports_streaming = True
    mock_model.capabilities.context_window = 200000
    mock_model.capabilities.cost_per_input_token = 0.000003
    mock_model.capabilities.cost_per_output_token = 0.000015

    # Mock async methods
    mock_model.generate = AsyncMock()
    mock_model.generate_stream = AsyncMock()
    mock_model.validate_connection = AsyncMock(return_value=True)
    mock_model.count_tokens = AsyncMock(return_value=100)

    return mock_model


@pytest.fixture
def sample_models(mock_openai_model, mock_anthropic_model):
    """Create sample models dictionary for testing."""
    return {
        "gpt-4": mock_openai_model,
        "claude-3": mock_anthropic_model
    }


@pytest.fixture
async def lm_agent(test_config, sample_models):
    """Create LMArenaAgent instance for testing."""
    agent = LMArenaAgent()

    # Register models
    for name, model in sample_models.items():
        agent.model_registry.register_model(model)

    await agent.initialize()
    yield agent

    await agent.shutdown()


@pytest.fixture
async def model_switcher(sample_models):
    """Create MonitoredModelSwitcher instance for testing."""
    switcher = MonitoredModelSwitcher(sample_models)
    yield switcher


@pytest.fixture
async def prompt_manager(test_config):
    """Create PromptManager instance for testing."""
    manager = PromptManager(test_config.prompts_dir)
    yield manager


@pytest.fixture
def sample_generation_request():
    """Create sample generation request for testing."""
    return GenerationRequest(
        prompt="Test prompt for LM Arena",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=100,
        system_prompt="You are a helpful assistant."
    )


@pytest.fixture
def sample_generation_response():
    """Create sample generation response for testing."""
    return GenerationResponse(
        content="This is a test response from the LM Arena system.",
        model_name="gpt-4",
        usage={
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        },
        finish_reason="stop",
        generation_time=1.5,
        metadata={
            "model": "gpt-4",
            "id": "test-response-id"
        }
    )


@pytest.fixture
def mock_metrics_collector():
    """Create a mock metrics collector for testing."""
    # Reset metrics collector
    metrics_collector.counters.clear()
    metrics_collector.gauges.clear()
    metrics_collector.histograms.clear()
    metrics_collector.timers.clear()
    metrics_collector.metrics.clear()

    return metrics_collector


@pytest.fixture
def sample_metrics():
    """Create sample metrics data for testing."""
    return {
        "model_requests_total": 100,
        "model_response_time_seconds": 2.5,
        "model_tokens_used_total": 5000,
        "model_cost_usd": 0.15,
        "model_error_rate": 2.0,
        "active_conversations": 10,
        "queue_size": 5,
        "memory_usage_bytes": 1024000
    }


@pytest.fixture
def sample_alert():
    """Create sample alert for testing."""
    return {
        "id": "test-alert-1",
        "name": "High Model Error Rate",
        "description": "Model error rate exceeds 10%",
        "severity": AlertSeverity.HIGH,
        "condition": "gt",
        "threshold": 10.0,
        "metric_name": "model_error_rate",
        "labels": {"model": "gpt-4"},
        "enabled": True
    }


@pytest.fixture
def test_metrics_data():
    """Create comprehensive test metrics data."""
    return {
        "counters": {
            "model_requests_total": {
                "gpt-4": 50,
                "claude-3": 30
            }
        },
        "gauges": {
            "model_error_rate": {
                "gpt-4": 2.0,
                "claude-3": 1.5
            },
            "active_conversations": 10
        },
        "histograms": {
            "model_response_time_seconds": {
                "gpt-4": [1.0, 1.5, 2.0, 2.5, 3.0],
                "claude-3": [0.8, 1.2, 1.6, 2.0, 2.4]
            }
        },
        "timers": {
            "generation_time": {
                "gpt-4": [1.2, 1.8, 2.1],
                "claude-3": [0.9, 1.3, 1.7]
            }
        }
    }


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing."""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def performance_test_data():
    """Create performance test data."""
    return {
        "concurrent_requests": 100,
        "duration_seconds": 60,
        "expected_response_time_ms": 500,
        "expected_error_rate": 5.0,
        "memory_limit_mb": 512
    }


@pytest.fixture
async def mock_external_api():
    """Create mock external API responses."""
    async def mock_openai_response(*args, **kwargs):
        return {
            "choices": [{
                "message": {"content": "Mock OpenAI response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }

    async def mock_anthropic_response(*args, **kwargs):
        return {
            "content": [{"text": "Mock Anthropic response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 18
            }
        }

    return {
        "openai": mock_openai_response,
        "anthropic": mock_anthropic_response
    }


# Helper functions for testing
@pytest.fixture
def create_test_request():
    """Helper function to create test requests."""
    def _create_request(prompt="Test prompt", model="gpt-4", **kwargs):
        return GenerationRequest(
            prompt=prompt,
            model_name=model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 100),
            system_prompt=kwargs.get("system_prompt"),
            metadata=kwargs.get("metadata", {})
        )
    return _create_request


@pytest.fixture
def create_test_response():
    """Helper function to create test responses."""
    def _create_response(content="Test response", model="gpt-4", **kwargs):
        return GenerationResponse(
            content=content,
            model_name=model,
            usage=kwargs.get("usage", {"total_tokens": 50}),
            finish_reason=kwargs.get("finish_reason", "stop"),
            generation_time=kwargs.get("generation_time", 1.0),
            metadata=kwargs.get("metadata", {})
        )
    return _create_response


# Async test helpers
@pytest.fixture
def async_test():
    """Helper for async test execution."""
    async def _run_async(coro):
        return await coro
    return _run_async


# Performance testing utilities
@pytest.fixture
def measure_performance():
    """Helper to measure performance metrics."""
    def _measure(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        return {
            "result": result,
            "duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time
        }
    return _measure


@pytest.fixture
def async_measure_performance():
    """Helper to measure async function performance."""
    async def _measure(coro, *args, **kwargs):
        start_time = time.time()
        result = await coro(*args, **kwargs)
        end_time = time.time()

        return {
            "result": result,
            "duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time
        }
    return _measure


# Test data generators
@pytest.fixture
def generate_test_data():
    """Generate test data for various scenarios."""
    def _generate_prompts(count=10):
        prompts = []
        for i in range(count):
            prompts.append(f"Test prompt {i+1} for comprehensive testing")
        return prompts

    def _generate_requests(count=10, model="gpt-4"):
        requests = []
        for i in range(count):
            requests.append(GenerationRequest(
                prompt=f"Test request {i+1}",
                model_name=model,
                temperature=0.5 + (i * 0.1),
                max_tokens=50 + (i * 10)
            ))
        return requests

    return {
        "prompts": _generate_prompts,
        "requests": _generate_requests
    }


# Cleanup utilities
@pytest.fixture(autouse=True)
async def cleanup_after_tests():
    """Cleanup after each test."""
    yield

    # Reset metrics collector
    metrics_collector.counters.clear()
    metrics_collector.gauges.clear()
    metrics_collector.histograms.clear()
    metrics_collector.timers.clear()
    metrics_collector.metrics.clear()
    metrics_collector.alerts.clear()

    # Reset model monitor
    model_monitor.model_health.clear()
    model_monitor.model_costs.clear()
    model_monitor.model_errors.clear()
    model_monitor.performance_history.clear()