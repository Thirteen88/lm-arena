"""
Test Suite for Monitoring Metrics Collection

Comprehensive tests for the metrics collection system, including
counters, gauges, histograms, timers, and alerting functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from monitoring.metrics import (
    metrics_collector, MetricType, Alert, AlertSeverity, MetricPoint,
    MetricDefinition
)


class TestMetricsCollector:
    """Test the core metrics collection functionality."""

    def test_metrics_collector_initialization(self):
        """Test that metrics collector initializes correctly."""
        assert metrics_collector is not None
        assert hasattr(metrics_collector, 'counters')
        assert hasattr(metrics_collector, 'gauges')
        assert hasattr(metrics_collector, 'histograms')
        assert hasattr(metrics_collector, 'timers')
        assert hasattr(metrics_collector, 'alerts')

    def test_define_metric(self):
        """Test metric definition functionality."""
        # Clear existing metrics for clean test
        metrics_collector.metric_definitions.clear()

        # Define a counter metric
        metrics_collector.define_metric(
            "test_counter",
            "Test counter metric",
            MetricType.COUNTER,
            ["model", "provider"]
        )

        assert "test_counter" in metrics_collector.metric_definitions
        metric_def = metrics_collector.metric_definitions["test_counter"]
        assert metric_def.name == "test_counter"
        assert metric_def.metric_type == MetricType.COUNTER
        assert "model" in metric_def.labels
        assert "provider" in metric_def.labels

    def test_increment_counter(self):
        """Test counter increment functionality."""
        # Clear counters
        metrics_collector.counters.clear()

        # Increment counter without labels
        metrics_collector.increment_counter("test_requests", 1.0)
        assert metrics_collector.counters["test_requests"] == 1.0

        # Increment counter with labels
        metrics_collector.increment_counter(
            "test_requests",
            2.0,
            {"model": "gpt-4", "provider": "openai"}
        )
        key = "test_requests[model=gpt-4,provider=openai]"
        assert metrics_collector.counters[key] == 2.0

        # Increment same counter again
        metrics_collector.increment_counter(
            "test_requests",
            1.0,
            {"model": "gpt-4", "provider": "openai"}
        )
        assert metrics_collector.counters[key] == 3.0

    def test_set_gauge(self):
        """Test gauge setting functionality."""
        # Clear gauges
        metrics_collector.gauges.clear()

        # Set gauge without labels
        metrics_collector.set_gauge("active_conversations", 10)
        assert metrics_collector.gauges["active_conversations"] == 10

        # Set gauge with labels
        metrics_collector.set_gauge(
            "model_health",
            0.95,
            {"model": "claude-3"}
        )
        key = "model_health[model=claude-3]"
        assert metrics_collector.gauges[key] == 0.95

        # Update same gauge
        metrics_collector.set_gauge(
            "model_health",
            0.87,
            {"model": "claude-3"}
        )
        assert metrics_collector.gauges[key] == 0.87

    def test_record_histogram(self):
        """Test histogram recording functionality."""
        # Clear histograms
        metrics_collector.histograms.clear()

        # Record histogram values
        test_values = [0.1, 0.5, 1.0, 2.5, 5.0]
        for value in test_values:
            metrics_collector.record_histogram("response_times", value)

        histogram_data = metrics_collector.histograms["response_times"]
        assert len(histogram_data) == 5
        assert all(val in histogram_data for val in test_values)

        # Test with labels
        labeled_values = [1.2, 1.8, 2.1]
        for value in labeled_values:
            metrics_collector.record_histogram(
                "response_times",
                value,
                {"model": "gpt-4"}
            )

        key = "response_times[model=gpt-4]"
        labeled_histogram = metrics_collector.histograms[key]
        assert len(labeled_histogram) == 3
        assert all(val in labeled_histogram for val in labeled_values)

    def test_record_timer(self):
        """Test timer recording functionality."""
        # Clear timers
        metrics_collector.timers.clear()

        # Record timer values
        test_durations = [0.8, 1.2, 1.5, 2.1, 3.0]
        for duration in test_durations:
            metrics_collector.record_timer("generation_time", duration)

        timer_data = metrics_collector.timers["generation_time"]
        assert len(timer_data) == len(test_durations)
        assert all(duration in timer_data for duration in test_durations)

    def test_get_metric_value(self):
        """Test metric value retrieval."""
        # Clear all metrics
        metrics_collector.counters.clear()
        metrics_collector.gauges.clear()
        metrics_collector.histograms.clear()
        metrics_collector.timers.clear()

        # Test counter retrieval
        metrics_collector.increment_counter("test_counter", 5.0)
        value = metrics_collector.get_metric_value("test_counter")
        assert value == 5.0

        # Test gauge retrieval
        metrics_collector.set_gauge("test_gauge", 42.0)
        value = metrics_collector.get_metric_value("test_gauge")
        assert value == 42.0

        # Test non-existent metric
        value = metrics_collector.get_metric_value("non_existent")
        assert value is None

    def test_get_metric_history(self):
        """Test metric history retrieval."""
        # Clear metrics history
        metrics_collector.metrics.clear()

        # Add some metric points
        metrics_collector.increment_counter("test_metric", 1.0)
        metrics_collector.increment_counter("test_metric", 2.0)

        time.sleep(0.01)  # Small delay to ensure different timestamps

        metrics_collector.set_gauge("test_metric", 3.0)

        # Get all history
        history = metrics_collector.get_metric_history("test_metric")
        assert len(history) == 3

        # Get history since a specific time
        since = datetime.utcnow() - timedelta(seconds=1)
        recent_history = metrics_collector.get_metric_history("test_metric", since=since)
        assert len(recent_history) == 3

        # Get history with labels filter
        # Add labeled metric
        metrics_collector.increment_counter("test_metric", 1.0, {"model": "gpt-4"})
        labeled_history = metrics_collector.get_metric_history(
            "test_metric",
            labels={"model": "gpt-4"}
        )
        assert len(labeled_history) == 1

    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export format."""
        # Clear metrics
        metrics_collector.counters.clear()
        metrics_collector.gauges.clear()
        metrics_collector.histograms.clear()
        metrics_collector.timers.clear()
        metrics_collector.metric_definitions.clear()

        # Define and add metrics
        metrics_collector.define_metric(
            "test_requests_total",
            "Total test requests",
            MetricType.COUNTER,
            ["model"]
        )

        metrics_collector.define_metric(
            "test_response_time",
            "Test response time",
            MetricType.GAUGE
        )

        metrics_collector.increment_counter("test_requests_total", 100)
        metrics_collector.set_gauge("test_response_time", 1.5)

        # Export to Prometheus format
        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Check for expected content
        assert "# HELP test_requests_total Total test requests" in prometheus_output
        assert "# TYPE test_requests_total counter" in prometheus_output
        assert "# HELP test_response_time Test response time" in prometheus_output
        assert "# TYPE test_response_time gauge" in prometheus_output
        assert "test_requests_total 100.0" in prometheus_output
        assert "test_response_time 1.5" in prometheus_output


class TestAlertSystem:
    """Test the alerting system functionality."""

    def test_create_alert(self):
        """Test alert creation."""
        # Clear existing alerts
        metrics_collector.alerts.clear()

        alert_id = metrics_collector.create_alert(
            name="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            condition="gt",
            threshold=10.0,
            metric_name="test_metric"
        )

        assert alert_id is not None
        assert alert_id in metrics_collector.alerts

        alert = metrics_collector.alerts[alert_id]
        assert alert.name == "Test Alert"
        assert alert.description == "Test alert description"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.condition == "gt"
        assert alert.threshold == 10.0
        assert alert.metric_name == "test_metric"
        assert alert.enabled is True

    def test_evaluate_alerts_triggering(self):
        """Test alert evaluation and triggering."""
        # Clear alerts and metrics
        metrics_collector.alerts.clear()
        metrics_collector.gauges.clear()

        # Create alert
        alert_id = metrics_collector.create_alert(
            name="High Error Rate",
            description="Error rate is too high",
            severity=AlertSeverity.CRITICAL,
            condition="gt",
            threshold=5.0,
            metric_name="error_rate"
        )

        # Set metric below threshold
        metrics_collector.set_gauge("error_rate", 3.0)
        metrics_collector.evaluate_alerts()

        alert = metrics_collector.alerts[alert_id]
        assert alert.last_triggered is None
        assert alert.trigger_count == 0

        # Set metric above threshold
        metrics_collector.set_gauge("error_rate", 8.0)
        metrics_collector.evaluate_alerts()

        alert = metrics_collector.alerts[alert_id]
        assert alert.last_triggered is not None
        assert alert.trigger_count == 1

    def test_alert_conditions(self):
        """Test different alert conditions."""
        # Clear alerts and metrics
        metrics_collector.alerts.clear()
        metrics_collector.gauges.clear()

        # Test greater than condition
        alert_gt = metrics_collector.create_alert(
            name="GT Alert",
            description="Greater than test",
            severity=AlertSeverity.MEDIUM,
            condition="gt",
            threshold=10.0,
            metric_name="test_metric"
        )

        # Test less than condition
        alert_lt = metrics_collector.create_alert(
            name="LT Alert",
            description="Less than test",
            severity=AlertSeverity.MEDIUM,
            condition="lt",
            threshold=5.0,
            metric_name="test_metric"
        )

        # Test greater than or equal condition
        alert_gte = metrics_collector.create_alert(
            name="GTE Alert",
            description="Greater than or equal test",
            severity=AlertSeverity.MEDIUM,
            condition="gte",
            threshold=10.0,
            metric_name="test_metric"
        )

        # Set metric value
        metrics_collector.set_gauge("test_metric", 10.0)
        metrics_collector.evaluate_alerts()

        # Check results
        assert metrics_collector.alerts[alert_gt].trigger_count == 0  # 10 is not > 10
        assert metrics_collector.alerts[alert_lt].trigger_count == 0   # 10 is not < 5
        assert metrics_collector.alerts[alert_gte].trigger_count == 1 # 10 is >= 10

    def test_alert_disable_enable(self):
        """Test alert enable/disable functionality."""
        # Create alert
        alert_id = metrics_collector.create_alert(
            name="Test Alert",
            description="Test alert",
            severity=AlertSeverity.LOW,
            condition="gt",
            threshold=1.0,
            metric_name="test_metric"
        )

        # Disable alert
        metrics_collector.alerts[alert_id].enabled = False

        # Set metric to trigger alert
        metrics_collector.gauges.clear()
        metrics_collector.set_gauge("test_metric", 5.0)
        metrics_collector.evaluate_alerts()

        # Alert should not trigger
        alert = metrics_collector.alerts[alert_id]
        assert alert.last_triggered is None
        assert alert.trigger_count == 0

        # Enable alert
        alert.enabled = True
        metrics_collector.evaluate_alerts()

        # Alert should trigger now
        assert alert.last_triggered is not None
        assert alert.trigger_count == 1

    def test_alert_callback(self):
        """Test alert callback functionality."""
        # Clear alerts and callbacks
        metrics_collector.alerts.clear()
        metrics_collector.alert_callbacks.clear()

        # Create mock callback
        callback_called = []
        callback_mock = Mock(side_effect=lambda alert: callback_called.append(alert))

        # Add callback
        metrics_collector.add_alert_callback(callback_mock)

        # Create and trigger alert
        alert_id = metrics_collector.create_alert(
            name="Callback Test",
            description="Test alert callback",
            severity=AlertSeverity.HIGH,
            condition="gt",
            threshold=1.0,
            metric_name="test_metric"
        )

        # Trigger alert
        metrics_collector.set_gauge("test_metric", 5.0)
        metrics_collector.evaluate_alerts()

        # Check callback was called
        assert len(callback_called) == 1
        assert callback_called[0].name == "Callback Test"


class TestMetricsSummary:
    """Test metrics summary functionality."""

    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        # Clear all metrics
        metrics_collector.counters.clear()
        metrics_collector.gauges.clear()
        metrics_collector.histograms.clear()
        metrics_collector.timers.clear()
        metrics_collector.alerts.clear()

        # Add some test data
        metrics_collector.increment_counter("test_counter", 10)
        metrics_collector.set_gauge("test_gauge", 5.0)
        metrics_collector.record_histogram("test_histogram", 1.5)
        metrics_collector.record_timer("test_timer", 2.0)

        # Create alerts
        metrics_collector.create_alert(
            "Test Alert 1",
            "Description 1",
            AlertSeverity.LOW,
            "gt",
            1.0,
            "test_metric"
        )
        metrics_collector.create_alert(
            "Test Alert 2",
            "Description 2",
            AlertSeverity.HIGH,
            "lt",
            10.0,
            "test_metric2"
        )

        # Get summary
        summary = metrics_collector.get_metrics_summary()

        # Check summary structure
        assert "timestamp" in summary
        assert "metrics_count" in summary
        assert "total_data_points" in summary
        assert "alerts" in summary

        # Check metrics count
        assert summary["metrics_count"]["counters"] == 1
        assert summary["metrics_count"]["gauges"] == 1
        assert summary["metrics_count"]["histograms"] == 1
        assert summary["metrics_count"]["timers"] == 1

        # Check alerts count
        assert summary["alerts"]["total"] == 2
        assert summary["alerts"]["enabled"] == 2
        assert summary["alerts"]["triggered_recently"] == 0


@pytest.mark.asyncio
class TestMetricsCollectionStartStop:
    """Test metrics collection start/stop functionality."""

    async def test_start_collection(self):
        """Test starting metrics collection."""
        # Stop any existing collection
        await metrics_collector.stop_collection()
        assert metrics_collector._running is False

        # Start collection
        metrics_collector.start_collection(interval=1.0)
        assert metrics_collector._running is True
        assert metrics_collector._collection_task is not None

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop collection
        await metrics_collector.stop_collection()
        assert metrics_collector._running is False

    async def test_stop_collection(self):
        """Test stopping metrics collection."""
        # Start collection first
        metrics_collector.start_collection(interval=1.0)
        assert metrics_collector._running is True

        # Stop collection
        await metrics_collector.stop_collection()
        assert metrics_collector._running is False
        assert metrics_collector._collection_task is None