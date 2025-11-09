"""
LM Arena - Advanced Metrics Collection System

Comprehensive metrics collection inspired by Agent 21 monitoring capabilities.
Provides real-time analytics for model performance, costs, and system health.
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we can collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = ""


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Evaluation condition
    threshold: float
    metric_name: str
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class MetricsCollector:
    """Advanced metrics collection system for LM Arena"""

    def __init__(self, max_history_points: int = 10000):
        self.max_history_points = max_history_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)

        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        self._lock = asyncio.Lock()
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None

        # Initialize default metrics
        self._initialize_default_metrics()

    def _initialize_default_metrics(self):
        """Initialize default LM Arena metrics"""

        # Model performance metrics
        self.define_metric(
            "model_requests_total",
            "Total number of requests per model",
            MetricType.COUNTER,
            ["model", "provider", "status"]
        )

        self.define_metric(
            "model_response_time_seconds",
            "Model response time in seconds",
            MetricType.HISTOGRAM,
            ["model", "provider"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.define_metric(
            "model_tokens_used_total",
            "Total tokens used per model",
            MetricType.COUNTER,
            ["model", "provider", "token_type"]
        )

        self.define_metric(
            "model_cost_usd",
            "Cost in USD per model",
            MetricType.COUNTER,
            ["model", "provider"]
        )

        self.define_metric(
            "model_error_rate",
            "Model error rate (percentage)",
            MetricType.GAUGE,
            ["model", "provider"]
        )

        # System metrics
        self.define_metric(
            "active_conversations",
            "Number of active conversations",
            MetricType.GAUGE
        )

        self.define_metric(
            "queue_size",
            "Current request queue size",
            MetricType.GAUGE
        )

        self.define_metric(
            "memory_usage_bytes",
            "Memory usage in bytes",
            MetricType.GAUGE
        )

        # Agent metrics
        self.define_metric(
            "agent_status",
            "Agent operational status (1=healthy, 0=unhealthy)",
            MetricType.GAUGE,
            ["component"]
        )

        self.define_metric(
            "model_switcher_decisions_total",
            "Total model switching decisions",
            MetricType.COUNTER,
            ["strategy", "from_model", "to_model"]
        )

        # API metrics
        self.define_metric(
            "http_requests_total",
            "Total HTTP requests",
            MetricType.COUNTER,
            ["method", "endpoint", "status"]
        )

        self.define_metric(
            "http_request_duration_seconds",
            "HTTP request duration",
            MetricType.HISTOGRAM,
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )

    def define_metric(self, name: str, description: str, metric_type: MetricType,
                     labels: List[str] = None, buckets: List[float] = None, unit: str = ""):
        """Define a new metric"""
        self.metric_definitions[name] = MetricDefinition(
            name=name,
            description=description,
            metric_type=metric_type,
            labels=labels or [],
            buckets=buckets,
            unit=unit
        )

    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        self._record_metric_point(name, float(value), labels)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self._record_metric_point(name, value, labels)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

        self._record_metric_point(name, value, labels)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer duration"""
        key = self._make_key(name, labels)
        self.timers[key].append(duration)

        # Keep only recent values
        if len(self.timers[key]) > 1000:
            self.timers[key] = self.timers[key][-1000:]

        self._record_metric_point(name, duration, labels)

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a key for metric storage"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

    def _record_metric_point(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric point in history"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {},
            metadata={"metric_type": self.metric_definitions.get(name, MetricDefinition("", "", MetricType.GAUGE)).metric_type.value}
        )

        self.metrics[name].append(point)

    def get_metric_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current value of a metric"""
        key = self._make_key(name, labels)

        if name in self.counters:
            return self.counters.get(key, 0.0)
        elif name in self.gauges:
            return self.gauges.get(key, 0.0)
        elif name in self.histograms and key in self.histograms:
            values = self.histograms[key]
            return sum(values) / len(values) if values else 0.0
        elif name in self.timers and key in self.timers:
            values = self.timers[key]
            return sum(values) / len(values) if values else 0.0

        return None

    def get_metric_history(self, name: str, since: Optional[datetime] = None,
                          labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get historical metric points"""
        if name not in self.metrics:
            return []

        points = list(self.metrics[name])

        # Filter by time
        if since:
            points = [p for p in points if p.timestamp >= since]

        # Filter by labels
        if labels:
            points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]

        return points

    def create_alert(self, name: str, description: str, severity: AlertSeverity,
                    condition: str, threshold: float, metric_name: str,
                    labels: Dict[str, str] = None) -> str:
        """Create a new alert"""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=condition,
            threshold=threshold,
            metric_name=metric_name,
            labels=labels or {}
        )

        self.alerts[alert_id] = alert
        return alert_id

    def evaluate_alerts(self):
        """Evaluate all alerts and trigger if needed"""
        current_time = datetime.utcnow()

        for alert in self.alerts.values():
            if not alert.enabled:
                continue

            try:
                current_value = self.get_metric_value(alert.metric_name, alert.labels)
                if current_value is None:
                    continue

                triggered = False

                if alert.condition == "gt":
                    triggered = current_value > alert.threshold
                elif alert.condition == "lt":
                    triggered = current_value < alert.threshold
                elif alert.condition == "eq":
                    triggered = abs(current_value - alert.threshold) < 0.001
                elif alert.condition == "gte":
                    triggered = current_value >= alert.threshold
                elif alert.condition == "lte":
                    triggered = current_value <= alert.threshold

                if triggered:
                    alert.last_triggered = current_time
                    alert.trigger_count += 1

                    logger.warning(
                        f"Alert triggered: {alert.name} - {alert.description} "
                        f"(Value: {current_value}, Threshold: {alert.threshold})"
                    )

                    # Call alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

            except Exception as e:
                logger.error(f"Error evaluating alert {alert.name}: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Export metrics definitions
        for name, definition in self.metric_definitions.items():
            # HELP line
            lines.append(f"# HELP {name} {definition.description}")

            # TYPE line
            prometheus_type = {
                MetricType.COUNTER: "counter",
                MetricType.GAUGE: "gauge",
                MetricType.HISTOGRAM: "histogram",
                MetricType.TIMER: "histogram"  # Timers are histograms in Prometheus
            }.get(definition.metric_type, "untyped")

            lines.append(f"# TYPE {name} {prometheus_type}")

            # Export values
            if definition.metric_type == MetricType.COUNTER:
                for key, value in self.counters.items():
                    if key.startswith(name):
                        labels = self._parse_labels_from_key(key, name)
                        label_str = self._format_labels(labels)
                        lines.append(f"{name}{label_str} {value}")

            elif definition.metric_type == MetricType.GAUGE:
                for key, value in self.gauges.items():
                    if key.startswith(name):
                        labels = self._parse_labels_from_key(key, name)
                        label_str = self._format_labels(labels)
                        lines.append(f"{name}{label_str} {value}")

            elif definition.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                # Export histogram buckets
                histogram_data = {}
                if name in self.histograms:
                    for key, values in self.histograms.items():
                        if key.startswith(name):
                            labels = self._parse_labels_from_key(key, name)
                            label_str = self._format_labels(labels)
                            histogram_data[f"{name}_bucket{label_str}"] = values

                if name in self.timers:
                    for key, values in self.timers.items():
                        if key.startswith(name):
                            labels = self._parse_labels_from_key(key, name)
                            label_str = self._format_labels(labels)
                            histogram_data[f"{name}_bucket{label_str}"] = values

                for bucket_key, values in histogram_data.items():
                    if definition.buckets:
                        for bucket in definition.buckets:
                            count = sum(1 for v in values if v <= bucket)
                            bucket_labels = self._parse_labels_from_key(bucket_key, f"{name}_bucket")
                            bucket_labels["le"] = str(bucket)
                            label_str = self._format_labels(bucket_labels)
                            lines.append(f"{bucket_key}{label_str} {count}")

                    # Add +Inf bucket
                    bucket_labels = self._parse_labels_from_key(bucket_key, f"{name}_bucket")
                    bucket_labels["le"] = "+Inf"
                    label_str = self._format_labels(bucket_labels)
                    lines.append(f"{bucket_key}{label_str} {len(values)}")

                    # Add count and sum
                    base_name = bucket_key.replace("_bucket", "")
                    lines.append(f"{base_name}_count{label_str.replace('le=\"+Inf\"', '')} {len(values)}")
                    lines.append(f"{base_name}_sum{label_str.replace('le=\"+Inf\"', '')} {sum(values)}")

        return "\n".join(lines)

    def _parse_labels_from_key(self, key: str, metric_name: str) -> Dict[str, str]:
        """Parse labels from a metric key"""
        if key == metric_name:
            return {}

        if "[" in key and key.endswith("]"):
            label_part = key[key.index("[") + 1:-1]
            labels = {}
            for pair in label_part.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip()
            return labels

        return {}

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels:
            return ""

        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_count": {
                "counters": len(self.counters),
                "gauges": len(self.gauges),
                "histograms": len(self.histograms),
                "timers": len(self.timers)
            },
            "total_data_points": sum(len(points) for points in self.metrics.values()),
            "alerts": {
                "total": len(self.alerts),
                "enabled": sum(1 for a in self.alerts.values() if a.enabled),
                "triggered_recently": sum(1 for a in self.alerts.values()
                                        if a.last_triggered and
                                        a.last_triggered > datetime.utcnow() - timedelta(hours=1))
            }
        }

        # Add model-specific metrics
        model_metrics = {}
        for name in self.metric_definitions:
            if "model" in name:
                current_value = self.get_metric_value(name)
                if current_value is not None:
                    model_metrics[name] = current_value

        if model_metrics:
            summary["model_metrics"] = model_metrics

        return summary

    def start_collection(self, interval: float = 10.0):
        """Start automatic metrics collection and alert evaluation"""
        if self._running:
            return

        self._running = True

        async def _collection_loop():
            while self._running:
                try:
                    # Evaluate alerts
                    self.evaluate_alerts()

                    # Clean old data points
                    cutoff_time = datetime.utcnow() - timedelta(hours=24)
                    for name, points in self.metrics.items():
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()

                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    await asyncio.sleep(interval)

        self._collection_task = asyncio.create_task(_collection_loop())

    async def stop_collection(self):
        """Stop metrics collection"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None


# Global metrics collector instance
metrics_collector = MetricsCollector()