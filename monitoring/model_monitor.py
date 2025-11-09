"""
LM Arena - Model Performance Monitoring

Integration with ModelSwitcher to provide comprehensive monitoring
of model performance, costs, and health metrics.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from core.agent import ModelInterface, GenerationRequest, GenerationResponse
from core.model_switcher import ModelSwitcher, SwitchingStrategy
from .metrics import metrics_collector, Alert, AlertSeverity

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitors individual model performance and health"""

    def __init__(self):
        self.model_health: Dict[str, Dict[str, Any]] = {}
        self.model_costs: Dict[str, Dict[str, float]] = {}
        self.model_errors: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}

        self.health_check_interval = 60  # seconds
        self.error_tracking_window = timedelta(hours=1)
        self.performance_window = timedelta(hours=24)

        # Initialize model alerts
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Setup default alerts for model monitoring"""

        # High error rate alert
        metrics_collector.create_alert(
            name="High Model Error Rate",
            description="Model error rate exceeds 10%",
            severity=AlertSeverity.HIGH,
            condition="gt",
            threshold=10.0,
            metric_name="model_error_rate"
        )

        # Slow response time alert
        metrics_collector.create_alert(
            name="Slow Model Response",
            description="Model response time exceeds 30 seconds",
            severity=AlertSeverity.MEDIUM,
            condition="gt",
            threshold=30.0,
            metric_name="model_response_time_seconds",
            labels={"le": "30.0"}
        )

        # High cost alert
        metrics_collector.create_alert(
            name="High Model Cost",
            description="Model cost exceeds $10 in 1 hour",
            severity=AlertSeverity.MEDIUM,
            condition="gt",
            threshold=10.0,
            metric_name="model_cost_usd"
        )

        # Model unavailable alert
        metrics_collector.create_alert(
            name="Model Unavailable",
            description="Model health check failed",
            severity=AlertSeverity.CRITICAL,
            condition="lt",
            threshold=1.0,
            metric_name="agent_status",
            labels={"component": "model"}
        )

    def track_request(self, model_name: str, provider: str, request: GenerationRequest,
                     response: Optional[GenerationResponse] = None, error: Optional[Exception] = None):
        """Track a model request and its outcome"""

        labels = {"model": model_name, "provider": provider}
        timestamp = datetime.utcnow()

        if error:
            # Track failed request
            metrics_collector.increment_counter(
                "model_requests_total",
                labels={**labels, "status": "error"}
            )

            # Record error
            if model_name not in self.model_errors:
                self.model_errors[model_name] = []

            self.model_errors[model_name].append({
                "timestamp": timestamp,
                "error": str(error),
                "error_type": type(error).__name__
            })

            # Clean old errors
            cutoff = timestamp - self.error_tracking_window
            self.model_errors[model_name] = [
                e for e in self.model_errors[model_name]
                if e["timestamp"] > cutoff
            ]

            logger.error(f"Model {model_name} error: {error}")

        elif response:
            # Track successful request
            metrics_collector.increment_counter(
                "model_requests_total",
                labels={**labels, "status": "success"}
            )

            # Track response time
            if response.generation_time:
                metrics_collector.record_histogram(
                    "model_response_time_seconds",
                    response.generation_time,
                    labels
                )

            # Track token usage
            usage = response.usage or {}
            if "prompt_tokens" in usage:
                metrics_collector.increment_counter(
                    "model_tokens_used_total",
                    usage["prompt_tokens"],
                    {**labels, "token_type": "prompt"}
                )
            if "completion_tokens" in usage:
                metrics_collector.increment_counter(
                    "model_tokens_used_total",
                    usage["completion_tokens"],
                    {**labels, "token_type": "completion"}
                )

            # Track cost
            cost = self._calculate_cost(model_name, provider, usage)
            if cost > 0:
                metrics_collector.increment_counter(
                    "model_cost_usd",
                    cost,
                    labels
                )

                # Update cost tracking
                if model_name not in self.model_costs:
                    self.model_costs[model_name] = {"hourly": 0.0, "daily": 0.0}

                self.model_costs[model_name]["hourly"] += cost
                self.model_costs[model_name]["daily"] += cost

            # Record performance
            self._record_performance(model_name, response)

            # Update model health
            self._update_model_health(model_name, True, timestamp)

    def _calculate_cost(self, model_name: str, provider: str, usage: Dict[str, Any]) -> float:
        """Calculate cost based on token usage and model pricing"""

        # Simplified cost calculation - in real implementation,
        # this would use actual pricing from model configurations
        pricing = {
            "openai": {
                "gpt-4": {"input": 0.00003, "output": 0.00006},
                "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
                "gpt-3.5-turbo": {"input": 0.000001, "output": 0.000002}
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.000015, "output": 0.000075},
                "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
                "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125}
            }
        }

        try:
            provider_pricing = pricing.get(provider, {})
            model_pricing = provider_pricing.get(model_name, {"input": 0.0, "output": 0.0})

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            cost = (input_tokens * model_pricing["input"] +
                   output_tokens * model_pricing["output"])

            return cost

        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0

    def _record_performance(self, model_name: str, response: GenerationResponse):
        """Record performance metrics for a model"""

        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        performance_point = {
            "timestamp": datetime.utcnow(),
            "generation_time": response.generation_time,
            "tokens_used": sum(response.usage.values()) if response.usage else 0,
            "finish_reason": response.finish_reason,
            "content_length": len(response.content) if response.content else 0
        }

        self.performance_history[model_name].append(performance_point)

        # Keep only recent performance data
        cutoff = datetime.utcnow() - self.performance_window
        self.performance_history[model_name] = [
            p for p in self.performance_history[model_name]
            if p["timestamp"] > cutoff
        ]

    def _update_model_health(self, model_name: str, is_healthy: bool, timestamp: datetime):
        """Update health status for a model"""

        if model_name not in self.model_health:
            self.model_health[model_name] = {
                "status": "unknown",
                "last_check": timestamp,
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "uptime_percentage": 100.0,
                "total_requests": 0,
                "successful_requests": 0
            }

        health = self.model_health[model_name]
        health["last_check"] = timestamp
        health["total_requests"] += 1

        if is_healthy:
            health["consecutive_successes"] += 1
            health["consecutive_failures"] = 0
            health["successful_requests"] += 1
        else:
            health["consecutive_failures"] += 1
            health["consecutive_successes"] = 0

        # Calculate uptime percentage
        if health["total_requests"] > 0:
            health["uptime_percentage"] = (
                health["successful_requests"] / health["total_requests"] * 100
            )

        # Update status
        if health["consecutive_failures"] >= 5:
            health["status"] = "unhealthy"
        elif health["consecutive_failures"] >= 2:
            health["status"] = "degraded"
        elif health["consecutive_successes"] >= 10:
            health["status"] = "healthy"
        else:
            health["status"] = "recovering"

        # Update agent status metric
        metrics_collector.set_gauge(
            "agent_status",
            1.0 if health["status"] == "healthy" else 0.0,
            {"component": f"model_{model_name}"}
        )

    def get_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive metrics for models"""

        if model_name:
            return self._get_single_model_metrics(model_name)

        # Get metrics for all models
        all_metrics = {}
        for model in self.model_health.keys():
            all_metrics[model] = self._get_single_model_metrics(model)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "models": all_metrics,
            "summary": self._get_metrics_summary()
        }

    def _get_single_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get metrics for a single model"""

        health = self.model_health.get(model_name, {})
        costs = self.model_costs.get(model_name, {})
        errors = self.model_errors.get(model_name, [])
        performance = self.performance_history.get(model_name, [])

        # Calculate error rate
        recent_requests = metrics_collector.get_metric_history(
            "model_requests_total",
            since=datetime.utcnow() - timedelta(hours=1),
            labels={"model": model_name}
        )

        error_count = len([r for r in recent_requests if r.labels.get("status") == "error"])
        total_count = len(recent_requests)
        error_rate = (error_count / total_count * 100) if total_count > 0 else 0.0

        # Update error rate metric
        metrics_collector.set_gauge(
            "model_error_rate",
            error_rate,
            {"model": model_name}
        )

        # Calculate average response time
        response_times = metrics_collector.get_metric_history(
            "model_response_time_seconds",
            since=datetime.utcnow() - timedelta(hours=1),
            labels={"model": model_name}
        )

        avg_response_time = (
            sum(p.value for p in response_times) / len(response_times)
        ) if response_times else 0.0

        # Calculate performance statistics
        performance_stats = {}
        if performance:
            generation_times = [p["generation_time"] for p in performance if p["generation_time"]]
            if generation_times:
                performance_stats = {
                    "avg_generation_time": sum(generation_times) / len(generation_times),
                    "min_generation_time": min(generation_times),
                    "max_generation_time": max(generation_times),
                    "total_requests": len(performance)
                }

        return {
            "model_name": model_name,
            "health": health,
            "costs": costs,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "recent_errors": len([e for e in errors
                               if e["timestamp"] > datetime.utcnow() - timedelta(hours=1)]),
            "performance": performance_stats
        }

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary"""

        total_models = len(self.model_health)
        healthy_models = sum(1 for h in self.model_health.values()
                           if h.get("status") == "healthy")
        total_requests = sum(h.get("total_requests", 0)
                           for h in self.model_health.values())
        total_cost = sum(c.get("hourly", 0.0)
                        for c in self.model_costs.values())

        return {
            "total_models": total_models,
            "healthy_models": healthy_models,
            "unhealthy_models": total_models - healthy_models,
            "total_requests_hour": total_requests,
            "total_cost_hour": total_cost,
            "health_percentage": (healthy_models / total_models * 100) if total_models > 0 else 0
        }

    async def start_health_checks(self, models: Dict[str, ModelInterface]):
        """Start periodic health checks for all models"""

        async def _health_check_loop():
            while True:
                try:
                    for model_name, model in models.items():
                        try:
                            # Validate connection
                            is_healthy = await model.validate_connection()
                            self._update_model_health(model_name, is_healthy, datetime.utcnow())

                        except Exception as e:
                            logger.error(f"Health check failed for {model_name}: {e}")
                            self._update_model_health(model_name, False, datetime.utcnow())

                    await asyncio.sleep(self.health_check_interval)

                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    await asyncio.sleep(self.health_check_interval)

        # Start health check task
        asyncio.create_task(_health_check_loop())

    def cleanup_old_data(self):
        """Clean up old monitoring data"""

        cutoff_time = datetime.utcnow() - timedelta(days=7)

        # Clean old errors
        for model_name in list(self.model_errors.keys()):
            self.model_errors[model_name] = [
                e for e in self.model_errors[model_name]
                if e["timestamp"] > cutoff_time
            ]
            if not self.model_errors[model_name]:
                del self.model_errors[model_name]

        # Clean old performance data
        for model_name in list(self.performance_history.keys()):
            self.performance_history[model_name] = [
                p for p in self.performance_history[model_name]
                if p["timestamp"] > cutoff_time
            ]
            if not self.performance_history[model_name]:
                del self.performance_history[model_name]

        # Reset cost counters periodically
        for model_name in self.model_costs:
            self.model_costs[model_name]["hourly"] = 0.0


# Global model monitor instance
model_monitor = ModelPerformanceMonitor()