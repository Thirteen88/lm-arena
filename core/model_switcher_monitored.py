"""
LM Arena - Enhanced Model Switching System with Monitoring

Advanced model switching with comprehensive monitoring, metrics collection,
and performance analytics integration.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import statistics
from datetime import datetime

import structlog

from core.agent import ModelInterface, GenerationRequest, GenerationResponse
from monitoring.metrics import metrics_collector
from monitoring.model_monitor import model_monitor

logger = structlog.get_logger(__name__)


class SwitchingStrategy(str, Enum):
    """Model switching strategies"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    PRIORITY_BASED = "priority_based"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class ModelStatus(str, Enum):
    """Model operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class ModelMetrics:
    """Enhanced performance metrics for a model with monitoring integration"""
    model_name: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    status: ModelStatus = ModelStatus.HEALTHY
    priority: int = 1
    cost_per_1k_tokens: float = 0.0
    max_requests_per_minute: int = 0
    current_minute_requests: int = 0
    current_minute_start: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def is_available(self) -> bool:
        """Check if model is currently available"""
        if self.status == ModelStatus.UNHEALTHY:
            return False
        if self.status == ModelStatus.MAINTENANCE:
            return False
        if self.status == ModelStatus.RATE_LIMITED:
            return self.current_minute_requests < self.max_requests_per_minute
        return True

    def update_request_metrics(self, success: bool, response_time: float,
                             tokens_used: int = 0, cost: float = 0.0):
        """Update metrics after a request"""
        now = datetime.utcnow()

        self.total_requests += 1
        self.last_request_time = now

        # Update minute counter
        if (not self.current_minute_start or
            (now - self.current_minute_start).total_seconds() >= 60):
            self.current_minute_requests = 0
            self.current_minute_start = now

        self.current_minute_requests += 1

        if success:
            self.successful_requests += 1
            self.last_success_time = now
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        # Update timing and cost metrics
        self.total_tokens_used += tokens_used
        self.total_cost += cost

        if self.total_requests > 0:
            total_time = self.average_response_time * (self.total_requests - 1) + response_time
            self.average_response_time = total_time / self.total_requests

        # Update status based on performance
        self._update_status()

        # Push to monitoring system
        self._push_metrics_to_monitoring(success, response_time, tokens_used, cost)

    def _update_status(self):
        """Update model status based on recent performance"""
        if self.consecutive_failures >= 5:
            self.status = ModelStatus.UNHEALTHY
        elif self.consecutive_failures >= 3:
            self.status = ModelStatus.DEGRADED
        elif self.success_rate < 50:
            self.status = ModelStatus.UNHEALTHY
        elif self.success_rate < 80:
            self.status = ModelStatus.DEGRADED
        elif (self.max_requests_per_minute > 0 and
              self.current_minute_requests >= self.max_requests_per_minute):
            self.status = ModelStatus.RATE_LIMITED
        else:
            self.status = ModelStatus.HEALTHY

    def _push_metrics_to_monitoring(self, success: bool, response_time: float,
                                  tokens_used: int, cost: float):
        """Push metrics to the monitoring system"""
        try:
            # Track request
            metrics_collector.increment_counter(
                "model_requests_total",
                labels={
                    "model": self.model_name,
                    "provider": self.provider,
                    "status": "success" if success else "error"
                }
            )

            # Track response time
            metrics_collector.record_histogram(
                "model_response_time_seconds",
                response_time,
                labels={"model": self.model_name, "provider": self.provider}
            )

            # Track tokens
            if tokens_used > 0:
                metrics_collector.increment_counter(
                    "model_tokens_used_total",
                    tokens_used,
                    {
                        "model": self.model_name,
                        "provider": self.provider,
                        "token_type": "total"
                    }
                )

            # Track cost
            if cost > 0:
                metrics_collector.increment_counter(
                    "model_cost_usd",
                    cost,
                    {"model": self.model_name, "provider": self.provider}
                )

        except Exception as e:
            logger.error(f"Failed to push metrics to monitoring: {e}")


class BaseSwitchingStrategy(ABC):
    """Abstract base class for model switching strategies"""

    @abstractmethod
    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select a model for the given request"""
        pass

    @abstractmethod
    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """Update strategy-specific metrics"""
        pass


class RoundRobinStrategy(BaseSwitchingStrategy):
    """Round-robin model selection"""

    def __init__(self):
        self.last_index = -1

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select next available model in round-robin order"""
        available_models = [
            name for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        self.last_index = (self.last_index + 1) % len(available_models)
        return available_models[self.last_index]

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class LoadBalancedStrategy(BaseSwitchingStrategy):
    """Load-balanced model selection based on current load"""

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select model with lowest current load"""
        available_models = [
            (name, metrics) for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        # Select model with minimum requests per minute
        selected_model = min(available_models,
                           key=lambda x: x[1].current_minute_requests)

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "load_balanced",
                "from_model": "unknown",
                "to_model": selected_model[0]
            }
        )

        return selected_model[0]

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class CostOptimizedStrategy(BaseSwitchingStrategy):
    """Cost-optimized model selection"""

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select cheapest available model"""
        available_models = [
            (name, metrics) for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        # Sort by cost per 1k tokens (cheapest first)
        available_models.sort(key=lambda x: x[1].cost_per_1k_tokens)

        selected_model = available_models[0]

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "cost_optimized",
                "from_model": "unknown",
                "to_model": selected_model[0]
            }
        )

        return selected_model[0]

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class PerformanceOptimizedStrategy(BaseSwitchingStrategy):
    """Performance-optimized model selection"""

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select best performing available model"""
        available_models = [
            (name, metrics) for name, metrics in models.items()
            if metrics.is_available and metrics.success_rate > 70
        ]

        if not available_models:
            # Fallback to any available model
            available_models = [
                (name, metrics) for name, metrics in models.items()
                if metrics.is_available
            ]

        if not available_models:
            return None

        # Score models based on success rate and response time
        def score_model(name_metrics):
            _, metrics = name_metrics
            success_weight = 0.6
            speed_weight = 0.4

            # Normalize response time (lower is better, cap at 10 seconds)
            normalized_speed = max(0, 1 - (metrics.average_response_time / 10))

            return (metrics.success_rate * success_weight +
                   normalized_speed * speed_weight)

        selected_model = max(available_models, key=score_model)

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "performance_optimized",
                "from_model": "unknown",
                "to_model": selected_model[0]
            }
        )

        return selected_model[0]

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class PriorityBasedStrategy(BaseSwitchingStrategy):
    """Priority-based model selection"""

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select highest priority available model"""
        available_models = [
            (name, metrics) for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        # Sort by priority (lower number = higher priority)
        available_models.sort(key=lambda x: x[1].priority)

        selected_model = available_models[0]

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "priority_based",
                "from_model": "unknown",
                "to_model": selected_model[0]
            }
        )

        return selected_model[0]

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class RandomStrategy(BaseSwitchingStrategy):
    """Random model selection"""

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select random available model"""
        available_models = [
            name for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        selected_model = random.choice(available_models)

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "random",
                "from_model": "unknown",
                "to_model": selected_model
            }
        )

        return selected_model

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """No strategy-specific metrics to update"""
        pass


class AdaptiveStrategy(BaseSwitchingStrategy):
    """Adaptive model selection based on performance trends"""

    def __init__(self):
        self.performance_history: Dict[str, List[float]] = {}
        self.max_history = 100

    async def select_model(self, models: Dict[str, ModelMetrics],
                         request: GenerationRequest,
                         preferred_model: Optional[str] = None) -> Optional[str]:
        """Select model using adaptive scoring"""
        available_models = [
            (name, metrics) for name, metrics in models.items()
            if metrics.is_available
        ]

        if not available_models:
            return None

        # Calculate adaptive scores
        scored_models = []
        for name, metrics in available_models:
            score = self._calculate_adaptive_score(name, metrics)
            scored_models.append((name, score))

        # Select highest scored model
        selected_model = max(scored_models, key=lambda x: x[1])

        # Track model switching decision
        metrics_collector.increment_counter(
            "model_switcher_decisions_total",
            labels={
                "strategy": "adaptive",
                "from_model": "unknown",
                "to_model": selected_model[0]
            }
        )

        return selected_model[0]

    def _calculate_adaptive_score(self, model_name: str, metrics: ModelMetrics) -> float:
        """Calculate adaptive score for a model"""
        base_score = metrics.success_rate

        # Factor in response time (faster is better)
        if metrics.average_response_time > 0:
            speed_bonus = max(0, 1 - (metrics.average_response_time / 5)) * 20
            base_score += speed_bonus

        # Factor in recent performance trends
        if model_name in self.performance_history:
            recent_performances = self.performance_history[model_name][-10:]
            if recent_performances:
                trend = statistics.mean(recent_performances)
                base_score = (base_score + trend) / 2

        # Factor in consecutive failures (penalty)
        failure_penalty = metrics.consecutive_failures * 5
        base_score -= failure_penalty

        return max(0, base_score)

    def update_metrics(self, model_name: str, metrics: ModelMetrics):
        """Update adaptive performance history"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        performance_score = metrics.success_rate
        self.performance_history[model_name].append(performance_score)

        # Keep only recent history
        if len(self.performance_history[model_name]) > self.max_history:
            self.performance_history[model_name] = self.performance_history[model_name][-self.max_history:]


class MonitoredModelSwitcher:
    """Enhanced model switcher with comprehensive monitoring"""

    def __init__(self, models: Dict[str, ModelInterface]):
        self.models = models
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.strategy = LoadBalancedStrategy()
        self.fallback_enabled = True
        self.max_retries = 3

        # Strategy mapping
        self.strategies = {
            SwitchingStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            SwitchingStrategy.LOAD_BALANCED: LoadBalancedStrategy(),
            SwitchingStrategy.COST_OPTIMIZED: CostOptimizedStrategy(),
            SwitchingStrategy.PERFORMANCE_OPTIMIZED: PerformanceOptimizedStrategy(),
            SwitchingStrategy.PRIORITY_BASED: PriorityBasedStrategy(),
            SwitchingStrategy.RANDOM: RandomStrategy(),
            SwitchingStrategy.ADAPTIVE: AdaptiveStrategy()
        }

        # Initialize model metrics
        self._initialize_model_metrics()

        # Start monitoring
        self._start_monitoring()

    def _initialize_model_metrics(self):
        """Initialize metrics for all models"""
        for name, model in self.models.items():
            self.model_metrics[name] = ModelMetrics(
                model_name=name,
                provider=model.provider,
                cost_per_1k_tokens=getattr(model.capabilities, 'cost_per_input_token', 0.001) * 1000,
                max_requests_per_minute=getattr(model.config, 'max_requests_per_minute', 0)
            )

    def _start_monitoring(self):
        """Start background monitoring tasks"""
        # Start model health checks
        asyncio.create_task(model_monitor.start_health_checks(self.models))

        # Start metrics collection
        metrics_collector.start_collection()

        logger.info("MonitoredModelSwitcher: Monitoring systems started")

    def set_strategy(self, strategy: SwitchingStrategy):
        """Change the switching strategy"""
        if strategy in self.strategies:
            self.strategy = self.strategies[strategy]
            logger.info(f"Model switching strategy changed to: {strategy.value}")

            # Track strategy change
            metrics_collector.increment_counter(
                "model_switcher_decisions_total",
                labels={
                    "strategy": "strategy_change",
                    "from_model": "unknown",
                    "to_model": strategy.value
                }
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def select_model(self, request: GenerationRequest,
                          preferred_model: Optional[str] = None,
                          excluded_models: List[str] = None) -> Tuple[Optional[ModelInterface], Optional[str]]:
        """Select best model for the request"""
        excluded_models = excluded_models or []

        # Filter out excluded models
        available_metrics = {
            name: metrics for name, metrics in self.model_metrics.items()
            if name not in excluded_models and metrics.is_available
        }

        if not available_metrics:
            logger.warning("No models available for selection")
            return None, None

        # Use strategy to select model
        selected_model_name = await self.strategy.select_model(
            available_metrics, request, preferred_model
        )

        if selected_model_name and selected_model_name in self.models:
            return self.models[selected_model_name], selected_model_name

        return None, None

    async def execute_with_switching(self, request: GenerationRequest,
                                   preferred_model: Optional[str] = None,
                                   max_retries: Optional[int] = None) -> GenerationResponse:
        """Execute request with automatic model switching on failure"""
        max_retries = max_retries or self.max_retries
        excluded_models = []
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Select model
                model, model_name = await self.select_model(
                    request, preferred_model, excluded_models
                )

                if not model:
                    raise Exception("No models available")

                logger.info(f"Attempting request with model: {model_name}")

                # Execute request
                start_time = time.time()
                response = await model.generate(request)
                generation_time = time.time() - start_time

                # Track successful request
                self._track_request(model_name, True, generation_time, response)

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed: {e}")

                # Track failed request
                self._track_request(model_name, False, 0, None, str(e))

                # Add to excluded models for next attempt
                if model_name:
                    excluded_models.append(model_name)

                # Update model status
                if model_name and model_name in self.model_metrics:
                    self.model_metrics[model_name].consecutive_failures += 1

        # All attempts failed
        error_msg = f"All {max_retries + 1} model attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _track_request(self, model_name: str, success: bool, generation_time: float,
                      response: Optional[GenerationResponse] = None,
                      error_msg: Optional[str] = None):
        """Track request metrics and update monitoring"""
        if model_name not in self.model_metrics:
            return

        metrics = self.model_metrics[model_name]

        # Extract metrics from response
        tokens_used = 0
        cost = 0.0

        if response and response.usage:
            tokens_used = sum(response.usage.values())
            # Calculate cost based on model pricing
            cost = self._calculate_cost(model_name, tokens_used)

        # Update model metrics
        metrics.update_request_metrics(success, generation_time, tokens_used, cost)

        # Track in monitoring system
        model_monitor.track_request(
            model_name=model_name,
            provider=metrics.provider,
            request=None,  # Will be filled by caller if needed
            response=response,
            error=Exception(error_msg) if error_msg else None
        )

    def _calculate_cost(self, model_name: str, tokens_used: int) -> float:
        """Calculate cost for token usage"""
        if model_name not in self.model_metrics:
            return 0.0

        metrics = self.model_metrics[model_name]
        return (tokens_used / 1000) * metrics.cost_per_1k_tokens

    def get_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for models"""
        if model_name:
            if model_name not in self.model_metrics:
                return {}
            return self.model_metrics[model_name].__dict__

        return {
            name: metrics.__dict__
            for name, metrics in self.model_metrics.items()
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including switching decisions"""
        model_metrics = self.get_model_metrics()

        # Calculate overall stats
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        total_successful = sum(m.successful_requests for m in self.model_metrics.values())
        total_cost = sum(m.total_cost for m in self.model_metrics.values())
        total_tokens = sum(m.total_tokens_used for m in self.model_metrics.values())

        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0

        return {
            "strategy": self.strategy.__class__.__name__,
            "total_models": len(self.models),
            "available_models": sum(1 for m in self.model_metrics.values() if m.is_available),
            "overall_stats": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "success_rate": overall_success_rate,
                "total_cost": total_cost,
                "total_tokens": total_tokens
            },
            "model_metrics": model_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics for monitoring"""
        return self.get_all_metrics()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models"""
        health_results = {}

        for name, model in self.models.items():
            try:
                is_healthy = await model.validate_connection()
                health_results[name] = {
                    "healthy": is_healthy,
                    "metrics": self.model_metrics[name].__dict__ if name in self.model_metrics else {}
                }
            except Exception as e:
                health_results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "metrics": self.model_metrics[name].__dict__ if name in self.model_metrics else {}
                }

        return {
            "overall_health": all(result["healthy"] for result in health_results.values()),
            "model_health": health_results,
            "timestamp": datetime.utcnow().isoformat()
        }