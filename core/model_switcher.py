"""
LM Arena - Model Switching System

Advanced model switching with load balancing, routing, and automatic failover.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import statistics

import structlog

from core.agent import ModelInterface, GenerationRequest, GenerationResponse

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
    """Performance metrics for a model"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    status: ModelStatus = ModelStatus.HEALTHY
    rate_limit_reset_time: Optional[float] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        return 100.0 - self.success_rate

    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request"""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests

    @property
    def average_tokens_per_request(self) -> float:
        """Calculate average tokens per request"""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens_used / self.total_requests

    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.response_times.append(response_time)

        # Keep only last 100 response times for rolling average
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

        self.average_response_time = statistics.mean(self.response_times)

    def record_success(self, tokens_used: int = 0, cost: float = 0.0):
        """Record a successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_tokens_used += tokens_used
        self.total_cost += cost
        self.last_used = time.time()
        self.consecutive_failures = 0

        # Update status if recovering
        if self.status in [ModelStatus.UNHEALTHY, ModelStatus.DEGRADED]:
            if self.consecutive_failures == 0 and self.success_rate > 80:
                self.status = ModelStatus.HEALTHY

    def record_failure(self, error: str):
        """Record a failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.last_error_time = time.time()
        self.consecutive_failures += 1

        # Update status based on failures
        if self.consecutive_failures >= 5:
            self.status = ModelStatus.UNHEALTHY
        elif self.consecutive_failures >= 3:
            self.status = ModelStatus.DEGRADED

    def check_rate_limit(self, reset_time: float):
        """Handle rate limiting"""
        self.status = ModelStatus.RATE_LIMITED
        self.rate_limit_reset_time = reset_time

    def is_rate_limited(self) -> bool:
        """Check if model is currently rate limited"""
        if (self.status == ModelStatus.RATE_LIMITED and
            self.rate_limit_reset_time and
            time.time() < self.rate_limit_reset_time):
            return True

        # Reset rate limit status if time has passed
        if self.status == ModelStatus.RATE_LIMITED and self.rate_limit_reset_time:
            if time.time() >= self.rate_limit_reset_time:
                self.status = ModelStatus.HEALTHY
                self.rate_limit_reset_time = None

        return False


@dataclass
class RoutingRule:
    """Rule for routing requests to specific models"""
    name: str
    condition: Callable[[GenerationRequest], bool]
    preferred_models: List[str]
    priority: int = 0
    enabled: bool = True


class ModelSwitcher:
    """Advanced model switching and routing system"""

    def __init__(self, models: Dict[str, ModelInterface]):
        self.models = models
        self.metrics: Dict[str, ModelMetrics] = {
            name: ModelMetrics(model_name=name) for name in models.keys()
        }
        self.routing_rules: List[RoutingRule] = []
        self.strategy = SwitchingStrategy.LOAD_BALANCED
        self.round_robin_index = 0
        self._lock = asyncio.Lock()

    async def select_model(
        self,
        request: GenerationRequest,
        preferred_model: Optional[str] = None,
        excluded_models: Optional[List[str]] = None
    ) -> Tuple[Optional[ModelInterface], Optional[str]]:
        """Select the best model for a request"""

        async with self._lock:
            # If specific model is requested, try to use it
            if preferred_model:
                if preferred_model in self.models:
                    model = self.models[preferred_model]
                    if await self._is_model_available(preferred_model):
                        return model, preferred_model
                    logger.warning("Preferred model not available", model=preferred_model)

            # Apply routing rules
            selected_model = await self._apply_routing_rules(request, excluded_models)
            if selected_model:
                return selected_model, selected_model.name

            # Apply strategy-based selection
            selected_model = await self._apply_strategy(request, excluded_models)
            if selected_model:
                return selected_model, selected_model.name

            # Fallback to any available model
            available_models = [
                (name, model) for name, model in self.models.items()
                if await self._is_model_available(name) and
                (not excluded_models or name not in excluded_models)
            ]

            if available_models:
                return available_models[0][1], available_models[0][0]

            logger.error("No models available for request")
            return None, None

    async def _apply_routing_rules(
        self,
        request: GenerationRequest,
        excluded_models: Optional[List[str]] = None
    ) -> Optional[ModelInterface]:
        """Apply routing rules to select model"""
        available_rules = [
            rule for rule in self.routing_rules
            if rule.enabled and rule.condition(request)
        ]

        if not available_rules:
            return None

        # Sort by priority (higher priority first)
        available_rules.sort(key=lambda r: r.priority, reverse=True)

        for rule in available_rules:
            for model_name in rule.preferred_models:
                if (model_name in self.models and
                    await self._is_model_available(model_name) and
                    (not excluded_models or model_name not in excluded_models)):

                    logger.info("Model selected by routing rule",
                              model=model_name, rule=rule.name)
                    return self.models[model_name]

        return None

    async def _apply_strategy(
        self,
        request: GenerationRequest,
        excluded_models: Optional[List[str]] = None
    ) -> Optional[ModelInterface]:
        """Apply the current switching strategy"""
        available_models = [
            (name, model) for name, model in self.models.items()
            if await self._is_model_available(name) and
            (not excluded_models or name not in excluded_models)
        ]

        if not available_models:
            return None

        if self.strategy == SwitchingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_models)
        elif self.strategy == SwitchingStrategy.LOAD_BALANCED:
            return self._load_balanced_selection(available_models)
        elif self.strategy == SwitchingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_selection(available_models)
        elif self.strategy == SwitchingStrategy.PERFORMANCE_OPTIMIZED:
            return self._performance_optimized_selection(available_models)
        elif self.strategy == SwitchingStrategy.PRIORITY_BASED:
            return self._priority_based_selection(available_models)
        elif self.strategy == SwitchingStrategy.RANDOM:
            return self._random_selection(available_models)
        elif self.strategy == SwitchingStrategy.ADAPTIVE:
            return self._adaptive_selection(available_models, request)
        else:
            return available_models[0][1]

    def _round_robin_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Round-robin model selection"""
        model = available_models[self.round_robin_index % len(available_models)][1]
        self.round_robin_index += 1
        return model

    def _load_balanced_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Select model based on current load (fewest requests)"""
        return min(
            available_models,
            key=lambda x: self.metrics[x[0]].total_requests
        )[1]

    def _cost_optimized_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Select cheapest model"""
        return min(
            available_models,
            key=lambda x: x[1].capabilities.cost_per_output_token
        )[1]

    def _performance_optimized_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Select fastest model"""
        return min(
            available_models,
            key=lambda x: self.metrics[x[0]].average_response_time or float('inf')
        )[1]

    def _priority_based_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Select model based on configured priority"""
        return max(
            available_models,
            key=lambda x: x[1].config.priority
        )[1]

    def _random_selection(self, available_models: List[Tuple[str, ModelInterface]]) -> ModelInterface:
        """Random model selection"""
        return random.choice(available_models)[1]

    def _adaptive_selection(
        self,
        available_models: List[Tuple[str, ModelInterface]],
        request: GenerationRequest
    ) -> ModelInterface:
        """Adaptive selection based on request characteristics and model performance"""
        # Score models based on multiple factors
        scored_models = []

        for name, model in available_models:
            metrics = self.metrics[name]
            score = 0.0

            # Success rate (30% weight)
            score += (metrics.success_rate / 100) * 0.3

            # Response time (25% weight) - lower is better
            if metrics.average_response_time > 0:
                # Normalize response times (faster gets higher score)
                max_response_time = max(
                    (m.average_response_time or float('inf'))
                    for m in self.metrics.values()
                )
                normalized_time = 1 - (metrics.average_response_time / max_response_time)
                score += normalized_time * 0.25

            # Cost (20% weight) - lower is better
            if model.capabilities.cost_per_output_token > 0:
                max_cost = max(
                    m.capabilities.cost_per_output_token
                    for m in self.models.values()
                )
                normalized_cost = 1 - (model.capabilities.cost_per_output_token / max_cost)
                score += normalized_cost * 0.2

            # Availability (15% weight)
            availability_score = 1.0 if metrics.status == ModelStatus.HEALTHY else 0.5
            score += availability_score * 0.15

            # Recent usage (10% weight) - prefer less used models
            max_requests = max(m.total_requests for m in self.metrics.values())
            if max_requests > 0:
                usage_score = 1 - (metrics.total_requests / max_requests)
                score += usage_score * 0.1

            scored_models.append((score, model))

        # Return model with highest score
        return max(scored_models, key=lambda x: x[0])[1]

    async def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available for requests"""
        if model_name not in self.models:
            return False

        model = self.models[model_name]
        metrics = self.metrics[model_name]

        # Check if model is active
        if not model.config.is_active:
            return False

        # Check if model is rate limited
        if metrics.is_rate_limited():
            return False

        # Check if model is in maintenance
        if metrics.status == ModelStatus.MAINTENANCE:
            return False

        # Check if model is unhealthy
        if metrics.status == ModelStatus.UNHEALTHY:
            # Allow some requests to test if it's recovered
            if random.random() < 0.1:  # 10% chance to try unhealthy model
                return True
            return False

        return True

    async def execute_with_switching(
        self,
        request: GenerationRequest,
        preferred_model: Optional[str] = None,
        excluded_models: Optional[List[str]] = None,
        max_retries: int = 2
    ) -> GenerationResponse:
        """Execute request with automatic model switching on failure"""

        last_error = None
        used_models = []

        for attempt in range(max_retries + 1):
            try:
                # Select model
                model, model_name = await self.select_model(
                    request,
                    preferred_model,
                    excluded_models + used_models
                )

                if not model:
                    raise Exception("No models available for request")

                # Execute request
                start_time = time.time()
                response = await model.generate(request)
                response_time = time.time() - start_time

                # Update metrics
                self.metrics[model_name].record_success(
                    tokens_used=response.usage.get("total_tokens", 0),
                    cost=self._calculate_cost(model, response.usage)
                )
                self.metrics[model_name].update_response_time(response_time)

                # Add model info to response
                response.metadata["selected_model"] = model_name
                response.metadata["selection_strategy"] = self.strategy.value
                if attempt > 0:
                    response.metadata["retry_count"] = attempt

                logger.info("Request completed successfully",
                          model=model_name,
                          attempt=attempt + 1,
                          response_time=response_time)

                return response

            except Exception as e:
                last_error = e

                # Record failure for the model that failed
                if model_name:
                    self.metrics[model_name].record_failure(str(e))
                    used_models.append(model_name)

                    # Special handling for rate limiting
                    if "rate limit" in str(e).lower():
                        # Estimate rate limit reset time (typically 1 minute)
                        reset_time = time.time() + 60
                        self.metrics[model_name].check_rate_limit(reset_time)

                logger.warning("Request failed, attempting switch",
                             attempt=attempt + 1,
                             error=str(e))

                # Don't wait on last attempt
                if attempt < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

        # All attempts failed
        raise Exception(f"All model attempts failed. Last error: {str(last_error)}")

    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)
        logger.info("Routing rule added", rule=rule.name)

    def remove_routing_rule(self, rule_name: str) -> bool:
        """Remove a routing rule"""
        for i, rule in enumerate(self.routing_rules):
            if rule.name == rule_name:
                removed_rule = self.routing_rules.pop(i)
                logger.info("Routing rule removed", rule=rule_name)
                return True
        return False

    def set_strategy(self, strategy: SwitchingStrategy):
        """Set the switching strategy"""
        self.strategy = strategy
        logger.info("Switching strategy updated", strategy=strategy.value)

    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model"""
        return self.metrics.get(model_name)

    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models"""
        return self.metrics.copy()

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_successful = sum(m.successful_requests for m in self.metrics.values())
        total_cost = sum(m.total_cost for m in self.metrics.values())
        total_tokens = sum(m.total_tokens_used for m in self.metrics.values())

        return {
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_requests - total_successful,
            "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "total_cost": total_cost,
            "total_tokens_used": total_tokens,
            "active_models": len([m for m in self.models.values() if m.config.is_active]),
            "healthy_models": len([
                m for m in self.metrics.values()
                if m.status == ModelStatus.HEALTHY
            ]),
            "current_strategy": self.strategy.value,
            "routing_rules_count": len(self.routing_rules)
        }

    def reset_metrics(self, model_name: Optional[str] = None):
        """Reset metrics for a model or all models"""
        if model_name:
            if model_name in self.metrics:
                self.metrics[model_name] = ModelMetrics(model_name=model_name)
                logger.info("Metrics reset for model", model=model_name)
        else:
            for name in self.metrics:
                self.metrics[name] = ModelMetrics(model_name=name)
            logger.info("All metrics reset")

    def _calculate_cost(self, model: ModelInterface, usage: Dict[str, int]) -> float:
        """Calculate generation cost"""
        input_cost = usage.get("input_tokens", 0) * model.capabilities.cost_per_input_token
        output_cost = usage.get("output_tokens", 0) * model.capabilities.cost_per_output_token
        return input_cost + output_cost