"""
LM Arena - Monitoring API Endpoints

REST API endpoints for accessing metrics, health status, and analytics.
Compatible with Prometheus and other monitoring systems.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import PlainTextResponse
import json

from .metrics import metrics_collector
from .model_monitor import model_monitor
from api.schemas import HealthResponse, StatsResponse

# Create monitoring router
monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@monitoring_router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Export metrics in Prometheus format"""
    try:
        return metrics_collector.get_prometheus_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@monitoring_router.get("/health")
async def get_monitoring_health():
    """Get monitoring system health status"""
    try:
        metrics_summary = metrics_collector.get_metrics_summary()
        model_metrics = model_monitor.get_model_metrics()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_collector": {
                "running": metrics_collector._running,
                "total_metrics": metrics_summary["total_data_points"],
                "alerts_enabled": metrics_summary["alerts"]["enabled"],
                "alerts_triggered": metrics_summary["alerts"]["triggered_recently"]
            },
            "model_monitor": {
                "models_tracked": len(model_monitor.model_health),
                "healthy_models": model_metrics["summary"]["healthy_models"],
                "total_requests_hour": model_metrics["summary"]["total_requests_hour"]
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@monitoring_router.get("/models/metrics")
async def get_models_metrics(
    model_name: Optional[str] = Query(None, description="Get metrics for specific model")
):
    """Get comprehensive model performance metrics"""
    try:
        return model_monitor.get_model_metrics(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")


@monitoring_router.get("/models/{model_name}/health")
async def get_model_health(model_name: str):
    """Get detailed health information for a specific model"""
    try:
        health_info = model_monitor.model_health.get(model_name)
        if not health_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        return {
            "model_name": model_name,
            "health": health_info,
            "recent_errors": model_monitor.model_errors.get(model_name, [])[-10:],  # Last 10 errors
            "costs": model_monitor.model_costs.get(model_name, {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model health: {str(e)}")


@monitoring_router.get("/models/{model_name}/performance")
async def get_model_performance(
    model_name: str,
    hours: int = Query(24, description="Hours of performance data to retrieve"),
    granularity: str = Query("minute", description="Data granularity: minute, hour")
):
    """Get detailed performance data for a model"""
    try:
        performance_data = model_monitor.performance_history.get(model_name, [])

        if not performance_data:
            raise HTTPException(status_code=404, detail=f"No performance data for model {model_name}")

        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filtered_data = [
            p for p in performance_data
            if p["timestamp"] > cutoff_time
        ]

        # Aggregate based on granularity
        if granularity == "hour":
            aggregated = {}
            for point in filtered_data:
                hour_key = point["timestamp"].strftime("%Y-%m-%d %H:00")
                if hour_key not in aggregated:
                    aggregated[hour_key] = {
                        "timestamp": hour_key,
                        "requests": 0,
                        "total_generation_time": 0,
                        "total_tokens": 0,
                        "errors": 0
                    }

                aggregated[hour_key]["requests"] += 1
                if point["generation_time"]:
                    aggregated[hour_key]["total_generation_time"] += point["generation_time"]
                aggregated[hour_key]["total_tokens"] += point.get("tokens_used", 0)
                if point["finish_reason"] == "error":
                    aggregated[hour_key]["errors"] += 1

            # Calculate averages
            for hour_data in aggregated.values():
                if hour_data["requests"] > 0:
                    hour_data["avg_generation_time"] = (
                        hour_data["total_generation_time"] / hour_data["requests"]
                    )
                    hour_data["avg_tokens"] = (
                        hour_data["total_tokens"] / hour_data["requests"]
                    )
                    hour_data["error_rate"] = (
                        hour_data["errors"] / hour_data["requests"] * 100
                    )

                # Remove intermediate totals
                del hour_data["total_generation_time"]
                del hour_data["total_tokens"]

            filtered_data = list(aggregated.values())

        return {
            "model_name": model_name,
            "time_range_hours": hours,
            "granularity": granularity,
            "data_points": len(filtered_data),
            "performance": filtered_data,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@monitoring_router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    enabled_only: bool = Query(True, description="Show only enabled alerts")
):
    """Get all configured alerts and their status"""
    try:
        alerts = []
        for alert in metrics_collector.alerts.values():
            if enabled_only and not alert.enabled:
                continue
            if severity and alert.severity.value != severity:
                continue

            alerts.append({
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "enabled": alert.enabled,
                "condition": alert.condition,
                "threshold": alert.threshold,
                "metric_name": alert.metric_name,
                "labels": alert.labels,
                "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                "trigger_count": alert.trigger_count
            })

        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@monitoring_router.post("/alerts/{alert_id}/enable")
async def enable_alert(alert_id: str):
    """Enable an alert"""
    try:
        if alert_id not in metrics_collector.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        metrics_collector.alerts[alert_id].enabled = True
        return {"message": f"Alert {alert_id} enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable alert: {str(e)}")


@monitoring_router.post("/alerts/{alert_id}/disable")
async def disable_alert(alert_id: str):
    """Disable an alert"""
    try:
        if alert_id not in metrics_collector.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        metrics_collector.alerts[alert_id].enabled = False
        return {"message": f"Alert {alert_id} disabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable alert: {str(e)}")


@monitoring_router.get("/analytics/costs")
async def get_cost_analytics(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    period: str = Query("day", description="Time period: hour, day, week")
):
    """Get cost analytics and breakdown"""
    try:
        # Calculate time period
        now = datetime.utcnow()
        if period == "hour":
            start_time = now - timedelta(hours=1)
        elif period == "day":
            start_time = now - timedelta(days=1)
        elif period == "week":
            start_time = now - timedelta(weeks=1)
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Use: hour, day, week")

        # Get cost data from metrics
        cost_data = {}

        if model_name:
            # Single model cost analysis
            hourly_cost = metrics_collector.get_metric_history(
                "model_cost_usd", since=start_time, labels={"model": model_name}
            )
            cost_data[model_name] = {
                "total_cost": sum(p.value for p in hourly_cost),
                "hourly_breakdown": [
                    {
                        "hour": p.timestamp.strftime("%Y-%m-%d %H:00"),
                        "cost": p.value
                    }
                    for p in hourly_cost
                ]
            }
        else:
            # All models cost analysis
            all_costs = metrics_collector.get_metric_history("model_cost_usd", since=start_time)

            # Group by model
            for point in all_costs:
                model = point.labels.get("model", "unknown")
                if model not in cost_data:
                    cost_data[model] = {"total_cost": 0, "hourly_breakdown": []}

                cost_data[model]["total_cost"] += point.value

            # Generate hourly breakdown for each model
            for model in cost_data.keys():
                model_costs = metrics_collector.get_metric_history(
                    "model_cost_usd", since=start_time, labels={"model": model}
                )
                cost_data[model]["hourly_breakdown"] = [
                    {
                        "hour": p.timestamp.strftime("%Y-%m-%d %H:00"),
                        "cost": p.value
                    }
                    for p in model_costs
                ]

        return {
            "period": period,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "cost_data": cost_data,
            "total_cost": sum(data["total_cost"] for data in cost_data.values()),
            "timestamp": now.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost analytics: {str(e)}")


@monitoring_router.get("/analytics/performance")
async def get_performance_analytics(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    metric: str = Query("response_time", description="Metric to analyze: response_time, error_rate, throughput"),
    period: str = Query("hour", description="Time period: minute, hour, day")
):
    """Get performance analytics and trends"""
    try:
        # Calculate time period
        now = datetime.utcnow()
        if period == "minute":
            start_time = now - timedelta(minutes=30)
        elif period == "hour":
            start_time = now - timedelta(hours=24)
        elif period == "day":
            start_time = now - timedelta(days=7)
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Use: minute, hour, day")

        analytics_data = {}

        if metric == "response_time":
            metric_name = "model_response_time_seconds"
        elif metric == "error_rate":
            # Calculate error rate from request counts
            success_requests = metrics_collector.get_metric_history(
                "model_requests_total", since=start_time, labels={"status": "success"}
            )
            error_requests = metrics_collector.get_metric_history(
                "model_requests_total", since=start_time, labels={"status": "error"}
            )

            # Group by model and calculate error rates
            model_data = {}
            for point in success_requests + error_requests:
                model = point.labels.get("model", "unknown")
                status = point.labels.get("status", "unknown")

                if model not in model_data:
                    model_data[model] = {"success": 0, "error": 0}

                model_data[model][status] = point.value

            for model, counts in model_data.items():
                total = counts["success"] + counts["error"]
                error_rate = (counts["error"] / total * 100) if total > 0 else 0
                analytics_data[model] = {"error_rate": error_rate}

            return {
                "metric": metric,
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": now.isoformat(),
                "analytics": analytics_data,
                "timestamp": now.isoformat()
            }

        elif metric == "throughput":
            # Calculate requests per minute/hour
            metric_name = "model_requests_total"

        # Handle other metrics
        if model_name:
            metric_data = metrics_collector.get_metric_history(
                metric_name, since=start_time, labels={"model": model_name}
            )
            analytics_data[model_name] = _analyze_metric_trends(metric_data)
        else:
            # Get all models
            all_metric_data = metrics_collector.get_metric_history(metric_name, since=start_time)

            # Group by model
            model_groups = {}
            for point in all_metric_data:
                model = point.labels.get("model", "unknown")
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(point)

            for model, data in model_groups.items():
                analytics_data[model] = _analyze_metric_trends(data)

        return {
            "metric": metric,
            "period": period,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "analytics": analytics_data,
            "timestamp": now.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")


def _analyze_metric_trends(data_points: List) -> Dict[str, Any]:
    """Analyze trends in metric data"""
    if not data_points:
        return {}

    values = [p.value for p in data_points]

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "latest": values[-1] if values else 0,
        "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing" if len(values) > 1 else "stable"
    }


@monitoring_router.get("/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # Get various metrics for dashboard
        metrics_summary = metrics_collector.get_metrics_summary()
        model_metrics = model_monitor.get_model_metrics()

        # Get recent alerts
        recent_alerts = []
        for alert in metrics_collector.alerts.values():
            if alert.last_triggered and alert.last_triggered > datetime.utcnow() - timedelta(hours=1):
                recent_alerts.append({
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "last_triggered": alert.last_triggered.isoformat()
                })

        # Get top performing models
        model_performance = {}
        for model_name, data in model_metrics.get("models", {}).items():
            model_performance[model_name] = {
                "health": data.get("health", {}).get("status", "unknown"),
                "error_rate": data.get("error_rate", 0),
                "avg_response_time": data.get("avg_response_time", 0),
                "cost_hour": data.get("costs", {}).get("hourly", 0)
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_models": model_metrics["summary"]["total_models"],
                "healthy_models": model_metrics["summary"]["healthy_models"],
                "total_requests_hour": model_metrics["summary"]["total_requests_hour"],
                "total_cost_hour": model_metrics["summary"]["total_cost_hour"],
                "health_percentage": model_metrics["summary"]["health_percentage"]
            },
            "recent_alerts": recent_alerts[:10],  # Top 10 recent alerts
            "model_performance": model_performance,
            "metrics_summary": metrics_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


# Add this router to the main app
def setup_monitoring_routes(app):
    """Setup monitoring routes in the FastAPI app"""
    app.include_router(monitoring_router)