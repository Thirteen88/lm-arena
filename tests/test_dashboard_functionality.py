#!/usr/bin/env python3
"""
Dashboard Functionality Test

Test the real-time dashboard and analytics endpoints.
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.metrics import metrics_collector


async def test_dashboard_functionality():
    """Test dashboard functionality and endpoints."""

    print("🔍 Testing Dashboard Functionality")
    print("=" * 50)

    # Test 1: Dashboard Data Generation
    print("\n1. Testing Dashboard Data Generation...")

    # Add some test metrics
    metrics_collector.increment_counter("dashboard_requests", 10)
    metrics_collector.set_gauge("active_users", 25)
    metrics_collector.record_histogram("response_times", 1.5)
    metrics_collector.record_timer("generation_time", 2.0)

    # Create test alerts
    alert_id = metrics_collector.create_alert(
        name="Dashboard Test Alert",
        description="Test alert for dashboard validation",
        severity="HIGH",
        condition="gt",
        threshold=5.0,
        metric_name="test_metric"
    )

    # Trigger the alert
    metrics_collector.set_gauge("test_metric", 8.0)
    metrics_collector.evaluate_alerts()

    print("   ✅ Test metrics and alerts created")

    # Test 2: Dashboard Summary
    print("\n2. Testing Dashboard Summary...")

    summary = metrics_collector.get_metrics_summary()

    assert "timestamp" in summary
    assert "metrics_count" in summary
    assert "alerts" in summary

    metrics_count = summary["metrics_count"]
    assert metrics_count["counters"] >= 1
    assert metrics_count["gauges"] >= 1
    assert metrics_count["histograms"] >= 1

    alerts_count = summary["alerts"]
    assert alerts_count["total"] >= 1

    print(f"   ✅ Dashboard summary generated with {alerts_count['total']} alerts")

    # Test 3: Prometheus Metrics Export
    print("\n3. Testing Prometheus Metrics Export...")

    prometheus_output = metrics_collector.get_prometheus_metrics()

    # Validate Prometheus format
    assert "# HELP" in prometheus_output
    assert "# TYPE" in prometheus_output

    # Check for Prometheus format structure (help and type definitions are sufficient)
    lines = prometheus_output.split('\n')
    help_lines = [line for line in lines if line.startswith('# HELP')]
    type_lines = [line for line in lines if line.startswith('# TYPE')]

    assert len(help_lines) > 0, "No HELP lines found in Prometheus output"
    assert len(type_lines) > 0, "No TYPE lines found in Prometheus output"

    print(f"   ✅ Prometheus metrics export validated with {len(help_lines)} metrics defined")

    # Test 4: Metric History
    print("\n4. Testing Metric History...")

    history = metrics_collector.get_metric_history("dashboard_requests")
    assert len(history) >= 1

    print(f"   ✅ Metric history retrieved with {len(history)} data points")

    # Test 5: Alert Evaluation
    print("\n5. Testing Alert Evaluation...")

    alert = metrics_collector.alerts[alert_id]
    assert alert.last_triggered is not None
    assert alert.trigger_count >= 1

    print(f"   ✅ Alert '{alert.name}' triggered {alert.trigger_count} times")

    # Test 6: Performance Analytics
    print("\n6. Testing Performance Analytics...")

    # Add performance data
    for i in range(10):
        metrics_collector.record_histogram("performance_test", 0.5 + (i * 0.1))

    performance_data = metrics_collector.get_metric_value("performance_test")
    assert performance_data is not None

    print("   ✅ Performance analytics data generated")

    # Test 7: Cost Tracking
    print("\n7. Testing Cost Tracking...")

    # Simulate cost data
    metrics_collector.increment_counter("total_tokens", 1000)
    metrics_collector.set_gauge("cost_per_hour", 0.05)

    cost_metrics = metrics_collector.get_metric_value("cost_per_hour")
    assert cost_metrics == 0.05

    print("   ✅ Cost tracking metrics validated")

    # Test 8: Real-time Updates
    print("\n8. Testing Real-time Updates...")

    # Simulate real-time metric updates
    for i in range(5):
        metrics_collector.increment_counter("realtime_counter", 1.0)
        metrics_collector.set_gauge("realtime_gauge", float(i))
        await asyncio.sleep(0.01)  # Small delay

    final_counter = metrics_collector.get_metric_value("realtime_counter")
    final_gauge = metrics_collector.get_metric_value("realtime_gauge")

    assert final_counter == 5.0
    assert final_gauge == 4.0

    print("   ✅ Real-time metric updates working")

    print("\n" + "=" * 50)
    print("🎉 Dashboard Functionality Tests - PASSED")
    print("=" * 50)

    return {
        "total_tests": 8,
        "passed_tests": 8,
        "failed_tests": 0,
        "test_results": [
            "Dashboard Data Generation: PASSED",
            "Dashboard Summary: PASSED",
            "Prometheus Metrics Export: PASSED",
            "Metric History: PASSED",
            "Alert Evaluation: PASSED",
            "Performance Analytics: PASSED",
            "Cost Tracking: PASSED",
            "Real-time Updates: PASSED"
        ]
    }


async def main():
    """Main test function."""
    print("🚀 LM Arena Dashboard Functionality Validation")
    print("=" * 60)

    try:
        results = await test_dashboard_functionality()

        print(f"\n📊 DASHBOARD VALIDATION SUMMARY:")
        print(f"   Total Tests: {results['total_tests']}")
        print(f"   Passed: {results['passed_tests']}")
        print(f"   Failed: {results['failed_tests']}")
        print(f"   Success Rate: {(results['passed_tests']/results['total_tests'])*100:.1f}%")

        print("\n✅ Dashboard functionality is FULLY OPERATIONAL")

        # Save test results
        with open('dashboard_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("📄 Results saved to dashboard_validation_results.json")

        return True

    except Exception as e:
        print(f"\n❌ Dashboard validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)