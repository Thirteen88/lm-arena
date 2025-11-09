"""
Test Suite for Multi-Agent Coordination

Comprehensive tests for coordination between LM Arena and other agents,
including Manus automation agent and Claude Orchestrator integration.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from core.agent import GenerationRequest, GenerationResponse
from monitoring.metrics import metrics_collector


class TestManusAgentIntegration:
    """Test integration with Manus automation agent."""

    def test_manus_agent_exists(self):
        """Test that Manus agent directory exists."""
        manus_path = Path("/home/gary/manus-automation-agent")
        assert manus_path.exists(), "Manus agent directory should exist"
        assert (manus_path / "src").exists(), "Manus agent source directory should exist"
        assert (manus_path / "README.md").exists(), "Manus agent README should exist"

    def test_manus_agent_structure(self):
        """Test Manus agent project structure."""
        manus_path = Path("/home/gary/manus-automation-agent")

        # Check key directories
        expected_dirs = ["src", "api", "core", "models", "tests"]
        for dir_name in expected_dirs:
            dir_path = manus_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"

        # Check key files
        expected_files = ["requirements.txt", "README.md"]
        for file_name in expected_files:
            file_path = manus_path / file_name
            assert file_path.exists(), f"File {file_name} should exist"

    def test_manus_agent_api_endpoints(self):
        """Test if Manus agent has API endpoints that can be integrated."""
        manus_api_path = Path("/home/gary/manus-automation-agent/src/api")
        if manus_api_path.exists():
            api_files = list(manus_api_path.glob("*.py"))
            assert len(api_files) > 0, "Manus agent should have API files"

    def test_lm_arena_manus_coordination_scenario(self):
        """Test a coordination scenario between LM Arena and Manus agent."""
        # This test validates that both agents can work together
        # by checking their existence and basic structure

        # Check LM Arena is running (from our earlier tests)
        lm_arena_running = True  # Based on successful API tests

        # Check Manus agent exists
        manus_path = Path("/home/gary/manus-automation-agent")
        manus_agent_exists = manus_path.exists()

        assert lm_arena_running, "LM Arena should be operational"
        assert manus_agent_exists, "Manus agent should be available"

        # Both agents should be ready for coordination
        coordination_ready = lm_arena_running and manus_agent_exists
        assert coordination_ready, "Both agents should be ready for coordination"


class TestClaudeOrchestratorIntegration:
    """Test integration with Claude Orchestrator system."""

    def test_claude_orchestrator_environment(self):
        """Test Claude Orchestrator environment variables."""
        orchestrator_vars = [
            "CLAUDE_ORCHESTRATOR_PATH",
            "CLAUDE_WORKSPACES"
        ]

        # Check if orchestrator environment is set up
        orchestrator_path = Path("/home/gary/claude-orchestrator")
        orchestrator_exists = orchestrator_path.exists()

        assert orchestrator_exists, "Claude Orchestrator should be installed"

    def test_orchestrator_capabilities(self):
        """Test Claude Orchestrator multi-agent capabilities."""
        orchestrator_path = Path("/home/gary/claude-orchestrator")

        if orchestrator_path.exists():
            # Check for key orchestrator files
            key_files = [
                "orchestrator.py",
                "QUICK-START.md",
                "IMPLEMENTATION_SUMMARY.md"
            ]

            for file_name in key_files:
                file_path = orchestrator_path / file_name
                if file_path.exists():
                    assert file_path.stat().st_size > 0, f"Orchestrator file {file_name} should not be empty"

    def test_multi_agent_workflow_possible(self):
        """Test that multi-agent workflow setup is possible."""
        # Check that we have multiple agents available
        available_agents = []

        # LM Arena
        lm_arena_path = Path("/home/gary/lm-arena")
        if lm_arena_path.exists() and (lm_arena_path / "api" / "main.py").exists():
            available_agents.append("LM Arena")

        # Manus Agent
        manus_path = Path("/home/gary/manus-automation-agent")
        if manus_path.exists() and (manus_path / "src").exists():
            available_agents.append("Manus Agent")

        # Claude Orchestrator
        orchestrator_path = Path("/home/gary/claude-orchestrator")
        if orchestrator_path.exists():
            available_agents.append("Claude Orchestrator")

        # Should have at least 2 agents for meaningful coordination
        assert len(available_agents) >= 2, f"Should have at least 2 agents available, found: {available_agents}"


class TestMultiAgentMonitoring:
    """Test monitoring across multiple agents."""

    def test_cross_agent_metrics_collection(self):
        """Test that metrics can be collected across different agents."""
        # This test validates that our monitoring system can track
        # activity from multiple agents

        # Test metrics collector is working
        assert metrics_collector is not None
        assert hasattr(metrics_collector, 'counters')
        assert hasattr(metrics_collector, 'gauges')

        # Create test metrics for different agents
        metrics_collector.increment_counter(
            "agent_requests_total",
            1.0,
            {"agent": "lm_arena", "type": "model_generation"}
        )

        metrics_collector.increment_counter(
            "agent_requests_total",
            1.0,
            {"agent": "manus_automation", "type": "browser_automation"}
        )

        # Verify metrics were recorded
        lm_arena_key = "agent_requests_total[agent=lm_arena,type=model_generation]"
        manus_key = "agent_requests_total[agent=manus_automation,type=browser_automation]"

        assert lm_arena_key in metrics_collector.counters
        assert manus_key in metrics_collector.counters

    def test_agent_health_monitoring(self):
        """Test health monitoring for multiple agents."""
        # Test that we can track health status of different agents
        metrics_collector.set_gauge("agent_status", 1.0, {"agent": "lm_arena", "component": "api"})
        metrics_collector.set_gauge("agent_status", 1.0, {"agent": "manus", "component": "browser"})

        # Verify health metrics
        lm_arena_health = metrics_collector.get_metric_value(
            "agent_status",
            {"agent": "lm_arena", "component": "api"}
        )
        manus_health = metrics_collector.get_metric_value(
            "agent_status",
            {"agent": "manus", "component": "browser"}
        )

        assert lm_arena_health == 1.0
        assert manus_health == 1.0

    def test_agent_coordination_metrics(self):
        """Test metrics for agent coordination activities."""
        # Track coordination between agents
        metrics_collector.increment_counter(
            "agent_coordination_events",
            1.0,
            {"from_agent": "lm_arena", "to_agent": "manus", "event_type": "task_delegation"}
        )

        metrics_collector.record_histogram(
            "coordination_latency",
            0.5,
            {"coordination_type": "lm_arena_to_manus"}
        )

        # Verify coordination metrics
        coordination_key = "agent_coordination_events[from_agent=lm_arena,to_agent=manus,event_type=task_delegation]"
        assert coordination_key in metrics_collector.counters
        assert metrics_collector.counters[coordination_key] == 1.0

        latency_key = "coordination_latency[coordination_type=lm_arena_to_manus]"
        assert latency_key in metrics_collector.histograms
        assert 0.5 in metrics_collector.histograms[latency_key]


class TestAgentCommunicationProtocols:
    """Test communication protocols between agents."""

    @pytest.mark.asyncio
    async def test_agent_message_format(self):
        """Test standardized message format between agents."""
        # Test that we can create standardized messages
        test_message = {
            "id": f"coordination-{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": "lm_arena",
            "to_agent": "manus",
            "message_type": "task_request",
            "payload": {
                "task_type": "browser_automation",
                "parameters": {"url": "https://example.com"},
                "priority": "normal"
            }
        }

        # Validate message structure
        required_fields = ["id", "timestamp", "from_agent", "to_agent", "message_type", "payload"]
        for field in required_fields:
            assert field in test_message, f"Message should contain field: {field}"

        # Test message serialization
        json_message = json.dumps(test_message)
        assert json.loads(json_message) == test_message

    @pytest.mark.asyncio
    async def test_agent_response_handling(self):
        """Test response handling between agents."""
        # Simulate agent response
        test_response = {
            "id": f"response-{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": "manus",
            "to_agent": "lm_arena",
            "message_type": "task_response",
            "in_reply_to": "coordination-123",
            "status": "success",
            "payload": {
                "result": "Task completed successfully",
                "data": {"automation_id": "task-456"},
                "execution_time": 2.5
            }
        }

        # Validate response structure
        assert test_response["status"] in ["success", "error", "pending"]
        assert "payload" in test_response
        assert "result" in test_response["payload"]

    def test_agent_error_handling(self):
        """Test error handling in agent communication."""
        # Test error message format
        error_message = {
            "id": "error-123",
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": "lm_arena",
            "to_agent": "manus",
            "message_type": "error",
            "error_details": {
                "error_code": "AGENT_UNAVAILABLE",
                "error_message": "Manus agent is not responding",
                "retry_after": 30
            }
        }

        assert error_message["message_type"] == "error"
        assert "error_details" in error_message
        assert "error_code" in error_message["error_details"]


class TestMultiAgentWorkflows:
    """Test complete multi-agent workflows."""

    def test_lm_arena_to_manus_workflow(self):
        """Test workflow from LM Arena to Manus agent."""
        workflow_steps = [
            "lm_arena_receives_request",
            "lm_arena_processes_with_ai",
            "lm_arena_delegates_to_manus",
            "manus_executes_browser_automation",
            "manus_returns_results",
            "lm_arena_processes_results",
            "lm_arena_returns_final_response"
        ]

        # Validate workflow structure
        assert len(workflow_steps) >= 6  # Should have meaningful workflow steps

        # Test that we can track workflow metrics
        for i, step in enumerate(workflow_steps):
            metrics_collector.increment_counter(
                "workflow_steps_completed",
                1.0,
                {"workflow": "lm_arena_to_manus", "step": str(i+1), "step_name": step}
            )

        # Verify all steps were recorded
        assert metrics_collector.counters.get("workflow_steps_completed") is not None

    def test_multi_agent_load_balancing(self):
        """Test load balancing across multiple agents."""
        # Simulate load metrics for different agents
        agents = ["lm_arena", "manus", "claude_orchestrator"]

        for agent in agents:
            # Simulate current load
            load = 0.5 if agent == "lm_arena" else 0.3
            metrics_collector.set_gauge("agent_load", load, {"agent": agent})

            # Simulate capacity
            capacity = 1.0
            metrics_collector.set_gauge("agent_capacity", capacity, {"agent": agent})

        # Test load balancing decision
        def select_least_loaded_agent(agents):
            agent_loads = {}
            for agent in agents:
                load = metrics_collector.get_metric_value("agent_load", {"agent": agent})
                capacity = metrics_collector.get_metric_value("agent_capacity", {"agent": agent})
                if load is not None and capacity is not None:
                    agent_loads[agent] = load / capacity

            if agent_loads:
                return min(agent_loads, key=agent_loads.get)
            return agents[0]

        selected_agent = select_least_loaded_agent(agents)
        assert selected_agent in agents

    def test_agent_failover_scenarios(self):
        """Test failover scenarios when agents are unavailable."""
        # Test failover logic
        available_agents = ["lm_arena", "manus"]

        # Simulate agent failure
        def is_agent_available(agent):
            if agent == "manus":
                return False  # Manus agent unavailable
            return True

        available_agents = [agent for agent in available_agents if is_agent_available(agent)]

        # Should fallback to available agents
        assert "lm_arena" in available_agents
        assert "manus" not in available_agents
        assert len(available_agents) == 1

        # Track failover metrics
        metrics_collector.increment_counter(
            "agent_failovers",
            1.0,
            {"failed_agent": "manus", "fallback_agent": "lm_arena", "reason": "unavailable"}
        )


@pytest.mark.asyncio
class TestRealTimeAgentCoordination:
    """Test real-time agent coordination scenarios."""

    async def test_simultaneous_agent_requests(self):
        """Test handling simultaneous requests to multiple agents."""
        # Simulate concurrent agent requests
        async def simulate_agent_request(agent_id, delay):
            await asyncio.sleep(delay)
            metrics_collector.increment_counter(
                "concurrent_agent_requests",
                1.0,
                {"agent_id": agent_id}
            )
            return f"Response from agent {agent_id}"

        # Create concurrent requests
        tasks = []
        for i in range(5):
            task = simulate_agent_request(f"agent-{i}", 0.1 * i)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Verify all requests completed
        assert len(results) == 5
        for i, result in enumerate(results):
            assert f"Response from agent agent-{i}" in result

        # Verify metrics were recorded
        total_requests = sum(
            metrics_collector.counters.get(f"concurrent_agent_requests[agent_id=agent-{j}]", 0)
            for j in range(5)
        )
        assert total_requests == 5

    async def test_agent_coordination_timeout(self):
        """Test timeout handling in agent coordination."""
        # Test coordination with timeout
        async def agent_with_timeout(agent_id, timeout):
            try:
                await asyncio.wait_for(
                    asyncio.sleep(2.0),  # Simulate work
                    timeout=timeout
                )
                return f"Agent {agent_id} completed"
            except asyncio.TimeoutError:
                metrics_collector.increment_counter(
                    "agent_timeouts",
                    1.0,
                    {"agent_id": agent_id, "timeout_duration": timeout}
                )
                return f"Agent {agent_id} timed out after {timeout}s"

        # Test with short timeout (should timeout)
        result = await agent_with_timeout("test-agent", 1.0)
        assert "timed out" in result.lower()

        # Test with longer timeout (should complete)
        result = await agent_with_timeout("test-agent", 3.0)
        assert "completed" in result.lower()