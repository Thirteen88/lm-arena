"""
LM Arena - Agent Tests

Unit tests for the core agent functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from lm_arena.core.agent import (
    LMArenaAgent, GenerationRequest, GenerationResponse,
    Message, MessageRole, Conversation, ModelRegistry
)
from lm_arena.core.model_switcher import ModelSwitcher, SwitchingStrategy
from lm_arena.models.openai_model import OpenAIModel, OpenAIConfig


class MockModel(OpenAIModel):
    """Mock model for testing"""

    def __init__(self, name: str, response_text: str = "Mock response"):
        config = OpenAIConfig(
            name=name,
            provider="mock",
            model_id="mock-model",
            api_key="mock-key"
        )
        super().__init__(config)
        self.response_text = response_text
        self.call_count = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Mock generation"""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate some processing time

        return GenerationResponse(
            content=self.response_text,
            model_name=self.name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            generation_time=0.01
        )

    async def generate_stream(self, request: GenerationRequest):
        """Mock streaming generation"""
        async for chunk in ["Hello", " ", "world", "!"]:
            yield {"type": "content", "content": chunk}

    async def validate_connection(self) -> bool:
        """Mock connection validation"""
        return True

    async def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        return len(text.split())


@pytest.fixture
async def mock_model():
    """Create a mock model for testing"""
    return MockModel("test-model", "Test response")


@pytest.fixture
async def agent():
    """Create an agent instance for testing"""
    agent = LMArenaAgent()
    # Initialize without model validation for testing
    agent.status = agent.AgentStatus.IDLE
    return agent


@pytest.fixture
async def agent_with_models():
    """Create an agent with registered models"""
    agent = LMArenaAgent()

    # Register mock models
    model1 = MockModel("model1", "Response from model 1")
    model2 = MockModel("model2", "Response from model 2")

    agent.model_registry.register_model(model1, is_default=True)
    agent.model_registry.register_model(model2)

    # Initialize without model validation for testing
    agent.status = agent.AgentStatus.IDLE
    return agent


class TestLMArenaAgent:
    """Test cases for LMArenaAgent"""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.status == agent.AgentStatus.IDLE
        assert agent.model_registry is not None
        assert agent.conversations == {}
        assert agent.current_conversation is None

    @pytest.mark.asyncio
    async def test_generate_response_basic(self, agent_with_models):
        """Test basic response generation"""
        request = GenerationRequest(
            prompt="Hello, world!",
            model_name="model1"
        )

        response = await agent_with_models.generate_response(request)

        assert response.content == "Response from model 1"
        assert response.model_name == "model1"
        assert response.usage["total_tokens"] == 30
        assert response.generation_time > 0

    @pytest.mark.asyncio
    async def test_generate_response_with_conversation(self, agent_with_models):
        """Test response generation with conversation context"""
        # Create conversation
        conv = agent_with_models.get_or_create_conversation("test-conv")
        conv.system_prompt = "You are a helpful assistant"

        request = GenerationRequest(
            prompt="What is 2+2?",
            conversation_id="test-conv"
        )

        response = await agent_with_models.generate_response(request, "test-conv")

        # Check conversation was updated
        assert len(conv.messages) == 2  # User message + assistant response
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_generate_response_default_model(self, agent_with_models):
        """Test response generation with default model"""
        request = GenerationRequest(
            prompt="Hello",
            model_name=None  # No specific model requested
        )

        response = await agent_with_models.generate_response(request)

        # Should use the default model
        assert response.content == "Response from model 1"

    @pytest.mark.asyncio
    async def test_generate_response_model_not_found(self, agent_with_models):
        """Test response generation with non-existent model"""
        request = GenerationRequest(
            prompt="Hello",
            model_name="non-existent-model"
        )

        with pytest.raises(ValueError, match="No model available"):
            await agent_with_models.generate_response(request)

    @pytest.mark.asyncio
    async def test_conversation_management(self, agent):
        """Test conversation management"""
        # Create conversation
        conv1 = agent.get_or_create_conversation("conv1")
        assert conv1.id == "conv1"
        assert len(agent.conversations) == 1

        # Get existing conversation
        conv2 = agent.get_or_create_conversation("conv1")
        assert conv1 is conv2
        assert len(agent.conversations) == 1

        # Create another conversation
        conv3 = agent.get_or_create_conversation("conv2")
        assert conv3.id == "conv2"
        assert len(agent.conversations) == 2

        # Delete conversation
        success = agent.delete_conversation("conv1")
        assert success is True
        assert len(agent.conversations) == 1

        # Delete non-existent conversation
        success = agent.delete_conversation("non-existent")
        assert success is False

    @pytest.mark.asyncio
    async def test_stats_tracking(self, agent_with_models):
        """Test statistics tracking"""
        initial_stats = agent_with_models.get_stats()
        assert initial_stats["total_requests"] == 0

        # Generate some responses
        request = GenerationRequest(prompt="Test")
        await agent_with_models.generate_response(request)
        await agent_with_models.generate_response(request)

        stats = agent_with_models.get_stats()
        assert stats["total_requests"] == 2
        assert stats["successful_requests"] == 2
        assert stats["failed_requests"] == 0
        assert stats["total_tokens_used"] == 60  # 30 per request
        assert "model1" in stats["models_used"]

    @pytest.mark.asyncio
    async def test_stats_reset(self, agent_with_models):
        """Test statistics reset"""
        # Generate a response
        request = GenerationRequest(prompt="Test")
        await agent_with_models.generate_response(request)

        # Verify stats are non-zero
        stats = agent_with_models.get_stats()
        assert stats["total_requests"] > 0

        # Reset stats
        agent_with_models.reset_stats()
        stats = agent_with_models.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0

    @pytest.mark.asyncio
    async def test_event_handlers(self, agent):
        """Test event handling system"""
        events_received = []

        def event_handler(data):
            events_received.append(data)

        # Register event handler
        agent.on("generation_started", event_handler)
        agent.on("generation_completed", event_handler)

        # Register mock model
        mock_model = MockModel("test-model")
        agent.model_registry.register_model(mock_model)

        # Generate response
        request = GenerationRequest(prompt="Test")
        await agent.generate_response(request)

        # Check events were received
        assert len(events_received) >= 1
        assert events_received[0]["model"] == "test-model"

        # Unregister event handler
        agent.off("generation_started", event_handler)


class TestConversation:
    """Test cases for Conversation class"""

    def test_conversation_creation(self):
        """Test conversation creation"""
        conv = Conversation()
        assert conv.id is not None
        assert len(conv.messages) == 0
        assert conv.system_prompt is None

    def test_add_message(self):
        """Test adding messages to conversation"""
        conv = Conversation()
        message = Message(role=MessageRole.USER, content="Hello")
        conv.add_message(message)

        assert len(conv.messages) == 1
        assert conv.messages[0] == message

    def test_get_context(self):
        """Test getting conversation context"""
        conv = Conversation()

        # Add some messages
        for i in range(10):
            conv.add_message(Message(role=MessageRole.USER, content=f"Message {i}"))

        # Get all context
        all_context = conv.get_context()
        assert len(all_context) == 10

        # Get limited context
        limited_context = conv.get_context(max_messages=5)
        assert len(limited_context) == 5
        assert limited_context[0].content == "Message 5"

    def test_clear_conversation(self):
        """Test clearing conversation"""
        conv = Conversation()

        # Add system and user messages
        conv.add_message(Message(role=MessageRole.SYSTEM, content="System prompt"))
        conv.add_message(Message(role=MessageRole.USER, content="User message"))

        # Clear conversation
        conv.clear()

        # Should keep system messages
        assert len(conv.messages) == 1
        assert conv.messages[0].role == MessageRole.SYSTEM


class TestMessage:
    """Test cases for Message class"""

    def test_message_creation(self):
        """Test message creation"""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
            metadata={"source": "test"}
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.metadata["source"] == "test"
        assert msg.timestamp > 0

    def test_message_with_tool_calls(self):
        """Test message with tool calls"""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"arg": "value"}'
                }
            }
        ]

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="I'll call a function for you.",
            tool_calls=tool_calls
        )

        assert msg.tool_calls == tool_calls
        assert len(msg.tool_calls) == 1


class TestModelRegistry:
    """Test cases for ModelRegistry"""

    def test_model_registration(self):
        """Test model registration"""
        registry = ModelRegistry()
        model = MockModel("test-model")

        # Register model
        registry.register_model(model, is_default=True)
        assert "test-model" in registry._models
        assert registry._default_model == "test-model"

        # Check default model
        default_model = registry.get_default_model()
        assert default_model is model

    def test_model_retrieval(self):
        """Test model retrieval"""
        registry = ModelRegistry()
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        registry.register_model(model1, is_default=True)
        registry.register_model(model2)

        # Get specific model
        retrieved = registry.get_model("model2")
        assert retrieved is model2

        # Get default model
        default = registry.get_model()
        assert default is model1

        # Get non-existent model
        non_existent = registry.get_model("non-existent")
        assert non_existent is None

    def test_model_listing(self):
        """Test model listing"""
        registry = ModelRegistry()
        model1 = MockModel("model1")
        model2 = MockModel("model2")
        model3 = MockModel("model3")

        # Set model2 as inactive
        model2.config.is_active = False

        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # List all models
        all_models = registry.list_models(active_only=False)
        assert len(all_models) == 3

        # List active models only
        active_models = registry.list_models(active_only=True)
        assert len(active_models) == 2
        assert model2 not in active_models

    def test_default_model_switching(self):
        """Test changing default model"""
        registry = ModelRegistry()
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        registry.register_model(model1, is_default=True)
        registry.register_model(model2)

        assert registry.get_default_model() is model1

        # Switch default model
        success = registry.set_default_model("model2")
        assert success is True
        assert registry.get_default_model() is model2

        # Try to set non-existent model as default
        success = registry.set_default_model("non-existent")
        assert success is False


@pytest.mark.asyncio
async def test_agent_with_mock_models():
    """Integration test with multiple mock models"""
    agent = LMArenaAgent()

    # Create models with different responses
    fast_model = MockModel("fast", "Fast response")
    slow_model = MockModel("slow", "Slow response")

    # Make slow model actually slower
    original_generate = slow_model.generate
    async def slow_generate(request):
        await asyncio.sleep(0.1)
        return await original_generate(request)
    slow_model.generate = slow_generate

    # Register models
    agent.model_registry.register_model(fast_model, is_default=True)
    agent.model_registry.register_model(slow_model)

    # Create model switcher
    switcher = ModelSwitcher(agent.model_registry.models)
    switcher.set_strategy(SwitchingStrategy.RANDOM)

    # Test multiple requests
    requests = [
        GenerationRequest(prompt=f"Request {i}")
        for i in range(5)
    ]

    responses = []
    for req in requests:
        response = await switcher.execute_with_switching(req)
        responses.append(response)

    # Verify all requests succeeded
    assert len(responses) == 5
    for response in responses:
        assert response.content in ["Fast response", "Slow response"]
        assert response.generation_time > 0

    # Check that both models were used (random strategy should distribute)
    models_used = set(r.metadata.get("selected_model") for r in responses)
    assert len(models_used) > 1  # Should have used both models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])