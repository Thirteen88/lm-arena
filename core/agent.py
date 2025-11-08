"""
LM Arena - Core Agent Framework

A flexible, multi-model agent framework with advanced prompt management
and model switching capabilities.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import json

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    THINKING = "thinking"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"
    OFFLINE = "offline"


class MessageRole(str, Enum):
    """Message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Message structure for conversations"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    model_info: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class Conversation:
    """Conversation context management"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    system_prompt: Optional[str] = None

    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = time.time()

    def get_context(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get conversation context with optional message limit"""
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:]

    def clear(self):
        """Clear all messages except system prompt"""
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        self.messages = system_messages
        self.updated_at = time.time()


class ModelCapabilities(BaseModel):
    """Model capability definitions"""
    max_tokens: int = 4096
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    context_window: int = 4096
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


class ModelConfig(BaseModel):
    """Configuration for AI models"""
    name: str
    provider: str
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    priority: int = 0  # Higher numbers = higher priority


class GenerationRequest(BaseModel):
    """Request structure for model generation"""
    prompt: str
    model_name: Optional[str] = None
    conversation_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None


class GenerationResponse(BaseModel):
    """Response structure for model generation"""
    content: str
    model_name: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    generation_time: float = 0.0
    cost: float = 0.0


class ModelInterface(ABC):
    """Abstract interface for AI models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.capabilities = config.capabilities
        self.name = config.name
        self.provider = config.provider

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response from the model"""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest):
        """Generate streaming response from the model"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate model connection and configuration"""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    def supports(self, capability: str) -> bool:
        """Check if model supports a specific capability"""
        return getattr(self.capabilities, f"supports_{capability}", False)


class ModelRegistry:
    """Registry for managing multiple AI models"""

    def __init__(self):
        self._models: Dict[str, ModelInterface] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._default_model: Optional[str] = None

    def register_model(self, model: ModelInterface, is_default: bool = False):
        """Register a model in the registry"""
        self._models[model.name] = model
        self._configs[model.name] = model.config

        if is_default or self._default_model is None:
            self._default_model = model.name

        logger.info("Model registered", model=model.name, is_default=is_default)

    def get_model(self, name: Optional[str] = None) -> Optional[ModelInterface]:
        """Get a model by name or default"""
        if name is None:
            name = self._default_model

        if name is None:
            return None

        return self._models.get(name)

    def list_models(self, active_only: bool = True) -> List[ModelInterface]:
        """List all registered models"""
        models = list(self._models.values())
        if active_only:
            models = [m for m in models if m.config.is_active]
        return models

    def get_default_model(self) -> Optional[ModelInterface]:
        """Get the default model"""
        return self.get_model(self._default_model)

    def set_default_model(self, name: str) -> bool:
        """Set the default model"""
        if name in self._models:
            self._default_model = name
            logger.info("Default model changed", model=name)
            return True
        return False

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration"""
        return self._configs.get(name)

    def update_model_config(self, name: str, config: ModelConfig) -> bool:
        """Update model configuration"""
        if name in self._models:
            self._configs[name] = config
            self._models[name].config = config
            self._models[name].capabilities = config.capabilities
            logger.info("Model config updated", model=name)
            return True
        return False


class LMArenaAgent:
    """Main LM Arena Agent class"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.model_registry = ModelRegistry()
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation: Optional[str] = None
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "models_used": set()
        }

        logger.info("LM Arena Agent initialized")

    async def initialize(self):
        """Initialize the agent and validate connections"""
        self.status = AgentStatus.THINKING

        try:
            # Validate all active models
            active_models = self.model_registry.list_models(active_only=True)
            if not active_models:
                raise ValueError("No active models found")

            for model in active_models:
                try:
                    is_valid = await model.validate_connection()
                    if not is_valid:
                        logger.warning("Model validation failed", model=model.name)
                        model.config.is_active = False
                    else:
                        logger.info("Model validated", model=model.name)
                except Exception as e:
                    logger.error("Model validation error", model=model.name, error=str(e))
                    model.config.is_active = False

            self.status = AgentStatus.IDLE
            logger.info("LM Arena Agent initialized successfully")

        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error("Agent initialization failed", error=str(e))
            raise

    async def generate_response(
        self,
        request: GenerationRequest,
        conversation_id: Optional[str] = None
    ) -> GenerationResponse:
        """Generate a response using the specified or default model"""

        self.status = AgentStatus.PROCESSING
        start_time = time.time()

        try:
            self.stats["total_requests"] += 1

            # Get model
            model = self.model_registry.get_model(request.model_name)
            if model is None:
                model = self.model_registry.get_default_model()

            if model is None:
                raise ValueError("No model available for generation")

            # Handle conversation context
            if conversation_id:
                conversation = self.get_or_create_conversation(conversation_id)
                if request.system_prompt:
                    conversation.system_prompt = request.system_prompt

                # Add user message to conversation
                user_msg = Message(
                    role=MessageRole.USER,
                    content=request.prompt,
                    metadata=request.metadata
                )
                conversation.add_message(user_msg)

            # Generate response
            self.status = AgentStatus.RESPONDING
            await self._emit_event("generation_started", {
                "model": model.name,
                "request": request.dict()
            })

            response = await model.generate(request)

            # Calculate timing and cost
            response.generation_time = time.time() - start_time
            response.cost = self._calculate_cost(model, response.usage)

            # Add response to conversation
            if conversation_id:
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    model_info={"name": model.name, "provider": model.provider},
                    metadata=response.metadata,
                    tool_calls=response.tool_calls
                )
                conversation.add_message(assistant_msg)

            # Update statistics
            self.stats["successful_requests"] += 1
            self.stats["total_tokens_used"] += response.usage.get("total_tokens", 0)
            self.stats["total_cost"] += response.cost
            self.stats["models_used"].add(model.name)

            await self._emit_event("generation_completed", {
                "model": model.name,
                "response": response.dict(),
                "generation_time": response.generation_time
            })

            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.status = AgentStatus.ERROR

            logger.error("Generation failed", error=str(e), request=request.dict())

            await self._emit_event("generation_failed", {
                "error": str(e),
                "request": request.dict()
            })

            raise
        finally:
            if self.status == AgentStatus.RESPONDING:
                self.status = AgentStatus.IDLE

    async def generate_stream(self, request: GenerationRequest):
        """Generate a streaming response"""
        self.status = AgentStatus.PROCESSING

        try:
            model = self.model_registry.get_model(request.model_name)
            if model is None:
                model = self.model_registry.get_default_model()

            if model is None:
                raise ValueError("No model available for streaming generation")

            if not model.supports("streaming"):
                raise ValueError("Model does not support streaming")

            async for chunk in model.generate_stream(request):
                yield chunk

        except Exception as e:
            logger.error("Stream generation failed", error=str(e))
            raise
        finally:
            self.status = AgentStatus.IDLE

    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get or create a conversation"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(id=conversation_id)

        self.current_conversation = conversation_id
        return self.conversations[conversation_id]

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a specific conversation"""
        return self.conversations.get(conversation_id)

    def list_conversations(self) -> List[Conversation]:
        """List all conversations"""
        return list(self.conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if self.current_conversation == conversation_id:
                self.current_conversation = None
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "models_used": list(self.stats["models_used"]),
            "active_models": len(self.model_registry.list_models(active_only=True)),
            "total_conversations": len(self.conversations),
            "current_status": self.status.value
        }

    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "models_used": set()
        }

    def on(self, event: str, handler: Callable):
        """Register an event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable):
        """Unregister an event handler"""
        if event in self._event_handlers:
            self._event_handlers[event].remove(handler)

    async def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error("Event handler error", event=event, error=str(e))

    def _calculate_cost(self, model: ModelInterface, usage: Dict[str, int]) -> float:
        """Calculate generation cost"""
        input_cost = usage.get("input_tokens", 0) * model.capabilities.cost_per_input_token
        output_cost = usage.get("output_tokens", 0) * model.capabilities.cost_per_output_token
        return input_cost + output_cost

    async def shutdown(self):
        """Shutdown the agent"""
        self.status = AgentStatus.OFFLINE
        logger.info("LM Arena Agent shutdown")