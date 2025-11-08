"""
LM Arena - API Schemas

Pydantic models for API request/response schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=10000, description="The user message")
    model: Optional[str] = Field(None, description="Preferred model to use")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "message": "Explain quantum computing in simple terms",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model that generated the response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason for generation completion")
    generation_time: float = Field(..., description="Time taken to generate response (seconds)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "content": "Quantum computing is a revolutionary approach...",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 15, "completion_tokens": 200, "total_tokens": 215},
                "generation_time": 2.3,
                "finish_reason": "stop"
            }
        }


class StreamChatRequest(BaseModel):
    """Request schema for streaming chat endpoint"""
    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = Field(None)
    conversation_id: Optional[str] = Field(None)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    system_prompt: Optional[str] = Field(None)


class ModelInfo(BaseModel):
    """Model information schema"""
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    model_id: str = Field(..., description="Provider-specific model ID")
    capabilities: Dict[str, Any] = Field(..., description="Model capabilities")
    is_active: bool = Field(..., description="Whether model is active")
    is_default: bool = Field(..., description="Whether model is default")

    class Config:
        schema_extra = {
            "example": {
                "name": "gpt-4",
                "provider": "openai",
                "model_id": "gpt-4",
                "capabilities": {
                    "max_tokens": 8192,
                    "supports_functions": True,
                    "supports_vision": False,
                    "supports_streaming": True,
                    "context_window": 8192
                },
                "is_active": True,
                "is_default": False
            }
        }


class ModelSwitchRequest(BaseModel):
    """Request schema for switching model strategy"""
    strategy: str = Field(..., description="Switching strategy to use")

    class Config:
        schema_extra = {
            "example": {
                "strategy": "load_balanced"
            }
        }


class PromptCreateRequest(BaseModel):
    """Request schema for creating prompts"""
    name: str = Field(..., min_length=1, max_length=100, description="Prompt name")
    content: str = Field(..., min_length=1, max_length=10000, description="Prompt content")
    type: str = Field(..., description="Prompt type (system, user, assistant, template)")
    category: str = Field(..., description="Prompt category")
    description: Optional[str] = Field(None, max_length=500, description="Prompt description")
    tags: Optional[List[str]] = Field(None, description="Prompt tags")
    variables: Optional[Dict[str, Any]] = Field(None, description="Default variables")

    class Config:
        schema_extra = {
            "example": {
                "name": "Code Review",
                "content": "Review the following code:\n\n{{ code }}\n\nProvide feedback on quality, security, and best practices.",
                "type": "user",
                "category": "coding",
                "description": "Template for code review requests",
                "tags": ["code", "review", "development"],
                "variables": {"code": ""}
            }
        }


class PromptInfo(BaseModel):
    """Prompt information schema"""
    id: str = Field(..., description="Prompt ID")
    name: str = Field(..., description="Prompt name")
    type: str = Field(..., description="Prompt type")
    category: str = Field(..., description="Prompt category")
    description: Optional[str] = Field(None, description="Prompt description")
    tags: List[str] = Field(default_factory=list, description="Prompt tags")
    usage_count: int = Field(..., description="Number of times used")
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")


class PromptListResponse(BaseModel):
    """Response schema for prompt listing"""
    prompts: List[PromptInfo] = Field(..., description="List of prompts")
    total: int = Field(..., description="Total number of prompts")


class ConversationCreateRequest(BaseModel):
    """Request schema for creating conversations"""
    system_prompt: Optional[str] = Field(None, description="System prompt for the conversation")

    class Config:
        schema_extra = {
            "example": {
                "system_prompt": "You are a helpful AI assistant specializing in Python programming."
            }
        }


class ConversationInfo(BaseModel):
    """Conversation information schema"""
    id: str = Field(..., description="Conversation ID")
    message_count: int = Field(..., description="Number of messages in conversation")
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")
    has_system_prompt: bool = Field(..., description="Whether conversation has system prompt")


class ConversationListResponse(BaseModel):
    """Response schema for conversation listing"""
    conversations: List[ConversationInfo] = Field(..., description="List of conversations")
    total: int = Field(..., description="Total number of conversations")


class AgentStats(BaseModel):
    """Agent statistics schema"""
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    total_tokens_used: int = Field(..., description="Total tokens consumed")
    total_cost: float = Field(..., description="Total cost incurred")
    models_used: List[str] = Field(..., description="Models that have been used")
    active_models: int = Field(..., description="Number of active models")
    total_conversations: int = Field(..., description="Total conversations")
    current_status: str = Field(..., description="Current agent status")


class ModelStats(BaseModel):
    """Model statistics schema"""
    total_requests: int = Field(..., description="Total requests for this model")
    success_rate: float = Field(..., description="Success rate percentage")
    average_response_time: float = Field(..., description="Average response time in seconds")
    total_cost: float = Field(..., description="Total cost for this model")
    status: str = Field(..., description="Current model status")


class OverallStats(BaseModel):
    """Overall system statistics schema"""
    total_requests: int = Field(..., description="Total system requests")
    successful_requests: int = Field(..., description="Total successful requests")
    failed_requests: int = Field(..., description="Total failed requests")
    overall_success_rate: float = Field(..., description="Overall success rate")
    total_cost: float = Field(..., description="Total system cost")
    total_tokens_used: int = Field(..., description="Total tokens used")
    active_models: int = Field(..., description="Number of active models")
    healthy_models: int = Field(..., description="Number of healthy models")
    current_strategy: str = Field(..., description="Current switching strategy")
    routing_rules_count: int = Field(..., description="Number of routing rules")


class StatsResponse(BaseModel):
    """Response schema for system statistics"""
    agent_stats: AgentStats = Field(..., description="Agent statistics")
    model_stats: Dict[str, ModelStats] = Field(..., description="Model-specific statistics")
    overall_stats: OverallStats = Field(..., description="Overall system statistics")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Overall system status")
    agent_status: str = Field(..., description="Agent status")
    active_models: int = Field(..., description="Number of active models")
    total_conversations: int = Field(..., description="Total conversations")
    uptime: str = Field(..., description="System uptime")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "agent_status": "idle",
                "active_models": 3,
                "total_conversations": 15,
                "uptime": "2 hours, 15 minutes"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    code: Optional[str] = Field(None, description="Error code")
    timestamp: Optional[float] = Field(None, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "The specified model 'gpt-5' is not available",
                "code": "MODEL_NOT_FOUND",
                "timestamp": 1703123456.789
            }
        }