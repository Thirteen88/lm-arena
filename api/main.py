"""
LM Arena - FastAPI Application

Main API application for LM Arena with model switching, prompt management,
and conversation handling.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import uvicorn
import structlog

from core.agent import LMArenaAgent, GenerationRequest, GenerationResponse
from core.model_switcher import ModelSwitcher, SwitchingStrategy
from models.openai_model import create_openai_model, create_openai_compatible_model
from models.anthropic_model import create_anthropic_model
from prompts.prompt_manager import PromptManager
from config.settings import get_config, load_config, validate_config
from api.schemas import (
    ChatRequest,
    ChatResponse,
    StreamChatRequest,
    ModelInfo,
    ModelSwitchRequest,
    PromptCreateRequest,
    PromptListResponse,
    ConversationListResponse,
    ConversationCreateRequest,
    StatsResponse,
    HealthResponse
)

logger = structlog.get_logger(__name__)

# Global variables
agent: Optional[LMArenaAgent] = None
prompt_manager: Optional[PromptManager] = None
model_switcher: Optional[ModelSwitcher] = None
config = None

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent, prompt_manager, model_switcher, config

    # Startup
    logger.info("Starting LM Arena API...")

    # Load configuration
    config = get_config()
    config.create_directories()

    # Validate configuration
    issues = validate_config(config)
    if issues:
        logger.warning("Configuration issues detected", issues=issues)

    # Initialize prompt manager
    prompt_manager = PromptManager(config.prompts_dir)

    # Initialize models
    models = {}
    await setup_models(models)

    # Initialize model switcher
    if models:
        model_switcher = ModelSwitcher(models)
        model_switcher.set_strategy(SwitchingStrategy(config.models.switching_strategy))

    # Initialize agent
    agent = LMArenaAgent()

    # Register models with agent
    for name, model in models.items():
        agent.model_registry.register_model(model)

    # Initialize agent
    await agent.initialize()

    logger.info("LM Arena API started successfully",
               models_count=len(models),
               strategy=config.models.switching_strategy)

    yield

    # Shutdown
    logger.info("Shutting down LM Arena API...")
    if agent:
        await agent.shutdown()


# Create FastAPI app
app = FastAPI(
    title="LM Arena API",
    description="Multi-model AI agent with advanced prompt management and model switching",
    version="1.0.0",
    docs_url="/docs" if get_config().api.enable_docs else None,
    redoc_url="/redoc" if get_config().api.enable_docs else None,
    lifespan=lifespan
)

# Add CORS middleware
if get_config().api.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_config().security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def setup_models(models: Dict[str, Any]):
    """Setup models based on configuration and environment variables"""
    # OpenAI models
    openai_api_key = get_config().security.secret_key  # For demo, use secret key as API key
    if openai_api_key and openai_api_key != "your-secret-key-here":
        # GPT models
        for model_name, model_id in [
            ("gpt-3.5-turbo", "gpt-3.5-turbo"),
            ("gpt-4", "gpt-4"),
            ("gpt-4-turbo", "gpt-4-turbo-preview"),
        ]:
            try:
                model = create_openai_model(model_name, model_id, openai_api_key)
                models[model_name] = model
                logger.info("OpenAI model registered", name=model_name, model_id=model_id)
            except Exception as e:
                logger.error("Failed to register OpenAI model", name=model_name, error=str(e))

    # Anthropic models (if API key is configured)
    anthropic_api_key = get_config().security.secret_key  # For demo
    if anthropic_api_key and anthropic_api_key != "your-secret-key-here":
        for model_name, model_id in [
            ("claude-3-haiku", "claude-3-haiku-20240307"),
            ("claude-3-sonnet", "claude-3-sonnet-20240229"),
            ("claude-3-opus", "claude-3-opus-20240229"),
        ]:
            try:
                model = create_anthropic_model(model_name, model_id, anthropic_api_key)
                models[model_name] = model
                logger.info("Anthropic model registered", name=model_name, model_id=model_id)
            except Exception as e:
                logger.error("Failed to register Anthropic model", name=model_name, error=str(e))

    # Local/OpenAI-compatible models
    local_api_bases = [
        "http://localhost:8000/v1",  # Common local API port
        "http://localhost:11434/v1",  # Ollama default
    ]

    for api_base in local_api_bases:
        try:
            model = create_openai_compatible_model(
                name=f"local-{api_base.split(':')[1].replace('/', '')}",
                model_id="llama2",  # Common local model
                api_base=api_base
            )
            models[model.name] = model
            logger.info("Local model registered", name=model.name, api_base=api_base)
        except Exception as e:
            logger.debug("Local model not available", api_base=api_base, error=str(e))


# Dependency functions
async def get_agent() -> LMArenaAgent:
    """Get the agent instance"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


async def get_prompt_manager() -> PromptManager:
    """Get the prompt manager instance"""
    if prompt_manager is None:
        raise HTTPException(status_code=503, detail="Prompt manager not initialized")
    return prompt_manager


async def get_model_switcher() -> Optional[ModelSwitcher]:
    """Get the model switcher instance"""
    return model_switcher


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check(agent_instance: LMArenaAgent = Depends(get_agent)):
    """Health check endpoint"""
    stats = agent_instance.get_stats()

    return HealthResponse(
        status="healthy" if agent_instance.status.value != "error" else "unhealthy",
        agent_status=agent_instance.status.value,
        active_models=stats["active_models"],
        total_conversations=stats["total_conversations"],
        uptime="N/A"  # TODO: Implement uptime tracking
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(agent_instance: LMArenaAgent = Depends(get_agent)):
    """Get system statistics"""
    agent_stats = agent_instance.get_stats()

    model_stats = {}
    if model_switcher:
        metrics = model_switcher.get_all_metrics()
        for name, metric in metrics.items():
            model_stats[name] = {
                "total_requests": metric.total_requests,
                "success_rate": metric.success_rate,
                "average_response_time": metric.average_response_time,
                "total_cost": metric.total_cost,
                "status": metric.status.value
            }

    return StatsResponse(
        agent_stats=agent_stats,
        model_stats=model_stats,
        overall_stats=model_switcher.get_overall_stats() if model_switcher else {}
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    agent_instance: LMArenaAgent = Depends(get_agent),
    switcher: Optional[ModelSwitcher] = Depends(get_model_switcher)
):
    """Chat endpoint for generating responses"""
    try:
        # Create generation request
        gen_request = GenerationRequest(
            prompt=request.message,
            model_name=request.model,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            system_prompt=request.system_prompt,
            metadata=request.metadata or {}
        )

        # Generate response
        if switcher and get_config().models.enable_model_switching:
            response = await switcher.execute_with_switching(
                gen_request,
                preferred_model=request.model,
                max_retries=get_config().models.max_retries
            )
        else:
            response = await agent_instance.generate_response(
                gen_request,
                request.conversation_id
            )

        return ChatResponse(
            content=response.content,
            model=response.metadata.get("selected_model", request.model),
            conversation_id=request.conversation_id,
            usage=response.usage,
            finish_reason=response.finish_reason,
            generation_time=response.generation_time,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def stream_chat(
    request: StreamChatRequest,
    agent_instance: LMArenaAgent = Depends(get_agent),
    switcher: Optional[ModelSwitcher] = Depends(get_model_switcher)
):
    """Streaming chat endpoint"""
    try:
        # Create generation request
        gen_request = GenerationRequest(
            prompt=request.message,
            model_name=request.model,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=True,
            system_prompt=request.system_prompt
        )

        async def generate():
            if switcher and get_config().models.enable_model_switching:
                # For streaming with model switching, we'll use the primary model
                model, _ = await switcher.select_model(gen_request, request.model)
                if not model:
                    raise Exception("No models available")

                async for chunk in model.generate_stream(gen_request):
                    yield f"data: {chunk}\n\n"
            else:
                model = agent_instance.model_registry.get_model(request.model)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")

                async for chunk in model.generate_stream(gen_request):
                    yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error("Stream chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def list_models(agent_instance: LMArenaAgent = Depends(get_agent)):
    """List available models"""
    models = agent_instance.model_registry.list_models(active_only=True)

    model_info = []
    for model in models:
        model_info.append(ModelInfo(
            name=model.name,
            provider=model.provider,
            model_id=model.config.model_id,
            capabilities={
                "max_tokens": model.capabilities.max_tokens,
                "supports_functions": model.capabilities.supports_functions,
                "supports_vision": model.capabilities.supports_vision,
                "supports_streaming": model.capabilities.supports_streaming,
                "context_window": model.capabilities.context_window
            },
            is_active=model.config.is_active,
            is_default=agent_instance.model_registry.get_default_model() == model
        ))

    return model_info


@app.post("/models/switch")
async def switch_model_strategy(
    request: ModelSwitchRequest,
    switcher: ModelSwitcher = Depends(get_model_switcher)
):
    """Switch model selection strategy"""
    if switcher is None:
        raise HTTPException(status_code=503, detail="Model switcher not available")

    try:
        strategy = SwitchingStrategy(request.strategy)
        switcher.set_strategy(strategy)

        return {
            "message": f"Model switching strategy changed to {request.strategy}",
            "current_strategy": strategy.value
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")


@app.post("/prompts", response_model=Dict[str, str])
async def create_prompt(
    request: PromptCreateRequest,
    prompt_manager_instance: PromptManager = Depends(get_prompt_manager)
):
    """Create a new prompt template"""
    try:
        from ..prompts.prompt_manager import PromptType, PromptCategory

        prompt_id = prompt_manager_instance.create_prompt(
            name=request.name,
            content=request.content,
            type=PromptType(request.type),
            category=PromptCategory(request.category),
            description=request.description,
            tags=request.tags or [],
            variables=request.variables or {}
        )

        return {"prompt_id": prompt_id, "message": "Prompt created successfully"}

    except Exception as e:
        logger.error("Failed to create prompt", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts", response_model=PromptListResponse)
async def list_prompts(
    category: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 50,
    prompt_manager_instance: PromptManager = Depends(get_prompt_manager)
):
    """List prompt templates"""
    try:
        from ..prompts.prompt_manager import PromptCategory, PromptType

        filters = {}
        if category:
            filters["category"] = PromptCategory(category)
        if type:
            filters["type"] = PromptType(type)

        filters["limit"] = limit

        templates = prompt_manager_instance.search(**filters)

        prompt_list = []
        for template in templates:
            prompt_list.append({
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "category": template.category.value,
                "description": template.description,
                "tags": template.tags,
                "usage_count": template.usage_count,
                "created_at": template.created_at,
                "updated_at": template.updated_at
            })

        return PromptListResponse(prompts=prompt_list, total=len(prompt_list))

    except Exception as e:
        logger.error("Failed to list prompts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations", response_model=Dict[str, str])
async def create_conversation(
    request: ConversationCreateRequest,
    agent_instance: LMArenaAgent = Depends(get_agent)
):
    """Create a new conversation"""
    try:
        conversation = agent_instance.get_or_create_conversation()

        if request.system_prompt:
            conversation.system_prompt = request.system_prompt

        return {
            "conversation_id": conversation.id,
            "message": "Conversation created successfully"
        }

    except Exception as e:
        logger.error("Failed to create conversation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    agent_instance: LMArenaAgent = Depends(get_agent)
):
    """List conversations"""
    try:
        conversations = agent_instance.list_conversations()

        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                "id": conv.id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "has_system_prompt": conv.system_prompt is not None
            })

        return ConversationListResponse(
            conversations=conversation_list,
            total=len(conversation_list)
        )

    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    agent_instance: LMArenaAgent = Depends(get_agent)
):
    """Delete a conversation"""
    try:
        success = agent_instance.delete_conversation(conversation_id)

        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete conversation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    agent_instance: LMArenaAgent = Depends(get_agent)
):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Create generation request
            gen_request = GenerationRequest(
                prompt=message_data.get("message", ""),
                model_name=message_data.get("model"),
                conversation_id=conversation_id,
                temperature=message_data.get("temperature"),
                max_tokens=message_data.get("max_tokens"),
                system_prompt=message_data.get("system_prompt")
            )

            # Send acknowledgment
            await websocket.send_text(json.dumps({
                "type": "status",
                "status": "processing"
            }))

            # Generate response
            response = await agent_instance.generate_response(gen_request, conversation_id)

            # Send response
            await websocket.send_text(json.dumps({
                "type": "response",
                "content": response.content,
                "model": response.metadata.get("selected_model"),
                "usage": response.usage,
                "generation_time": response.generation_time
            }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", conversation_id=conversation_id)
    except Exception as e:
        logger.error("WebSocket error", conversation_id=conversation_id, error=str(e))
        await websocket.close(code=1011, reason=str(e))


# Run the application
def run():
    """Run the FastAPI application"""
    config = get_config()

    uvicorn.run(
        "lm_arena.api.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.logging.level.lower()
    )


if __name__ == "__main__":
    run()