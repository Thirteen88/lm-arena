# LM Arena - Multi-Model AI Agent Framework

A flexible, extensible AI agent framework that supports multiple models, advanced prompt management, and intelligent model switching.

## 🚀 Features

### Core Capabilities
- **Multi-Model Support**: Seamless integration with OpenAI, Anthropic Claude, and OpenAI-compatible models
- **Intelligent Model Switching**: Load balancing, cost optimization, and performance-based routing
- **Advanced Prompt Management**: Template system with versioning, categories, and variable substitution
- **Conversation Management**: Persistent conversation history with context tracking
- **Real-time Streaming**: WebSocket support for real-time AI responses
- **Comprehensive Monitoring**: Performance metrics, cost tracking, and health monitoring

### Technical Highlights
- **Async Architecture**: Built with FastAPI for high-performance async operations
- **Plugin System**: Easy addition of new models and capabilities
- **Configuration-Driven**: Flexible configuration with environment variables
- **REST + WebSocket APIs**: Complete API coverage for all functionality
- **Production Ready**: Error handling, retries, and circuit breakers

## 📋 Requirements

- Python 3.8+
- Optional: OpenAI API key, Anthropic API key
- Optional: Redis for caching (recommended for production)

## 🛠️ Installation

### Quick Start

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd lm-arena
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   # Copy example configuration
   cp config.yaml.example config.yaml

   # Or set environment variables
   export LM_ARENA_OPENAI_API_KEY="your-openai-key"
   export LM_ARENA_ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Start the server**
   ```bash
   python -m lm_arena.api.main
   ```

### Docker Setup

```bash
# Build image
docker build -t lm-arena .

# Run container
docker run -p 8000:8000 \
  -e LM_ARENA_OPENAI_API_KEY="your-key" \
  lm-arena
```

## 📖 Usage

### Basic Chat

```python
import asyncio
from lm_arena import LMArenaAgent
from lm_arena.models.openai_model import create_openai_model

async def main():
    # Create agent
    agent = LMArenaAgent()

    # Add model
    model = create_openai_model("gpt-4", "gpt-4", "your-api-key")
    agent.model_registry.register_model(model, is_default=True)

    # Initialize
    await agent.initialize()

    # Generate response
    from lm_arena.core.agent import GenerationRequest
    request = GenerationRequest(prompt="Explain quantum computing")
    response = await agent.generate_response(request)

    print(f"Response: {response.content}")
    print(f"Model: {response.model_name}")
    print(f"Cost: ${response.cost:.4f}")

asyncio.run(main())
```

### Using Model Switching

```python
from lm_arena.core.model_switcher import ModelSwitcher, SwitchingStrategy

# Create switcher with multiple models
switcher = ModelSwitcher(models)
switcher.set_strategy(SwitchingStrategy.LOAD_BALANCED)

# Generate with automatic model switching
response = await switcher.execute_with_switching(
    GenerationRequest(prompt="Hello"),
    max_retries=3
)
```

### Prompt Templates

```python
from lm_arena.prompts.prompt_manager import PromptManager

# Create prompt manager
prompt_manager = PromptManager()

# Create a template
prompt_manager.create_prompt(
    name="Code Review",
    content="Review this code:\\n\\n{{ code }}\\n\\nProvide feedback on quality and security.",
    category="coding",
    variables={"code": ""}
)

# Use template
rendered = prompt_manager.use_prompt(
    "Code Review",
    code="def hello(): print('Hello, World!')"
)
```

## 🌐 API Usage

### REST API

```bash
# Start a conversation
curl -X POST "http://localhost:8000/conversations" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a helpful assistant"}'

# Chat with the agent
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain machine learning",
    "model": "gpt-4",
    "conversation_id": "your-conversation-id"
  }'
```

### Streaming Chat

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a story",
    "stream": true
  }'
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/conversation-id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'response') {
    console.log('Response:', data.content);
  }
};

ws.send(JSON.stringify({
  message: "Hello, AI!",
  model: "gpt-4"
}));
```

## ⚙️ Configuration

### Environment Variables

```bash
# Core settings
LM_ARENA_ENVIRONMENT=development
LM_ARENA_DEBUG=true
LM_ARENA_HOST=0.0.0.0
LM_ARENA_PORT=8000

# API Keys
LM_ARENA_OPENAI_API_KEY=sk-...
LM_ARENA_ANTHROPIC_API_KEY=sk-ant-...

# Models
LM_ARENA_DEFAULT_MODEL=gpt-4
LM_ARENA_FALLBACK_MODELS=gpt-3.5-turbo,claude-3-haiku
LM_ARENA_MODEL_SWITCHING_STRATEGY=load_balanced

# Database (optional)
LM_ARENA_DATABASE_URL=postgresql://user:pass@localhost/lmarena
LM_ARENA_REDIS_URL=redis://localhost:6379

# Security
LM_ARENA_SECRET_KEY=your-secret-key
LM_ARENA_CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Configuration File

```yaml
# config.yaml
environment: production
debug: false

api:
  host: 0.0.0.0
  port: 8000
  workers: 4

models:
  default_model: "gpt-4"
  fallback_models: ["gpt-3.5-turbo", "claude-3-haiku"]
  enable_model_switching: true
  switching_strategy: "load_balanced"

security:
  secret_key: "your-production-secret-key"
  cors_origins: ["https://yourdomain.com"]

monitoring:
  enable_metrics: true
  metrics_port: 9090
```

## 🎯 Model Switching Strategies

### Available Strategies

1. **Round Robin**: Rotate through models evenly
2. **Load Balanced**: Route to model with fewest active requests
3. **Cost Optimized**: Prefer cheapest available model
4. **Performance Optimized**: Prefer fastest responding model
5. **Priority Based**: Use models with highest priority first
6. **Adaptive**: Intelligent selection based on multiple factors
7. **Random**: Random model selection

### Routing Rules

Create custom routing rules for specific request patterns:

```python
from lm_arena.core.model_switcher import RoutingRule

# Route coding requests to specialized models
coding_rule = RoutingRule(
    name="coding_requests",
    condition=lambda req: "code" in req.prompt.lower() or "programming" in req.prompt.lower(),
    preferred_models=["gpt-4", "claude-3-opus"],
    priority=10
)

switcher.add_routing_rule(coding_rule)
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Statistics

```bash
curl http://localhost:8000/stats
```

### Metrics (Prometheus)

Available at `http://localhost:9090/metrics` when monitoring is enabled.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lm_arena --cov-report=html

# Run specific tests
pytest tests/test_agent.py -v
```

## 📦 Deployment

### Production with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  lm-arena:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LM_ARENA_ENVIRONMENT=production
      - LM_ARENA_DATABASE_URL=postgresql://postgres:password@db:5432/lmarena
      - LM_ARENA_REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: lmarena
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 🔧 Extending LM Arena

### Adding New Models

```python
from lm_arena.core.agent import ModelInterface, ModelConfig

class CustomModel(ModelInterface):
    async def generate(self, request):
        # Your custom model implementation
        pass

# Register the model
model = CustomModel(config)
agent.model_registry.register_model(model)
```

### Custom Prompt Templates

```python
from lm_arena.prompts.prompt_manager import PromptTemplate, PromptType

template = PromptTemplate(
    name="Custom Analysis",
    content="Analyze: {{ text }} with {{ perspective }}",
    type=PromptType.USER,
    category="analysis"
)

prompt_manager.library.add_template(template)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [OpenAI](https://openai.com/) - GPT models and API
- [Anthropic](https://anthropic.com/) - Claude models and API
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

---

**Built with ❤️ for AI enthusiasts and developers**