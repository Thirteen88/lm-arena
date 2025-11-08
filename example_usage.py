#!/usr/bin/env python3
"""
LM Arena - Example Usage

Demonstrates how to use LM Arena for various tasks.
"""

import asyncio
import os
from lm_arena import (
    LMArenaAgent,
    create_openai_model,
    create_anthropic_model,
    PromptManager,
    GenerationRequest,
    SwitchingStrategy,
    ModelSwitcher
)


async def basic_example():
    """Basic usage example"""
    print("🤖 LM Arena - Basic Example")
    print("=" * 50)

    # Create agent
    agent = LMArenaAgent()

    # Add OpenAI model (if API key is available)
    openai_key = os.getenv("LM_ARENA_OPENAI_API_KEY")
    if openai_key:
        model = create_openai_model("gpt-3.5-turbo", "gpt-3.5-turbo", openai_key)
        agent.model_registry.register_model(model, is_default=True)
        print("✅ OpenAI GPT-3.5-turbo registered")
    else:
        print("⚠️  No OpenAI API key found")

    # Initialize agent
    try:
        await agent.initialize()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return

    # Generate response
    request = GenerationRequest(
        prompt="Explain quantum computing in simple terms",
        max_tokens=200
    )

    try:
        response = await agent.generate_response(request)
        print(f"\n📝 Response: {response.content}")
        print(f"🤖 Model: {response.model_name}")
        print(f"💰 Cost: ${response.cost:.4f}")
        print(f"⏱️  Time: {response.generation_time:.2f}s")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

    # Show stats
    stats = agent.get_stats()
    print(f"\n📊 Stats: {stats['total_requests']} requests, ${stats['total_cost']:.4f} total")


async def model_switching_example():
    """Model switching example"""
    print("\n🔄 LM Arena - Model Switching Example")
    print("=" * 50)

    # Create models
    models = {}
    openai_key = os.getenv("LM_ARENA_OPENAI_API_KEY")
    anthropic_key = os.getenv("LM_ARENA_ANTHROPIC_API_KEY")

    if openai_key:
        models["gpt-3.5"] = create_openai_model("gpt-3.5", "gpt-3.5-turbo", openai_key)
        models["gpt-4"] = create_openai_model("gpt-4", "gpt-4", openai_key)
        print("✅ OpenAI models registered")

    if anthropic_key:
        models["claude-3-haiku"] = create_anthropic_model("claude-3-haiku", "claude-3-haiku-20240307", anthropic_key)
        print("✅ Anthropic models registered")

    if not models:
        print("❌ No API keys found, skipping model switching example")
        return

    # Create model switcher
    switcher = ModelSwitcher(models)
    switcher.set_strategy(SwitchingStrategy.LOAD_BALANCED)
    print("✅ Model switcher initialized")

    # Test multiple requests
    requests = [
        "What is machine learning?",
        "Explain photosynthesis",
        "Write a haiku about programming",
        "What are the benefits of meditation?"
    ]

    for i, prompt in enumerate(requests, 1):
        print(f"\n📝 Request {i}: {prompt}")

        try:
            request = GenerationRequest(prompt=prompt)
            response = await switcher.execute_with_switching(request, max_retries=2)

            print(f"🤖 Model: {response.metadata.get('selected_model', 'Unknown')}")
            print(f"📄 Response: {response.content[:100]}...")
            print(f"⏱️  Time: {response.generation_time:.2f}s")

        except Exception as e:
            print(f"❌ Request failed: {e}")

    # Show overall stats
    overall_stats = switcher.get_overall_stats()
    print(f"\n📊 Overall Stats:")
    print(f"   Total requests: {overall_stats['total_requests']}")
    print(f"   Success rate: {overall_stats['overall_success_rate']:.1f}%")
    print(f"   Total cost: ${overall_stats['total_cost']:.4f}")


async def prompt_management_example():
    """Prompt management example"""
    print("\n📚 LM Arena - Prompt Management Example")
    print("=" * 50)

    # Create prompt manager
    prompt_manager = PromptManager()

    # Create custom prompt templates
    prompt_manager.create_prompt(
        name="Code Reviewer",
        content="""You are an expert code reviewer. Please analyze the following code and provide:

1. Security issues
2. Performance considerations
3. Code quality and readability
4. Best practices adherence

Code to review:
{{ code }}

Language: {{ language }}""",
        category="coding",
        description="Template for code review requests",
        tags=["code", "review", "security", "performance"],
        variables={"code": "", "language": "python"}
    )

    prompt_manager.create_prompt(
        name="Business Email",
        content="""Write a professional business email with the following details:

To: {{ recipient }}
Subject: {{ subject }}
Tone: {{ tone }}
Key points: {{ key_points }}

Requirements:
- Professional and courteous tone
- Clear and concise language
- Proper email structure
- Call to action if needed""",
        category="business",
        description="Template for writing business emails",
        tags=["email", "business", "communication"],
        variables={
            "recipient": "",
            "subject": "",
            "tone": "professional",
            "key_points": ""
        }
    )

    print("✅ Created prompt templates")

    # Use the code reviewer template
    code_to_review = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total
"""

    rendered_prompt = prompt_manager.use_prompt(
        "Code Reviewer",
        code=code_to_review,
        language="python"
    )

    print(f"\n📝 Code Review Prompt:")
    print(rendered_prompt)

    # Use the business email template
    email_prompt = prompt_manager.use_prompt(
        "Business Email",
        recipient="team@company.com",
        subject="Project Update - Q4 Results",
        tone="professional",
        key_points="Project completed on time, under budget, positive client feedback"
    )

    print(f"\n📧 Business Email Prompt:")
    print(email_prompt)

    # Search prompts
    coding_prompts = prompt_manager.search(category="coding", limit=5)
    print(f"\n🔍 Found {len(coding_prompts)} coding prompts")
    for prompt in coding_prompts:
        print(f"   - {prompt.name}: {prompt.description}")


async def conversation_example():
    """Conversation management example"""
    print("\n💬 LM Arena - Conversation Management Example")
    print("=" * 50)

    # Create agent
    agent = LMArenaAgent()

    # Add mock model for demo (since we might not have API keys)
    from lm_arena.models.openai_model import create_openai_compatible_model

    try:
        # Try to create a local model
        model = create_openai_compatible_model(
            name="local-demo",
            model_id="llama2",
            api_base="http://localhost:11434/v1"  # Ollama default
        )
        agent.model_registry.register_model(model, is_default=True)
        print("✅ Local model registered")
    except:
        print("⚠️  No local model available, using mock for demo")
        # In real usage, you'd have proper API keys
        return

    # Initialize agent
    try:
        await agent.initialize()
    except:
        print("⚠️  Agent initialization failed, continuing with demo")

    # Create conversation
    conversation = agent.get_or_create_conversation("demo-conversation")
    conversation.system_prompt = "You are a helpful AI assistant specializing in Python programming."
    print("✅ Created conversation with system prompt")

    # Simulate conversation
    messages = [
        "What are the benefits of using list comprehensions in Python?",
        "Can you show me an example?",
        "How do they compare to traditional for loops in terms of performance?"
    ]

    for i, message in enumerate(messages, 1):
        print(f"\n👤 User {i}: {message}")

        try:
            request = GenerationRequest(
                prompt=message,
                conversation_id="demo-conversation"
            )
            response = await agent.generate_response(request, "demo-conversation")

            print(f"🤖 Assistant: {response.content[:150]}...")
            print(f"   (Model: {response.model_name}, Tokens: {response.usage.get('total_tokens', 0)})")

        except Exception as e:
            print(f"   Error: {e}")

    # Show conversation details
    print(f"\n📊 Conversation details:")
    print(f"   Messages: {len(conversation.messages)}")
    print(f"   Duration: {conversation.updated_at - conversation.created_at:.1f}s")
    print(f"   Has system prompt: {conversation.system_prompt is not None}")

    # List all conversations
    conversations = agent.list_conversations()
    print(f"\n📋 Total conversations: {len(conversations)}")


async def main():
    """Run all examples"""
    print("🚀 LM Arena - Example Usage")
    print("=" * 60)

    # Run examples
    await basic_example()
    await model_switching_example()
    await prompt_management_example()
    await conversation_example()

    print("\n✅ Examples completed!")
    print("\nTo use with real models:")
    print("1. Set API keys: export LM_ARENA_OPENAI_API_KEY='your-key'")
    print("2. Install required: pip install openai anthropic")
    print("3. Run the server: python -m lm_arena.main")


if __name__ == "__main__":
    asyncio.run(main())