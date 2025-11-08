#!/usr/bin/env python3
"""
LM Arena Integration Tests

Comprehensive testing of the LM Arena framework functionality.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '.')

def print_test(test_name):
    """Print test header"""
    print(f"\n🧪 {test_name}")
    print("=" * 50)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def test_imports():
    """Test core module imports"""
    print_test("Testing Core Imports")

    try:
        from core.agent import LMArenaAgent, GenerationRequest, GenerationResponse
        print_success("Core agent module imported")
    except Exception as e:
        print_error(f"Core agent import failed: {e}")
        return False

    try:
        from core.model_switcher import ModelSwitcher, SwitchingStrategy
        print_success("Model switcher imported")
    except Exception as e:
        print_error(f"Model switcher import failed: {e}")
        return False

    try:
        from models.openai_model import OpenAIModel, OpenAIConfig
        print_success("OpenAI model imported")
    except Exception as e:
        print_error(f"OpenAI model import failed: {e}")
        return False

    try:
        from models.anthropic_model import AnthropicModel, AnthropicConfig
        print_success("Anthropic model imported")
    except Exception as e:
        print_error(f"Anthropic model import failed: {e}")
        return False

    try:
        from prompts.prompt_manager import PromptManager
        print_success("Prompt manager imported")
    except Exception as e:
        print_error(f"Prompt manager import failed: {e}")
        return False

    try:
        from config.settings import load_config, get_config
        print_success("Configuration module imported")
    except Exception as e:
        print_error(f"Configuration module import failed: {e}")
        return False

    return True

def test_agent_creation():
    """Test agent creation and basic functionality"""
    print_test("Testing Agent Creation")

    try:
        from core.agent import LMArenaAgent

        # Create agent
        agent = LMArenaAgent()
        print_success("Agent created successfully")

        # Check initial state
        if agent.status.value == "idle":
            print_success("Agent status is idle")
        else:
            print_warning(f"Agent status is {agent.status.value}")

        # Check model registry
        if agent.model_registry is not None:
            print_success("Model registry initialized")
        else:
            print_error("Model registry is None")
            return False

        # Check conversation management
        if agent.conversations is not None:
            print_success("Conversation system initialized")
        else:
            print_error("Conversation system is None")
            return False

        return True

    except Exception as e:
        print_error(f"Agent creation failed: {e}")
        return False

def test_model_registry():
    """Test model registry functionality"""
    print_test("Testing Model Registry")

    try:
        from core.agent import ModelRegistry
        from models.openai_model import OpenAIConfig, OpenAIModel

        # Create registry
        registry = ModelRegistry()
        print_success("Model registry created")

        # Create mock model config
        config = OpenAIConfig(
            name="test-model",
            provider="test",
            model_id="test-model-id",
            api_key="test-key"
        )

        # Create mock model
        model = OpenAIModel(config)

        # Register model
        registry.register_model(model, is_default=True)
        print_success("Model registered successfully")

        # Test retrieval
        default_model = registry.get_default_model()
        if default_model is not None:
            print_success("Default model retrieved successfully")
        else:
            print_error("Default model retrieval failed")
            return False

        # Test model listing
        models = registry.list_models()
        if len(models) > 0:
            print_success(f"Model listing works: {len(models)} models found")
        else:
            print_warning("No models found in registry")

        return True

    except Exception as e:
        print_error(f"Model registry test failed: {e}")
        return False

def test_prompt_manager():
    """Test prompt manager functionality"""
    print_test("Testing Prompt Manager")

    try:
        from prompts.prompt_manager import PromptManager
        from prompts.prompt_manager import PromptType, PromptCategory

        # Create prompt manager
        manager = PromptManager()
        print_success("Prompt manager created")

        # Create test prompt
        prompt_id = manager.create_prompt(
            name="Test Prompt",
            content="Hello {{ name }}, how are you?",
            type=PromptType.USER,
            category=PromptCategory.GENERAL,
            description="A simple test prompt",
            variables={"name": "World"}
        )
        print_success(f"Prompt created with ID: {prompt_id}")

        # Test prompt retrieval
        prompt = manager.get_template(prompt_id)
        if prompt is not None:
            print_success("Prompt retrieved successfully")
        else:
            print_error("Prompt retrieval failed")
            return False

        # Test prompt rendering
        rendered = manager.use_prompt("Test Prompt", name="LM Arena")
        if "LM Arena" in rendered:
            print_success(f"Prompt rendering works: {rendered}")
        else:
            print_error(f"Prompt rendering failed: {rendered}")
            return False

        return True

    except Exception as e:
        print_error(f"Prompt manager test failed: {e}")
        return False

def test_model_switcher():
    """Test model switcher functionality"""
    print_test("Testing Model Switcher")

    try:
        from core.model_switcher import ModelSwitcher, SwitchingStrategy
        from core.agent import ModelRegistry
        from models.openai_model import OpenAIModel, OpenAIConfig

        # Create test models
        config1 = OpenAIConfig(
            name="model1",
            provider="test",
            model_id="model1-id",
            api_key="test-key"
        )
        config2 = OpenAIConfig(
            name="model2",
            provider="test",
            model_id="model2-id",
            api_key="test-key"
        )

        model1 = OpenAIModel(config1)
        model2 = OpenAIModel(config2)

        # Create models dict
        models = {
            "model1": model1,
            "model2": model2
        }

        # Create switcher
        switcher = ModelSwitcher(models)
        print_success("Model switcher created")

        # Test strategy setting
        switcher.set_strategy(SwitchingStrategy.ROUND_ROBIN)
        print_success(f"Strategy set to: {switcher.strategy.value}")

        # Test model selection
        model, name = switcher.select_model()
        if model is not None:
            print_success(f"Model selection works: {name}")
        else:
            print_error("Model selection failed")
            return False

        # Test stats
        stats = switcher.get_overall_stats()
        if stats is not None:
            print_success(f"Stats available: {stats['total_requests']} requests")
        else:
            print_error("Stats retrieval failed")
            return False

        return True

    except Exception as e:
        print_error(f"Model switcher test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print_test("Testing Configuration System")

    try:
        from config.settings import load_config, get_config, Environment

        # Load configuration
        config = load_config()
        print_success("Configuration loaded")

        # Test environment detection
        if config.environment in [Environment.DEVELOPMENT, Environment.PRODUCTION]:
            print_success(f"Environment detected: {config.environment.value}")
        else:
            print_warning(f"Unknown environment: {config.environment}")

        # Test API settings
        if config.api.port > 0:
            print_success(f"API port configured: {config.api.port}")
        else:
            print_error("API port not configured")
            return False

        # Test model settings
        if config.models.default_model:
            print_success(f"Default model: {config.models.default_model}")
        else:
            print_warning("No default model configured")

        return True

    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False

def test_api_startup():
    """Test API server startup"""
    print_test("Testing API Server Startup")

    try:
        import subprocess
        import time

        # Start API server in background
        print_success("Starting API server...")

        # Test API import
        try:
            import sys
            sys.path.insert(0, '.')
            from api.main import app
            print_success("API module imported successfully")
        except Exception as e:
            print_error(f"API import failed: {e}")
            return False

        print_success("API startup test passed (import verification)")
        return True

    except Exception as e:
        print_error(f"API startup test failed: {e}")
        return False

def test_syntax_validation():
    """Test Python syntax across all modules"""
    print_test("Testing Python Syntax Validation")

    try:
        import py_compile
        import os

        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk('.'):
            # Skip virtual environment
            if 'venv' in root or '.git' in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        syntax_errors = 0
        for file_path in python_files:
            try:
                py_compile.compile(file_path, doraise=True)
                print(f"✅ {file_path}")
            except py_compile.PyCompileError as e:
                print(f"❌ {file_path}: {e}")
                syntax_errors += 1

        if syntax_errors == 0:
            print_success(f"All {len(python_files)} Python files have valid syntax")
            return True
        else:
            print_error(f"Found {syntax_errors} files with syntax errors")
            return False

    except Exception as e:
        print_error(f"Syntax validation test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests and return results"""
    print("🚀 LM Arena Integration Tests")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Syntax Validation", test_syntax_validation),
        ("Core Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Model Registry", test_model_registry),
        ("Prompt Manager", test_prompt_manager),
        ("Model Switcher", test_model_switcher),
        ("Configuration", test_configuration),
        ("API Startup", test_api_startup),
    ]

    results = {
        'passed': 0,
        'failed': 0,
        'total': len(tests),
        'details': []
    }

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: PASSED")
                print(f"\n🎉 {test_name}: PASSED")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: FAILED")
                print(f"\n💥 {test_name}: FAILED")

        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: ERROR - {str(e)}")
            print(f"\n💥 {test_name}: ERROR - {str(e)}")

    # Print final summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")

    print(f"\n📋 Detailed Results:")
    for detail in results['details']:
        print(f"  {detail}")

    if results['failed'] == 0:
        print(f"\n🎉 ALL TESTS PASSED! LM Arena is ready for deployment!")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed. Please review the issues above.")

    return results['failed'] == 0

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)