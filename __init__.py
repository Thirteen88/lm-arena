"""
LM Arena - Multi-Model AI Agent Framework

A flexible, extensible framework for building AI agents with multiple model support,
intelligent model switching, and advanced prompt management.
"""

__version__ = "1.0.0"
__author__ = "LM Arena Team"
__email__ = "team@lm-arena.com"

# Core imports
from .core.agent import LMArenaAgent, GenerationRequest, GenerationResponse
from .core.model_switcher import ModelSwitcher, SwitchingStrategy
from .models.openai_model import create_openai_model, create_openai_compatible_model
from .models.anthropic_model import create_anthropic_model
from .prompts.prompt_manager import PromptManager
from .config.settings import load_config, get_config

__all__ = [
    # Core
    "LMArenaAgent",
    "GenerationRequest",
    "GenerationResponse",

    # Model Switching
    "ModelSwitcher",
    "SwitchingStrategy",

    # Models
    "create_openai_model",
    "create_openai_compatible_model",
    "create_anthropic_model",

    # Prompt Management
    "PromptManager",

    # Configuration
    "load_config",
    "get_config",
]