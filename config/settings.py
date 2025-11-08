"""
LM Arena - Configuration Management System

Centralized configuration management with environment variables and file-based config.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./lm_arena.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    max_connections: int = 10
    retry_on_timeout: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_hosts: List[str] = field(default_factory=lambda: ["*"])
    enable_https: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_structlog: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    performance_tracking: bool = True
    enable_tracing: bool = False


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    timeout: int = 30
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    enable_docs: bool = True
    enable_cors: bool = True


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    default_model: str = "gpt-3.5-turbo"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo"])
    timeout: int = 60
    max_retries: int = 3
    rate_limit_buffer: int = 5  # Buffer for rate limit handling
    enable_model_switching: bool = True
    switching_strategy: str = "load_balanced"


class LMArenaConfig(BaseModel):
    """Main LM Arena configuration"""

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Core components
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

    # Storage paths
    data_dir: str = "./data"
    prompts_dir: str = "./prompts"
    models_dir: str = "./models"
    logs_dir: str = "./logs"

    # Feature flags
    enable_prompt_management: bool = True
    enable_conversation_history: bool = True
    enable_model_metrics: bool = True
    enable_webhooks: bool = False

    class Config:
        use_enum_values = True


class ConfigManager:
    """Configuration manager with environment variable support and file loading"""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self._config: Optional[LMArenaConfig] = None
        self._env_prefix = "LM_ARENA_"

    def load_config(self) -> LMArenaConfig:
        """Load configuration from file and environment variables"""
        # Start with default configuration
        config_dict = {}

        # Load from file if provided
        if self.config_file and self.config_file.exists():
            file_config = self._load_from_file()
            config_dict.update(file_config)
            logger.info("Configuration loaded from file", file=str(self.config_file))

        # Override with environment variables
        env_config = self._load_from_environment()
        config_dict = self._merge_configs(config_dict, env_config)

        # Validate and create configuration
        self._config = LMArenaConfig(**config_dict)
        logger.info("Configuration loaded successfully",
                   environment=self._config.environment.value,
                   debug=self._config.debug)

        return self._config

    def get_config(self) -> LMArenaConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)"""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif self.config_file.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_file.suffix}")
        except Exception as e:
            logger.error("Failed to load config file", file=str(self.config_file), error=str(e))
            return {}

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}

        # Environment-specific mappings
        env_mappings = {
            # Basic settings
            'ENVIRONMENT': ('environment', lambda x: x.lower()),
            'DEBUG': ('debug', lambda x: x.lower() in ['true', '1', 'yes']),

            # API settings
            'API_HOST': ('api.host', str),
            'API_PORT': ('api.port', int),
            'API_WORKERS': ('api.workers', int),
            'API_RELOAD': ('api.reload', lambda x: x.lower() in ['true', '1', 'yes']),

            # Database settings
            'DATABASE_URL': ('database.url', str),
            'DATABASE_POOL_SIZE': ('database.pool_size', int),

            # Redis settings
            'REDIS_URL': ('redis.url', str),
            'REDIS_PASSWORD': ('redis.password', str),

            # Security settings
            'SECRET_KEY': ('security.secret_key', str),
            'CORS_ORIGINS': ('security.cors_origins', lambda x: x.split(',')),

            # Logging settings
            'LOG_LEVEL': ('logging.level', lambda x: x.upper()),
            'LOG_FILE': ('logging.file_path', str),

            # Model settings
            'DEFAULT_MODEL': ('models.default_model', str),
            'FALLBACK_MODELS': ('models.fallback_models', lambda x: x.split(',')),
            'MODEL_TIMEOUT': ('models.timeout', int),

            # Storage paths
            'DATA_DIR': ('data_dir', str),
            'PROMPTS_DIR': ('prompts_dir', str),
            'MODELS_DIR': ('models_dir', str),
            'LOGS_DIR': ('logs_dir', str),

            # Feature flags
            'ENABLE_PROMPT_MANAGEMENT': ('enable_prompt_management', lambda x: x.lower() in ['true', '1', 'yes']),
            'ENABLE_CONVERSATION_HISTORY': ('enable_conversation_history', lambda x: x.lower() in ['true', '1', 'yes']),
            'ENABLE_MODEL_METRICS': ('enable_model_metrics', lambda x: x.lower() in ['true', '1', 'yes']),
        }

        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(f"{self._env_prefix}{env_var}")
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_value(config, config_path, converted_value)
                except Exception as e:
                    logger.warning("Failed to parse environment variable",
                                 env_var=env_var, value=env_value, error=str(e))

        return config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested value in the configuration dictionary"""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save_config(self, file_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration loaded")

        target_file = Path(file_path) if file_path else self.config_file
        if not target_file:
            raise ValueError("No file path specified")

        # Ensure directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        config_dict = self._config.dict()

        try:
            with open(target_file, 'w') as f:
                if target_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif target_file.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {target_file.suffix}")

            logger.info("Configuration saved", file=str(target_file))

        except Exception as e:
            logger.error("Failed to save config", file=str(target_file), error=str(e))
            raise

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self._config is None:
            self._config = self.load_config()

        # Convert to dict, update, and create new config
        config_dict = self._config.dict()
        merged_dict = self._merge_configs(config_dict, updates)
        self._config = LMArenaConfig(**merged_dict)

        logger.info("Configuration updated", updates=updates)

    def get_database_url(self) -> str:
        """Get database URL from configuration"""
        config = self.get_config()
        return config.database.url

    def get_redis_url(self) -> str:
        """Get Redis URL from configuration"""
        config = self.get_config()
        return config.redis.url

    def is_development(self) -> bool:
        """Check if running in development mode"""
        config = self.get_config()
        return config.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        config = self.get_config()
        return config.environment == Environment.PRODUCTION

    def create_directories(self):
        """Create necessary directories based on configuration"""
        config = self.get_config()

        directories = [
            config.data_dir,
            config.prompts_dir,
            config.models_dir,
            config.logs_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        logger.info("Directories created/verified")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> LMArenaConfig:
    """Get global configuration instance"""
    return config_manager.get_config()


def load_config(config_file: Optional[Union[str, Path]] = None) -> LMArenaConfig:
    """Load configuration from file and environment"""
    global config_manager

    if config_file:
        config_manager = ConfigManager(config_file)

    return config_manager.load_config()


# Environment detection utilities
def detect_environment() -> Environment:
    """Detect current environment"""
    env_var = os.getenv(f"{config_manager._env_prefix}ENVIRONMENT", "").lower()

    if env_var:
        try:
            return Environment(env_var)
        except ValueError:
            pass

    # Auto-detect based on common indicators
    if os.getenv("PYTEST_CURRENT_TEST"):
        return Environment.TESTING
    elif os.getenv("GITHUB_ACTIONS") or os.getenv("CI"):
        return Environment.STAGING
    elif os.getenv("NODE_ENV") == "production" or os.getenv("FLASK_ENV") == "production":
        return Environment.PRODUCTION

    return Environment.DEVELOPMENT


# Configuration validation
def validate_config(config: LMArenaConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    # Check required fields
    if not config.security.secret_key or config.security.secret_key == "your-secret-key-here":
        issues.append("Security secret key should be set for production")

    # Check database URL for production
    if config.environment == Environment.PRODUCTION:
        if config.database.url.startswith("sqlite:///"):
            issues.append("SQLite is not recommended for production")

    # Check CORS settings
    if config.environment == Environment.PRODUCTION and "*" in config.security.cors_origins:
        issues.append("Wildcard CORS origins not recommended for production")

    # Check model configuration
    if not config.models.default_model:
        issues.append("Default model should be specified")

    # Check directory paths
    for path_attr in ["data_dir", "prompts_dir", "models_dir", "logs_dir"]:
        path = getattr(config, path_attr)
        if not path:
            issues.append(f"{path_attr} should be specified")

    return issues