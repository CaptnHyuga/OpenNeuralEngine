"""Configuration management with .env support and validation.

Centralized config module for web interfaces (Gradio, Streamlit, API) and
CLI tools. Uses Pydantic Settings for validation with fallback to dataclass.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Load .env file if present (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Try Pydantic Settings first (preferred), fallback to dataclass
try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass, field

# Manual .env loading fallback (only if dotenv not available)
if ENV_FILE.exists() and not PYDANTIC_AVAILABLE:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except ImportError:
        with ENV_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    if key not in os.environ:
                        os.environ[key] = value


if PYDANTIC_AVAILABLE:
    class AppConfig(BaseSettings):
        """Unified configuration with validation."""
        model_config = SettingsConfigDict(
            env_file=str(ENV_FILE),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )
        
        # API Keys & Auth
        puzzle_api_key: str = Field(default="", alias="PUZZLE_API_KEY")
        gradio_user: str = Field(default="admin", alias="GRADIO_USER")
        gradio_pass: str = Field(default="", alias="GRADIO_PASS")
        
        # Server Configuration
        api_host: str = Field(default="127.0.0.1", alias="API_HOST")
        api_port: int = Field(default=8000, ge=1024, le=65535, alias="API_PORT")
        gradio_host: str = Field(default="127.0.0.1", alias="GRADIO_HOST")
        gradio_port: int = Field(default=7860, ge=1024, le=65535, alias="GRADIO_PORT")
        
        # Rate Limiting
        rate_limit_requests: int = Field(default=60, ge=1, alias="RATE_LIMIT_REQUESTS")
        rate_limit_window: int = Field(default=60, ge=1, alias="RATE_LIMIT_WINDOW")
        
        # Input Limits
        max_prompt_length: int = Field(default=8192, ge=1, alias="MAX_PROMPT_LENGTH")
        max_tokens_limit: int = Field(default=2048, ge=1, le=32768, alias="MAX_TOKENS_LIMIT")
        
        # Logging
        log_level: str = Field(default="INFO", alias="LOG_LEVEL")
        log_file: Optional[str] = Field(default=None, alias="LOG_FILE")
        
        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            v = v.upper()
            if v not in allowed:
                raise ValueError(f"log_level must be one of {allowed}")
            return v
        
        @property
        def api_key(self) -> str:
            """Alias for backward compatibility."""
            return self.puzzle_api_key
        
        @property
        def gradio_auth(self) -> Optional[tuple[str, str]]:
            if self.gradio_user and self.gradio_pass:
                return (self.gradio_user, self.gradio_pass)
            return None
        
        def validate(self) -> list[str]:
            """Backward compatibility validation method."""
            issues = []
            if not self.puzzle_api_key:
                issues.append("PUZZLE_API_KEY not set")
            if not self.gradio_pass:
                issues.append("GRADIO_PASS not set")
            return issues

else:
    # Fallback to dataclass if pydantic-settings not available
    @dataclass
    class AppConfig:
        """Unified configuration for all interfaces."""

        # API Keys & Auth
        api_key: str = field(default_factory=lambda: os.environ.get("PUZZLE_API_KEY", ""))
        gradio_user: str = field(default_factory=lambda: os.environ.get("GRADIO_USER", "admin"))
        gradio_pass: str = field(default_factory=lambda: os.environ.get("GRADIO_PASS", ""))

        # Server Configuration
        api_host: str = field(default_factory=lambda: os.environ.get("API_HOST", "127.0.0.1"))
        api_port: int = field(default_factory=lambda: int(os.environ.get("API_PORT", "8000")))
        gradio_host: str = field(default_factory=lambda: os.environ.get("GRADIO_HOST", "127.0.0.1"))
        gradio_port: int = field(default_factory=lambda: int(os.environ.get("GRADIO_PORT", "7860")))

        # Rate Limiting
        rate_limit_requests: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_REQUESTS", "60")))
        rate_limit_window: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_WINDOW", "60")))

        # Input Limits
        max_prompt_length: int = field(default_factory=lambda: int(os.environ.get("MAX_PROMPT_LENGTH", "8192")))
        max_tokens_limit: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS_LIMIT", "2048")))

        # Logging
        log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
        log_file: Optional[str] = field(default_factory=lambda: os.environ.get("LOG_FILE"))

        def validate(self) -> list[str]:
            issues = []
            if not self.api_key:
                issues.append("PUZZLE_API_KEY not set")
            if not self.gradio_pass:
                issues.append("GRADIO_PASS not set")
            return issues

        @property
        def gradio_auth(self) -> Optional[tuple[str, str]]:
            if self.gradio_user and self.gradio_pass:
                return (self.gradio_user, self.gradio_pass)
            return None

        def __post_init__(self):
            import logging
            logger = logging.getLogger(__name__)
            
            # Validate port ranges
            for port_name in ("api_port", "gradio_port"):
                port_val = getattr(self, port_name)
                if not (1024 <= port_val <= 65535):
                    logger.warning("Config: %s=%d outside valid range [1024, 65535]", port_name, port_val)
            
            # Validate positive integers
            for int_name in ("rate_limit_requests", "rate_limit_window", "max_prompt_length", "max_tokens_limit"):
                int_val = getattr(self, int_name)
                if int_val < 1:
                    logger.warning("Config: %s=%d must be >= 1", int_name, int_val)
            
            # Validate max_tokens_limit upper bound
            if self.max_tokens_limit > 32768:
                logger.warning("Config: max_tokens_limit=%d exceeds 32768", self.max_tokens_limit)
            
            # Validate log level
            allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if self.log_level.upper() not in allowed_levels:
                logger.warning("Config: log_level=%s not in %s", self.log_level, allowed_levels)
            
            # Report missing credentials
            issues = self.validate()
            for issue in issues:
                logger.warning("Config: %s", issue)


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    global _config
    _config = AppConfig()
    return _config
