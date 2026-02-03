"""
Configuration module for the Multi-Agent Chat Threading System.
Loads sensitive environment variables and provides typed configuration access.

Note: Non-sensitive constants and prompts are in constants.py
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Only contains sensitive or deployment-specific configuration.
    """
    
    # Server configuration
    port: int = Field(default=8000, description="Server port")
    env: str = Field(default="development", description="Environment name")
    
    # Database configuration (sensitive)
    database_url: str = Field(
        default="postgresql+asyncpg://chatuser:chatpass@localhost:5432/chatdb",
        description="PostgreSQL connection string"
    )
    db_pool_size: int = Field(default=20, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, description="Max overflow connections")
    
    # OpenRouter API configuration (sensitive)
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key (required)"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    openrouter_timeout: int = Field(default=60, description="API timeout in seconds")
    
    # Model settings
    default_model: str = Field(
        default="openai/gpt-4-turbo",
        description="Default model for new threads"
    )
    summarization_model: str = Field(
        default="openai/gpt-4-turbo",
        description="Model used for generating summaries"
    )
    summarization_message_threshold: int = Field(
        default=10,
        description="Number of messages before triggering summarization"
    )
    
    # Context management
    max_context_messages: int = Field(
        default=20,
        description="Maximum messages to include in context"
    )
    max_context_tokens: int = Field(
        default=8000,
        description="Maximum tokens in context window"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or console)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

