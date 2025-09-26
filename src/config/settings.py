"""
Centralized configuration management for the AI Trading System.

This module uses pydantic-settings to manage configuration from environment
variables and .env files, providing a structured and validated way to
access settings throughout the application.
"""
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """
    Configuration for the Language Model client.
    """
    model_config = SettingsConfigDict(env_prefix='LLM_')

    SITE_URL: str = "https://my-site.com"
    APP_NAME: str = "AI Trading System"
    BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "anthropic/claude-3-haiku"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour default TTL for LLM response caching


class PortfolioSettings(BaseSettings):
    """
    Configuration for the portfolio.
    """
    model_config = SettingsConfigDict(env_prefix='PORTFOLIO_')

    STARTING_CASH: float = 100000.0


class DataSettings(BaseSettings):
    """
    Configuration for data providers and caching.
    """
    model_config = SettingsConfigDict(env_prefix='DATA_')

    CACHE_ENABLED: bool = True
    ALPHA_VANTAGE_API_KEY: Optional[str] = None


class APISettings(BaseSettings):
    """
    Configuration for the FastAPI application.
    """
    model_config = SettingsConfigDict(env_prefix='API_')

    CORS_ALLOWED_ORIGINS: List[str] = ["*"]


class Settings(BaseSettings):
    """
    Main settings object that aggregates all other settings.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    llm: LLMSettings = LLMSettings()
    portfolio: PortfolioSettings = PortfolioSettings()
    data: DataSettings = DataSettings()
    api: APISettings = APISettings()

    # Direct environment variable access
    OPENROUTER_API_KEY: Optional[str] = None


settings = Settings()
