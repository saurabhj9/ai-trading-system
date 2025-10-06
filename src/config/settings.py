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


class RegimeDetectionSettings(BaseSettings):
    """
    Configuration for the market regime detection system.
    """
    model_config = SettingsConfigDict(env_prefix='REGIME_DETECTION_')

    ENABLED: bool = True
    ADX_PERIOD: int = 14
    ATR_PERIOD: int = 14
    HURST_EXPONENT_LAG: int = 100
    TREND_STRENGTH_THRESHOLD: float = 25.0
    RANGING_THRESHOLD: float = 20.0
    VOLATILITY_THRESHOLD_PERCENT: float = 2.5
    CONFIRMATION_PERIODS: int = 3


class SignalGenerationSettings(BaseSettings):
    """
    Configuration for the signal generation system integration.
    """
    model_config = SettingsConfigDict(env_prefix='SIGNAL_GENERATION_')

    # Integration mode settings
    LOCAL_SIGNAL_GENERATION_ENABLED: bool = False  # Feature flag for gradual rollout
    HYBRID_MODE_ENABLED: bool = False  # Enable hybrid local+LLM operation
    FALLBACK_TO_LLM_ON_ERROR: bool = True  # Fall back to LLM if local generation fails

    # Gradual rollout controls
    ROLLOUT_PERCENTAGE: float = 0.0  # Percentage of requests to use local generation (0.0-1.0)
    ENABLED_SYMBOLS: List[str] = []  # Specific symbols to enable for local generation
    ENABLED_TIMEFRAMES: List[str] = []  # Specific timeframes to enable for local generation

    # Performance comparison settings
    ENABLE_SIDE_BY_SIDE_COMPARISON: bool = False  # Generate both local and LLM signals for comparison
    COMPARISON_SAMPLE_RATE: float = 0.1  # Percentage of requests to sample for comparison (0.0-1.0)

    # Escalation settings
    ESCALATION_ENABLED: bool = True  # Enable escalation to LLM for complex cases
    ESCALATION_CONFIDENCE_THRESHOLD: float = 0.3  # Escalate if local confidence is below this
    ESCALATION_CONFLICT_THRESHOLD: int = 2  # Escalate if this many conflicts are detected

    # Performance monitoring
    TRACK_PERFORMANCE_METRICS: bool = True  # Track performance comparison metrics
    METRICS_RETENTION_DAYS: int = 30  # Days to retain performance metrics


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
    regime_detection: RegimeDetectionSettings = RegimeDetectionSettings()
    signal_generation: SignalGenerationSettings = SignalGenerationSettings()

    # Direct environment variable access
    OPENROUTER_API_KEY: Optional[str] = None


settings = Settings()
