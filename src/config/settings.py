"""
Centralized configuration management for the AI Trading System.

This module implements a 3-tier configuration architecture for optimal
separation of concerns and maintainability:

Tier 1: Code Defaults (settings.py)
- Default values for all feature flags and business logic
- Version controlled, visible in PRs
- Provides sensible production-ready defaults

Tier 2: Config Files (Optional: config/*.yaml)
- Environment-specific configurations (dev/staging/prod)
- Complex nested configurations
- Version controlled, easy to review

Tier 3: Environment Variables (.env)
- Only secrets and credentials (API keys, tokens)
- Never committed to git
- Can override any Tier 1/2 setting for local development

Example Override Pattern:
- Default in code: LOCAL_SIGNAL_GENERATION_ENABLED = True
- Override in .env: SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED=false

Benefits:
✅ Clear separation of secrets vs configuration
✅ Feature flag changes visible in PRs
✅ Better documentation through code defaults
✅ Easier to maintain defaults
✅ Can still override locally via .env

This module uses pydantic-settings to manage configuration from environment
variables and .env files, providing a structured and validated way to
access settings throughout the application.
"""
from typing import List, Optional, Dict
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

    # Provider-specific configuration
    PROVIDER: str = "openrouter"  # openrouter, openai_direct, anthropic_direct, etc.

    # Direct provider configurations (when not using OpenRouter as proxy)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-haiku-20240307"

    # Provider model mappings
    PROVIDER_MODELS: Dict[str, str] = {
        "openrouter": "anthropic/claude-3-haiku",
        "openai_direct": "gpt-4o-mini",
        "anthropic_direct": "claude-3-haiku-20240307",
    }


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
    MARKETAUX_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None


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

    This provides production-ready defaults for the local signal generation framework.
    These defaults are version-controlled and can be overridden via environment variables.
    """
    model_config = SettingsConfigDict(env_prefix='SIGNAL_GENERATION_')

    # Integration mode settings
    LOCAL_SIGNAL_GENERATION_ENABLED: bool = True  # Enable local rule-based signals for faster, cost-effective analysis
    HYBRID_MODE_ENABLED: bool = True  # Try local first, escalate to LLM when confidence is low
    FALLBACK_TO_LLM_ON_ERROR: bool = True  # Use LLM as fallback if local generation fails

    # Gradual rollout controls
    ROLLOUT_PERCENTAGE: float = 1.0  # Percentage of requests to use local generation (1.0 = 100%)
    ENABLED_SYMBOLS: List[str] = []  # Specific symbols to enable for local generation (empty = all symbols)
    ENABLED_TIMEFRAMES: List[str] = []  # Specific timeframes to enable (empty = all timeframes)

    # Performance comparison settings
    ENABLE_SIDE_BY_SIDE_COMPARISON: bool = False  # Generate both local and LLM signals for research
    COMPARISON_SAMPLE_RATE: float = 0.1  # Percentage of requests to sample for comparison when enabled

    # Escalation settings (only used when HYBRID_MODE_ENABLED=True)
    ESCALATION_CONFIDENCE_THRESHOLD: float = 0.3  # Minimum confidence to avoid LLM escalation

    # Performance monitoring
    TRACK_PERFORMANCE_METRICS: bool = True  # Track performance comparison between local vs LLM
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
