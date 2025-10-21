"""
Configuration for the enhanced caching system.

This module defines cache settings, TTL strategies, and market-aware
caching rules for different types of financial data.
"""
from datetime import time, timedelta
from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class CacheDataType(str, Enum):
    """Types of data that can be cached."""
    HISTORICAL_OHLCV = "historical_ohlcv"
    CURRENT_PRICE = "current_price"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_INDICATORS = "technical_indicators"
    COMPANY_PROFILE = "company_profile"
    MARKET_REGIME = "market_regime"
    FUNDAMENTAL_DATA = "fundamental_data"


class MarketHours(str, Enum):
    """Market hours for different exchanges."""
    NYSE_OPEN = time(9, 30)  # 9:30 AM EST
    NYSE_CLOSE = time(16, 0)  # 4:00 PM EST
    NASDAQ_OPEN = time(9, 30)  # 9:30 AM EST
    NASDAQ_CLOSE = time(16, 0)  # 4:00 PM EST


class TTLConfig(BaseModel):
    """Configuration for TTL (Time-To-Live) settings."""

    # Market hours TTL (shorter for real-time data)
    current_price_market_hours: timedelta = Field(
        default=timedelta(minutes=1),
        description="TTL for current prices during market hours"
    )
    current_price_after_hours: timedelta = Field(
        default=timedelta(minutes=15),
        description="TTL for current prices after market hours"
    )

    # Historical data TTL (longer, as data doesn't change)
    historical_ohlcv: timedelta = Field(
        default=timedelta(hours=23),  # Until next market day
        description="TTL for historical OHLCV data"
    )

    # News and sentiment TTL
    news_sentiment: timedelta = Field(
        default=timedelta(minutes=30),
        description="TTL for news sentiment data"
    )

    # Technical indicators TTL
    technical_indicators: timedelta = Field(
        default=timedelta(hours=23),  # Tied to underlying data
        description="TTL for technical indicators"
    )

    # Company profiles and fundamentals
    company_profile: timedelta = Field(
        default=timedelta(days=7),
        description="TTL for company profile data"
    )
    fundamental_data: timedelta = Field(
        default=timedelta(days=1),
        description="TTL for fundamental financial data"
    )

    # Market regime analysis
    market_regime: timedelta = Field(
        default=timedelta(hours=4),
        description="TTL for market regime analysis"
    )


class RedisConfig(BaseModel):
    """Configuration for Redis connection and settings."""

    # Connection settings
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    # Connection pool settings
    max_connections: int = Field(default=10, description="Maximum Redis connections")
    retry_on_timeout: bool = Field(default=True, description="Retry on connection timeout")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")

    # Performance settings
    decode_responses: bool = Field(default=True, description="Decode responses to strings")
    compression: bool = Field(default=True, description="Enable Redis compression")

    # Key prefix for this application
    key_prefix: str = Field(
        default="ai_trading:",
        description="Prefix for all cache keys"
    )


class CacheConfig(BaseModel):
    """Main configuration for the caching system."""

    # Enable/disable caching
    enabled: bool = Field(default=True, description="Enable caching system")
    fallback_to_memory: bool = Field(
        default=True,
        description="Fallback to in-memory cache if Redis unavailable"
    )

    # Sub-configurations
    ttl: TTLConfig = Field(default_factory=TTLConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # Cache size limits (for memory fallback)
    max_memory_items: int = Field(
        default=10000,
        description="Maximum items in memory cache"
    )

    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable cache performance metrics")
    metrics_retention_hours: int = Field(
        default=24,
        description="How long to retain cache metrics"
    )


class CacheKeyBuilder:
    """Utility class for building consistent cache keys."""

    @staticmethod
    def build_key(
        data_type: CacheDataType,
        symbol: str,
        identifier: str = "",
        provider: str = ""
    ) -> str:
        """
        Build a standardized cache key.

        Args:
            data_type: Type of data being cached
            symbol: Stock symbol or identifier
            identifier: Additional identifier (date range, etc.)
            provider: Data provider name

        Returns:
            Standardized cache key string
        """
        parts = [data_type.value, symbol]

        if provider:
            parts.append(f"provider:{provider}")

        if identifier:
            parts.append(identifier)

        return ":".join(parts)

    @staticmethod
    def build_historical_key(
        symbol: str,
        start_date: str,
        end_date: str,
        provider: str = ""
    ) -> str:
        """Build cache key for historical data."""
        identifier = f"{start_date}:{end_date}"
        return CacheKeyBuilder.build_key(
            CacheDataType.HISTORICAL_OHLCV,
            symbol,
            identifier,
            provider
        )

    @staticmethod
    def build_price_key(symbol: str, provider: str = "") -> str:
        """Build cache key for current price data."""
        return CacheKeyBuilder.build_key(
            CacheDataType.CURRENT_PRICE,
            symbol,
            "",
            provider
        )

    @staticmethod
    def build_news_key(symbol: str, limit: int = 20, provider: str = "") -> str:
        """Build cache key for news sentiment data."""
        identifier = f"limit:{limit}"
        return CacheKeyBuilder.build_key(
            CacheDataType.NEWS_SENTIMENT,
            symbol,
            identifier,
            provider
        )

    @staticmethod
    def build_indicators_key(
        symbol: str,
        indicators_hash: str,
        provider: str = ""
    ) -> str:
        """Build cache key for technical indicators."""
        return CacheKeyBuilder.build_key(
            CacheDataType.TECHNICAL_INDICATORS,
            symbol,
            indicators_hash,
            provider
        )


# Default configuration instance
default_cache_config = CacheConfig()
