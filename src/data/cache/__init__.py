"""
Enhanced caching framework for the AI Trading System.

This module provides a comprehensive caching solution with Redis backend,
market-aware TTL calculation, and performance monitoring capabilities.

Components:
- CacheConfig: Configuration management for caching settings
- MarketAwareTTL: Intelligent TTL calculation based on market hours
- RedisCacheManager: Enhanced Redis-based cache manager
- CacheMonitor: Performance monitoring and metrics collection
- CacheManager: Enhanced cache manager with Redis and simple cache support
"""

from .cache_config import (
    CacheConfig,
    CacheDataType,
    TTLConfig,
    RedisConfig,
    CacheKeyBuilder
)

from .market_aware_ttl import MarketAwareTTL
from .redis_cache_manager import RedisCacheManager
from .cache_monitor import CacheMonitor
from .cache_manager import CacheManager

__all__ = [
    "CacheConfig",
    "CacheDataType",
    "TTLConfig",
    "RedisConfig",
    "CacheKeyBuilder",
    "MarketAwareTTL",
    "RedisCacheManager",
    "CacheMonitor",
    "CacheManager"
]
