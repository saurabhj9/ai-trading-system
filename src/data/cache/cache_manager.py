"""
Enhanced cache manager with Redis backend and market-aware TTL support.

This module provides a unified interface to both simple in-memory caching
and advanced Redis-based caching with intelligent TTL calculation and
comprehensive monitoring.
"""
import time
import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Union

from .redis_cache_manager import RedisCacheManager
from .cache_config import CacheConfig, CacheDataType

logger = logging.getLogger(__name__)


class SimpleCacheManager:
    """
    A simple in-memory cache with Time-To-Live (TTL) support.

    This manager stores data in a dictionary and automatically handles the
    expiration of cached items. Maintained for backward compatibility.
    """

    def __init__(self):
        """Initializes the cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl_seconds: int):
        """
        Adds an item to the cache with a specified TTL.

        Args:
            key: The key to store the item under.
            value: The item to store.
            ttl_seconds: The time-to-live for the item, in seconds.
        """
        if ttl_seconds <= 0:
            return  # Do not cache if TTL is non-positive

        expires_at = time.time() + ttl_seconds
        self._cache[key] = {"value": value, "expires_at": expires_at}
        logger.debug(f"SimpleCache SET for key: {key} with TTL: {ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and has not expired.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached item, or None if it is not found or has expired.
        """
        item = self._cache.get(key)
        if item is None:
            logger.debug(f"SimpleCache MISS for key: {key}")
            return None

        if time.time() > item["expires_at"]:
            # Item has expired, so remove it from the cache
            del self._cache[key]
            logger.debug(f"SimpleCache EXPIRED for key: {key}")
            return None

        logger.debug(f"SimpleCache HIT for key: {key}")
        return item["value"]

    def clear(self):
        """Clears all items from the cache."""
        self._cache.clear()
        logger.info("SimpleCache CLEARED.")


class CacheManager:
    """
    Enhanced cache manager with Redis backend and market-aware TTL support.

    This class provides a unified interface to both simple in-memory caching
    and advanced Redis-based caching with intelligent TTL calculation and
    comprehensive monitoring.
    """

    def __init__(
        self,
        enable_redis: bool = True,
        cache_config: Optional[CacheConfig] = None,
        fallback_to_simple: bool = True
    ):
        """
        Initialize the enhanced cache manager.

        Args:
            enable_redis: Whether to enable Redis caching
            cache_config: Configuration for Redis cache
            fallback_to_simple: Whether to fallback to simple cache if Redis fails
        """
        self.enable_redis = enable_redis
        self.cache_config = cache_config or CacheConfig()
        self.fallback_to_simple = fallback_to_simple

        # Initialize Redis cache if enabled
        self.redis_cache = None
        if self.enable_redis:
            try:
                self.redis_cache = RedisCacheManager(self.cache_config)
                logger.info("Enhanced Redis cache manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                if not self.fallback_to_simple:
                    logger.error("Cache disabled - Redis unavailable and fallback disabled")
                    self.enable_redis = False
                else:
                    logger.info("Falling back to simple in-memory cache")

        # Initialize simple cache as fallback
        self.simple_cache = SimpleCacheManager()

        # Determine which cache to use
        self.use_redis = self.enable_redis and self.redis_cache is not None
        self.cache_type = "redis" if self.use_redis else "simple"

        logger.info(f"Cache manager initialized using {self.cache_type} backend")

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int,
        data_type: Optional[Union[str, CacheDataType]] = None
    ) -> bool:
        """
        Adds an item to the cache with a specified TTL.

        Args:
            key: The key to store the item under.
            value: The item to store.
            ttl_seconds: The time-to-live for the item, in seconds.
            data_type: Type of data for intelligent TTL (Redis cache only)

        Returns:
            True if successful, False otherwise
        """
        if self.use_redis and data_type is not None:
            # Use enhanced Redis cache with data type awareness
            if isinstance(data_type, str):
                try:
                    data_type = CacheDataType(data_type)
                except ValueError:
                    data_type = CacheDataType.HISTORICAL_OHLCV

            return self.redis_cache.set(key, value, data_type)

        # Use simple cache with basic TTL
        if self.use_redis:
            # Redis cache without data type (backwards compatibility)
            return self.redis_cache.set(key, value, CacheDataType.HISTORICAL_OHLCV,
                                      timedelta(seconds=ttl_seconds))
        else:
            # Simple cache
            self.simple_cache.set(key, value, ttl_seconds)
            return True

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and has not expired.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached item, or None if it is not found or has expired.
        """
        if self.use_redis:
            # Use Redis cache (will fallback to memory if configured)
            return self.redis_cache.get(key, CacheDataType.HISTORICAL_OHLCV)
        else:
            # Use simple cache
            return self.simple_cache.get(key)

    def delete(self, key: str) -> bool:
        """
        Deletes an item from the cache.

        Args:
            key: The key of the item to delete.

        Returns:
            True if successful, False otherwise
        """
        if self.use_redis:
            return self.redis_cache.delete(key)
        else:
            if key in self.simple_cache._cache:
                del self.simple_cache._cache[key]
                return True
            return False

    def clear(self) -> bool:
        """Clears all items from the cache."""
        if self.use_redis:
            return self.redis_cache.clear()
        else:
            self.simple_cache.clear()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if self.use_redis:
            return self.redis_cache.get_stats()
        else:
            # Simple cache stats
            cache_size = len(self.simple_cache._cache)
            return {
                "cache_type": "simple",
                "cache_size": cache_size,
                "enabled": True,
                "redis_available": False
            }

    def cleanup_expired(self) -> int:
        """
        Clean up expired entries from memory cache.

        Returns:
            Number of items cleaned up
        """
        if self.use_redis:
            return self.redis_cache.cleanup_expired()
        else:
            # Simple cache cleanup
            current_time = time.time()
            expired_keys = [
                key for key, item in self.simple_cache._cache.items()
                if current_time > item["expires_at"]
            ]

            for key in expired_keys:
                del self.simple_cache._cache[key]

            return len(expired_keys)

    def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cached item.

        Args:
            key: Cache key

        Returns:
            Dictionary with cache information or None if not found
        """
        if self.use_redis:
            return self.redis_cache.get_cache_info(key, CacheDataType.HISTORICAL_OHLCV)
        else:
            # Simple cache info
            item = self.simple_cache._cache.get(key)
            if item:
                from datetime import datetime
                return {
                    "key": key,
                    "cached_at": datetime.fromtimestamp(item["expires_at"] - item.get("ttl", 3600)).isoformat(),
                    "expires_at": datetime.fromtimestamp(item["expires_at"]).isoformat(),
                    "source": "simple",
                    "cache_type": "simple"
                }
            return None


# Backward compatibility alias
CacheManagerV1 = SimpleCacheManager
