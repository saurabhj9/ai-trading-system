"""
Enhanced Redis-based cache manager with market-aware TTL and fallback support.

This module provides a production-ready caching system with Redis backend,
market-aware TTL calculation, and comprehensive monitoring capabilities.
"""
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import redis
from redis.exceptions import ConnectionError, RedisError

from .cache_config import CacheConfig, CacheDataType, CacheKeyBuilder
from .market_aware_ttl import MarketAwareTTL
from .cache_monitor import CacheMonitor

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Enhanced cache manager with Redis backend and market-aware TTL.
    
    Provides intelligent caching with automatic fallback to in-memory storage,
    market-aware TTL calculation, and comprehensive performance monitoring.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the Redis cache manager.
        
        Args:
            config: Cache configuration settings
        """
        self.config = config or CacheConfig()
        self.key_builder = CacheKeyBuilder()
        self.ttl_calculator = MarketAwareTTL(self.config.ttl)
        self.monitor = CacheMonitor() if self.config.enable_metrics else None
        
        # Initialize Redis connection
        self._redis_client: Optional[redis.Redis] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "redis_hits": 0,
            "memory_hits": 0,
            "sets": 0
        }
        
        if self.config.enabled:
            self._initialize_redis()
    
    def _initialize_redis(self) -> bool:
        """
        Initialize Redis connection.
        
        Returns:
            True if Redis connection successful, False otherwise
        """
        try:
            self._redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                password=self.config.redis.password,
                decode_responses=self.config.redis.decode_responses,
                socket_timeout=self.config.redis.socket_timeout,
                retry_on_timeout=self.config.redis.retry_on_timeout,
                max_connections=self.config.redis.max_connections
            )
            
            # Test connection
            self._redis_client.ping()
            logger.info("Redis cache manager initialized successfully")
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis connection failed: {e}")
            if self.config.fallback_to_memory:
                logger.info("Falling back to in-memory cache")
            else:
                logger.error("Caching disabled - Redis unavailable and fallback disabled")
            return False
    
    def _is_redis_available(self) -> bool:
        """
        Check if Redis is available.
        
        Returns:
            True if Redis is available and connected
        """
        if not self._redis_client:
            return False
        
        try:
            self._redis_client.ping()
            return True
        except (ConnectionError, RedisError):
            logger.warning("Redis connection lost")
            self._redis_client = None
            return False
    
    def _serialize_value(self, value: Any) -> str:
        """
        Serialize value for Redis storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value as string
        """
        try:
            # Handle different data types
            if isinstance(value, pd.DataFrame):
                # Serialize DataFrames with metadata
                data = {
                    "type": "dataframe",
                    "data": value.to_dict(),
                    "index": value.index.tolist(),
                    "columns": value.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()}
                }
                return json.dumps(data)
            
            elif isinstance(value, (dict, list, str, int, float, bool)):
                # JSON-serializable types
                return json.dumps({"type": "json", "data": value})
            
            else:
                # Use pickle for complex objects
                data = {
                    "type": "pickle",
                    "data": pickle.dumps(value).hex()
                }
                return json.dumps(data)
                
        except Exception as e:
            logger.error(f"Error serializing cache value: {e}")
            raise
    
    def _deserialize_value(self, serialized_value: str) -> Any:
        """
        Deserialize value from Redis storage.
        
        Args:
            serialized_value: Serialized value string
            
        Returns:
            Deserialized value
        """
        try:
            data = json.loads(serialized_value)
            data_type = data.get("type")
            
            if data_type == "dataframe":
                # Reconstruct DataFrame
                df = pd.DataFrame(
                    data=data["data"],
                    index=data["index"],
                    columns=data["columns"]
                )
                # Convert data types back
                for col, dtype_str in data["dtypes"].items():
                    try:
                        df[col] = df[col].astype(dtype_str)
                    except:
                        pass  # Keep current dtype if conversion fails
                return df
            
            elif data_type == "json":
                return data["data"]
            
            elif data_type == "pickle":
                # Deserialize pickled object
                pickle_data = bytes.fromhex(data["data"])
                return pickle.loads(pickle_data)
            
            else:
                # Unknown type, return as-is
                return serialized_value
                
        except Exception as e:
            logger.error(f"Error deserializing cache value: {e}")
            return None
    
    def _get_memory_cache(self, key: str) -> Optional[Any]:
        """
        Get value from in-memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._memory_cache:
            return None
        
        item = self._memory_cache[key]
        cached_at = item["cached_at"]
        
        # Check if expired
        if self.ttl_calculator.should_refresh(item["data_type"], cached_at):
            del self._memory_cache[key]
            return None
        
        # Check memory cache size limit
        if len(self._memory_cache) > self.config.max_memory_items:
            # Remove oldest item
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k]["cached_at"])
            del self._memory_cache[oldest_key]
        
        return item["value"]
    
    def _set_memory_cache(
        self,
        key: str,
        value: Any,
        data_type: CacheDataType
    ) -> None:
        """
        Set value in in-memory cache.
        
        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data being cached
        """
        self._memory_cache[key] = {
            "value": value,
            "data_type": data_type,
            "cached_at": datetime.utcnow()
        }
    
    def get(self, key: str, data_type: CacheDataType) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            data_type: Type of data for TTL calculation
            
        Returns:
            Cached value or None if not found
        """
        if not self.config.enabled:
            return None
        
        try:
            # Try Redis first
            if self._is_redis_available():
                serialized_value = self._redis_client.get(
                    self.config.redis.key_prefix + key
                )
                
                if serialized_value:
                    value = self._deserialize_value(serialized_value)
                    self._stats["hits"] += 1
                    self._stats["redis_hits"] += 1
                    
                    if self.monitor:
                        self.monitor.record_hit(key, data_type, "redis")
                    
                    logger.debug(f"Cache HIT (Redis): {key}")
                    return value
            
            # Fallback to memory cache
            if self.config.fallback_to_memory:
                value = self._get_memory_cache(key)
                if value is not None:
                    self._stats["hits"] += 1
                    self._stats["memory_hits"] += 1
                    
                    if self.monitor:
                        self.monitor.record_hit(key, data_type, "memory")
                    
                    logger.debug(f"Cache HIT (Memory): {key}")
                    return value
            
            # Cache miss
            self._stats["misses"] += 1
            
            if self.monitor:
                self.monitor.record_miss(key, data_type)
            
            logger.debug(f"Cache MISS: {key}")
            return None
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        data_type: CacheDataType,
        custom_ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data for TTL calculation
            custom_ttl: Custom TTL override
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False
        
        try:
            # Calculate TTL
            if custom_ttl:
                ttl_seconds = int(custom_ttl.total_seconds())
            else:
                ttl = self.ttl_calculator.calculate_ttl(data_type)
                ttl_seconds = int(ttl.total_seconds())
            
            # Set in Redis
            redis_success = False
            if self._is_redis_available():
                serialized_value = self._serialize_value(value)
                self._redis_client.setex(
                    self.config.redis.key_prefix + key,
                    ttl_seconds,
                    serialized_value
                )
                redis_success = True
            
            # Set in memory cache as fallback
            if self.config.fallback_to_memory or not redis_success:
                self._set_memory_cache(key, value, data_type)
            
            self._stats["sets"] += 1
            
            if self.monitor:
                self.monitor.record_set(key, data_type, ttl_seconds)
            
            logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Redis
            redis_success = False
            if self._is_redis_available():
                self._redis_client.delete(self.config.redis.key_prefix + key)
                redis_success = True
            
            # Delete from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
            
            logger.debug(f"Cache DELETE: {key}")
            return True
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear Redis
            redis_success = False
            if self._is_redis_available():
                pattern = self.config.redis.key_prefix + "*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                redis_success = True
            
            # Clear memory cache
            self._memory_cache.clear()
            
            # Reset statistics
            self._stats = {
                "hits": 0,
                "misses": 0,
                "errors": 0,
                "redis_hits": 0,
                "memory_hits": 0,
                "sets": 0
            }
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self._memory_cache),
            "redis_available": self._is_redis_available(),
            "enabled": self.config.enabled
        }
        
        # Add monitor stats if available
        if self.monitor:
            stats["monitor"] = self.monitor.get_stats()
        
        return stats
    
    def get_cache_info(self, key: str, data_type: CacheDataType) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cached item.
        
        Args:
            key: Cache key
            data_type: Type of cached data
            
        Returns:
            Dictionary with cache information or None if not found
        """
        try:
            # Check Redis
            if self._is_redis_available():
                ttl = self._redis_client.ttl(self.config.redis.key_prefix + key)
                if ttl >= 0:
                    return {
                        "key": key,
                        "data_type": data_type.value,
                        "source": "redis",
                        "ttl_seconds": ttl,
                        "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
                    }
            
            # Check memory cache
            if key in self._memory_cache:
                item = self._memory_cache[key]
                return self.ttl_calculator.get_cache_info(
                    data_type, key, item["cached_at"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return None
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries from memory cache.
        
        Returns:
            Number of items cleaned up
        """
        if not self.config.fallback_to_memory:
            return 0
        
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, item in self._memory_cache.items():
            if self.ttl_calculator.should_refresh(item["data_type"], item["cached_at"], current_time):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
