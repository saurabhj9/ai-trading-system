"""
Unit tests for the enhanced caching framework.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json
import pandas as pd

from src.data.cache.cache_config import CacheConfig, CacheDataType
from src.data.cache.market_aware_ttl import MarketAwareTTL
from src.data.cache.redis_cache_manager import RedisCacheManager
from src.data.cache.cache_monitor import CacheMonitor
from src.data.cache import CacheManager


class TestCacheConfig:
    """Test suite for CacheConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.fallback_to_memory is True
        assert config.max_memory_items == 10000
        assert config.enable_metrics is True

        # TTL config
        assert config.ttl.current_price_market_hours == timedelta(minutes=1)
        assert config.ttl.historical_ohlcv == timedelta(hours=23)
        assert config.ttl.news_sentiment == timedelta(minutes=30)

        # Redis config
        assert config.redis.host == "localhost"
        assert config.redis.port == 6379
        assert config.redis.decode_responses is True

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_config = {
            "enabled": False,
            "ttl": {
                "current_price_market_hours": timedelta(minutes=2),
                "historical_ohlcv": timedelta(hours=25)
            },
            "redis": {
                "host": "custom-host",
                "port": 6380
            }
        }

        config = CacheConfig(**custom_config)

        assert config.enabled is False
        assert config.ttl.current_price_market_hours == timedelta(minutes=2)
        assert config.ttl.historical_ohlcv == timedelta(hours=25)
        assert config.redis.host == "custom-host"
        assert config.redis.port == 6380


class TestMarketAwareTTL:
    """Test suite for MarketAwareTTL class."""

    @pytest.fixture
    def ttl_calculator(self):
        """Create MarketAwareTTL instance for testing."""
        return MarketAwareTTL()

    def test_calculate_ttl_current_price_market_hours(self, ttl_calculator):
        """Test TTL calculation for current prices during market hours."""
        # Mock market hours time (10:30 AM ET)
        market_time = datetime(2024, 1, 15, 14, 30, 0)  # 2:30 PM UTC = 10:30 AM ET

        ttl = ttl_calculator.calculate_ttl(CacheDataType.CURRENT_PRICE, market_time)

        assert ttl == ttl_calculator.ttl_config.current_price_market_hours
        assert ttl.total_seconds() == 60  # 1 minute

    def test_calculate_ttl_current_price_after_hours(self, ttl_calculator):
        """Test TTL calculation for current prices after market hours."""
        # Mock after-hours time (8:00 PM ET)
        after_hours_time = datetime(2024, 1, 15, 1, 0, 0)  # 1:00 AM UTC = 8:00 PM ET

        ttl = ttl_calculator.calculate_ttl(CacheDataType.CURRENT_PRICE, after_hours_time)

        assert ttl == ttl_calculator.ttl_config.current_price_after_hours
        assert ttl.total_seconds() == 900  # 15 minutes

    def test_calculate_ttl_historical_data(self, ttl_calculator):
        """Test TTL calculation for historical data."""
        current_time = datetime(2024, 1, 15, 12, 0, 0)

        ttl = ttl_calculator.calculate_ttl(CacheDataType.HISTORICAL_OHLCV, current_time)

        assert ttl == ttl_calculator.ttl_config.historical_ohlcv
        assert ttl.total_seconds() == 82800  # 23 hours

    def test_calculate_ttl_news_sentiment(self, ttl_calculator):
        """Test TTL calculation for news sentiment."""
        current_time = datetime(2024, 1, 15, 12, 0, 0)

        ttl = ttl_calculator.calculate_ttl(CacheDataType.NEWS_SENTIMENT, current_time)

        assert ttl == ttl_calculator.ttl_config.news_sentiment
        assert ttl.total_seconds() == 1800  # 30 minutes

    def test_should_refresh_true(self, ttl_calculator):
        """Test cache refresh determination for stale data."""
        cached_time = datetime(2024, 1, 15, 10, 0, 0)
        current_time = datetime(2024, 1, 15, 10, 30, 0)  # 30 minutes later

        # Should refresh for data with 1 minute TTL
        should_refresh = ttl_calculator.should_refresh(
            CacheDataType.CURRENT_PRICE, cached_time, current_time
        )
        assert should_refresh is True

    def test_should_refresh_false(self, ttl_calculator):
        """Test cache refresh determination for fresh data."""
        cached_time = datetime(2024, 1, 15, 10, 0, 0)
        current_time = datetime(2024, 1, 15, 10, 0, 30)  # 30 seconds later

        # Should not refresh for data with 1 minute TTL
        should_refresh = ttl_calculator.should_refresh(
            CacheDataType.CURRENT_PRICE, cached_time, current_time
        )
        assert should_refresh is False

    def test_get_cache_expiry(self, ttl_calculator):
        """Test cache expiry time calculation."""
        current_time = datetime(2024, 1, 15, 12, 0, 0)
        ttl = ttl_calculator.calculate_ttl(CacheDataType.NEWS_SENTIMENT, current_time)

        expiry_time = ttl_calculator.get_cache_expiry(CacheDataType.NEWS_SENTIMENT, current_time)

        expected_expiry = current_time + ttl
        assert expiry_time == expected_expiry


class TestCacheMonitor:
    """Test suite for CacheMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create CacheMonitor instance for testing."""
        return CacheMonitor(retention_hours=1)

    def test_record_hit(self, monitor):
        """Test recording cache hits."""
        monitor.record_hit("test_key", CacheDataType.CURRENT_PRICE, "redis")

        stats = monitor.get_stats()
        assert stats["performance"]["hits"] == 1
        assert stats["performance"]["total_requests"] == 1
        assert stats["by_data_type"]["current_price"]["hits"] == 1
        assert stats["by_source"]["redis"] == 1

    def test_record_miss(self, monitor):
        """Test recording cache misses."""
        monitor.record_miss("test_key", CacheDataType.HISTORICAL_OHLCV)

        stats = monitor.get_stats()
        assert stats["performance"]["misses"] == 1
        assert stats["performance"]["total_requests"] == 1
        assert stats["by_data_type"]["historical_ohlcv"]["misses"] == 1

    def test_record_set(self, monitor):
        """Test recording cache sets."""
        monitor.record_set("test_key", CacheDataType.NEWS_SENTIMENT, 300)

        stats = monitor.get_stats()
        assert stats["performance"]["sets"] == 1
        assert stats["by_data_type"]["news_sentiment"]["sets"] == 1

    def test_record_error(self, monitor):
        """Test recording cache errors."""
        monitor.record_error("get", "Connection failed")

        stats = monitor.get_stats()
        assert stats["performance"]["errors"] == 1
        assert len(stats["recent_errors"]) == 1
        assert stats["recent_errors"][0]["operation"] == "get"
        assert stats["recent_errors"][0]["error"] == "Connection failed"

    def test_hit_rate_calculation(self, monitor):
        """Test hit rate calculation."""
        # Record some hits and misses
        for _ in range(7):
            monitor.record_hit("key", CacheDataType.CURRENT_PRICE, "redis")
        for _ in range(3):
            monitor.record_miss("key", CacheDataType.CURRENT_PRICE)

        stats = monitor.get_stats()
        assert stats["performance"]["hits"] == 7
        assert stats["performance"]["misses"] == 3
        assert stats["performance"]["total_requests"] == 10
        assert stats["performance"]["hit_rate_percent"] == 70.0

    def test_reset_metrics(self, monitor):
        """Test resetting metrics."""
        # Add some data
        monitor.record_hit("key", CacheDataType.CURRENT_PRICE, "redis")
        monitor.record_miss("key", CacheDataType.CURRENT_PRICE)

        # Verify data exists
        stats = monitor.get_stats()
        assert stats["performance"]["hits"] == 1
        assert stats["performance"]["misses"] == 1

        # Reset and verify empty
        monitor.reset_metrics()
        stats = monitor.get_stats()
        assert stats["performance"]["hits"] == 0
        assert stats["performance"]["misses"] == 0
        assert stats["performance"]["total_requests"] == 0

    def test_performance_report(self, monitor):
        """Test performance report generation."""
        # Add some sample data
        monitor.record_hit("key1", CacheDataType.CURRENT_PRICE, "redis")
        monitor.record_hit("key2", CacheDataType.HISTORICAL_OHLCV, "memory")
        monitor.record_miss("key3", CacheDataType.NEWS_SENTIMENT)

        report = monitor.get_performance_report()

        assert "Cache Performance Report" in report
        assert "Performance Summary" in report
        assert "Response Times" in report
        assert "Performance by Data Type" in report


class TestRedisCacheManager:
    """Test suite for RedisCacheManager class."""

    @pytest.fixture
    def cache_config(self):
        """Create CacheConfig for testing."""
        return CacheConfig(enabled=True)

    @pytest.fixture
    def mock_redis_cache(self, cache_config):
        """Create RedisCacheManager with mocked Redis."""
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            # Mock successful ping
            mock_client.ping.return_value = True

            cache = RedisCacheManager(cache_config)
            cache._redis_client = mock_client

            yield cache, mock_client

    def test_init_with_redis_available(self):
        """Test initialization when Redis is available."""
        config = CacheConfig()

        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            cache = RedisCacheManager(config)

            assert cache.config.enabled is True
            assert cache._redis_client is not None

    def test_init_with_redis_unavailable(self):
        """Test initialization when Redis is unavailable."""
        config = CacheConfig(fallback_to_memory=True)

        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            cache = RedisCacheManager(config)

            assert cache._redis_client is None
            assert cache.config.fallback_to_memory is True

    def test_serialize_dataframe(self, mock_redis_cache):
        """Test DataFrame serialization."""
        cache, _ = mock_redis_cache

        # Create test DataFrame
        df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'Close': [105.0, 106.0],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))

        serialized = cache._serialize_value(df)
        data = json.loads(serialized)

        assert data["type"] == "dataframe"
        assert data["columns"] == ["Open", "Close", "Volume"]
        assert len(data["data"]["Open"]) == 2
        assert data["data"]["Open"][0] == 100.0

    def test_deserialize_dataframe(self, mock_redis_cache):
        """Test DataFrame deserialization."""
        cache, _ = mock_redis_cache

        # Create serialized DataFrame data
        data = {
            "type": "dataframe",
            "data": {
                "Open": [100.0, 101.0],
                "Close": [105.0, 106.0],
                "Volume": [1000, 1100]
            },
            "index": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            "columns": ["Open", "Close", "Volume"],
            "dtypes": {"Open": "float64", "Close": "float64", "Volume": "int64"}
        }

        serialized = json.dumps(data)
        result = cache._deserialize_value(serialized)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Open", "Close", "Volume"]
        assert result.iloc[0]["Open"] == 100.0

    def test_get_cache_hit_redis(self, mock_redis_cache):
        """Test cache hit from Redis."""
        cache, mock_redis = mock_redis_cache

        # Mock Redis get with test data
        test_data = {"type": "json", "data": {"price": 150.25}}
        mock_redis.get.return_value = json.dumps(test_data)

        result = cache.get("test_key", CacheDataType.CURRENT_PRICE)

        assert result == {"price": 150.25}
        assert cache._stats["hits"] == 1
        assert cache._stats["redis_hits"] == 1
        mock_redis.get.assert_called_once_with("ai_trading:test_key")

    def test_get_cache_miss_redis(self, mock_redis_cache):
        """Test cache miss from Redis."""
        cache, mock_redis = mock_redis_cache

        # Mock Redis get returning None
        mock_redis.get.return_value = None

        result = cache.get("test_key", CacheDataType.CURRENT_PRICE)

        assert result is None
        assert cache._stats["misses"] == 1
        mock_redis.get.assert_called_once_with("ai_trading:test_key")

    def test_set_cache_success(self, mock_redis_cache):
        """Test successful cache set."""
        cache, mock_redis = mock_redis_cache

        test_value = {"price": 150.25}

        result = cache.set("test_key", test_value, CacheDataType.CURRENT_PRICE)

        assert result is True
        assert cache._stats["sets"] == 1
        mock_redis.setex.assert_called_once()

        # Verify setex was called with correct arguments
        args, kwargs = mock_redis.setex.call_args
        assert args[0] == "ai_trading:test_key"  # key with prefix
        assert isinstance(args[1], int)  # TTL in seconds
        assert json.loads(args[2])["data"] == test_value  # serialized value

    def test_delete_cache(self, mock_redis_cache):
        """Test cache deletion."""
        cache, mock_redis = mock_redis_cache

        result = cache.delete("test_key")

        assert result is True
        mock_redis.delete.assert_called_once_with("ai_trading:test_key")

    def test_clear_cache(self, mock_redis_cache):
        """Test cache clearing."""
        cache, mock_redis = mock_redis_cache

        # Mock Redis keys command
        mock_redis.keys.return_value = ["ai_trading:key1", "ai_trading:key2"]
        mock_redis.delete.return_value = 2

        result = cache.clear()

        assert result is True
        mock_redis.keys.assert_called_once_with("ai_trading:*")
        mock_redis.delete.assert_called_once_with("ai_trading:key1", "ai_trading:key2")

    def test_get_stats(self, mock_redis_cache):
        """Test statistics retrieval."""
        cache, mock_redis = mock_redis_cache

        # Add some test stats
        cache._stats["hits"] = 10
        cache._stats["misses"] = 5
        cache._stats["sets"] = 8
        cache._stats["errors"] = 1

        stats = cache.get_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["sets"] == 8
        assert stats["errors"] == 1
        assert stats["total_requests"] == 15
        assert stats["hit_rate_percent"] == 66.67
        assert stats["redis_available"] is True
        assert stats["enabled"] is True


class TestEnhancedCacheManager:
    """Test suite for enhanced CacheManager class."""

    def test_init_with_redis_enabled(self):
        """Test initialization with Redis enabled."""
        config = CacheConfig()

        with patch('src.data.cache.redis_cache_manager.RedisCacheManager') as mock_redis:
            mock_cache = MagicMock()
            mock_redis.return_value = mock_cache

            cache = CacheManager(enable_redis=True, cache_config=config)

            assert cache.enable_redis is True
            assert cache.redis_cache is mock_cache
            assert cache.use_redis is True
            assert cache.cache_type == "redis"

    def test_init_with_redis_disabled(self):
        """Test initialization with Redis disabled."""
        cache = CacheManager(enable_redis=False)

        assert cache.enable_redis is False
        assert cache.redis_cache is None
        assert cache.use_redis is False
        assert cache.cache_type == "simple"
        assert cache.simple_cache is not None

    def test_init_with_redis_fallback_to_simple(self):
        """Test initialization falling back to simple cache."""
        config = CacheConfig()

        with patch('src.data.cache.redis_cache_manager.RedisCacheManager') as mock_redis:
            mock_redis.side_effect = Exception("Redis failed")

            cache = CacheManager(enable_redis=True, cache_config=config, fallback_to_simple=True)

            assert cache.enable_redis is False
            assert cache.redis_cache is None
            assert cache.use_redis is False
            assert cache.cache_type == "simple"

    def test_set_with_enhanced_caching(self):
        """Test setting cache with enhanced caching."""
        with patch('src.data.cache.redis_cache_manager.RedisCacheManager') as mock_redis:
            mock_cache = MagicMock()
            mock_cache.set.return_value = True
            mock_redis.return_value = mock_cache

            cache = CacheManager(enable_redis=True)
            result = cache.set("test_key", {"data": "value"}, 300, "current_price")

            assert result is True
            mock_cache.set.assert_called_once_with("test_key", {"data": "value"}, CacheDataType.CURRENT_PRICE)

    def test_set_with_simple_caching(self):
        """Test setting cache with simple caching."""
        cache = CacheManager(enable_redis=False)

        result = cache.set("test_key", {"data": "value"}, 300)

        assert result is True
        # Verify it's in simple cache
        cached_value = cache.simple_cache.get("test_key")
        assert cached_value == {"data": "value"}

    def test_get_with_enhanced_caching(self):
        """Test getting cache with enhanced caching."""
        with patch('src.data.cache.redis_cache_manager.RedisCacheManager') as mock_redis:
            mock_cache = MagicMock()
            mock_cache.get.return_value = {"data": "value"}
            mock_redis.return_value = mock_cache

            cache = CacheManager(enable_redis=True)
            result = cache.get("test_key")

            assert result == {"data": "value"}
            mock_cache.get.assert_called_once_with("test_key", CacheDataType.HISTORICAL_OHLCV)

    def test_get_with_simple_caching(self):
        """Test getting cache with simple caching."""
        cache = CacheManager(enable_redis=False)

        # Set value first
        cache.simple_cache.set("test_key", {"data": "value"}, 300)

        result = cache.get("test_key")

        assert result == {"data": "value"}

    def test_get_stats_redis(self):
        """Test getting statistics from Redis cache."""
        with patch('src.data.cache.redis_cache_manager.RedisCacheManager') as mock_redis:
            mock_cache = MagicMock()
            mock_cache.get_stats.return_value = {
                "hits": 10,
                "misses": 5,
                "cache_type": "redis"
            }
            mock_redis.return_value = mock_cache

            cache = CacheManager(enable_redis=True)
            stats = cache.get_stats()

            assert stats["hits"] == 10
            assert stats["misses"] == 5
            assert stats["cache_type"] == "redis"

    def test_get_stats_simple(self):
        """Test getting statistics from simple cache."""
        cache = CacheManager(enable_redis=False)

        # Add some data to simple cache
        cache.simple_cache._cache["key1"] = {"value": "data1", "expires_at": 9999999999}
        cache.simple_cache._cache["key2"] = {"value": "data2", "expires_at": 9999999999}

        stats = cache.get_stats()

        assert stats["cache_type"] == "simple"
        assert stats["cache_size"] == 2
        assert stats["redis_available"] is False
        assert stats["enabled"] is True
