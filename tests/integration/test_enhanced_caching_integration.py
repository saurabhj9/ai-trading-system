"""
Integration tests for the enhanced caching framework.

These tests demonstrate the enhanced caching capabilities with Redis backend
and market-aware TTL calculation.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from src.data.cache import CacheManager
from src.data.cache.cache_config import CacheConfig, CacheDataType
from src.agents.data_structures import MarketData


@pytest.mark.integration
@pytest.mark.asyncio
class TestEnhancedCachingIntegration:
    """Integration tests for enhanced caching framework."""

    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing."""
        config = CacheConfig(
            enabled=True,
            fallback_to_memory=True,
            enable_metrics=True
        )
        return CacheManager(
            enable_redis=True,
            cache_config=config,
            fallback_to_simple=True
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return MarketData(
            symbol="AAPL",
            price=150.25,
            volume=1000000,
            timestamp=datetime.utcnow(),
            ohlc={"Open": 148.50, "High": 152.00, "Low": 147.75, "Close": 150.25},
            technical_indicators={
                "RSI": 65.5,
                "MACD": 1.25,
                "MACD_signal": 1.10,
                "MACD_hist": 0.15
            },
            fundamental_data={
                "market_cap": 2500000000000,
                "pe_ratio": 28.5
            }
        )

    async def test_market_data_caching(self, cache_manager, sample_market_data):
        """Test caching of market data with enhanced features."""
        cache_key = f"market_data_{sample_market_data.symbol}"

        # Test initial cache miss
        result = cache_manager.get(cache_key)
        assert result is None

        # Test setting with enhanced caching
        success = cache_manager.set(
            cache_key,
            sample_market_data,
            ttl_seconds=300,  # 5 minutes
            data_type=CacheDataType.HISTORICAL_OHLCV
        )
        assert success is True

        # Test cache hit
        result = cache_manager.get(cache_key)
        assert result is not None
        assert result.symbol == sample_market_data.symbol
        assert result.price == sample_market_data.price
        assert isinstance(result.technical_indicators, dict)
        assert result.technical_indicators["RSI"] == 65.5

    async def test_dataframe_caching(self, cache_manager):
        """Test caching of pandas DataFrames."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'Close': [104.0, 105.0, 106.0, 107.0, 108.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2024-01-01', periods=5))

        cache_key = "ohlcv_data_AAPL_2024-01-01_2024-01-05"

        # Test setting DataFrame
        success = cache_manager.set(
            cache_key,
            df,
            ttl_seconds=3600,  # 1 hour
            data_type=CacheDataType.HISTORICAL_OHLCV
        )
        assert success is True

        # Test retrieving DataFrame
        result = cache_manager.get(cache_key)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 5
        assert result.iloc[0]['Close'] == 104.0
        assert result.iloc[-1]['Close'] == 108.0

    async def test_cache_statistics(self, cache_manager, sample_market_data):
        """Test cache performance statistics."""
        cache_key = f"stats_test_{sample_market_data.symbol}"

        # Perform cache operations
        # Initial miss
        cache_manager.get(cache_key)

        # Set operation
        cache_manager.set(
            cache_key,
            sample_market_data,
            ttl_seconds=300,
            data_type=CacheDataType.CURRENT_PRICE
        )

        # Hit operation
        cache_manager.get(cache_key)

        # Get statistics
        stats = cache_manager.get_stats()

        # Verify statistics
        assert "hits" in stats or "performance" in stats
        assert stats.get("enabled", True) is True

        if "performance" in stats:  # Enhanced cache with monitor
            perf = stats["performance"]
            assert perf["total_requests"] >= 2
            assert perf["hits"] >= 1
            assert perf["misses"] >= 1
            assert perf["hit_rate_percent"] >= 0
        else:  # Simple cache
            assert "cache_size" in stats
            assert stats["cache_type"] in ["simple", "redis"]

    async def test_cache_info(self, cache_manager, sample_market_data):
        """Test cache information retrieval."""
        cache_key = f"info_test_{sample_market_data.symbol}"

        # Set cache item
        cache_manager.set(
            cache_key,
            sample_market_data,
            ttl_seconds=300,
            data_type=CacheDataType.CURRENT_PRICE
        )

        # Get cache info
        info = cache_manager.get_cache_info(cache_key)

        # Verify info structure (may vary between Redis and simple cache)
        if info:
            assert "key" in info
            assert info["key"] == cache_key
            assert "source" in info  # "redis" or "simple"
            assert "cache_type" in info

    async def test_cache_cleanup(self, cache_manager):
        """Test cache cleanup functionality."""
        # Add some expired items to simple cache
        cache_manager.simple_cache.set("old_key1", "value1", 1)  # 1 second TTL
        cache_manager.simple_cache.set("old_key2", "value2", 1)  # 1 second TTL
        cache_manager.simple_cache.set("valid_key", "value3", 300)  # 5 minute TTL

        # Wait for expiration
        await asyncio.sleep(2)

        # Cleanup expired items
        cleaned_count = cache_manager.cleanup_expired()

        # Verify cleanup (simple cache)
        if not cache_manager.use_redis:
            assert cleaned_count >= 2  # At least 2 expired items removed

        # Verify valid item still exists
        valid_item = cache_manager.get("valid_key")
        assert valid_item == "value3"

    async def test_fallback_behavior(self):
        """Test fallback behavior when Redis is unavailable."""
        # Create cache with forced Redis failure
        config = CacheConfig(
            enabled=True,
            fallback_to_memory=True
        )

        cache = CacheManager(
            enable_redis=True,
            cache_config=config,
            fallback_to_simple=True
        )

        # If Redis failed, should fallback to simple cache
        if not cache.use_redis:
            assert cache.cache_type == "simple"
            assert cache.simple_cache is not None

            # Test basic operations
            cache.set("fallback_test", "test_value", 300)
            result = cache.get("fallback_test")
            assert result == "test_value"

    async def test_different_data_types(self, cache_manager):
        """Test caching different data types with appropriate TTL handling."""
        test_data = {
            "current_price": 150.25,
            "news_sentiment": {"title": "Test News", "sentiment": "positive"},
            "technical_indicators": {"RSI": 65.5, "MACD": 1.25},
            "company_profile": {"name": "Apple Inc.", "sector": "Technology"},
            "market_regime": {"regime": "bullish", "confidence": 0.85}
        }

        for data_type_str, data_value in test_data.items():
            cache_key = f"test_{data_type_str}"

            # Map string to CacheDataType
            data_type_map = {
                "current_price": CacheDataType.CURRENT_PRICE,
                "news_sentiment": CacheDataType.NEWS_SENTIMENT,
                "technical_indicators": CacheDataType.TECHNICAL_INDICATORS,
                "company_profile": CacheDataType.COMPANY_PROFILE,
                "market_regime": CacheDataType.MARKET_REGIME
            }

            data_type = data_type_map[data_type_str]

            # Test set and get
            success = cache_manager.set(
                cache_key,
                data_value,
                ttl_seconds=300,
                data_type=data_type
            )
            assert success is True

            result = cache_manager.get(cache_key)
            assert result is not None
            assert result == data_value

    async def test_concurrent_access(self, cache_manager):
        """Test concurrent cache access."""
        cache_key = "concurrent_test"
        test_value = {"test": "concurrent_value"}

        # Set initial value
        cache_manager.set(cache_key, test_value, 300, CacheDataType.HISTORICAL_OHLCV)

        # Create multiple concurrent tasks
        async def read_task(task_id):
            result = cache_manager.get(cache_key)
            await asyncio.sleep(0.1)  # Small delay
            return f"task_{task_id}_{result['test']}"

        # Run concurrent reads
        tasks = [read_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all tasks got the same result
        expected = "task_0_concurrent_value"
        for result in results:
            assert result == expected.replace("0", result.split("_")[1])
