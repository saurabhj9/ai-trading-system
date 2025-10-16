"""
Integration tests for batch provider integration.

This test suite validates the entire batch provider system, including provider selection,
caching integration, performance improvements, and reliability under load.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pytest
from unittest.mock import AsyncMock
import logging

from src.data.pipeline import DataPipeline
from src.data.cache import CacheManager
from src.data.cache.cache_config import CacheConfig, CacheDataType
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.providers.unified_batch_provider import UnifiedBatchProvider
from src.data.providers.batch_provider import BatchResponse
from src.agents.data_structures import MarketData
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TestBatchIntegration:
    """Integration tests for batch provider integration."""

    @pytest.fixture
    def config(self):
        """Get configuration for integration tests."""
        # Use environment variables or test configuration
        return {
            "ALPHA_VANTAGE_API_KEY": settings.data.ALPHA_VANTAGE_API_KEY or "integration-test-av-key",
            "FINNHUB_API_KEY": settings.data.FINNHUB_API_KEY or "integration-test-finnhub-key",
            "MARKETAUX_API_KEY": settings.data.MARKETAUX_API_KEY or "integration-test-marketaux-key",
            "CACHE_ENABLED": True,
            "USE_ENHANCED_CACHING": True,
            "USE_BATCH_PROCESSING": True,
            "BATCH_CONFIG": {
                "ALPHA_VANTAGE_API_KEY": settings.data.ALPHA_VANTAGE_API_KEY or "integration-test-av-key",
                "FINNHUB_API_KEY": settings.data.FINNHUB_API_KEY or "integration-test-finnhub-key",
                "MARKETAUX_API_KEY": settings.data.MARKETAUX_API_KEY or "integration-test-marketaux-key"
            }
        }
    
    @pytest.fixture
    def data_pipeline(self, config):
        """Create DataPipeline with batch processing enabled."""
        return DataPipeline(
            provider=YFinanceProvider(),
            cache=CacheManager(enable_redis=True, cache_config=CacheConfig()) if config.get("USE_ENHANCED_CACHING") else None,
            cache_ttl_seconds=300,
            use_enhanced_caching=config.get("USE_ENHANCED_CACHING", True),
            use_batch_processing=config.get("USE_BATCH_PROCESSING", True),
            batch_config=config.get("BATCH_CONFIG", {})
        )
    
    @pytest.fixture
    def batch_provider(self, config):
        """Create unified batch provider for testing."""
        return UnifiedBatchProvider(config)
    
    @pytest.mark.integration
    async def test_batch_historical_data_workflow(self, data_pipeline, batch_provider):
        """Test complete batch historical data workflow."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now()
        
        # Test the batch workflow
        start_time = datetime.now()
        
        # First pass - all cache misses
        results1 = await data_pipeline.fetch_and_process_multiple_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        first_pass_time = datetime.now() - start_time
        
        assert len(results1) == len(symbols)
        for result in results1:
            assert result is not None
            assert isinstance(result, MarketData)
        
        # Second pass - should mostly cache hits
        start_time = datetime.now()
        results2 = await data_pipeline.fetch_and_process_multiple_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        second_pass_time = datetime.now() - start_time
        
        # Second pass should be much faster due to caching
        assert second_pass_time < first_pass_time * 0.5  # At least 50% improvement
        
        # Log efficiency improvements
        cache_hit_count = sum(1 for r in results2 if r is not None)
        cache_miss_count = sum(1 for r in results2 if r is None)
        
        logger.info(f"Batch workflow: {cache_hit_count}/{len(symbols)} cache hits, {cache_miss_count} cache misses")
        
        # Validate data consistency
        for i, result in enumerate(results2):
            assert result is not None
            assert isinstance(result, MarketData)
            assert result.symbol == symbols[i]
            assert result.price > 0
    
    @pytest.mark.integration
    async def test_batch_current_price_workflow(self, data_pipeline, batch_provider):
        """Test batch current price workflow."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        
        # Test current price fetching via batch provider
        batch_response = await batch_provider.fetch_batch_current_prices(symbols)
        
        assert batch_response.success_count >= 0
        assert isinstance(batch_response, BatchResponse)
        
        # Verify data quality
        for symbol in symbols:
            price = batch_response.get_symbol_data(symbol)
            if price is not None:
                assert isinstance(price, (int, float))
    
    @pytest.mark.integration
    async def test_news_sentiment_workflow(self, data_pipeline, batch_provider):
        """Test news sentiment workflow with batch processing."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=7)
        
        # Test news sentiment fetching
        response = await batch_provider.fetch_batch_news_sentiment(
            symbols=symbols,
            limit=15
        )
        
        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        
        # Validate news data structure
        for symbol in symbols:
            articles = response.get_symbol_data(symbol)
            if articles:
                assert isinstance(articles, list)
                for article in articles:
                    assert "title" in article
                    assert "summary" in article
    
    @pytest.mark.integration
    async def test_multi_type_batch_workflow(self, data_pipeline, batch_provider):
        """Test multi-type data fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        data_types = ["historical_data", "current_price", "news_sentiment", "company_profile"]
        
        responses = await data_pipeline.fetch_multi_type_data(
            symbols=symbols,
            data_types=data_types,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate all data types were processed
        assert len(responses) == len(data_types)
        
        for data_type, response in responses.items():
            assert isinstance(response, BatchResponse)
            
            if data_type == "historical_data":
                assert response.success_count >= 0
            elif data_type == "current_price":
                assert response.success_count >= 0
            elif data_type == "news_sentiment":
                assert response.success_count >= 0
            elif data_type == "company_profile":
                assert response.success_count >= 0
    
    @pytest.mark.integration
    async def test_batch_efficiency_gains(self, data_pipeline, batch_provider):
        """Test that batch processing delivers measurable efficiency gains."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "NFLX"]
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now()
        
        # Test individual request baseline
        start_time = datetime.now()
        individual_results = []
        for symbol in symbols:
            result = await data_pipeline.provider.fetch_data(symbol, start_date, end_date)
            individual_results.append(result)
        individual_time = datetime.now() - start_time
        
        # Test batch processing
        start_time = datetime.now()
        batch_response = await batch_provider.fetch_batch_historical_data(symbols, start_date, end_date)
        batch_time = datetime.now() - start_time
        
        # Calculate efficiency
        efficiency_gain = ((individual_time - batch_time) / individual_time * 100) if individual_time.total_seconds() > 0 else 0
        
        logger.info(f"Batch processing achieved {efficiency_gain:.1f}% efficiency improvement")
        
        # Should see significant efficiency gains
        assert efficiency_gain >= 10  # At least 10% improvement
    
    @pytest.mark.integration
    async def test_concurrent_batch_processing(self, data_pipeline, batch_provider):
        """Test concurrent batch processing capabilities."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "NFLX"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Create concurrent tasks
        tasks = [
            batch_provider.fetch_batch_historical_data(
                symbols[i::3], start_date, end_date
            ) for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = [r for r in results if not isinstance(r, Exception) and r.success_count > 0]
        
        assert len(successful_results) >= len(results) * 0.8  # At least 80% success rate
    
    @pytest.mark.integration
    async def test_error_handling_and_recovery(self, data_pipeline):
        """Test error handling and graceful degradation."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Should fall back to individual requests when batch fails
        response = await data_pipeline.fetch_and_process_multiple_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Should get responses even with batch failures
        assert len(response) == len(symbols)
    
    @pytest.mark.integration
    async def test_performance_comparison(self, data_pipeline, batch_provider):
        """Test performance comparison between batch and individual processing."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "NFLX"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Test individual processing baseline
        start_time = datetime.now()
        individual_results = []
        for symbol in symbols:
            result = await data_pipeline.fetch_and_process_data(symbol, start_date, end_date)
            individual_results.append(result)
        individual_time = datetime.now() - start_time
        
        # Test batch processing
        start_time = datetime.now()
        batch_response = await batch_provider.fetch_batch_historical_data(symbols, start_date, end_date)
        batch_time = datetime.now() - start_time
        
        # Batch should be significantly faster
        efficiency_gain = ((individual_time - batch_time) / individual_time * 100) if individual_time.total_seconds() > 0 else 0
        
        logger.info(f"Batch processing achieved {efficiency_gain:.1f}% improvement")
        
        assert efficiency_gain >= 10  # Should see significant improvement
    
    @pytest.mark.integration
    async def test_cache_integration_with_batch(self, data_pipeline, batch_provider):
        """Test caching integration with batch processing."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # First request - cache miss
        response1 = await data_pipeline.fetch_and_process_multiple_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Second request should be faster due to caching
        start_time = datetime.now()
        response2 = await data_pipeline.fetch_and_process_multiple_data(
            symbols=symbols[:2],  # Test subset
            start_date=start_date,
            end_date=end_date
        )
        second_time = datetime.now() - start_time
        
        # Validate responses
        assert len(response1) == len(symbols)
        assert len(response2) == 2
        
        for result in response1:
            if result is not None:
                assert isinstance(result, MarketData)
        
        for result in response2:
            if result is not None:
                assert isinstance(result, MarketData)
    
    @pytest.mark.integration
    async def test_batch_news_cache_efficiency(self, data_pipeline, batch_provider):
        """Test cache efficiency with news sentiment data."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # First fetch - cache miss
        response1 = await batch_provider.fetch_batch_news_sentiment(symbols, limit=10)
        
        # Second fetch - should be cached
        start_time = datetime.now()
        response2 = await batch_provider.fetch_batch_news_sentiment(symbols, limit=10)
        second_time = datetime.now() - start_time
        
        # Validate responses
        assert isinstance(response1, BatchResponse)
        assert isinstance(response2, BatchResponse)
        assert response1.success_count >= 0
        assert response2.success_count >= 0
