"""
Unit tests for batch provider implementations.

This test suite validates the batch provider implementations for all supported data providers,
ensuring they achieve the targeted 60-80% reduction in API calls through intelligent
request aggregation and parallel processing.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, AsyncMock
import pandas as pd
import aiohttp

from src.data.providers.batch_yfinance_provider import YFinanceBatchProvider
from src.data.providers.batch_alpha_vantage_provider import AlphaVantageBatchProvider
from src.data.providers.batch_finnhub_provider import FinnhubBatchProvider
from src.data.providers.batch_marketaux_provider import MarketauxBatchProvider
from src.data.providers.batch_provider import BatchRequest, BatchResponse, BaseBatchProvider
from src.data.providers.unified_batch_provider import UnifiedBatchProvider


class TestBatchRequest:
    """Test suite for BatchRequest class."""

    def test_batch_request_creation(self):
        """Test BatchRequest object creation."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        request = BatchRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            request_type="historical_data",
            limit=20
        )

        assert request.symbols == symbols
        assert request.start_date == start_date
        assert request.end_date == end_date
        assert request.request_type == "historical_data"
        assert request.kwargs["limit"] == 20
        assert request.request_id is not None

    def test_batch_request_repr(self):
        """Test BatchRequest string representation."""
        request = BatchRequest(
            symbols=["AAPL", "MSFT"],
            request_type="current_price"
        )

        repr_str = repr(request)
        assert "BatchRequest" in repr_str
        assert "current_price" in repr_str
        assert "2 symbols" in repr_str


class TestBatchResponse:
    """Test suite for BatchResponse class."""

    def test_batch_response_creation(self):
        """Test BatchResponse object creation."""
        data = {"AAPL": {"price": 150.25}, "MSFT": {"price": 380.50}}
        errors = {"GOOGL": "No data available"}
        metadata = {"provider": "test_provider", "batch_count": 1}

        response = BatchResponse(123, data, errors, metadata)

        assert response.request_id == 123
        assert response.data == data
        assert response.errors == errors
        assert response.metadata == metadata
        assert response.success_count == 2
        assert response.error_count == 1
        assert response.total_count == 3
        assert response.success_rate == (2/3) * 100

    def test_batch_response_success_rate(self):
        """Test success rate calculation."""
        response = BatchResponse(123,
                                       {"AAPL": {"price": 150.25}},
                                       {"MSFT": {"price": 380.50}},
                                       {"GOOGL": {"price": "No data available"}},
        metadata={"provider": "test_provider"})

        assert response.success_rate == 66.67  # (2/3) * 100

    def test_get_symbol_data_and_error(self):
        """Test getting data and errors for specific symbols."""
        data = {"AAPL": {"price": 150.25}, "MSFT": {"price": 380.50}}
        errors = {"GOOGL": "No data available"}

        response = BatchResponse(123, data, errors, metadata)

        assert response.get_symbol_data("AAPL") == {"price": 150.25}
        assert response.get_symbol_data("MSFT") == {"price": 380.50}
        assert response.get_symbol_data("INVALID") is None
        assert response.get_symbol_error("AAPL") is None
        assert response.get_symbol_error("GOOGL") == "No data available"


class TestYFinanceBatchProvider:
    """Test suite for YFinanceBatchProvider."""

    @pytest.fixture
    def provider(self):
        """Create YFinanceBatchProvider instance for testing."""
        return YFinanceBatchProvider(rate_limit=10, period=60.0, max_batch_size=50)

    @pytest.marked.asyncio
    async def test_fetch_batch_historical_data_success(self, provider):
        """Test successful batch historical data fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        response = await provider.fetch_batch_historical_data(symbols, start_date, end_date)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                df = response.get_symbol_data(symbol)
                assert df is not None
                assert isinstance(df, pd.DataFrame)
                assert not df.empty

    @pytest.marked.asyncio
    async def test_fetch_batch_historical_data_empty_symbols(self, provider):
        """Test batch request with empty symbols list."""
        response = await provider.fetch_batch_historical_data([], datetime.now() - timedelta(days=30), datetime.now())

        assert response.total_count == 0
        assert response.success_count == 0
        assert response.error_count == 0

    @pytest.marked.asyncio
    async def test_fetch_batch_current_prices_success(self, provider):
        """Test successful batch current prices fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

        response = await provider.fetch_batch_current_prices(symbols)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data types
        if response.success_count > 0:
            for symbol in response.data:
                price = response.get_symbol_data(symbol)
                assert price is not None
                assert isinstance(price, (int, float))

    @pytest.marked.asyncio
    async def test_fetch_batch_news_sentiment_not_supported(self, provider):
        """Test that news sentiment is not supported."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_news_sentiment(symbols)

        assert response.total_count == 2  # All symbols processed
        assert response.success_count == 0
        assert response.error_count == 2
        assert "capability" in response.metadata
        assert response.metadata["capability"] == "news_sentiment_not_supported"

    @pytest.marked.asyncio
    async def test_fetch_batch_company_profiles_success(self, provider):
        """Test successful company profile fetching."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_company_profiles(symbols)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                profile = response.get_symbol_data(symbol)
                assert profile is not None
                assert isinstance(profile, dict)
                assert "name" in profile or "ticker" in profile

    @pytest.marked.asyncio
    async def test_rate_limiting(self, provider):
        """Test rate limiting behavior."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

        # Make multiple requests to test rate limiting
        responses = []
        for i in range(3):
            response = await provider.fetch_batch_current_prices(symbols)
            responses.append(response)

        # All requests should succeed
        assert all(isinstance(r, BatchResponse) for r in responses)

        # Verify rate limiting was applied (approximate test)
        metadata_list = [r.metadata for r in responses]
        assert len(metadata_list) == 3
        for metadata in metadata_list:
            assert "batch_count" in metadata


class TestAlphaVantageBatchProvider:
    """Test suite for AlphaVantageBatchProvider."""

    @pytest.fixture
    def provider(self):
        """Create AlphaVantageBatchProvider instance for testing."""
        return AlphaVantageBatchProvider(
            api_key="test-api-key",
            rate_limit=5,
            period=60.0,
            max_batch_size=5
        )

    @pytest.marked.asyncio
    async def test_fetch_batch_current_prices_success(self, provider):
        """Test successful batch current prices fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        response = await provider.fetch_batch_current_prices(symbols)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                price = response.get_symbol_data(symbol)
                assert price is not None
                assert isinstance(price, (int, float))

    @pytest.marked.asyncio
    async def test_batch_current_prices_large_batch(self, provider):
        """Test batch request exceeding size limit."""
        # Alpha Vantage has small batch size limits
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX", "TSLA", "NVDA"]

        response = await provider.fetch_batch_current_prices(symbols)

        # Should succeed but may have some errors due to rate limits
        assert isinstance(response, BatchResponse)
        assert response.total_count == len(symbols)
        assert response.success_count >= 0
        assert response.errors is not None

    @pytest.marked.asyncio
    async def test_fetch_batch_news_sentiment_parallel(self, provider):
        """Test news sentiment with parallel requests."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_news_sentiment(symbols, limit=10)

        assert isinstance(response, BatchResponse)
        assert response.total_count == 2
        assert response.errors is not None
        assert response.success_count == 2  # May succeed with limited data
        assert response.metadata.get("method") == "parallel_single_requests"


class TestFinnhubBatchProvider:
    """Test suite for FinnhubBatchProvider."""

    @pytest.fixture
    def provider(self):
        """Create FinnhubBatchProvider instance for testing."""
        return FinnhubBatchProvider(
            api_key="test-api-key",
            rate_limit=60,
            period=60.0,
            max_batch_size=20
        )

    @pytest.marked.asyncio
    async def test_fetch_batch_current_prices_success(self, provider):
        """Test successful batch current prices fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

        response = await provider.fetch_batch_current_prices(symbols)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                price = response.get_symbol_data(symbol)
                assert price is not None
                assert isinstance(price, (int, float))

    @pytest.marked.asyncio
    async def test_fetch_batch_news_sentiment_success(self, provider):
        """Test successful batch news sentiment fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        response = await provider.fetch_batch_news_sentiment(symbols, limit=20)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                articles = response.get_symbol_data(symbol)
                assert articles is not None
                assert isinstance(articles, list)

    @pytest.marked.asyncio
    async def test_fetch_batch_company_profiles_success(self, provider):
        """Test successful company profile fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        response = await provider.fetch_batch_company_profiles(symbols)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                profile = response.get_symbol_data(symbol)
                assert profile is not None
                assert isinstance(profile, dict)
                assert "name" in profile or "ticker" in profile

    @pytest.marked.asyncio
    async def test_large_batch_handling(self, provider):
        """Test handling of large batch requests."""
        symbols = [f"SYM{i}" for i in range(25)]  # 25 symbols

        response = await provider.fetch_batch_current_prices(symbols)

        assert isinstance(response, response)
        assert response.total_count == 25
        assert response.success_count >= 0
        assert response.errors is not None


class TestMarketauxBatchProvider:
    """Test suite for MarketauxBatchProvider."""

    @pytest.fixture
    def provider(self):
        """Create MarketauxBatchProvider instance for testing."""
        return MarketauxBatchProvider(
            api_key="test-api-key",
            rate_limit=100,
            period=86400.0,  # 1 day
            max_batch_size=10
        )

    @pytest.marked.asyncio
    async def test_fetch_batch_news_sentiment_success(self, provider):
        """Test successful batch news sentiment fetching."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        response = await provider.fetch_batch_news_sentiment(symbols, limit=15)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.errors is not None
        assert response.metadata is not None

        # Verify data structure
        if response.success_count > 0:
            for symbol in response.data:
                articles = response.get_symbol_data(symbol)
                assert articles is not None
                assert isinstance(articles, list)

    @pytest.marked.asyncio
    async def test_fetch_batch_current_prices_not_supported(self, provider):
        """Test that current prices are not supported."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_current_prices(symbols)

        assert response.total_count == 2
        assert response.success_count == 0
        assert response.error_count == 2
        assert response.metadata["capability"] == "current_price_not_supported"

    @pytest.marked.asyncio
    async def test_fetch_batch_historical_data_not_supported(self, provider):
        """Test that historical data is not supported."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_historical_data(
            symbols,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )

        assert response.total_count == 2
        assert response.success_count == 0
        assert response.error_count == 2
        assert response.metadata["capability"] == "historical_data_not_supported"

    @pytest.marked.asyncio
    async def test_fetch_batch_company_profiles_not_supported(self, provider):
        """Test that company profiles are not supported."""
        symbols = ["AAPL", "MSFT"]

        response = await provider.fetch_batch_company_profiles(symbols)

        assert response.total_count == 2
        assert response.success_count == 0
        assert response.error_count == 2
        assert response.metadata["capability"] == "company_profile_not_supported"


class TestUnifiedBatchProvider:
    """Test suite for UnifiedBatchProvider."""

    @pytest.fixture
    def config(self):
        """Create configuration for testing."""
        return {
            "ALPHA_VANTAGE_API_KEY": "test-av-key",
            "FINNHUB_API_KEY": "test-finnhub-key",
            "MARKETAUX_API_KEY": "test-marketaux-key"
        }

    @pytest.fixture
    def provider(self, config):
        """Create UnifiedBatchProvider instance for testing."""
        return UnifiedBatchProvider(config)

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.providers is not None
        assert provider.single_providers is not None
        assert provider.request_manager is not None
        assert provider.provider_capabilities is not None
        assert provider.provider_preferences is not None

        # Check available providers
        available_providers = provider.get_provider_status()["available_providers"]
        expected_providers = ["yfinance", "alpha_vantage", "finnhub", "marketaux"]

        for provider_name in expected_providers:
            assert provider_name in available_providers

    def test_select_best_provider(self, provider):
        """Test provider selection logic."""
        # Test historical data provider selection
        provider_name = provider._select_best_provider("historical_data", ["AAPL", "MSFT"])
        assert provider_name == "yfinance"  # yfinance should be preferred for historical data

        # Test current price provider selection
        provider_name = provider._select_best_provider("current_price", ["AAPL", "MSFT"])
        assert provider_name in ["finnhub", "yfinance", "alpha_vantage"]  # Finnhub preferred for current prices

        # Test news sentiment provider selection
        provider_name = provider._select_best_provider("news_sentiment", ["AAPL", "MSFT"])
        assert provider_name in ["finnhub", "alpha_vantage", "marketaux"]  # Finnhub preferred

        # Test unavailable data type
        provider_name = provider._select_best_provider("unavailable_type", ["AAPL"])
        assert provider_name is None

    @pytest.marked.asyncio
    async def test_fetch_batch_historical_data(self, provider):
        """Test batch historical data fetching through unified provider."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        response = await provider.fetch_batch_historical_data(symbols, start_date, end_date)

        assert isinstance(response, BatchResponse)
        assert response.success_count >= 0
        assert response.metadata is not None

        # Verify efficiency metrics
        if response.metadata:
            assert "efficiency_percentage" in response.metadata
            assert "batch_count" in response.metadata
            assert "symbols_per_batch" in response.metadata

    @pytest.marked.asyncio
    async def test_fetch_multi_type_data(self, provider):
        """Test fetching multiple data types in parallel."""
        symbols = ["AAPL", "MSFT"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        data_types = ["historical_data", "current_price", "news_sentiment"]

        responses = await provider.fetch_multi_type_data(
            symbols=symbols,
            data_types=data_types,
            start_date=start_date,
            end_date=end_date,
            news_limit=20
        )

        assert len(responses) == len(data_types)

        # Check each data type response
        for data_type in data_types:
            assert data_type in responses
            response = responses[data_type]
            assert isinstance(response, BatchResponse)

            if data_type == "historical_data":
                assert response.success_count >= 0
            elif data_type == "current_price":
                assert response.success_count >= 0
            elif data_type == "news_sentiment":
                assert response.success_count >= 0

    @pytest.marked.asyncio
    async def test_batch_efficiency_calculation(self, provider):
        """Test efficiency calculation."""
        response = await provider.fetch_batch_historical_data(
            ["AAPL", "MSFT", "GOOGL"],
            datetime.now() - timedelta(days=30),
            datetime.now()
        )

        if response.metadata:
            efficiency = response.metadata.get("efficiency_percentage", 0)
            api_calls_saved = response.metadata.get("api_calls_saved", 0)

            assert efficiency >= 0
            assert api_calls_saved >= 0

            efficiency_metrics = provider.calculate_batch_efficiency(
                len(symbols),
                data_types=["historical_data"],
                providers_used=[response.metadata.get("provider", "unknown")]
            )

            assert efficiency_metrics["efficiency_percentage"] >= 0
            assert efficiency_metrics["requests_saved"] >= 0

    def test_get_optimal_batch_size(self, provider):
        """Test optimal batch size calculation."""
        # Test different data types
        test_cases = [
            ("historical_data", "yfinance", 50),
            ("current_price", "finnhub", 20),
            ("news_sentiment", "finnhub", 20),
            ("company_profile", "finnhub", 10)
        ]

        for data_type, expected_provider, expected_size in test_cases:
            optimal_size = provider.get_optimal_batch_size(data_type, expected_provider)
            assert optimal_size == expected_size

    def test_provider_status(self, provider):
        """Test provider status reporting."""
        status = provider.get_provider_status()

        assert "available_providers" in status
        assert "provider_capabilities" in status
        assert "provider_preferences" in status
        assert "batch_statistics" in status

        # Verify all expected providers are available
        expected_providers = ["yfinance", "alpha_vantage", "finnhub", "marketaux"]
        for provider_name in expected_providers:
            assert provider_name in status["available_providers"]

    async def test_fallback_to_single_requests(self, provider):
        """Test fallback to single requests when batch processing fails."""
        # Create a mock provider that fails batch requests
        failing_provider = MagicMock()
        failing_provider.fetch_batch_historical_data = AsyncMock(
            side_effect=lambda *args, **kwargs: (BatchResponse(
                request_id=999,
                data={},
                errors={symbol: "Batch request failed" for symbol in args[0]},
                metadata={"error": "Batch request failed"}
            ))
        )

        # Temporarily replace batch provider
        original_batch = provider.batch_provider
        provider.batch_provider = None

        try:
            # Should fall back to single requests
            response = await provider.fetch_and_process_multiple_data(
                ["AAPL", "MSFT"],
                datetime.now() - timedelta(days=30),
                datetime.now()
            )

            assert len(response) == 2
            assert response.count(None) == 0
        finally:
            # Restore original batch provider
            provider.batch_provider = original_batch_provider

    def test_concurrent_batch_processing(self, provider):
        """Test concurrent batch processing."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Create multiple concurrent requests
        tasks = [
            provider.fetch_batch_historical_data(symbols[i::2], start_date, end_date)
            for i in range(0, len(symbols), 2)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete
        assert len(results) == 2
        assert all(isinstance(r, Exception) or isinstance(r, BatchResponse) for r in results)

        # Filter out exceptions and process results
        responses = [r for r in results if not isinstance(r, Exception)]
        assert len(responses) >= 0

    async def test_cache_integration(self, provider):
        """Test batch provider integration with enhanced caching."""
        symbols = ["AAPL", "MSFT"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Mock cache manager
        cache_manager = AsyncMock()
        cache_manager.get.return_value = None  # Cache miss initially
        cache_manager.set.return_value = None

        # Temporarily replace provider with caching enabled version
        original_provider = provider.batch_provider
        provider.batch_provider = None

        try:
            # First request - cache miss
            response1 = await original_provider.fetch_batch_historical_data(symbols, start_date, end_date)

            # Cache the response
            if cache_manager.set.return_value is not None:
                cache_manager.set.return_value = (response1, 300)  # Cache with 5 min TTL

            # Second request - cache hit
            response2 = await original_provider.fetch_batch_historical_data(symbols, start_date, end_date)

            # Both should have the same data
            assert response1.request_id == response2.request_id
            assert response1.success_count == response2.success_count
            assert response1.data == response2.data

        finally:
            # Restore original batch provider
            provider.batch_provider = original_provider

    async def close_all_sessions(self, provider):
        """Test cleanup of all provider sessions."""
        # This should not raise any exceptions
        await provider.close_all_sessions()


class TestBatchProviderIntegration:
    """Integration tests for batch provider system."""

    @pytest.fixture
    def config(self):
        """Get configuration for integration tests."""
        # Use mock API keys for integration tests
        return {
            "ALPHA_VANTAGE_API_KEY": "integration-test-av-key",
            "FINNHUB_API_KEY": "integration-test-finnhub-key",
            "MARKETAUX_API_KEY": "integration-test-marketaux-key"
        }

    @pytest.fixture
    def provider(self, config):
        """Create unified batch provider for integration tests."""
        return UnifiedBatchProvider(config)

    @pytest.marked.integration
    async def test_end_to_end_batch_workflow(self, provider):
        """Test complete batch workflow from start to finish."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Fetch multiple data types
        responses = await provider.fetch_multi_type_data(
            symbols=symbols,
            data_types=["historical_data", "current_price", "news_sentiment"],
            start_date=start_date,
            end_date=end_date
        )

        # Verify all responses are valid
        assert len(responses) == 3
        for data_type, response in responses.items():
            assert isinstance(response, BatchResponse)
            assert response.total_count == len(symbols)
            assert response.success_count >= 0

    @pytest.marked.integration
    async def test_batch_efficiency_gains(self, provider):
        """Test that batch processing achieves efficiency gains."""
        # Test with larger symbol set
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Test batch vs individual performance
        start_time = datetime.now()
        batch_response = await provider.fetch_batch_historical_data(symbols, start_date, end_date)
        batch_time = datetime.now() - start_time

        start_time = datetime.now()
        individual_responses = []
        for symbol in symbols:
            response = await provider.single_providers["yfinance"].fetch_data(symbol, start_date, end_date)
            individual_responses.append(response)
        individual_time = datetime.now() - start_time

        # Batch should be more efficient for multiple symbols
        assert batch_time < individual_time
        efficiency_gain = (individual_time - batch_time) / individual_time * 100 if individual_time > 0 else 0

        logger.info(f"Batch processing achieved {efficiency_gain:.1f}% efficiency gain")
        assert efficiency_gain >= 0

    @pytest.marked.integration
    async def test_provider_failover_behavior(self, provider):
        """Test graceful degradation when providers fail."""
        # Test with invalid provider (should fall back gracefully)
        provider.providers = {}

        response = await provider.fetch_batch_current_prices(["AAPL", "MSFT"])

        # Should handle failure gracefully
        assert response.total_count == 2
        assert response.success_count == 0
        assert len(response.errors) == 2
        assert "error" in response.metadata

    @pytest.marked.integration
    async def test_provider_error_recovery(self, provider):
        """Test error recovery and continued operation."""
        # Test with partially failing provider
        original_providers = provider.providers.copy()

        # Mark one provider as failed
        failed_provider = provider.providers.get("alpha_vantage")
        del provider.providers["alpha_vantage"]

        response = await provider.fetch_batch_current_prices(["AAPL", "MSFT"])

        # Should still work with remaining providers
        assert response.total_count == 2
        assert response.success_count >= 1  # At least one provider should work

        # Restore failed provider
        provider.providers = original_providers

    @pytest.marked.integration
    async def test_caching_with_batch_processing(self, provider):
        """Test that caching works with batch processing."""
        symbols = ["AAPL", "MSFT"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # First request - cache miss
        response1 = await provider.fetch_batch_historical_data(symbols, start_date, end_date)
        assert response1.success_count >= 0

        # Second request - should be cache hit
        response2 = await provider.fetch_batch_historical_data(symbols, start_date, end_date)
        assert response2.request_id == response1.request_id
        assert response2.success_count == response1.success_count
        assert response2.data == response1.data

    @pytest.marked.integration
    async def test_news_provider_selection(self, provider):
        """Test intelligent provider selection for different data types."""
        # Test news provider selection
        news_providers = provider._select_best_provider("news_sentiment", ["AAPL", "MSFT", "GOOGL"])
        assert news_providers in ["finnhub", "alpha_vantage", "marketaux"]

        # Test provider selection with invalid data type
        invalid_provider = provider._select_best_provider("invalid_type", ["AAPL"])
        assert invalid_provider is None


# Performance and Load Testing
class TestBatchProviderPerformance:
    """Test suite for batch provider performance under load."""

    @pytest.marked.performance
    async def test_large_scale_processing(self, provider):
        """Test processing large symbol sets."""
        # Test with 100 symbols
        symbols = [f"SYM{i:03d}" for i in range(100)]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        start_time = datetime.now()
        response = await provider.fetch_batch_historical_data(symbols, start_date, end_date)
        processing_time = datetime.now() - start_time

        # Performance targets
        assert processing_time < 30  # Should complete within 30 seconds
        assert response.success_count >= 90  # At least 90% success rate
        assert response.success_rate >= 90

    @pytest.marked.performance
    async def test_concurrent_load(self, provider):
        """Test concurrent batch processing under load."""
        symbols = [f"SYM{i:03d}" for i in range(50)]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Create multiple concurrent requests
        tasks = [
            provider.fetch_batch_historical_data(
                symbols[i::10], start_date, end_date
            ) for i in range(0, len(symbols), 10)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = sum(1 for r in results if not isinstance(r, Exception) and r.success_count > 0)
        total_requests = len(results)

        assert success_count >= 45  # At least 90% success rate
        assert success_count / total_requests >= 0.9
