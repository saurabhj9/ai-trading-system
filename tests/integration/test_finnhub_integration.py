"""
Integration tests for the Finnhub data provider.

These tests require a real Finnhub API key and will make actual API calls.
Run these tests sparingly to avoid rate limits.
"""
import pytest
import os
from datetime import datetime, timedelta

from src.data.providers.finnhub_provider import FinnhubProvider
from src.config.settings import settings


@pytest.mark.integration
@pytest.mark.asyncio
class TestFinnhubIntegration:
    """Integration tests for FinnhubProvider with real API calls."""

    @pytest.fixture
    def api_key(self):
        """Get Finnhub API key from environment."""
        api_key = os.getenv("FINNHUB_API_KEY") or settings.data.FINNHUB_API_KEY
        if not api_key or api_key == "YOUR-API-KEY-HERE":
            pytest.skip("FINNHUB_API_KEY not configured")
        return api_key

    @pytest.fixture
    def provider(self, api_key):
        """Create a FinnhubProvider with real API key."""
        return FinnhubProvider(api_key)

    async def test_fetch_real_data(self, provider):
        """Test fetching real historical data for a major stock."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=1)

        result = await provider.fetch_data("AAPL", start_date, end_date)

        assert result is not None
        assert len(result) > 0
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.name == "Date"

    async def test_get_real_current_price(self, provider):
        """Test fetching real current price for a major stock."""
        result = await provider.get_current_price("AAPL")

        assert result is not None
        assert isinstance(result, (int, float))
        assert result > 0

    async def test_fetch_real_news_sentiment(self, provider):
        """Test fetching real news sentiment for a major stock."""
        result = await provider.fetch_news_sentiment("AAPL", limit=5)

        assert result is not None
        assert isinstance(result, list)
        if len(result) > 0:  # News might not always be available
            article = result[0]
            assert "title" in article
            assert "url" in article
            assert "source" in article

    async def test_fetch_real_company_profile(self, provider):
        """Test fetching real company profile for a major stock."""
        result = await provider.fetch_company_profile("AAPL")

        assert result is not None
        assert "name" in result
        assert "ticker" in result
        assert result["ticker"] == "AAPL"

    async def test_real_multiple_quotes(self, provider):
        """Test fetching multiple real quotes in parallel."""
        symbols = ["AAPL", "MSFT"]
        
        result = await provider.get_multiple_quotes(symbols)

        assert isinstance(result, dict)
        assert len(result) == 2
        for symbol in symbols:
            assert symbol in result
            if result[symbol] is not None:  # Prices might be None if markets closed
                assert isinstance(result[symbol], (int, float))
                assert result[symbol] > 0

    async def test_invalid_symbol_handling(self, provider):
        """Test handling of invalid symbols with real API."""
        # Test with clearly invalid symbol
        result = await provider.get_current_price("INVALID_SYMBOL_12345")
        assert result is None

        # Test with invalid historical data request
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=1)
        result = await provider.fetch_data("INVALID_SYMBOL_12345", start_date, end_date)
        assert result is None
