"""
Unit tests for the Finnhub data provider.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import aiohttp

from src.data.providers.finnhub_provider import FinnhubProvider


class TestFinnhubProvider:
    """Test suite for FinnhubProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a FinnhubProvider instance for testing."""
        return FinnhubProvider("test-api-key")

    @pytest.fixture
    def mock_ohlcv_response(self):
        """Mock Finnhub OHLCV API response."""
        return {
            "s": "ok",
            "t": [1609459200, 1609545600, 1609632000],  # Unix timestamps
            "o": [100.0, 101.0, 102.0],  # Open prices
            "h": [105.0, 106.0, 107.0],  # High prices
            "l": [99.0, 100.0, 101.0],   # Low prices
            "c": [104.0, 105.0, 106.0],  # Close prices
            "v": [1000000, 1100000, 1200000]  # Volumes
        }

    @pytest.fixture
    def mock_quote_response(self):
        """Mock Finnhub quote API response."""
        return {
            "c": 150.25,  # Current price
            "h": 155.0,   # High of the day
            "l": 148.5,   # Low of the day
            "o": 149.0,   # Open price
            "pc": 148.75  # Previous close
        }

    @pytest.fixture
    def mock_news_response(self):
        """Mock Finnhub news API response."""
        return [
            {
                "category": "general",
                "datetime": 1609459200,
                "headline": "Test News Article 1",
                "id": 12345,
                "image": "https://example.com/image1.jpg",
                "related": "AAPL",
                "source": "Test Source",
                "summary": "This is a test news article summary.",
                "url": "https://example.com/article1"
            },
            {
                "category": "general",
                "datetime": 1609372800,
                "headline": "Test News Article 2",
                "id": 12346,
                "image": "https://example.com/image2.jpg",
                "related": "AAPL",
                "source": "Test Source",
                "summary": "This is another test news article summary.",
                "url": "https://example.com/article2"
            }
        ]

    @pytest.mark.asyncio
    async def test_init_with_api_key(self):
        """Test provider initialization with API key."""
        provider = FinnhubProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://finnhub.io/api/v1"
        assert provider.throttler.rate_limit == 60
        assert provider.throttler.period == 60.0

    @pytest.mark.asyncio
    async def test_init_without_api_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(ValueError, match="Finnhub API key is required"):
            FinnhubProvider("")

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, provider, mock_ohlcv_response):
        """Test successful historical data fetching."""
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 1, 3)

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_ohlcv_response)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_data("AAPL", start_date, end_date)

        # Verify the result is a properly formatted DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.name == "Date"
        
        # Verify data types
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        
        # Verify sample values
        assert result.iloc[0]["Open"] == 100.0
        assert result.iloc[0]["Close"] == 104.0
        assert result.iloc[0]["Volume"] == 1000000

    @pytest.mark.asyncio
    async def test_fetch_data_api_error(self, provider):
        """Test handling of API errors during data fetching."""
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 1, 3)

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock(side_effect=aiohttp.ClientError("API Error"))
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_data("INVALID", start_date, end_date)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_data_no_data(self, provider):
        """Test handling of no data response from API."""
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 1, 3)

        # Mock response indicating no data
        mock_response_data = {"s": "no_data", "error": "No data found"}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_data("INVALID", start_date, end_date)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_price_success(self, provider, mock_quote_response):
        """Test successful current price fetching."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_quote_response)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.get_current_price("AAPL")

        assert result == 150.25

    @pytest.mark.asyncio
    async def test_get_current_price_no_data(self, provider):
        """Test handling of missing current price data."""
        # Mock response with no current price
        mock_response_data = {"error": "Symbol not found"}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.get_current_price("INVALID")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_success(self, provider, mock_news_response):
        """Test successful news sentiment fetching."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_news_response)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_news_sentiment("AAPL", limit=2)

        # Verify the result is a list with correct format
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Verify article format matches Alpha Vantage format
        article = result[0]
        assert "title" in article
        assert "url" in article
        assert "summary" in article
        assert "source" in article
        assert "overall_sentiment_label" in article
        assert "ticker_sentiment" in article
        assert article["title"] == "Test News Article 1"
        assert article["overall_sentiment_label"] == "Neutral"

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_limit(self, provider, mock_news_response):
        """Test news sentiment fetching with limit parameter."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_news_response)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_news_sentiment("AAPL", limit=1)

        # Verify only 1 article is returned despite 2 in mock response
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_no_data(self, provider):
        """Test handling of no news data."""
        # Mock response with no news
        mock_response_data = {"error": "No news found"}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_news_sentiment("INVALID")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_company_profile_success(self, provider):
        """Test successful company profile fetching."""
        mock_profile_data = {
            "name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
            "ipo": "1980-12-12",
            "marketCapitalization": 2500000000000,
            "shareOutstanding": 16000000000,
            "weburl": "https://www.apple.com",
            "logo": "https://logo.clearbit.com/apple.com",
            "finnhubIndustry": "Technology"
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_profile_data)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_company_profile("AAPL")

        assert result is not None
        assert result["name"] == "Apple Inc."
        assert result["ticker"] == "AAPL"
        assert result["exchange"] == "NASDAQ"

    @pytest.mark.asyncio
    async def test_fetch_company_profile_no_data(self, provider):
        """Test handling of missing company profile."""
        mock_response_data = {"error": "Symbol not found"}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await provider.fetch_company_profile("INVALID")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_multiple_quotes(self, provider):
        """Test fetching multiple quotes in parallel."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        expected_prices = [150.25, 300.50, 2500.75]

        # Mock the get_current_price method
        async def mock_get_current_price(symbol):
            index = symbols.index(symbol)
            return expected_prices[index]

        provider.get_current_price = mock_get_current_price

        result = await provider.get_multiple_quotes(symbols)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["AAPL"] == 150.25
        assert result["MSFT"] == 300.50
        assert result["GOOGL"] == 2500.75

    @pytest.mark.asyncio
    async def test_get_multiple_quotes_with_errors(self, provider):
        """Test fetching multiple quotes with some errors."""
        symbols = ["AAPL", "INVALID", "MSFT"]
        expected_prices = [150.25, None, 300.50]

        # Mock the get_current_price method with some failures
        async def mock_get_current_price(symbol):
            if symbol == "INVALID":
                return None
            if symbol == "AAPL":
                return 150.25
            if symbol == "MSFT":
                return 300.50

        provider.get_current_price = mock_get_current_price

        result = await provider.get_multiple_quotes(symbols)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["AAPL"] == 150.25
        assert result["INVALID"] is None
        assert result["MSFT"] == 300.50

    def test_to_market_data_inherited(self, provider):
        """Test that to_market_data method is inherited from base class."""
        # Create sample DataFrame
        data = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [104.0],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex(['2021-01-01']))

        # This method should be inherited from BaseDataProvider
        market_data = provider.to_market_data(data, "AAPL")
        
        assert market_data.symbol == "AAPL"
        assert market_data.price == 104.0
        assert market_data.volume == 1000000
