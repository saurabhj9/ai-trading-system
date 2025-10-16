"""
Unit tests for CompositeNewsProvider that verifies the fallback mechanism
between Alpha Vantage and Marketaux for news sentiment data.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.data.providers.composite_news_provider import CompositeNewsProvider
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.marketaux_provider import MarketauxProvider


@pytest.fixture
def mock_alpha_vantage_provider():
    """Create a mock Alpha Vantage provider."""
    provider = AsyncMock(spec=AlphaVantageProvider)
    return provider


@pytest.fixture
def mock_marketaux_provider():
    """Create a mock Marketaux provider."""
    provider = AsyncMock(spec=MarketauxProvider)
    return provider


@pytest.fixture
def composite_provider(mock_alpha_vantage_provider, mock_marketaux_provider):
    """Create a CompositeNewsProvider with mocked providers."""
    with patch('src.data.providers.composite_news_provider.AlphaVantageProvider', return_value=mock_alpha_vantage_provider), \
         patch('src.data.providers.composite_news_provider.MarketauxProvider', return_value=mock_marketaux_provider):
        provider = CompositeNewsProvider("test_alpha_key", "test_marketaux_key")
        provider.alpha_vantage = mock_alpha_vantage_provider
        provider.marketaux = mock_marketaux_provider
        return provider


@pytest.fixture
def sample_alpha_vantage_news():
    """Sample news data from Alpha Vantage."""
    return [
        {
            "title": "Alpha Vantage News 1",
            "url": "https://example.com/news1",
            "time_published": "20231009T12:30:00",
            "authors": ["Author 1"],
            "summary": "Summary of news 1",
            "banner_image": "https://example.com/image1.jpg",
            "source": "Source 1",
            "category_within_source": "Technology",
            "source_domain": "example.com",
            "topics": [{"topic": "Technology", "relevance_score": "0.8"}],
            "overall_sentiment_score": 0.2,
            "overall_sentiment_label": "Positive",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_score": 0.3,
                    "ticker_sentiment_label": "Positive"
                }
            ]
        },
        {
            "title": "Alpha Vantage News 2",
            "url": "https://example.com/news2",
            "time_published": "20231009T13:45:00",
            "authors": ["Author 2"],
            "summary": "Summary of news 2",
            "banner_image": "https://example.com/image2.jpg",
            "source": "Source 2",
            "category_within_source": "Finance",
            "source_domain": "example.com",
            "topics": [{"topic": "Finance", "relevance_score": "0.7"}],
            "overall_sentiment_score": -0.1,
            "overall_sentiment_label": "Negative",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.8",
                    "ticker_sentiment_score": -0.2,
                    "ticker_sentiment_label": "Negative"
                }
            ]
        }
    ]


@pytest.fixture
def sample_marketaux_news():
    """Sample news data from Marketaux (in Alpha Vantage format)."""
    return [
        {
            "title": "Marketaux News 1",
            "url": "https://marketaux.com/news1",
            "time_published": "20231009T14:30:00",
            "authors": ["Marketaux"],
            "summary": "Summary of Marketaux news 1",
            "banner_image": "https://marketaux.com/image1.jpg",
            "source": "Marketaux Source",
            "category_within_source": "Markets",
            "source_domain": "marketaux.com",
            "topics": [],
            "overall_sentiment_score": 0.5,
            "overall_sentiment_label": "Positive",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": 1.0,
                    "ticker_sentiment_score": 0.5,
                    "ticker_sentiment_label": "Positive"
                }
            ]
        }
    ]


class TestCompositeNewsProvider:
    """Test cases for CompositeNewsProvider."""

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_alpha_vantage_success(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_alpha_vantage_news
    ):
        """Test successful news fetch from Alpha Vantage without fallback."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = sample_alpha_vantage_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result == sample_alpha_vantage_news
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_alpha_vantage_http_error_fallback_to_marketaux(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_marketaux_news
    ):
        """Test fallback to Marketaux when Alpha Vantage fails with HTTP error."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("HTTP Error")
        mock_marketaux_provider.fetch_news_sentiment.return_value = sample_marketaux_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result == sample_marketaux_news
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_alpha_vantage_empty_fallback_to_marketaux(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_marketaux_news
    ):
        """Test fallback to Marketaux when Alpha Vantage returns empty list."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = []
        mock_marketaux_provider.fetch_news_sentiment.return_value = sample_marketaux_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result == sample_marketaux_news
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_alpha_vantage_none_fallback_to_marketaux(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_marketaux_news
    ):
        """Test fallback to Marketaux when Alpha Vantage returns None."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = None
        mock_marketaux_provider.fetch_news_sentiment.return_value = sample_marketaux_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result == sample_marketaux_news
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_both_providers_fail(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test when both Alpha Vantage and Marketaux fail."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("Alpha Vantage Error")
        mock_marketaux_provider.fetch_news_sentiment.side_effect = Exception("Marketaux Error")
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result is None
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_both_providers_return_empty(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test when both providers return empty results."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = []
        mock_marketaux_provider.fetch_news_sentiment.return_value = []
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result is None
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_both_providers_return_none(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test when both providers return None."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = None
        mock_marketaux_provider.fetch_news_sentiment.return_value = None
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result is None
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("AAPL", 10)

    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_marketaux_success_as_fallback(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_marketaux_news
    ):
        """Test Marketaux succeeding as a fallback provider."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("Alpha Vantage Error")
        mock_marketaux_provider.fetch_news_sentiment.return_value = sample_marketaux_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("MSFT", 15)
        
        # Verify
        assert result == sample_marketaux_news
        mock_alpha_vantage_provider.fetch_news_sentiment.assert_called_once_with("MSFT", 15)
        mock_marketaux_provider.fetch_news_sentiment.assert_called_once_with("MSFT", 15)

    @pytest.mark.asyncio
    async def test_fetch_data_uses_alpha_vantage_only(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test that fetch_data only uses Alpha Vantage."""
        # Setup
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        mock_df = MagicMock()
        mock_alpha_vantage_provider.fetch_data.return_value = mock_df
        
        # Execute
        result = await composite_provider.fetch_data("AAPL", start_date, end_date)
        
        # Verify
        assert result == mock_df
        mock_alpha_vantage_provider.fetch_data.assert_called_once_with("AAPL", start_date, end_date)
        mock_marketaux_provider.fetch_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_current_price_uses_alpha_vantage_only(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test that get_current_price only uses Alpha Vantage."""
        # Setup
        mock_alpha_vantage_provider.get_current_price.return_value = 150.25
        
        # Execute
        result = await composite_provider.get_current_price("AAPL")
        
        # Verify
        assert result == 150.25
        mock_alpha_vantage_provider.get_current_price.assert_called_once_with("AAPL")
        mock_marketaux_provider.get_current_price.assert_not_called()

    @pytest.mark.asyncio
    async def test_logging_alpha_vantage_success(
        self, composite_provider, mock_alpha_vantage_provider, sample_alpha_vantage_news
    ):
        """Test that successful Alpha Vantage fetch is logged correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = sample_alpha_vantage_news
        
        with patch.object(composite_provider.logger, 'info') as mock_logger_info:
            # Execute
            await composite_provider.fetch_news_sentiment("AAPL", 10)
            
            # Verify
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Alpha Vantage")
            mock_logger_info.assert_any_call(f"Successfully fetched {len(sample_alpha_vantage_news)} news articles from Alpha Vantage for AAPL")

    @pytest.mark.asyncio
    async def test_logging_alpha_vantage_empty_fallback(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test that Alpha Vantage empty result fallback is logged correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = []
        mock_marketaux_provider.fetch_news_sentiment.return_value = None
        
        with patch.object(composite_provider.logger, 'warning') as mock_logger_warning, \
             patch.object(composite_provider.logger, 'info') as mock_logger_info:
            
            # Execute
            await composite_provider.fetch_news_sentiment("AAPL", 10)
            
            # Verify
            mock_logger_warning.assert_any_call("Alpha Vantage returned no news for AAPL, falling back to Marketaux")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Alpha Vantage")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Marketaux (fallback)")

    @pytest.mark.asyncio
    async def test_logging_alpha_vantage_error_fallback(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test that Alpha Vantage error fallback is logged correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("Connection Error")
        mock_marketaux_provider.fetch_news_sentiment.return_value = None
        
        with patch.object(composite_provider.logger, 'warning') as mock_logger_warning, \
             patch.object(composite_provider.logger, 'info') as mock_logger_info:
            
            # Execute
            await composite_provider.fetch_news_sentiment("AAPL", 10)
            
            # Verify
            mock_logger_warning.assert_any_call("Alpha Vantage failed for AAPL with error: Connection Error, falling back to Marketaux")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Alpha Vantage")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Marketaux (fallback)")

    @pytest.mark.asyncio
    async def test_logging_both_providers_fail(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider
    ):
        """Test that both providers failing is logged correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("Alpha Vantage Error")
        mock_marketaux_provider.fetch_news_sentiment.side_effect = Exception("Marketaux Error")
        
        with patch.object(composite_provider.logger, 'error') as mock_logger_error, \
             patch.object(composite_provider.logger, 'warning') as mock_logger_warning, \
             patch.object(composite_provider.logger, 'info') as mock_logger_info:
            
            # Execute
            await composite_provider.fetch_news_sentiment("AAPL", 10)
            
            # Verify
            mock_logger_error.assert_any_call("Both Alpha Vantage and Marketaux failed for AAPL. Marketaux error: Marketaux Error")
            mock_logger_warning.assert_any_call("Alpha Vantage failed for AAPL with error: Alpha Vantage Error, falling back to Marketaux")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Alpha Vantage")
            mock_logger_info.assert_any_call("Fetching news sentiment for AAPL using Marketaux (fallback)")

    @pytest.mark.asyncio
    async def test_data_format_validation_alpha_vantage(
        self, composite_provider, mock_alpha_vantage_provider, sample_alpha_vantage_news
    ):
        """Test that Alpha Vantage data format is preserved correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.return_value = sample_alpha_vantage_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result is not None
        assert len(result) == 2
        
        # Check first article
        article = result[0]
        assert "title" in article
        assert "url" in article
        assert "time_published" in article
        assert "authors" in article
        assert "summary" in article
        assert "banner_image" in article
        assert "source" in article
        assert "category_within_source" in article
        assert "source_domain" in article
        assert "topics" in article
        assert "overall_sentiment_score" in article
        assert "overall_sentiment_label" in article
        assert "ticker_sentiment" in article
        
        # Check ticker sentiment structure
        ticker_sentiment = article["ticker_sentiment"][0]
        assert "ticker" in ticker_sentiment
        assert "relevance_score" in ticker_sentiment
        assert "ticker_sentiment_score" in ticker_sentiment
        assert "ticker_sentiment_label" in ticker_sentiment

    @pytest.mark.asyncio
    async def test_data_format_validation_marketaux(
        self, composite_provider, mock_alpha_vantage_provider, mock_marketaux_provider, sample_marketaux_news
    ):
        """Test that Marketaux data format is preserved correctly."""
        # Setup
        mock_alpha_vantage_provider.fetch_news_sentiment.side_effect = Exception("Alpha Vantage Error")
        mock_marketaux_provider.fetch_news_sentiment.return_value = sample_marketaux_news
        
        # Execute
        result = await composite_provider.fetch_news_sentiment("AAPL", 10)
        
        # Verify
        assert result is not None
        assert len(result) == 1
        
        # Check article structure
        article = result[0]
        assert "title" in article
        assert "url" in article
        assert "time_published" in article
        assert "authors" in article
        assert "summary" in article
        assert "banner_image" in article
        assert "source" in article
        assert "category_within_source" in article
        assert "source_domain" in article
        assert "topics" in article
        assert "overall_sentiment_score" in article
        assert "overall_sentiment_label" in article
        assert "ticker_sentiment" in article
        
        # Check ticker sentiment structure
        ticker_sentiment = article["ticker_sentiment"][0]
        assert "ticker" in ticker_sentiment
        assert "relevance_score" in ticker_sentiment
        assert "ticker_sentiment_score" in ticker_sentiment
        assert "ticker_sentiment_label" in ticker_sentiment