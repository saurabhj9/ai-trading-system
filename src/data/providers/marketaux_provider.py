"""
Data provider implementation for fetching news data from Marketaux.
"""
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
import pandas as pd
from asyncio_throttle import Throttler

from src.data.providers.base_provider import BaseDataProvider
from src.data.symbol_validator import SymbolValidator


class MarketauxProvider(BaseDataProvider):
    """
    A data provider that fetches news data from Marketaux.

    This class implements the BaseDataProvider interface to retrieve news data
    using the Marketaux API. An API key is required.
    It includes rate limiting to adhere to API usage policies (100 requests/day for free tier).
    """

    def __init__(self, api_key: str, rate_limit: int = 100, period: float = 86400.0):
        """
        Initialize the Marketaux provider.

        Args:
            api_key: Marketaux API key
            rate_limit: Number of requests allowed per period (default: 100 for free tier)
            period: Time period in seconds for rate limit (default: 86400 = 1 day)
        """
        if not api_key:
            raise ValueError("Marketaux API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1/news/all"
        self.throttler = Throttler(rate_limit, period)
        self.symbol_validator = SymbolValidator()

    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical market data from Marketaux.

        Note: Marketaux is primarily a news API, so this is a basic implementation
        that returns minimal OHLCV data. For comprehensive historical data,
        consider using another provider like Alpha Vantage or YFinance.
        """
        # Marketaux doesn't provide historical OHLCV data, so we return None
        # This is a placeholder implementation to satisfy the interface
        print(f"Marketaux does not support historical market data for {symbol}")
        return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price from Marketaux.

        Note: Marketaux is primarily a news API, so this is a basic implementation
        that attempts to extract price information from news articles.
        For accurate current prices, consider using another provider.
        """
        # Marketaux doesn't provide direct price data, so we return None
        # This is a placeholder implementation to satisfy the interface
        print(f"Marketaux does not support direct current price data for {symbol}")
        return None

    async def fetch_news_sentiment(self, symbol: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches news and sentiment data from Marketaux.

        Args:
            symbol: The stock ticker to fetch news for.
            limit: The maximum number of news articles to return.

        Returns:
            A list of news articles, each as a dictionary with the same format as Alpha Vantage.
        """
        params = {
            "api_token": self.api_key,
            "symbols": symbol,
            "limit": str(limit),
        }

        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if "data" in data and data["data"]:
                # Transform Marketaux format to match Alpha Vantage format for compatibility
                transformed_articles = []
                for article in data["data"]:
                    transformed_article = {
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "time_published": article.get("published_at", ""),
                        "authors": [article.get("source", "")],
                        "summary": article.get("description", ""),
                        "banner_image": article.get("image_url", ""),
                        "source": article.get("source", ""),
                        "category_within_source": article.get("category", ""),
                        "source_domain": article.get("source", ""),
                        "topics": [],
                        "overall_sentiment_score": self._convert_sentiment_score(article.get("sentiment", "")),
                        "overall_sentiment_label": self._convert_sentiment_label(article.get("sentiment", "")),
                        "ticker_sentiment": [{
                            "ticker": symbol,
                            "relevance_score": 1.0,
                            "ticker_sentiment_score": self._convert_sentiment_score(article.get("sentiment", "")),
                            "ticker_sentiment_label": self._convert_sentiment_label(article.get("sentiment", ""))
                        }]
                    }
                    transformed_articles.append(transformed_article)

                return transformed_articles
            else:
                # Handle error cases
                if "error" in data:
                    error_message = data.get("error", {}).get("message", "")
                    if error_message:
                        if "Invalid API token" in error_message:
                            print(f"Marketaux API error: Invalid API token")
                        elif "rate limit" in error_message.lower():
                            print(f"Marketaux rate limit reached. Please wait and try again.")
                        else:
                            print(f"Marketaux API error: {error_message}")
                    else:
                        print(f"Marketaux API error: Unknown error occurred")
                else:
                    # Empty response - validate symbol
                    is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                    if validation_error:
                        print(f"Marketaux news error for {symbol}: {validation_error.message}")
                    else:
                        print(f"Marketaux: No news data available for '{symbol}'")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching news for {symbol} from Marketaux: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching news for {symbol}: {e}")
            return None

    def _convert_sentiment_score(self, sentiment: str) -> float:
        """
        Convert Marketaux sentiment string to numeric score.

        Args:
            sentiment: Sentiment string from Marketaux (e.g., "positive", "negative", "neutral")

        Returns:
            Numeric sentiment score compatible with Alpha Vantage format
        """
        if not sentiment:
            return 0.0

        sentiment_lower = sentiment.lower()
        if sentiment_lower == "positive":
            return 0.5
        elif sentiment_lower == "negative":
            return -0.5
        elif sentiment_lower == "neutral":
            return 0.0
        else:
            # For any other sentiment, default to neutral
            return 0.0

    def _convert_sentiment_label(self, sentiment: str) -> str:
        """
        Convert Marketaux sentiment string to Alpha Vantage compatible label.

        Args:
            sentiment: Sentiment string from Marketaux

        Returns:
            Sentiment label compatible with Alpha Vantage format
        """
        if not sentiment:
            return "Neutral"

        sentiment_lower = sentiment.lower()
        if sentiment_lower == "positive":
            return "Positive"
        elif sentiment_lower == "negative":
            return "Negative"
        elif sentiment_lower == "neutral":
            return "Neutral"
        else:
            # For any other sentiment, default to neutral
            return "Neutral"
