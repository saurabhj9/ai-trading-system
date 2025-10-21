"""
Composite data provider that implements fallback mechanism for news providers.
Tries Alpha Vantage first and falls back to Marketaux if Alpha Vantage fails.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.marketaux_provider import MarketauxProvider
from src.data.providers.base_provider import BaseDataProvider


class CompositeNewsProvider(BaseDataProvider):
    """
    A composite data provider that implements fallback logic for news sentiment.

    This provider tries Alpha Vantage first for all operations, and falls back to
    Marketaux if Alpha Vantage fails or returns no news. For historical data and
    current price, it only uses Alpha Vantage since Marketaux doesn't provide these.
    """

    def __init__(self, alpha_vantage_api_key: str, marketaux_api_key: str):
        """
        Initialize the composite provider with both API keys.

        Args:
            alpha_vantage_api_key: API key for Alpha Vantage
            marketaux_api_key: API key for Marketaux
        """
        self.alpha_vantage = AlphaVantageProvider(alpha_vantage_api_key)
        self.marketaux = MarketauxProvider(marketaux_api_key)
        self.logger = logging.getLogger(__name__)

    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical market data using Alpha Vantage only.

        Args:
            symbol: The stock ticker or symbol to fetch data for.
            start_date: The start date of the data range.
            end_date: The end date of the data range.

        Returns:
            A pandas DataFrame containing the OHLCV data, or None if fetching fails.
        """
        self.logger.info(f"Fetching historical data for {symbol} using Alpha Vantage")
        return await self.alpha_vantage.fetch_data(symbol, start_date, end_date)

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price using Alpha Vantage only.

        Args:
            symbol: The stock ticker or symbol to fetch the price for.

        Returns:
            The current price as a float, or None if fetching fails.
        """
        self.logger.info(f"Fetching current price for {symbol} using Alpha Vantage")
        return await self.alpha_vantage.get_current_price(symbol)

    async def fetch_news_sentiment(self, symbol: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches news and sentiment data with fallback logic.

        Tries Alpha Vantage first, and falls back to Marketaux if:
        - Alpha Vantage raises an exception
        - Alpha Vantage returns an empty list
        - Alpha Vantage returns None

        Args:
            symbol: The stock ticker to fetch news for.
            limit: The maximum number of news articles to return.

        Returns:
            A list of news articles, each as a dictionary, or None if both providers fail.
        """
        # Try Alpha Vantage first
        self.logger.info(f"Fetching news sentiment for {symbol} using Alpha Vantage")
        try:
            alpha_vantage_result = await self.alpha_vantage.fetch_news_sentiment(symbol, limit)

            # Check if Alpha Vantage returned valid data
            if alpha_vantage_result is not None and len(alpha_vantage_result) > 0:
                self.logger.info(f"Successfully fetched {len(alpha_vantage_result)} news articles from Alpha Vantage for {symbol}")
                return alpha_vantage_result
            else:
                self.logger.warning(f"Alpha Vantage returned no news for {symbol}, falling back to Marketaux")
        except Exception as e:
            self.logger.warning(f"Alpha Vantage failed for {symbol} with error: {str(e)}, falling back to Marketaux")

        # Fall back to Marketaux
        self.logger.info(f"Fetching news sentiment for {symbol} using Marketaux (fallback)")
        try:
            marketaux_result = await self.marketaux.fetch_news_sentiment(symbol, limit)

            # Check if Marketaux returned valid data
            if marketaux_result is not None and len(marketaux_result) > 0:
                self.logger.info(f"Successfully fetched {len(marketaux_result)} news articles from Marketaux for {symbol}")
                return marketaux_result
            else:
                self.logger.warning(f"Marketaux also returned no news for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Both Alpha Vantage and Marketaux failed for {symbol}. Marketaux error: {str(e)}")
            return None
