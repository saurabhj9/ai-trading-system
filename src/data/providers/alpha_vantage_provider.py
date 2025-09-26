"""
Data provider implementation for fetching market data from Alpha Vantage.
"""
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
import pandas as pd
from asyncio_throttle import Throttler

from src.data.providers.base_provider import BaseDataProvider


class AlphaVantageProvider(BaseDataProvider):
    """
    A data provider that fetches market data from Alpha Vantage.

    This class implements the BaseDataProvider interface to retrieve data
    using the Alpha Vantage API. An API key is required.
    It includes rate limiting to adhere to API usage policies.
    """

    def __init__(self, api_key: str, rate_limit: int = 5, period: float = 60.0):
        if not api_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.throttler = Throttler(rate_limit, period)

    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical market data from Alpha Vantage.
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if "Time Series (Daily)" not in data:
                print(f"Error from Alpha Vantage API for {symbol}: {data.get('Note') or data}")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "4. close": "Close", "5. adjusted close": "Adj Close", "6. volume": "Volume"
            }, inplace=True)

            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])

            df = df[(df.index >= start_date) & (df.index <= end_date)]
            df.sort_index(ascending=True, inplace=True)

            return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching data for {symbol} from Alpha Vantage: {e}")
            return None
        except Exception as e:
            print(f"Error processing data for {symbol} from Alpha Vantage: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price from Alpha Vantage using GLOBAL_QUOTE.
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                return float(data["Global Quote"]["05. price"])
            else:
                print(f"Could not retrieve current price for {symbol} from Alpha Vantage: {data.get('Note') or data}")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching current price for {symbol} from Alpha Vantage: {e}")
            return None
        except (ValueError, KeyError) as e:
            print(f"Error parsing current price for {symbol} from Alpha Vantage: {e}")
            return None

    async def fetch_news_sentiment(self, symbol: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches news and sentiment data from Alpha Vantage.

        Args:
            symbol: The stock ticker to fetch news for.
            limit: The maximum number of news articles to return.

        Returns:
            A list of news articles, each as a dictionary.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_key,
            "limit": str(limit),
        }
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if "feed" in data:
                return data["feed"]
            else:
                print(f"Could not retrieve news for {symbol} from Alpha Vantage: {data.get('Note') or data}")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching news for {symbol} from Alpha Vantage: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching news for {symbol}: {e}")
            return None
