"""
Data provider implementation for fetching market data from Finnhub.

Finnhub provides free access to real-time and historical market data,
financial statements, news sentiment, and more. Free tier supports
60 API calls per minute, which is 12x better than Alpha Vantage.
"""
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import aiohttp
import pandas as pd
from asyncio_throttle import Throttler

from src.data.providers.base_provider import BaseDataProvider
from src.data.symbol_validator import SymbolValidator


class FinnhubProvider(BaseDataProvider):
    """
    A data provider that fetches market data from Finnhub.

    This class implements the BaseDataProvider interface to retrieve data
    using the Finnhub API. An API key is required.
    It includes rate limiting to adhere to API usage policies (60 calls/minute).
    """

    def __init__(self, api_key: str, rate_limit: int = 60, period: float = 60.0):
        """
        Initialize the Finnhub provider.
        
        Args:
            api_key: Finnhub API key
            rate_limit: Number of requests allowed per period (default: 60 for free tier)
            period: Time period in seconds for rate limit (default: 60 = 1 minute)
        """
        if not api_key:
            raise ValueError("Finnhub API key is required.")
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.throttler = Throttler(rate_limit, period)
        self.symbol_validator = SymbolValidator()

    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical market data from Finnhub.
        
        Note: Finnhub candle data is limited to recent history (typically 1-2 years max).
        For longer historical data, consider using yfinance as primary provider.
        """
        # Convert dates to Unix timestamps (required by Finnhub API)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        params = {
            "symbol": symbol,
            "resolution": "D",  # Daily resolution
            "from": start_timestamp,
            "to": end_timestamp,
            "token": self.api_key,
        }
        
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/stock/candle", params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Check if API returned valid data
            if data.get("s") != "ok" or not data.get("c"):
                error_msg = data.get("error", "Unknown error")
                if "no data" in error_msg.lower() or "not found" in error_msg.lower():
                    # Try to provide better symbol error message
                    is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                    if validation_error:
                        print(f"Finnhub symbol error for {symbol}: {validation_error.message}")
                    else:
                        print(f"Finnhub: No historical data available for '{symbol}' (date range: {start_date.date()} to {end_date.date()})")
                else:
                    print(f"Finnhub API error for {symbol}: {error_msg}")
                return None

            # Convert Finnhub response to DataFrame
            # Finnhub returns: c=[close prices], h=[high prices], l=[low prices], 
            # o=[open prices], s=status, t=[timestamps], v=[volumes]
            df_data = {
                "Open": data["o"],
                "High": data["h"], 
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"]
            }
            
            df = pd.DataFrame(df_data)
            
            # Convert timestamps to datetime index
            df.index = pd.to_datetime(data["t"], unit='s')
            df.index.name = "Date"
            
            # Sort by date (ascending)
            df.sort_index(inplace=True)
            
            # Convert to numeric types
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            return df

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching data for {symbol} from Finnhub: {e}")
            return None
        except Exception as e:
            print(f"Error processing data for {symbol} from Finnhub: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price from Finnhub.
        """
        params = {
            "symbol": symbol,
            "token": self.api_key,
        }
        
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/quote", params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if "c" in data and data["c"] is not None:
                return float(data["c"])
            else:
                # Better error parsing for current price
                error_msg = data.get("error", "")
                if error_msg:
                    if "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                        is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                        if validation_error:
                            print(f"Finnhub current price error for {symbol}: {validation_error.message}")
                        else:
                            print(f"Finnhub current price error: Invalid symbol '{symbol}'")
                    else:
                        print(f"Finnhub current price error for {symbol}: {error_msg}")
                else:
                    # Empty response
                    is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                    if validation_error:
                        print(f"Finnhub current price error for {symbol}: {validation_error.message}")
                    else:
                        print(f"Finnhub: No current price data available for '{symbol}'")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching current price for {symbol} from Finnhub: {e}")
            return None
        except (ValueError, KeyError) as e:
            print(f"Error parsing current price for {symbol} from Finnhub: {e}")
            return None

    async def fetch_news_sentiment(self, symbol: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches news and sentiment data from Finnhub.

        Args:
            symbol: The stock ticker to fetch news for.
            limit: The maximum number of news articles to return.

        Returns:
            A list of news articles, each as a dictionary with the same format as Alpha Vantage.
        """
        params = {
            "category": "general",
            "id": symbol,
            "token": self.api_key,
            "minId": 0,  # Start from most recent
        }
        
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/news", params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if isinstance(data, list) and data:
                # Transform Finnhub format to match Alpha Vantage format for compatibility
                transformed_articles = []
                for article in data[:limit]:  # Limit to requested number
                    transformed_article = {
                        "title": article.get("headline", ""),
                        "url": article.get("url", ""),
                        "time_published": article.get("datetime", ""),
                        "authors": [article.get("source", "")],
                        "summary": article.get("summary", ""),
                        "banner_image": article.get("image", ""),
                        "source": article.get("source", ""),
                        "category_within_source": article.get("category", "general"),
                        "source_domain": article.get("source", ""),
                        "topics": [],
                        "overall_sentiment_score": 0.0,  # Finnhub doesn't provide sentiment scores
                        "overall_sentiment_label": "Neutral",
                        "ticker_sentiment": [{
                            "ticker": symbol,
                            "relevance_score": 1.0,
                            "ticker_sentiment_score": 0.0,  # Neutral by default
                            "ticker_sentiment_label": "Neutral"
                        }]
                    }
                    transformed_articles.append(transformed_article)
                
                return transformed_articles
            else:
                # Handle error cases
                if isinstance(data, dict) and "error" in data:
                    error_message = data.get("error", "")
                    if "Invalid" in error_message or "not found" in error_message.lower():
                        is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                        if validation_error:
                            print(f"Finnhub news error for {symbol}: {validation_error.message}")
                        else:
                            print(f"Finnhub news error: Invalid symbol '{symbol}'")
                    else:
                        print(f"Finnhub news error for {symbol}: {error_message}")
                else:
                    # Empty response
                    is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                    if validation_error:
                        print(f"Finnhub news error for {symbol}: {validation_error.message}")
                    else:
                        print(f"Finnhub: No news data available for '{symbol}'")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching news for {symbol} from Finnhub: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching news for {symbol}: {e}")
            return None

    async def fetch_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches company profile information from Finnhub.
        
        Args:
            symbol: The stock ticker to fetch profile for.
            
        Returns:
            Dictionary with company profile information or None if failed.
        """
        params = {
            "symbol": symbol,
            "token": self.api_key,
        }
        
        try:
            async with self.throttler, aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/stock/profile2", params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if data.get("name") and data.get("ticker"):
                return data
            else:
                # Handle error cases
                if isinstance(data, dict) and "error" in data:
                    error_message = data.get("error", "")
                    print(f"Finnhub profile error for {symbol}: {error_message}")
                else:
                    print(f"Finnhub: No profile data available for '{symbol}'")
                return None

        except aiohttp.ClientError as e:
            print(f"HTTP Error fetching profile for {symbol} from Finnhub: {e}")
            return None
        except Exception as e:
            print(f"Error processing profile for {symbol} from Finnhub: {e}")
            return None

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Fetches current prices for multiple symbols in parallel.
        
        Args:
            symbols: List of stock tickers to fetch prices for.
            
        Returns:
            Dictionary mapping symbols to their current prices (or None if failed).
        """
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return dict(zip(symbols, results))
