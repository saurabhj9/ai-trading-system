"""
Batch provider implementation for Marketaux with optimized multi-symbol fetching.

This module provides batch operations for Marketaux news data provider,
leveraging Marketaux's support for multiple symbols in news requests to
achieve significant efficiency improvements.
"""
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any

from .batch_provider import BaseBatchProvider, BatchResponse
from ..symbol_validator import SymbolValidator


class MarketauxBatchProvider(BaseBatchProvider):
    """
    Batch provider for Marketaux with optimized multi-symbol operations.
    
    Marketaux primarily provides news data and supports multiple symbols in
    news requests, making it ideal for batch news operations.
    """
    
    def __init__(self, api_key: str, rate_limit: int = 100, period: float = 86400.0, max_batch_size: int = 10):
        """
        Initialize Marketaux batch provider.
        
        Args:
            api_key: Marketaux API key
            rate_limit: Rate limit for requests (100 per day for free tier)
            period: Time period for rate limiting (86400 seconds = 1 day)
            max_batch_size: Maximum number of symbols per batch request
        """
        super().__init__(None)  # We'll use Marketaux API directly
        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1/news/all"
        self.rate_limit = rate_limit
        self.period = period
        self.max_batch_size = max_batch_size
        self.symbol_validator = SymbolValidator()
        
        # Create session for reuse
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _rate_limit_request(self):
        """Apply rate limiting to requests."""
        await asyncio.sleep(self.period / self.rate_limit)
    
    def _split_symbols_batch(self, symbols: List[str]) -> List[List[str]]:
        """
        Split symbols into smaller batches to respect rate limits.
        
        Args:
            symbols: List of symbols to split
            
        Returns:
            List of symbol batches
        """
        batches = []
        for i in range(0, len(symbols), self.max_batch_size):
            batches.append(symbols[i:i + self.max_batch_size])
        return batches
    
    async def fetch_batch_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> BatchResponse:
        """
        Fetch historical data for multiple symbols in batch.
        
        Marketaux does not provide historical OHLCV data, so this will
        return empty data with appropriate error messages.
        
        Args:
            symbols: List of stock symbols to fetch data for
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            BatchResponse indicating no historical data available
        """
        all_errors = {}
        for symbol in symbols:
            all_errors[symbol] = "Marketaux does not provide historical OHLCV data"
        
        return BatchResponse(
            request_id=id(symbols),
            data={},
            errors=all_errors,
            metadata={
                "provider": "marketaux",
                "capability": "historical_data_not_supported"
            }
        )
    
    async def fetch_batch_current_prices(self, symbols: List[str]) -> BatchResponse:
        """
        Fetch current prices for multiple symbols in batch.
        
        Marketaux does not provide current price data, so this will
        return empty data with appropriate error messages.
        
        Args:
            symbols: List of stock symbols to fetch prices for
            
        Returns:
            BatchResponse indicating no price data available
        """
        all_errors = {}
        for symbol in symbols:
            all_errors[symbol] = "Marketaux does not provide current price data"
        
        return BatchResponse(
            request_id=id(symbols),
            data={},
            errors=all_errors,
            metadata={
                "provider": "marketaux",
                "capability": "current_price_not_supported"
            }
        )
    
    async def fetch_batch_news_sentiment(self, symbols: List[str], limit: int = 20) -> BatchResponse:
        """
        Fetch news sentiment data for multiple symbols in batch.
        
        Marketaux supports multiple symbols in news requests, which is very
        efficient for batch news operations.
        
        Args:
            symbols: List of stock symbols to fetch news for
            limit: Maximum number of news articles per symbol
            
        Returns:
            BatchResponse with news data for all symbols
        """
        all_data = {}
        all_errors = {}
        
        # Split into batches if too many symbols
        symbol_batches = self._split_symbols_batch(symbols)
        
        for batch_symbols in symbol_batches:
            await self._rate_limit_request()
            
            try:
                session = await self._get_session()
                
                # Create symbols parameter
                symbols_param = ",".join(batch_symbols)
                params = {
                    "api_token": self.api_key,
                    "symbols": symbols_param,
                    "limit": str(limit * len(batch_symbols)),  # Get more to allow filtering
                }
                
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                
                if "data" in data and data["data"]:
                    # Group news by symbol
                    symbol_news = {symbol: [] for symbol in batch_symbols}
                    
                    for article in data["data"]:
                        # Check which symbols this article relates to
                        article_symbols = article.get("symbols", "")
                        if isinstance(article_symbols, str):
                            article_symbols = [s.strip() for s in article_symbols.split(",")]
                        elif not isinstance(article_symbols, list):
                            article_symbols = []
                        
                        for symbol in batch_symbols:
                            if symbol.upper() in [s.upper() for s in article_symbols]:
                                # Transform to Alpha Vantage compatible format
                                transformed_article = {
                                    "title": article.get("title", ""),
                                    "url": article.get("url", ""),
                                    "time_published": article.get("published_at", ""),
                                    "authors": [article.get("source", "")],
                                    "summary": article.get("description", ""),
                                    "banner_image": article.get("image_url", ""),
                                    "source": article.get("source", ""),
                                    "category_within_source": article.get("category", "general"),
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
                                symbol_news[symbol].append(transformed_article)
                    
                    # Apply limit and add to results
                    for symbol in batch_symbols:
                        news_list = symbol_news[symbol][:limit]
                        if news_list:
                            all_data[symbol] = news_list
                        else:
                            all_errors[symbol] = f"No news data available for {symbol}"
                else:
                    # Handle error cases
                    if "error" in data:
                        error_message = data.get("error", {})
                        if error_message:
                            if "Invalid API token" in str(error_message):
                                for symbol in batch_symbols:
                                    all_errors[symbol] = "Invalid API token"
                            elif "rate limit" in str(error_message).lower():
                                for symbol in batch_symbols:
                                    all_errors[symbol] = "Rate limit reached"
                            else:
                                for symbol in batch_symbols:
                                    all_errors[symbol] = f"API error: {error_message}"
                        else:
                            for symbol in batch_symbols:
                                all_errors[symbol] = "Unknown API error"
                    else:
                        for symbol in batch_symbols:
                            all_errors[symbol] = f"No news data available for {symbol}"
                
            except Exception as e:
                error_msg = f"Batch news request error: {e}"
                for symbol in batch_symbols:
                    all_errors[symbol] = error_msg
        
        # Calculate efficiency metrics
        api_calls_saved = len(symbols) - len(symbol_batches)
        efficiency = self.calculate_batch_efficiency(len(symbols), api_calls_saved)
        
        return BatchResponse(
            request_id=id(symbols),
            data=all_data,
            errors=all_errors,
            metadata={
                "provider": "marketaux",
                "batch_count": len(symbol_batches),
                "symbols_per_batch": self.max_batch_size,
                "rate_limit": f"{self.rate_limit}/{self.period}s",
                **efficiency
            }
        )
    
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
    
    async def fetch_batch_company_profiles(self, symbols: List[str]) -> BatchResponse:
        """
        Fetch company profiles for multiple symbols in batch.
        
        Marketaux does not provide company profile data, so this will
        return empty data with appropriate error messages.
        
        Args:
            symbols: List of stock symbols to fetch profiles for
            
        Returns:
            BatchResponse indicating no profile data available
        """
        all_errors = {}
        for symbol in symbols:
            all_errors[symbol] = "Marketaux does not provide company profile data"
        
        return BatchResponse(
            request_id=id(symbols),
            data={},
            errors=all_errors,
            metadata={
                "provider": "marketaux",
                "capability": "company_profile_not_supported"
            }
        )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
