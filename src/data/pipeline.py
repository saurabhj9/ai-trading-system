"""
Data pipeline for fetching, processing, and enriching market data.
"""
from datetime import datetime
from typing import Optional, List

import pandas as pd
import pandas_ta as ta
import asyncio

from src.agents.data_structures import MarketData
from src.data.cache import CacheManager
from src.data.providers.base_provider import BaseDataProvider


class DataPipeline:
    """
    Orchestrates fetching, processing, and caching market data.
    """

    def __init__(self, provider: BaseDataProvider, cache: Optional[CacheManager] = None, cache_ttl_seconds: int = 300):
        self.provider = provider
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

    async def fetch_and_process_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[MarketData]:
        """
        Fetches data, calculates indicators, and returns a MarketData object.
        """
        cache_key = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

        ohlcv_df = await self.provider.fetch_data(symbol, start_date, end_date)
        if ohlcv_df is None or ohlcv_df.empty:
            print(f"DataPipeline: No data received for {symbol} from the provider.")
            return None

        try:
            ohlcv_df.ta.rsi(append=True)
            ohlcv_df.ta.macd(append=True)
            ohlcv_df.rename(columns={
                "MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_hist", "MACDs_12_26_9": "MACD_signal"
            }, inplace=True)
        except Exception as e:
            print(f"DataPipeline: Error calculating technical indicators for {symbol}: {e}")
            pass

        latest_data = ohlcv_df.iloc[-1]
        technical_indicators = {}
        indicator_keys = ['RSI_14', 'MACD', 'MACD_hist', 'MACD_signal']
        for key in indicator_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                simple_key = key.replace('_14', '').upper()
                technical_indicators[simple_key] = latest_data[key]

        market_data = MarketData(
            symbol=symbol,
            price=latest_data["Close"],
            volume=latest_data["Volume"],
            timestamp=latest_data.name.to_pydatetime(),
            ohlc={
                "Open": latest_data["Open"], "High": latest_data["High"],
                "Low": latest_data["Low"], "Close": latest_data["Close"],
            },
            technical_indicators=technical_indicators,
        )

        if self.cache:
            self.cache.set(cache_key, market_data, self.cache_ttl_seconds)

        return market_data
    async def fetch_and_process_multiple_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[Optional[MarketData]]:
        """
        Fetches and processes data for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols.
            start_date: Start date for data.
            end_date: End date for data.

        Returns:
            List of MarketData objects, one per symbol (None if failed).
        """
        tasks = [self.fetch_and_process_data(symbol, start_date, end_date) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)
