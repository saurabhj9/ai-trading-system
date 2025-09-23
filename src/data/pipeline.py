"""
Data pipeline for fetching, processing, and enriching market data.
"""
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_ta as ta

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

        """
        Fetches data, calculates indicators, and returns a MarketData object.
        Uses a cache to avoid redundant processing.
        """
        # Generate a unique cache key for the request
        cache_key = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"

        # 1. Check cache first
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

        # 2. If not in cache, fetch from provider
        ohlcv_df = await self.provider.fetch_data(symbol, start_date, end_date)
        if ohlcv_df is None or ohlcv_df.empty:
            print(f"DataPipeline: No data received for {symbol} from the provider.")
            return None

        # Ensure the index is a simple DatetimeIndex before calculating indicators
        if isinstance(ohlcv_df.index, pd.MultiIndex):
            ohlcv_df = ohlcv_df.reset_index()
            ohlcv_df = ohlcv_df.set_index("Date")

        # 3. Calculate technical indicators
        try:
            ohlcv_df.ta.rsi(append=True)
            ohlcv_df.ta.macd(append=True)
            ohlcv_df.rename(columns={
                "MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_hist", "MACDs_12_26_9": "MACD_signal"
            }, inplace=True)
        except Exception as e:
            print(f"DataPipeline: Error calculating technical indicators for {symbol}: {e}")
            pass  # Continue without indicators

        # 4. Extract latest data and assemble MarketData object
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

        # 5. Store the result in the cache before returning
        if self.cache:
            self.cache.set(cache_key, market_data, self.cache_ttl_seconds)

        return market_data
