"""
Data provider implementation for fetching market data from Yahoo Finance.
"""
import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from asyncio_throttle import Throttler

from src.data.providers.base_provider import BaseDataProvider


class YFinanceProvider(BaseDataProvider):
    """
    A data provider that fetches market data from Yahoo Finance.
    """

    def __init__(self, rate_limit: int = 10, period: float = 60.0):
        self.throttler = Throttler(rate_limit, period)

    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and cleans historical market data from Yahoo Finance.
        Includes retry logic with exponential backoff for reliability.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.throttler:
                    data = await asyncio.to_thread(
                        yf.download, symbol, start=start_date, end=end_date, progress=False, auto_adjust=True
                    )
                if data.empty:
                    print(f"No data found for {symbol} from yfinance.")
                    return None

                # --- DataFrame Cleaning ---
                # Flatten the column index if it's a MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Ensure index is a DatetimeIndex
                data.index = pd.to_datetime(data.index)

                # Standardize column names
                data.rename(columns=str.capitalize, inplace=True)
                # --- End Cleaning ---

                return data
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Error fetching data for {symbol} from yfinance (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
                    return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the most recent market price for a given symbol.
        """
        try:
            ticker = yf.Ticker(symbol)
            async with self.throttler:
                hist = await asyncio.to_thread(ticker.history, period="2d")

            if hist is not None and not hist.empty:
                return hist['Close'].iloc[-1]

            async with self.throttler:
                info = await asyncio.to_thread(getattr, ticker, 'info')

            if info:
                price = info.get('regularMarketPrice') or info.get('currentPrice')
                if price:
                    return price

            print(f"Could not retrieve current price for {symbol} from yfinance.")
            return None
        except Exception as e:
            print(f"Error fetching current price for {symbol} from yfinance: {e}")
            return None
