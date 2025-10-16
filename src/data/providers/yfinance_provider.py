"""
Data provider implementation for fetching market data from Yahoo Finance.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import yfinance as yf
from asyncio_throttle import Throttler

from src.data.providers.base_provider import BaseDataProvider
from src.data.symbol_validator import SymbolValidator

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """
    A data provider that fetches market data from Yahoo Finance.
    """

    def __init__(self, rate_limit: int = 10, period: float = 60.0):
        self.throttler = Throttler(rate_limit, period)
        self.symbol_validator = SymbolValidator()

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
                    # Use symbol validator to provide better error message
                    is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol)
                    if validation_error:
                        logger.error(f"Symbol validation error for {symbol}: {validation_error.message}")
                    else:
                        logger.warning(f"No data found for {symbol} from yfinance.")
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
                error_str = str(e)
                wait_time = 2 ** attempt  # Exponential backoff
                
                # Try to extract meaningful error information
                if "yftzmissingerror" in error_str.lower():
                    if "possibly delisted" in error_str.lower():
                        logger.warning(f"Symbol '{symbol}' appears to be delisted or inactive")
                    elif "no timezone found" in error_str.lower():
                        # Try to get a suggestion
                        _, validation_error = await self.symbol_validator.validate_symbol(symbol)
                        if validation_error and validation_error.suggestion:
                            logger.error(f"Invalid symbol '{symbol}'. Did you mean '{validation_error.suggestion}'?")
                        else:
                            logger.error(f"Invalid symbol '{symbol}' - no timezone or exchange data found")
                    else:
                        logger.error(f"Symbol '{symbol}' not found or appears to be invalid")
                elif "http error 404" in error_str.lower():
                    logger.error(f"Symbol '{symbol}' not found (404 error) - check symbol spelling")
                else:
                    logger.error(f"Error fetching data for {symbol} from yfinance (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
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

            # Try to provide better error message for invalid symbol
            is_valid, validation_error = await self.symbol_validator.validate_symbol(symbol) 
            if validation_error:
                logger.error(f"Current price error for {symbol}: {validation_error.message}")
            else:
                logger.warning(f"Could not retrieve current price for {symbol} from yfinance.")
            return None
        except Exception as e:
            error_str = str(e)
            if "yftzmissingerror" in error_str.lower():
                if "possibly delisted" in error_str.lower():
                    logger.warning(f"Current price error: Symbol '{symbol}' appears to be delisted or inactive")
                elif "no timezone found" in error_str.lower():
                    logger.error(f"Current price error: Symbol '{symbol}' is invalid or not supported")
                else:
                    logger.error(f"Current price error: Symbol '{symbol}' not found or invalid")
            elif "http error 404" in error_str.lower():
                logger.error(f"Current price error: Symbol '{symbol}' not found (404 error)")
            else:
                logger.error(f"Error fetching current price for {symbol} from yfinance: {e}")
            return None
