"""
Abstract base class for all data providers in the AI Trading System.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from src.agents.data_structures import MarketData


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.

    This class defines the interface that all data provider implementations
    must adhere to. It ensures that the rest of the system can interact
    with any data source in a consistent manner.
    """

    @abstractmethod
    async def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical market data for a given symbol.

        Args:
            symbol: The stock ticker or symbol to fetch data for.
            start_date: The start date of the data range.
            end_date: The end date of the data range.

        Returns:
            A pandas DataFrame containing the OHLCV data, or None if fetching fails.
        """
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price for a given symbol.

        Args:
            symbol: The stock ticker or symbol to fetch the price for.

        Returns:
            The current price as a float, or None if fetching fails.
        """
        pass

    def to_market_data(self, data: pd.DataFrame, symbol: str) -> MarketData:
        """
        Converts a DataFrame into a MarketData object.
        This is a placeholder and will be more robustly implemented.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        latest = data.iloc[-1]
        return MarketData(
            symbol=symbol,
            price=latest["Close"],
            volume=latest["Volume"],
            timestamp=latest.name.to_pydatetime(),
            ohlc={"Open": latest["Open"], "High": latest["High"], "Low": latest["Low"], "Close": latest["Close"]},
            technical_indicators={},  # To be filled by the pipeline
            fundamental_data=None,  # To be filled by another provider or pipeline
        )
