"""
Unit tests for statistical indicators in the AI Trading System.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.data.pipeline import DataPipeline
from src.data.providers.base_provider import BaseDataProvider
from src.agents.data_structures import MarketData


class MockDataProvider(BaseDataProvider):
    """Mock data provider for testing."""

    def __init__(self, data_length=200):
        self.data_length = data_length

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Limit to data_length
        dates = dates[:self.data_length]

        # Generate price data with some trend and noise
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0

        # Create a trending series for Hurst Exponent testing
        trend = np.linspace(0, 20, len(dates))
        noise = np.random.normal(0, 1, len(dates))
        close_prices = base_price + trend + noise

        # Generate OHLC data
        high_prices = close_prices + np.random.uniform(0, 2, len(dates))
        low_prices = close_prices - np.random.uniform(0, 2, len(dates))
        open_prices = low_prices + np.random.uniform(0, high_prices - low_prices)

        # Generate volume data
        volumes = np.random.randint(1000000, 5000000, len(dates))

        # Create DataFrame
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

        return df

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Return a mock current price for testing."""
        return 100.0  # Fixed price for testing


class TestStatisticalIndicators(unittest.TestCase):
    """Test cases for statistical indicators."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockDataProvider()
        self.pipeline = DataPipeline(self.provider)

    async def test_hurst_exponent_calculation(self):
        """Test Hurst Exponent calculation."""
        # Test with trending data (should have H > 0.5)
        trending_data = pd.Series(np.linspace(100, 200, 200) + np.random.normal(0, 1, 200))
        hurst = self.pipeline._calculate_hurst_exponent(trending_data)

        self.assertIsNotNone(hurst)
        self.assertGreater(hurst, 0.5, "Trending data should have Hurst Exponent > 0.5")
        self.assertLessEqual(hurst, 1.0, "Hurst Exponent should be <= 1.0")

        # Test with mean-reverting data (should have H < 0.5)
        mean_reverting_data = pd.Series([100 + 10 * np.sin(i/10) + np.random.normal(0, 1) for i in range(200)])
        hurst_mr = self.pipeline._calculate_hurst_exponent(mean_reverting_data)

        self.assertIsNotNone(hurst_mr)
        self.assertLess(hurst_mr, 0.5, "Mean-reverting data should have Hurst Exponent < 0.5")
        self.assertGreaterEqual(hurst_mr, 0.0, "Hurst Exponent should be >= 0.0")

        # Test with insufficient data
        short_data = pd.Series([100, 101, 102, 103, 104])
        hurst_short = self.pipeline._calculate_hurst_exponent(short_data)
        self.assertIsNone(hurst_short, "Should return None for insufficient data")

    async def test_z_score_calculation(self):
        """Test Z-Score calculation."""
        # Create test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)

        # Fetch and process data
        market_data = await self.pipeline.fetch_and_process_data("TEST", start_date, end_date)

        self.assertIsNotNone(market_data)
        self.assertIn("Z_SCORE", market_data.statistical_indicators)

        z_score = market_data.statistical_indicators["Z_SCORE"]
        self.assertIsInstance(z_score, (float, np.floating))

        # Z-Score should be a reasonable value (not extremely high)
        self.assertLess(abs(z_score), 5.0, "Z-Score should not be extremely high")

    async def test_correlation_calculation(self):
        """Test Correlation calculation."""
        # Create test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)

        # Fetch and process data
        market_data = await self.pipeline.fetch_and_process_data("TEST", start_date, end_date)

        self.assertIsNotNone(market_data)
        self.assertIn("CORRELATION", market_data.statistical_indicators)

        correlation = market_data.statistical_indicators["CORRELATION"]
        self.assertIsInstance(correlation, (float, np.floating))

        # Correlation should be between -1 and 1
        self.assertGreaterEqual(correlation, -1.0, "Correlation should be >= -1.0")
        self.assertLessEqual(correlation, 1.0, "Correlation should be <= 1.0")

    async def test_statistical_indicators_integration(self):
        """Test integration of all statistical indicators in the pipeline."""
        # Create test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)  # Longer period for Hurst Exponent

        # Fetch and process data
        market_data = await self.pipeline.fetch_and_process_data("TEST", start_date, end_date)

        self.assertIsNotNone(market_data)
        self.assertIsInstance(market_data, MarketData)

        # Check that statistical indicators are present
        self.assertIsInstance(market_data.statistical_indicators, dict)
        self.assertGreater(len(market_data.statistical_indicators), 0)

        # Check that historical statistical indicators are present
        self.assertIsInstance(market_data.historical_statistical, list)
        self.assertGreater(len(market_data.historical_statistical), 0)

        # Check specific indicators
        if "HURST" in market_data.statistical_indicators:
            hurst = market_data.statistical_indicators["HURST"]
            self.assertGreaterEqual(hurst, 0.0, "Hurst Exponent should be >= 0.0")
            self.assertLessEqual(hurst, 1.0, "Hurst Exponent should be <= 1.0")

        if "Z_SCORE" in market_data.statistical_indicators:
            z_score = market_data.statistical_indicators["Z_SCORE"]
            self.assertIsInstance(z_score, (float, np.floating))

        if "CORRELATION" in market_data.statistical_indicators:
            correlation = market_data.statistical_indicators["CORRELATION"]
            self.assertGreaterEqual(correlation, -1.0, "Correlation should be >= -1.0")
            self.assertLessEqual(correlation, 1.0, "Correlation should be <= 1.0")

    async def test_insufficient_data_handling(self):
        """Test handling of insufficient data for statistical indicators."""
        # Create a mock provider with insufficient data
        short_provider = MockDataProvider(data_length=10)
        short_pipeline = DataPipeline(short_provider)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=20)

        # Fetch and process data
        market_data = await short_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        self.assertIsNotNone(market_data)

        # With insufficient data, some indicators might be missing
        # This is expected behavior, so we just check that the pipeline doesn't crash
        self.assertIsInstance(market_data.statistical_indicators, dict)

    def test_hurst_exponent_edge_cases(self):
        """Test Hurst Exponent calculation with edge cases."""
        # Test with constant data
        constant_data = pd.Series([100.0] * 200)
        hurst_constant = self.pipeline._calculate_hurst_exponent(constant_data)
        # Constant data might return None or a specific value
        self.assertTrue(hurst_constant is None or 0.0 <= hurst_constant <= 1.0)

        # Test with NaN values
        data_with_nan = pd.Series([100.0, np.nan, 102.0, 103.0] * 50)
        hurst_nan = self.pipeline._calculate_hurst_exponent(data_with_nan)
        # Should handle NaN values gracefully
        self.assertTrue(hurst_nan is None or 0.0 <= hurst_nan <= 1.0)

        # Test with empty data
        empty_data = pd.Series([])
        hurst_empty = self.pipeline._calculate_hurst_exponent(empty_data)
        self.assertIsNone(hurst_empty, "Should return None for empty data")


if __name__ == '__main__':
    unittest.main()
