"""
Unit tests for momentum indicators implementation.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.indicators_metadata import get_indicator_metadata, get_all_momentum_indicators


class TestMomentumIndicators(unittest.TestCase):
    """Test cases for momentum indicators implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = YFinanceProvider()
        self.pipeline = DataPipeline(provider=self.provider)

    def test_indicator_metadata(self):
        """Test that indicator metadata is properly defined."""
        # Test STOCH_K metadata
        stoch_k_metadata = get_indicator_metadata('STOCH_K')
        self.assertEqual(stoch_k_metadata['name'], 'Stochastic Oscillator %K')
        self.assertEqual(stoch_k_metadata['parameters']['k_period'], 14)
        self.assertEqual(stoch_k_metadata['signal_thresholds']['overbought'], 80)

        # Test WILLR metadata
        willr_metadata = get_indicator_metadata('WILLR')
        self.assertEqual(willr_metadata['name'], 'Williams %R')
        self.assertEqual(willr_metadata['parameters']['period'], 14)
        self.assertEqual(willr_metadata['signal_thresholds']['overbought'], -20)

        # Test CCI metadata
        cci_metadata = get_indicator_metadata('CCI')
        self.assertEqual(cci_metadata['name'], 'Commodity Channel Index')
        self.assertEqual(cci_metadata['parameters']['period'], 20)
        self.assertEqual(cci_metadata['signal_thresholds']['overbought'], 100)

    def test_all_momentum_indicators(self):
        """Test that all momentum indicators are defined."""
        all_indicators = get_all_momentum_indicators()
        expected_indicators = ['STOCH_K', 'STOCH_D', 'WILLR', 'CCI']

        for indicator in expected_indicators:
            self.assertIn(indicator, all_indicators)

    def test_momentum_indicators_calculation(self):
        """Test that momentum indicators are calculated correctly."""
        async def run_test():
            # Use a well-known stock for testing
            symbol = "AAPL"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get enough data for indicators

            try:
                market_data = await self.pipeline.fetch_and_process_data(symbol, start_date, end_date)

                # Check that market data is not None
                self.assertIsNotNone(market_data)

                # Check that momentum indicators are present
                self.assertIn('momentum_indicators', market_data.__dict__)
                momentum_indicators = market_data.momentum_indicators

                # Check that all expected indicators are calculated
                # Note: Some indicators might not be calculated if there's insufficient data
                if 'STOCH_K' in momentum_indicators:
                    self.assertIsInstance(momentum_indicators['STOCH_K'], float)
                    self.assertGreaterEqual(momentum_indicators['STOCH_K'], 0)
                    self.assertLessEqual(momentum_indicators['STOCH_K'], 100)

                if 'STOCH_D' in momentum_indicators:
                    self.assertIsInstance(momentum_indicators['STOCH_D'], float)
                    self.assertGreaterEqual(momentum_indicators['STOCH_D'], 0)
                    self.assertLessEqual(momentum_indicators['STOCH_D'], 100)

                if 'WILLR' in momentum_indicators:
                    self.assertIsInstance(momentum_indicators['WILLR'], float)
                    self.assertGreaterEqual(momentum_indicators['WILLR'], -100)
                    self.assertLessEqual(momentum_indicators['WILLR'], 0)

                if 'CCI' in momentum_indicators:
                    self.assertIsInstance(momentum_indicators['CCI'], float)
                    # CCI can have very wide ranges, so we just check it's a number

                # Check that historical momentum indicators are present
                self.assertIn('historical_momentum', market_data.__dict__)
                self.assertIsInstance(market_data.historical_momentum, list)

            except Exception as e:
                self.fail(f"Error testing momentum indicators calculation: {e}")

        # Run the async test
        asyncio.run(run_test())

    def test_momentum_indicators_validation(self):
        """Test that momentum indicators are validated correctly."""
        # This test would require mocking the data pipeline to test edge cases
        # For now, we'll just test the metadata validation
        stoch_k_metadata = get_indicator_metadata('STOCH_K')
        self.assertIn('signal_thresholds', stoch_k_metadata)
        self.assertIn('overbought', stoch_k_metadata['signal_thresholds'])
        self.assertIn('oversold', stoch_k_metadata['signal_thresholds'])


if __name__ == '__main__':
    unittest.main()
