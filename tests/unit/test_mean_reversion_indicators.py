"""
Unit tests for mean reversion indicators implementation.
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
from src.data.indicators_metadata import get_indicator_metadata, get_all_mean_reversion_indicators


class TestMeanReversionIndicators(unittest.TestCase):
    """Test cases for mean reversion indicators implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = YFinanceProvider()
        self.pipeline = DataPipeline(provider=self.provider)

    def test_indicator_metadata(self):
        """Test that indicator metadata is properly defined."""
        # Test Bollinger Bands metadata
        bb_upper_metadata = get_indicator_metadata('BB_UPPER')
        self.assertEqual(bb_upper_metadata['name'], 'Bollinger Bands Upper')
        self.assertEqual(bb_upper_metadata['parameters']['period'], 20)
        self.assertEqual(bb_upper_metadata['parameters']['std_dev'], 2)

        # Test Keltner Channels metadata
        kc_upper_metadata = get_indicator_metadata('KC_UPPER')
        self.assertEqual(kc_upper_metadata['name'], 'Keltner Channels Upper')
        self.assertEqual(kc_upper_metadata['parameters']['ema_period'], 20)
        self.assertEqual(kc_upper_metadata['parameters']['atr_period'], 10)
        self.assertEqual(kc_upper_metadata['parameters']['multiplier'], 2)

        # Test Donchian Channels metadata
        dc_upper_metadata = get_indicator_metadata('DC_UPPER')
        self.assertEqual(dc_upper_metadata['name'], 'Donchian Channels Upper')
        self.assertEqual(dc_upper_metadata['parameters']['period'], 20)

    def test_all_mean_reversion_indicators(self):
        """Test that all mean reversion indicators are defined."""
        all_indicators = get_all_mean_reversion_indicators()
        expected_indicators = [
            'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_WIDTH',
            'KC_UPPER', 'KC_MIDDLE', 'KC_LOWER',
            'DC_UPPER', 'DC_MIDDLE', 'DC_LOWER'
        ]

        for indicator in expected_indicators:
            self.assertIn(indicator, all_indicators)

    def test_mean_reversion_indicators_calculation(self):
        """Test that mean reversion indicators are calculated correctly."""
        async def run_test():
            # Use a well-known stock for testing
            symbol = "AAPL"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get enough data for indicators

            try:
                market_data = await self.pipeline.fetch_and_process_data(symbol, start_date, end_date)

                # Check that market data is not None
                self.assertIsNotNone(market_data)

                # Check that mean reversion indicators are present
                self.assertIn('mean_reversion_indicators', market_data.__dict__)
                mean_reversion_indicators = market_data.mean_reversion_indicators

                # Check that all expected indicators are calculated
                # Note: Some indicators might not be calculated if there's insufficient data
                if 'BB_UPPER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['BB_UPPER'], float)
                    self.assertGreater(mean_reversion_indicators['BB_UPPER'], 0)

                if 'BB_MIDDLE' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['BB_MIDDLE'], float)
                    self.assertGreater(mean_reversion_indicators['BB_MIDDLE'], 0)

                if 'BB_LOWER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['BB_LOWER'], float)
                    self.assertGreater(mean_reversion_indicators['BB_LOWER'], 0)

                if 'BB_WIDTH' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['BB_WIDTH'], float)
                    self.assertGreaterEqual(mean_reversion_indicators['BB_WIDTH'], 0)

                if 'KC_UPPER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['KC_UPPER'], float)
                    self.assertGreater(mean_reversion_indicators['KC_UPPER'], 0)

                if 'KC_MIDDLE' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['KC_MIDDLE'], float)
                    self.assertGreater(mean_reversion_indicators['KC_MIDDLE'], 0)

                if 'KC_LOWER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['KC_LOWER'], float)
                    self.assertGreater(mean_reversion_indicators['KC_LOWER'], 0)

                if 'DC_UPPER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['DC_UPPER'], float)
                    self.assertGreater(mean_reversion_indicators['DC_UPPER'], 0)

                if 'DC_MIDDLE' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['DC_MIDDLE'], float)
                    self.assertGreater(mean_reversion_indicators['DC_MIDDLE'], 0)

                if 'DC_LOWER' in mean_reversion_indicators:
                    self.assertIsInstance(mean_reversion_indicators['DC_LOWER'], float)
                    self.assertGreater(mean_reversion_indicators['DC_LOWER'], 0)

                # Check that historical mean reversion indicators are present
                self.assertIn('historical_mean_reversion', market_data.__dict__)
                self.assertIsInstance(market_data.historical_mean_reversion, list)

            except Exception as e:
                self.fail(f"Error testing mean reversion indicators calculation: {e}")

        # Run the async test
        asyncio.run(run_test())

    def test_mean_reversion_indicators_validation(self):
        """Test that mean reversion indicators are validated correctly."""
        # This test would require mocking the data pipeline to test edge cases
        # For now, we'll just test the metadata validation
        bb_upper_metadata = get_indicator_metadata('BB_UPPER')
        self.assertIn('signal_thresholds', bb_upper_metadata)
        self.assertIn('overbought', bb_upper_metadata['signal_thresholds'])

        bb_lower_metadata = get_indicator_metadata('BB_LOWER')
        self.assertIn('signal_thresholds', bb_lower_metadata)
        self.assertIn('oversold', bb_lower_metadata['signal_thresholds'])

    def test_mean_reversion_indicators_consistency(self):
        """Test that mean reversion indicators maintain logical consistency."""
        async def run_test():
            # Use a well-known stock for testing
            symbol = "MSFT"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get enough data for indicators

            try:
                market_data = await self.pipeline.fetch_and_process_data(symbol, start_date, end_date)

                # Check that market data is not None
                self.assertIsNotNone(market_data)

                mean_reversion_indicators = market_data.mean_reversion_indicators

                # Test Bollinger Bands consistency
                if all(ind in mean_reversion_indicators for ind in ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']):
                    self.assertGreater(mean_reversion_indicators['BB_UPPER'], mean_reversion_indicators['BB_MIDDLE'])
                    self.assertGreater(mean_reversion_indicators['BB_MIDDLE'], mean_reversion_indicators['BB_LOWER'])

                # Test Keltner Channels consistency
                if all(ind in mean_reversion_indicators for ind in ['KC_UPPER', 'KC_MIDDLE', 'KC_LOWER']):
                    self.assertGreater(mean_reversion_indicators['KC_UPPER'], mean_reversion_indicators['KC_MIDDLE'])
                    self.assertGreater(mean_reversion_indicators['KC_MIDDLE'], mean_reversion_indicators['KC_LOWER'])

                # Test Donchian Channels consistency
                if all(ind in mean_reversion_indicators for ind in ['DC_UPPER', 'DC_MIDDLE', 'DC_LOWER']):
                    self.assertGreater(mean_reversion_indicators['DC_UPPER'], mean_reversion_indicators['DC_MIDDLE'])
                    self.assertGreater(mean_reversion_indicators['DC_MIDDLE'], mean_reversion_indicators['DC_LOWER'])

            except Exception as e:
                self.fail(f"Error testing mean reversion indicators consistency: {e}")

        # Run the async test
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
