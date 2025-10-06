"""
Unit tests for volume indicators implementation.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.indicators_metadata import get_indicator_metadata, get_all_volume_indicators


class TestVolumeIndicators(unittest.TestCase):
    """Test cases for volume indicators implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = YFinanceProvider()
        self.pipeline = DataPipeline(provider=self.provider)

    def test_indicator_metadata(self):
        """Test that volume indicator metadata is properly defined."""
        # Test OBV metadata
        obv_metadata = get_indicator_metadata('OBV')
        self.assertEqual(obv_metadata['name'], 'On-Balance Volume')
        self.assertEqual(obv_metadata['category'].value, 'volume')
        self.assertEqual(obv_metadata['reliability'].value, 'high')

        # Test VWAP metadata
        vwap_metadata = get_indicator_metadata('VWAP')
        self.assertEqual(vwap_metadata['name'], 'Volume Weighted Average Price')
        self.assertEqual(vwap_metadata['category'].value, 'volume')
        self.assertEqual(vwap_metadata['reliability'].value, 'high')

        # Test MFI metadata
        mfi_metadata = get_indicator_metadata('MFI')
        self.assertEqual(mfi_metadata['name'], 'Money Flow Index')
        self.assertEqual(mfi_metadata['category'].value, 'volume')
        self.assertEqual(mfi_metadata['parameters']['period'], 14)
        self.assertEqual(mfi_metadata['signal_thresholds']['overbought'], 80)

        # Test ADL metadata
        adl_metadata = get_indicator_metadata('ADL')
        self.assertEqual(adl_metadata['name'], 'Accumulation/Distribution Line')
        self.assertEqual(adl_metadata['category'].value, 'volume')
        self.assertEqual(adl_metadata['reliability'].value, 'medium')

    def test_all_volume_indicators(self):
        """Test that all volume indicators are defined."""
        all_indicators = get_all_volume_indicators()
        expected_indicators = ['OBV', 'VWAP', 'MFI', 'ADL']

        for indicator in expected_indicators:
            self.assertIn(indicator, all_indicators)

    def test_volume_indicators_calculation(self):
        """Test that volume indicators are calculated correctly."""
        async def run_test():
            # Use a well-known stock for testing
            symbol = "AAPL"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get enough data for indicators

            try:
                market_data = await self.pipeline.fetch_and_process_data(symbol, start_date, end_date)

                # Check that market data is not None
                self.assertIsNotNone(market_data)

                # Check that volume indicators are present
                self.assertIn('volume_indicators', market_data.__dict__)
                volume_indicators = market_data.volume_indicators

                # Check that all expected indicators are calculated
                # Note: Some indicators might not be calculated if there's insufficient data
                if 'OBV' in volume_indicators:
                    self.assertIsInstance(volume_indicators['OBV'], float)

                if 'VWAP' in volume_indicators:
                    self.assertIsInstance(volume_indicators['VWAP'], float)
                    self.assertGreater(volume_indicators['VWAP'], 0)

                if 'MFI' in volume_indicators:
                    self.assertIsInstance(volume_indicators['MFI'], float)
                    self.assertGreaterEqual(volume_indicators['MFI'], 0)
                    self.assertLessEqual(volume_indicators['MFI'], 100)

                if 'ADL' in volume_indicators:
                    self.assertIsInstance(volume_indicators['ADL'], float)

                # Check that historical volume indicators are present
                self.assertIn('historical_volume', market_data.__dict__)
                self.assertIsInstance(market_data.historical_volume, list)

            except Exception as e:
                self.fail(f"Error testing volume indicators calculation: {e}")

        # Run the async test
        asyncio.run(run_test())

    def test_volume_indicators_validation(self):
        """Test that volume indicators are validated correctly."""
        # This test would require mocking the data pipeline to test edge cases
        # For now, we'll just test the metadata validation
        mfi_metadata = get_indicator_metadata('MFI')
        self.assertIn('signal_thresholds', mfi_metadata)
        self.assertIn('overbought', mfi_metadata['signal_thresholds'])
        self.assertIn('oversold', mfi_metadata['signal_thresholds'])

        vwap_metadata = get_indicator_metadata('VWAP')
        self.assertIn('signal_thresholds', vwap_metadata)
        self.assertIn('bullish', vwap_metadata['signal_thresholds'])
        self.assertIn('bearish', vwap_metadata['signal_thresholds'])

    def test_volume_indicators_data_requirements(self):
        """Test that volume indicators have proper data requirements defined."""
        obv_metadata = get_indicator_metadata('OBV')
        self.assertIn('data_requirements', obv_metadata)
        self.assertIn('Close', obv_metadata['data_requirements'])
        self.assertIn('Volume', obv_metadata['data_requirements'])

        vwap_metadata = get_indicator_metadata('VWAP')
        self.assertIn('data_requirements', vwap_metadata)
        self.assertEqual(vwap_metadata['data_requirements'], ['OHLCV'])

        mfi_metadata = get_indicator_metadata('MFI')
        self.assertIn('data_requirements', mfi_metadata)
        self.assertEqual(mfi_metadata['data_requirements'], ['OHLCV'])

        adl_metadata = get_indicator_metadata('ADL')
        self.assertIn('data_requirements', adl_metadata)
        self.assertEqual(adl_metadata['data_requirements'], ['OHLCV'])


if __name__ == '__main__':
    unittest.main()
