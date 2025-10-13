"""
Test script to demonstrate volume indicators implementation.
"""

import asyncio
from datetime import datetime, timedelta
import sys
import os

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.indicators_metadata import get_all_volume_indicators


@pytest.mark.asyncio
async def test_volume_indicators():
    """Test volume indicators implementation with a sample stock."""

    # Set up data pipeline
    provider = YFinanceProvider()
    pipeline = DataPipeline(provider=provider)

    # Test with a liquid stock that should have good volume data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Get enough data for indicators

    print(f"Testing volume indicators for {symbol}")
    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print("-" * 50)

    try:
        # Fetch and process data
        market_data = await pipeline.fetch_and_process_data(symbol, start_date, end_date)

        if market_data is None:
            print(f"Failed to fetch data for {symbol}")
            return

        # Display volume indicators
        print("Volume Indicators:")
        if market_data.volume_indicators:
            for indicator, value in market_data.volume_indicators.items():
                print(f"  {indicator}: {value:.4f}")
        else:
            print("  No volume indicators calculated")

        print("\nVolume Indicator Metadata:")
        all_volume_indicators = get_all_volume_indicators()
        for indicator_name, metadata in all_volume_indicators.items():
            print(f"\n{indicator_name}:")
            print(f"  Name: {metadata['name']}")
            print(f"  Category: {metadata['category'].value}")
            print(f"  Reliability: {metadata['reliability'].value}")
            print(f"  Description: {metadata['description']}")
            print(f"  Data Requirements: {metadata['data_requirements']}")

            # Show signal thresholds if available
            if 'signal_thresholds' in metadata:
                print(f"  Signal Thresholds: {metadata['signal_thresholds']}")

            # Show typical usage
            if 'typical_usage' in metadata:
                print(f"  Typical Usage: {', '.join(metadata['typical_usage'])}")

        # Show historical volume indicators (last 3 periods)
        print("\nHistorical Volume Indicators (Last 3 periods):")
        if market_data.historical_volume:
            for i, volume_data in enumerate(market_data.historical_volume[-3:]):
                print(f"  Period {i+1}:")
                for indicator, value in volume_data.items():
                    print(f"    {indicator}: {value:.4f}")
        else:
            print("  No historical volume indicators available")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error testing volume indicators: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_volume_indicators())
