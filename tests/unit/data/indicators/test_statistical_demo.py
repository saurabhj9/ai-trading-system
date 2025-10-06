"""
Demo script to test the statistical indicators implementation.
"""

import asyncio
from datetime import datetime, timedelta
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider


async def test_statistical_indicators():
    """Test the statistical indicators with real data."""

    # Initialize the data pipeline with YFinance provider
    provider = YFinanceProvider()
    pipeline = DataPipeline(provider)

    # Set date range for testing (last 6 months to ensure enough data for Hurst Exponent)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    # Test symbol
    symbol = "AAPL"

    print(f"Testing statistical indicators for {symbol}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("-" * 50)

    try:
        # Fetch and process data
        market_data = await pipeline.fetch_and_process_data(symbol, start_date, end_date)

        if market_data is None:
            print(f"Failed to fetch data for {symbol}")
            return

        # Display basic information
        print(f"Symbol: {market_data.symbol}")
        print(f"Current Price: ${market_data.price:.2f}")
        print(f"Timestamp: {market_data.timestamp}")
        print()

        # Display statistical indicators
        print("Statistical Indicators:")
        print("-" * 30)

        if market_data.statistical_indicators:
            for indicator, value in market_data.statistical_indicators.items():
                if indicator == "HURST":
                    if value > 0.5:
                        trend_type = "Trending"
                    elif value < 0.5:
                        trend_type = "Mean-reverting"
                    else:
                        trend_type = "Random Walk"
                    print(f"{indicator}: {value:.4f} ({trend_type})")
                elif indicator == "Z_SCORE":
                    if abs(value) > 2:
                        signal = "Extreme"
                    elif abs(value) > 1:
                        signal = "Moderate"
                    else:
                        signal = "Normal"
                    print(f"{indicator}: {value:.4f} ({signal})")
                elif indicator == "CORRELATION":
                    if abs(value) > 0.7:
                        corr_type = "High"
                    elif abs(value) > 0.3:
                        corr_type = "Medium"
                    else:
                        corr_type = "Low"
                    print(f"{indicator}: {value:.4f} ({corr_type} correlation)")
                else:
                    print(f"{indicator}: {value:.4f}")
        else:
            print("No statistical indicators calculated")

        print()

        # Display historical statistical indicators (last 5 periods)
        print("Historical Statistical Indicators (Last 5 periods):")
        print("-" * 55)

        if market_data.historical_statistical:
            for i, indicators in enumerate(market_data.historical_statistical[-5:]):
                print(f"Period {i+1}:")
                for indicator, value in indicators.items():
                    print(f"  {indicator}: {value:.4f}")
                print()
        else:
            print("No historical statistical indicators available")

        print()
        print("Statistical indicators test completed successfully!")

    except Exception as e:
        print(f"Error testing statistical indicators: {e}")


if __name__ == "__main__":
    asyncio.run(test_statistical_indicators())
