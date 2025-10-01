"""
Unit tests for volatility indicators implementation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.data.pipeline import DataPipeline
from src.data.providers.base_provider import BaseDataProvider
from src.agents.data_structures import MarketData
from src.data.indicators_metadata import get_indicator_metadata


class TestVolatilityIndicators:
    """Test cases for volatility indicators."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        np.random.seed(42)  # For reproducible results

        # Create realistic price data with some volatility
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 30)  # Daily returns with 2% volatility
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLC data
        data = {
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        }

        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def mock_provider(self):
        """Create a mock data provider."""
        provider = MagicMock(spec=BaseDataProvider)
        provider.fetch_data = AsyncMock()
        return provider

    @pytest.fixture
    def data_pipeline(self, mock_provider):
        """Create a data pipeline with mock provider."""
        return DataPipeline(provider=mock_provider, cache=None)

    @pytest.mark.asyncio
    async def test_atr_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test Average True Range (ATR) calculation."""
        # Setup mock provider to return sample data
        mock_provider.fetch_data.return_value = sample_ohlcv_data

        # Fetch and process data
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 30)
        )

        # Verify ATR is calculated
        assert market_data is not None
        assert "ATR" in market_data.volatility_indicators
        atr_value = market_data.volatility_indicators["ATR"]

        # ATR should be positive
        assert atr_value > 0

        # ATR should be reasonable (not extremely high)
        assert atr_value < 10  # Very high ATR would indicate calculation error

        # Verify historical ATR values
        assert len(market_data.historical_volatility) > 0
        assert "ATR" in market_data.historical_volatility[-1]

    @pytest.mark.asyncio
    async def test_historical_volatility_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test Historical Volatility calculation."""
        # Setup mock provider to return sample data
        mock_provider.fetch_data.return_value = sample_ohlcv_data

        # Fetch and process data
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 30)
        )

        # Verify Historical Volatility is calculated
        assert market_data is not None
        assert "HISTORICAL_VOLATILITY" in market_data.volatility_indicators
        hv_value = market_data.volatility_indicators["HISTORICAL_VOLATILITY"]

        # Historical Volatility should be positive
        assert hv_value > 0

        # Historical Volatility should be annualized and reasonable
        assert 0.05 < hv_value < 2.0  # Between 5% and 200% annualized volatility

        # Verify historical HV values
        assert len(market_data.historical_volatility) > 0
        assert "HISTORICAL_VOLATILITY" in market_data.historical_volatility[-1]

    @pytest.mark.asyncio
    async def test_chaikin_volatility_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test Chaikin Volatility calculation."""
        # Setup mock provider to return sample data
        mock_provider.fetch_data.return_value = sample_ohlcv_data

        # Fetch and process data
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 30)
        )

        # Verify Chaikin Volatility is calculated
        assert market_data is not None
        assert "CHAIKIN_VOLATILITY" in market_data.volatility_indicators
        cv_value = market_data.volatility_indicators["CHAIKIN_VOLATILITY"]

        # Chaikin Volatility can be positive or negative
        assert isinstance(cv_value, (int, float))
        assert not np.isnan(cv_value)

        # Verify historical CV values
        assert len(market_data.historical_volatility) > 0
        assert "CHAIKIN_VOLATILITY" in market_data.historical_volatility[-1]

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, data_pipeline, mock_provider):
        """Test handling of insufficient data for volatility indicators."""
        # Create minimal data (less than required for indicators)
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = {
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        }
        minimal_df = pd.DataFrame(data, index=dates)

        # Setup mock provider to return minimal data
        mock_provider.fetch_data.return_value = minimal_df

        # Fetch and process data
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10)
        )

        # Market data should still be created but without volatility indicators
        assert market_data is not None
        # Volatility indicators might be empty due to insufficient data
        # This is expected behavior

    @pytest.mark.asyncio
    async def test_error_handling_in_volatility_calculations(self, data_pipeline, mock_provider):
        """Test error handling in volatility indicator calculations."""
        # Create data that might cause calculation issues
        dates = pd.date_range(start='2023-01-01', periods=25, freq='D')

        # Create data with some zero values that might cause issues
        data = {
            'Open': [100] * 25,
            'High': [100] * 25,  # High equals Open (no range)
            'Low': [100] * 25,   # Low equals Open (no range)
            'Close': [100] * 25,
            'Volume': [1000000] * 25
        }
        problematic_df = pd.DataFrame(data, index=dates)

        # Setup mock provider to return problematic data
        mock_provider.fetch_data.return_value = problematic_df

        # Fetch and process data - should not raise exceptions
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 25)
        )

        # Market data should still be created
        assert market_data is not None

    def test_volatility_indicator_metadata(self):
        """Test that volatility indicators have proper metadata."""
        # Test ATR metadata
        atr_metadata = get_indicator_metadata("ATR")
        assert atr_metadata is not None
        assert atr_metadata["name"] == "Average True Range"
        assert atr_metadata["category"].value == "volatility"
        assert "period" in atr_metadata["parameters"]
        assert atr_metadata["parameters"]["period"] == 14

        # Test Historical Volatility metadata
        hv_metadata = get_indicator_metadata("HISTORICAL_VOLATILITY")
        assert hv_metadata is not None
        assert hv_metadata["name"] == "Historical Volatility"
        assert hv_metadata["category"].value == "volatility"
        assert "period" in hv_metadata["parameters"]
        assert hv_metadata["parameters"]["period"] == 20

        # Test Chaikin Volatility metadata
        cv_metadata = get_indicator_metadata("CHAIKIN_VOLATILITY")
        assert cv_metadata is not None
        assert cv_metadata["name"] == "Chaikin Volatility"
        assert cv_metadata["category"].value == "volatility"
        assert "ema_period" in cv_metadata["parameters"]
        assert "roc_period" in cv_metadata["parameters"]

    @pytest.mark.asyncio
    async def test_volatility_indicators_with_realistic_data(self, data_pipeline, mock_provider):
        """Test volatility indicators with more realistic market data."""
        # Create more realistic data with clear volatility patterns
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        np.random.seed(123)

        # Simulate a stock with changing volatility
        base_price = 100
        prices = [base_price]

        # First 20 days: low volatility
        for i in range(1, 20):
            ret = np.random.normal(0.0005, 0.005)  # 0.5% daily volatility
            prices.append(prices[-1] * (1 + ret))

        # Next 20 days: high volatility
        for i in range(20, 40):
            ret = np.random.normal(0.001, 0.03)  # 3% daily volatility
            prices.append(prices[-1] * (1 + ret))

        # Last 10 days: medium volatility
        for i in range(40, 50):
            ret = np.random.normal(0.0008, 0.015)  # 1.5% daily volatility
            prices.append(prices[-1] * (1 + ret))

        # Create OHLC data
        data = {
            'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }

        df = pd.DataFrame(data, index=dates)

        # Setup mock provider to return realistic data
        mock_provider.fetch_data.return_value = df

        # Fetch and process data
        market_data = await data_pipeline.fetch_and_process_data(
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 20)
        )

        # Verify all volatility indicators are calculated
        assert market_data is not None
        assert "ATR" in market_data.volatility_indicators
        assert "HISTORICAL_VOLATILITY" in market_data.volatility_indicators
        assert "CHAIKIN_VOLATILITY" in market_data.volatility_indicators

        # Verify historical data is populated
        assert len(market_data.historical_volatility) > 0

        # Check that ATR reflects the changing volatility
        atr_values = [item.get("ATR") for item in market_data.historical_volatility if "ATR" in item]
        if len(atr_values) > 10:
            # ATR should vary with changing volatility
            atr_std = np.std(atr_values)
            assert atr_std > 0  # There should be some variation in ATR
