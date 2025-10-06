"""
Unit tests for trend indicators implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.data.pipeline import DataPipeline
from src.data.providers.base_provider import BaseDataProvider
from src.agents.data_structures import MarketData
from src.data.indicators_metadata import get_all_trend_indicators


class TestTrendIndicators:
    """Test cases for trend indicators calculation and validation."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock data provider."""
        provider = MagicMock(spec=BaseDataProvider)
        return provider

    @pytest.fixture
    def data_pipeline(self, mock_provider):
        """Create a data pipeline with mock provider."""
        return DataPipeline(provider=mock_provider)

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        # Generate 250 days of sample data (enough for 200-period EMA)
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=250, freq='D')

        # Create a trending price series
        np.random.seed(42)  # For reproducible results
        base_price = 100
        trend = np.linspace(0, 50, 250)  # Upward trend
        noise = np.random.normal(0, 2, 250)  # Random noise
        close_prices = base_price + trend + noise

        # Create OHLCV data
        data = {
            'Open': close_prices + np.random.normal(0, 0.5, 250),
            'High': close_prices + np.abs(np.random.normal(0, 1, 250)),
            'Low': close_prices - np.abs(np.random.normal(0, 1, 250)),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 250)
        }

        # Ensure High >= Close >= Low and Open is reasonable
        for i in range(250):
            high = max(data['Open'][i], data['Close'][i]) + abs(np.random.normal(0, 0.5))
            low = min(data['Open'][i], data['Close'][i]) - abs(np.random.normal(0, 0.5))
            data['High'][i] = high
            data['Low'][i] = low

        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.mark.asyncio
    async def test_ema_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test EMA calculation with different periods."""
        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=sample_ohlcv_data)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 10, 1)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None
        assert isinstance(result, MarketData)

        # Check that trend indicators are populated
        assert hasattr(result, 'trend_indicators')
        assert 'EMA_5' in result.trend_indicators
        assert 'EMA_10' in result.trend_indicators
        assert 'EMA_20' in result.trend_indicators
        assert 'EMA_50' in result.trend_indicators
        assert 'EMA_200' in result.trend_indicators

        # Verify EMA values are positive and logical
        assert result.trend_indicators['EMA_5'] > 0
        assert result.trend_indicators['EMA_10'] > 0
        assert result.trend_indicators['EMA_20'] > 0
        assert result.trend_indicators['EMA_50'] > 0
        assert result.trend_indicators['EMA_200'] > 0

        # Check EMA hierarchy (shorter period EMA should be more responsive to recent prices)
        # In an uptrend, shorter EMAs should be above longer EMAs
        assert result.trend_indicators['EMA_5'] >= result.trend_indicators['EMA_10']
        assert result.trend_indicators['EMA_10'] >= result.trend_indicators['EMA_20']
        assert result.trend_indicators['EMA_20'] >= result.trend_indicators['EMA_50']
        assert result.trend_indicators['EMA_50'] >= result.trend_indicators['EMA_200']

    @pytest.mark.asyncio
    async def test_ema_with_insufficient_data(self, data_pipeline, mock_provider):
        """Test EMA calculation with insufficient data."""
        # Create minimal data (only 10 days)
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=10, freq='D')
        data = {
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'Volume': [1000000] * 10
        }
        df = pd.DataFrame(data, index=dates)

        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=df)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None

        # Check that only available EMAs are calculated
        assert 'EMA_5' in result.trend_indicators
        assert 'EMA_10' in result.trend_indicators
        assert 'EMA_20' not in result.trend_indicators
        assert 'EMA_50' not in result.trend_indicators
        assert 'EMA_200' not in result.trend_indicators

    @pytest.mark.asyncio
    async def test_adx_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test ADX calculation."""
        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=sample_ohlcv_data)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 10, 1)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None

        # Check that ADX indicators are populated
        assert 'ADX' in result.trend_indicators
        assert 'DI_PLUS' in result.trend_indicators
        assert 'DI_MINUS' in result.trend_indicators

        # Verify ADX values are in expected range [0, 100]
        assert 0 <= result.trend_indicators['ADX'] <= 100
        assert 0 <= result.trend_indicators['DI_PLUS'] <= 100
        assert 0 <= result.trend_indicators['DI_MINUS'] <= 100

    @pytest.mark.asyncio
    async def test_aroon_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test Aroon calculation."""
        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=sample_ohlcv_data)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 10, 1)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None

        # Check that Aroon indicators are populated
        assert 'AROON_UP' in result.trend_indicators
        assert 'AROON_DOWN' in result.trend_indicators
        assert 'AROON_OSC' in result.trend_indicators

        # Verify Aroon values are in expected range
        assert 0 <= result.trend_indicators['AROON_UP'] <= 100
        assert 0 <= result.trend_indicators['AROON_DOWN'] <= 100
        assert -100 <= result.trend_indicators['AROON_OSC'] <= 100

        # Verify Aroon Oscillator is the difference between Aroon Up and Aroon Down
        expected_osc = result.trend_indicators['AROON_UP'] - result.trend_indicators['AROON_DOWN']
        assert abs(result.trend_indicators['AROON_OSC'] - expected_osc) < 0.1

    @pytest.mark.asyncio
    async def test_psar_calculation(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test Parabolic SAR calculation."""
        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=sample_ohlcv_data)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 10, 1)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None

        # Check that PSAR indicator is populated
        assert 'PSAR' in result.trend_indicators

        # Verify PSAR value is positive
        assert result.trend_indicators['PSAR'] > 0

    @pytest.mark.asyncio
    async def test_trend_indicators_validation(self, data_pipeline, mock_provider):
        """Test trend indicators validation with extreme values."""
        # Create data with extreme values that might cause calculation errors
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=50, freq='D')
        data = {
            'Open': [100] * 50,
            'High': [100] * 50,
            'Low': [100] * 50,
            'Close': [100] * 50,  # Flat price
            'Volume': [1000000] * 50
        }
        df = pd.DataFrame(data, index=dates)

        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=df)

        # Fetch and process data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 20)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

        # Verify result is not None
        assert result is not None

        # Check that trend indicators are still calculated despite flat prices
        assert 'EMA_5' in result.trend_indicators
        assert 'ADX' in result.trend_indicators
        assert 'AROON_UP' in result.trend_indicators
        assert 'PSAR' in result.trend_indicators

    @pytest.mark.asyncio
    async def test_historical_trend_indicators(self, data_pipeline, mock_provider, sample_ohlcv_data):
        """Test historical trend indicators collection."""
        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=sample_ohlcv_data)

        # Fetch and process data with historical periods
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 10, 1)
        result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date, historical_periods=5)

        # Verify result is not None
        assert result is not None

        # Check that historical trend indicators are populated
        assert hasattr(result, 'historical_trend')
        assert len(result.historical_trend) == 5

        # Check that each historical period has trend indicators
        for period in result.historical_trend:
            assert 'EMA_5' in period
            assert 'EMA_10' in period
            assert 'EMA_20' in period
            assert 'EMA_50' in period
            assert 'EMA_200' in period
            assert 'ADX' in period
            assert 'DI_PLUS' in period
            assert 'DI_MINUS' in period
            assert 'AROON_UP' in period
            assert 'AROON_DOWN' in period
            assert 'AROON_OSC' in period
            assert 'PSAR' in period

    def test_trend_indicators_metadata(self):
        """Test that trend indicators metadata is properly defined."""
        trend_indicators = get_all_trend_indicators()

        # Check that all expected trend indicators have metadata
        expected_indicators = [
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
            'ADX', 'DI_PLUS', 'DI_MINUS',
            'AROON_UP', 'AROON_DOWN', 'AROON_OSC',
            'PSAR'
        ]

        for indicator in expected_indicators:
            assert indicator in trend_indicators
            metadata = trend_indicators[indicator]

            # Check required metadata fields
            assert 'name' in metadata
            assert 'category' in metadata
            assert 'reliability' in metadata
            assert 'description' in metadata
            assert 'parameters' in metadata
            assert 'signal_thresholds' in metadata
            assert 'typical_usage' in metadata
            assert 'data_requirements' in metadata

            # Check that category is 'trend'
            assert metadata['category'].value == 'trend'

    @pytest.mark.asyncio
    async def test_error_handling_in_trend_indicators(self, data_pipeline, mock_provider):
        """Test error handling in trend indicators calculation."""
        # Create data that might cause errors
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=20, freq='D')
        data = {
            'Open': [100] * 20,
            'High': [100] * 20,
            'Low': [100] * 20,
            'Close': [100] * 20,
            'Volume': [1000000] * 20
        }
        df = pd.DataFrame(data, index=dates)

        # Setup mock provider
        mock_provider.fetch_data = AsyncMock(return_value=df)

        # Mock pandas_ta functions to raise exceptions
        import pandas_ta as ta
        original_ema = ta.ema
        original_adx = ta.adx
        original_aroon = ta.aroon
        original_psar = ta.psar

        def mock_ema(*args, **kwargs):
            raise Exception("EMA calculation error")

        def mock_adx(*args, **kwargs):
            raise Exception("ADX calculation error")

        def mock_aroon(*args, **kwargs):
            raise Exception("Aroon calculation error")

        def mock_psar(*args, **kwargs):
            raise Exception("PSAR calculation error")

        # Apply mocks
        ta.ema = mock_ema
        ta.adx = mock_adx
        ta.aroon = mock_aroon
        ta.psar = mock_psar

        try:
            # Fetch and process data
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 20)
            result = await data_pipeline.fetch_and_process_data("TEST", start_date, end_date)

            # Verify result is not None despite errors
            assert result is not None

            # Check that trend indicators dict exists but might be empty
            assert hasattr(result, 'trend_indicators')

        finally:
            # Restore original functions
            ta.ema = original_ema
            ta.adx = original_adx
            ta.aroon = original_aroon
            ta.psar = original_psar
