"""
Data pipeline for fetching, processing, and enriching market data.
"""
from datetime import datetime
from typing import Optional, List

import pandas as pd
import pandas_ta as ta
import numpy as np
import asyncio

from src.agents.data_structures import MarketData
from src.data.cache import CacheManager
from src.data.providers.base_provider import BaseDataProvider
from src.data.indicators_metadata import get_indicator_metadata


class DataPipeline:
    """
    Orchestrates fetching, processing, and caching market data.
    """

    def __init__(self, provider: BaseDataProvider, cache: Optional[CacheManager] = None, cache_ttl_seconds: int = 300):
        self.provider = provider
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

    async def fetch_and_process_data(
        self, symbol: str, start_date: datetime, end_date: datetime, historical_periods: int = 10
    ) -> Optional[MarketData]:
        """
        Fetches data, calculates indicators, and returns a MarketData object.

        Args:
            symbol: The stock symbol to fetch data for.
            start_date: Start date for historical data.
            end_date: End date for historical data.
            historical_periods: Number of historical periods to include in historical_indicators.

        Returns:
            MarketData object with current and historical indicators.
        """
        cache_key = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

        ohlcv_df = await self.provider.fetch_data(symbol, start_date, end_date)
        if ohlcv_df is None or ohlcv_df.empty:
            print(f"DataPipeline: No data received for {symbol} from the provider.")
            return None

        try:
            # Check if we have enough data for indicator calculations
            min_periods = 20  # Maximum period required for our indicators
            if len(ohlcv_df) < min_periods:
                print(f"DataPipeline: Insufficient data for {symbol}. Need at least {min_periods} periods, got {len(ohlcv_df)}")
                # Still calculate what we can with available data

            # Calculate existing technical indicators
            try:
                ohlcv_df.ta.rsi(append=True)
            except Exception as e:
                print(f"DataPipeline: Error calculating RSI for {symbol}: {e}")

            try:
                ohlcv_df.ta.macd(append=True)
                ohlcv_df.rename(columns={
                    "MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_hist", "MACDs_12_26_9": "MACD_signal"
                }, inplace=True)
            except Exception as e:
                print(f"DataPipeline: Error calculating MACD for {symbol}: {e}")

            # Calculate momentum indicators with individual error handling
            try:
                # Stochastic Oscillator with k_period=14, d_period=3, smoothing=3
                if len(ohlcv_df) >= 14:  # Minimum period for Stochastic
                    ohlcv_df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
                else:
                    print(f"DataPipeline: Insufficient data for Stochastic Oscillator for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Stochastic Oscillator for {symbol}: {e}")

            try:
                # Williams %R with period=14
                if len(ohlcv_df) >= 14:  # Minimum period for Williams %R
                    ohlcv_df.ta.willr(length=14, append=True)
                else:
                    print(f"DataPipeline: Insufficient data for Williams %R for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Williams %R for {symbol}: {e}")

            try:
                # Commodity Channel Index (CCI) with period=20
                if len(ohlcv_df) >= 20:  # Minimum period for CCI
                    ohlcv_df.ta.cci(length=20, append=True)
                else:
                    print(f"DataPipeline: Insufficient data for CCI for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating CCI for {symbol}: {e}")

            # Calculate mean reversion indicators with individual error handling
            try:
                # Bollinger Bands with period=20, std_dev=2
                if len(ohlcv_df) >= 20:  # Minimum period for Bollinger Bands
                    ohlcv_df.ta.bbands(length=20, std=2, append=True)
                    # Calculate Bollinger Bands Width
                    if all(col in ohlcv_df.columns for col in ['BBU_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBL_20_2.0_2.0']):
                        ohlcv_df['BBW_20_2.0_2.0'] = (ohlcv_df['BBU_20_2.0_2.0'] - ohlcv_df['BBL_20_2.0_2.0']) / ohlcv_df['BBM_20_2.0_2.0']
                else:
                    print(f"DataPipeline: Insufficient data for Bollinger Bands for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Bollinger Bands for {symbol}: {e}")

            try:
                # Keltner Channels with ema_period=20, atr_period=10, multiplier=2
                if len(ohlcv_df) >= 20:  # Minimum period for Keltner Channels
                    ohlcv_df.ta.kc(length=20, scalar=2, mamode='ema', append=True)
                else:
                    print(f"DataPipeline: Insufficient data for Keltner Channels for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Keltner Channels for {symbol}: {e}")

            try:
                # Donchian Channels with period=20 (custom implementation)
                if len(ohlcv_df) >= 20:  # Minimum period for Donchian Channels
                    period = 20
                    ohlcv_df[f'DC_UPPER_{period}'] = ohlcv_df['High'].rolling(window=period).max()
                    ohlcv_df[f'DC_LOWER_{period}'] = ohlcv_df['Low'].rolling(window=period).min()
                    ohlcv_df[f'DC_MIDDLE_{period}'] = (ohlcv_df[f'DC_UPPER_{period}'] + ohlcv_df[f'DC_LOWER_{period}']) / 2
                else:
                    print(f"DataPipeline: Insufficient data for Donchian Channels for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Donchian Channels for {symbol}: {e}")

            # Calculate volatility indicators with individual error handling
            try:
                # Average True Range (ATR) with period=14
                if len(ohlcv_df) >= 14:  # Minimum period for ATR
                    ohlcv_df.ta.atr(length=14, append=True)
                else:
                    print(f"DataPipeline: Insufficient data for ATR for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating ATR for {symbol}: {e}")

            try:
                # Historical Volatility with period=20 (custom implementation)
                if len(ohlcv_df) >= 20:  # Minimum period for Historical Volatility
                    period = 20
                    # Calculate log returns
                    ohlcv_df['log_return'] = np.log(ohlcv_df['Close'] / ohlcv_df['Close'].shift(1))
                    # Calculate rolling standard deviation of log returns
                    ohlcv_df['HV_20'] = ohlcv_df['log_return'].rolling(window=period).std() * np.sqrt(252)
                    # Clean up temporary column
                    ohlcv_df.drop('log_return', axis=1, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for Historical Volatility for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Historical Volatility for {symbol}: {e}")

            try:
                # Chaikin Volatility with ema_period=10, roc_period=10 (custom implementation)
                if len(ohlcv_df) >= 20:  # Need at least 20 periods for 10-period EMA and 10-period ROC
                    ema_period = 10
                    roc_period = 10
                    # Calculate High-Low spread
                    ohlcv_df['hl_spread'] = ohlcv_df['High'] - ohlcv_df['Low']
                    # Calculate EMA of the spread
                    ohlcv_df['hl_spread_ema'] = ohlcv_df['hl_spread'].ewm(span=ema_period).mean()
                    # Calculate rate of change of the EMA
                    ohlcv_df['CHAIKIN_VOL_10_10'] = ((ohlcv_df['hl_spread_ema'] - ohlcv_df['hl_spread_ema'].shift(roc_period)) /
                                                     ohlcv_df['hl_spread_ema'].shift(roc_period)) * 100
                    # Clean up temporary columns
                    ohlcv_df.drop(['hl_spread', 'hl_spread_ema'], axis=1, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for Chaikin Volatility for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Chaikin Volatility for {symbol}: {e}")

            # Calculate trend indicators with individual error handling
            try:
                # Exponential Moving Averages (EMAs) with multiple periods
                if len(ohlcv_df) >= 200:  # Need at least 200 periods for 200-period EMA
                    # Calculate EMAs for different periods
                    for period in [5, 10, 20, 50, 200]:
                        ohlcv_df.ta.ema(length=period, append=True)
                        # Rename columns to be more descriptive
                        ohlcv_df.rename(columns={f"EMA_{period}": f"EMA_{period}"}, inplace=True)
                elif len(ohlcv_df) >= 50:  # Can calculate up to 50-period EMA
                    for period in [5, 10, 20, 50]:
                        ohlcv_df.ta.ema(length=period, append=True)
                        ohlcv_df.rename(columns={f"EMA_{period}": f"EMA_{period}"}, inplace=True)
                elif len(ohlcv_df) >= 20:  # Can calculate up to 20-period EMA
                    for period in [5, 10, 20]:
                        ohlcv_df.ta.ema(length=period, append=True)
                        ohlcv_df.rename(columns={f"EMA_{period}": f"EMA_{period}"}, inplace=True)
                elif len(ohlcv_df) >= 10:  # Can calculate up to 10-period EMA
                    for period in [5, 10]:
                        ohlcv_df.ta.ema(length=period, append=True)
                        ohlcv_df.rename(columns={f"EMA_{period}": f"EMA_{period}"}, inplace=True)
                elif len(ohlcv_df) >= 5:  # Can calculate only 5-period EMA
                    ohlcv_df.ta.ema(length=5, append=True)
                    ohlcv_df.rename(columns={f"EMA_5": f"EMA_5"}, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for EMAs for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating EMAs for {symbol}: {e}")

            try:
                # Average Directional Index (ADX) with period=14
                if len(ohlcv_df) >= 14:  # Minimum period for ADX
                    ohlcv_df.ta.adx(length=14, append=True)
                    # Rename columns to be more descriptive
                    if 'ADX_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"ADX_14": "ADX"}, inplace=True)
                    if 'DMP_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"DMP_14": "DI_PLUS"}, inplace=True)
                    if 'DMN_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"DMN_14": "DI_MINUS"}, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for ADX for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating ADX for {symbol}: {e}")

            try:
                # Aroon Indicator with period=14
                if len(ohlcv_df) >= 14:  # Minimum period for Aroon
                    ohlcv_df.ta.aroon(length=14, append=True)
                    # Rename columns to be more descriptive
                    if 'AROONU_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"AROONU_14": "AROON_UP"}, inplace=True)
                    if 'AROOND_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"AROOND_14": "AROON_DOWN"}, inplace=True)
                    if 'AROONOSC_14' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"AROONOSC_14": "AROON_OSC"}, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for Aroon for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Aroon for {symbol}: {e}")

            try:
                # Parabolic SAR with acceleration=0.02, maximum=0.2
                if len(ohlcv_df) >= 5:  # Minimum period for PSAR
                    ohlcv_df.ta.psar(af=0.02, max_af=0.2, append=True)
                    # Rename columns to be more descriptive
                    if 'PSARl_0.02_0.2' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"PSARl_0.02_0.2": "PSAR_LONG"}, inplace=True)
                    if 'PSARs_0.02_0.2' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"PSARs_0.02_0.2": "PSAR_SHORT"}, inplace=True)

                    # Combine long and short PSAR into a single PSAR value
                    if 'PSAR_LONG' in ohlcv_df.columns and 'PSAR_SHORT' in ohlcv_df.columns:
                        # Use PSAR_LONG when it's not NaN, otherwise use PSAR_SHORT
                        ohlcv_df['PSAR'] = ohlcv_df['PSAR_LONG'].fillna(ohlcv_df['PSAR_SHORT'])
                        # Drop the individual long/short columns
                        ohlcv_df.drop(['PSAR_LONG', 'PSAR_SHORT'], axis=1, inplace=True)
                    elif 'PSAR_LONG' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"PSAR_LONG": "PSAR"}, inplace=True)
                    elif 'PSAR_SHORT' in ohlcv_df.columns:
                        ohlcv_df.rename(columns={"PSAR_SHORT": "PSAR"}, inplace=True)
                else:
                    print(f"DataPipeline: Insufficient data for Parabolic SAR for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Parabolic SAR for {symbol}: {e}")

            # Calculate volume indicators with individual error handling
            try:
                # On-Balance Volume (OBV)
                if 'Volume' in ohlcv_df.columns and not ohlcv_df['Volume'].isna().all():
                    ohlcv_df.ta.obv(append=True)
                else:
                    print(f"DataPipeline: Volume data missing or all NaN for OBV calculation for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating OBV for {symbol}: {e}")

            try:
                # Volume Weighted Average Price (VWAP) - custom implementation that resets daily
                if all(col in ohlcv_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    # Calculate typical price (H+L+C)/3
                    ohlcv_df['typical_price'] = (ohlcv_df['High'] + ohlcv_df['Low'] + ohlcv_df['Close']) / 3
                    # Calculate typical price * volume
                    ohlcv_df['pv'] = ohlcv_df['typical_price'] * ohlcv_df['Volume']

                    # Group by date to reset VWAP daily
                    ohlcv_df['date'] = ohlcv_df.index.date

                    # Calculate cumulative sum of PV and Volume within each day
                    ohlcv_df['cumulative_pv'] = ohlcv_df.groupby('date')['pv'].cumsum()
                    ohlcv_df['cumulative_volume'] = ohlcv_df.groupby('date')['Volume'].cumsum()

                    # Calculate VWAP
                    ohlcv_df['VWAP'] = ohlcv_df['cumulative_pv'] / ohlcv_df['cumulative_volume']

                    # Clean up temporary columns
                    ohlcv_df.drop(['typical_price', 'pv', 'date', 'cumulative_pv', 'cumulative_volume'], axis=1, inplace=True)
                else:
                    print(f"DataPipeline: Required OHLCV data missing for VWAP calculation for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating VWAP for {symbol}: {e}")

            try:
                # Money Flow Index (MFI) with period=14
                if len(ohlcv_df) >= 14 and all(col in ohlcv_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    ohlcv_df.ta.mfi(length=14, append=True)
                else:
                    print(f"DataPipeline: Insufficient data or missing OHLCV for MFI calculation for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating MFI for {symbol}: {e}")

            try:
                # Accumulation/Distribution Line
                if all(col in ohlcv_df.columns for col in ['High', 'Low', 'Close', 'Volume']):
                    ohlcv_df.ta.ad(append=True)
                else:
                    print(f"DataPipeline: Required OHLCV data missing for Accumulation/Distribution Line calculation for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Accumulation/Distribution Line for {symbol}: {e}")

            # Calculate statistical indicators with individual error handling
            try:
                # Hurst Exponent with custom implementation
                if len(ohlcv_df) >= 100:  # Need sufficient data for Hurst Exponent
                    min_window = 10
                    max_window = min(100, len(ohlcv_df) // 2)  # Ensure we have enough data points
                    num_windows = 5

                    # Calculate Hurst Exponent using rescaled range analysis
                    hurst_exponent = self._calculate_hurst_exponent(ohlcv_df['Close'], min_window, max_window, num_windows)
                    if hurst_exponent is not None:
                        ohlcv_df['HURST'] = hurst_exponent
                else:
                    print(f"DataPipeline: Insufficient data for Hurst Exponent for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Hurst Exponent for {symbol}: {e}")

            try:
                # Z-Score with period=20
                if len(ohlcv_df) >= 20:  # Minimum period for Z-Score
                    period = 20
                    # Calculate rolling mean and standard deviation
                    rolling_mean = ohlcv_df['Close'].rolling(window=period).mean()
                    rolling_std = ohlcv_df['Close'].rolling(window=period).std()
                    # Calculate Z-Score
                    ohlcv_df['Z_SCORE_20'] = (ohlcv_df['Close'] - rolling_mean) / rolling_std
                else:
                    print(f"DataPipeline: Insufficient data for Z-Score for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Z-Score for {symbol}: {e}")

            try:
                # Correlation analysis with period=20 (requires market index data)
                if len(ohlcv_df) >= 20:  # Minimum period for Correlation
                    period = 20
                    # For now, we'll calculate autocorrelation as a placeholder
                    # In a real implementation, this would fetch market index data (e.g., SPY)
                    # and calculate correlation between the symbol and the index
                    ohlcv_df['CORRELATION_20'] = ohlcv_df['Close'].rolling(window=period).apply(
                        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
                    )
                else:
                    print(f"DataPipeline: Insufficient data for Correlation analysis for {symbol}")
            except Exception as e:
                print(f"DataPipeline: Error calculating Correlation analysis for {symbol}: {e}")

        except Exception as e:
            print(f"DataPipeline: Critical error in indicator calculation for {symbol}: {e}")
            # Continue with available data even if indicators fail

        latest_data = ohlcv_df.iloc[-1]
        technical_indicators = {}
        indicator_keys = ['RSI_14', 'MACD', 'MACD_hist', 'MACD_signal']
        for key in indicator_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                simple_key = key.replace('_14', '').upper()
                technical_indicators[simple_key] = latest_data[key]

        # Extract momentum indicators with validation
        momentum_indicators = {}
        momentum_keys = ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'WILLR_14', 'CCI_20_0.015']

        for key in momentum_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                # Simplify the key names for easier access
                if key == 'STOCHk_14_3_3':
                    momentum_indicators['STOCH_K'] = latest_data[key]
                elif key == 'STOCHd_14_3_3':
                    momentum_indicators['STOCH_D'] = latest_data[key]
                elif key == 'WILLR_14':
                    momentum_indicators['WILLR'] = latest_data[key]
                elif key == 'CCI_20_0.015':
                    momentum_indicators['CCI'] = latest_data[key]

        # Validate momentum indicators against expected ranges
        for indicator, value in momentum_indicators.items():
            metadata = get_indicator_metadata(indicator)
            if metadata:
                # Check for extreme values that might indicate calculation errors
                if indicator == 'STOCH_K' or indicator == 'STOCH_D':
                    if not (0 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 100] for {symbol}")
                elif indicator == 'WILLR':
                    if not (-100 <= value <= 0):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [-100, 0] for {symbol}")
                elif indicator == 'CCI':
                    # CCI can have very wide ranges, but extremely large values might indicate issues
                    if abs(value) > 1000:
                        print(f"DataPipeline: Warning - {indicator} value {value} extremely high for {symbol}")

        # Extract mean reversion indicators with validation
        mean_reversion_indicators = {}
        mean_reversion_keys = [
            'BBU_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBL_20_2.0_2.0', 'BBW_20_2.0_2.0',
            'KCUe_20_2', 'KCBe_20_2', 'KCLe_20_2',
            'DC_UPPER_20', 'DC_MIDDLE_20', 'DC_LOWER_20'
        ]

        for key in mean_reversion_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                # Simplify the key names for easier access
                if key == 'BBU_20_2.0_2.0':
                    mean_reversion_indicators['BB_UPPER'] = latest_data[key]
                elif key == 'BBM_20_2.0_2.0':
                    mean_reversion_indicators['BB_MIDDLE'] = latest_data[key]
                elif key == 'BBL_20_2.0_2.0':
                    mean_reversion_indicators['BB_LOWER'] = latest_data[key]
                elif key == 'BBW_20_2.0_2.0':
                    mean_reversion_indicators['BB_WIDTH'] = latest_data[key]
                elif key == 'KCUe_20_2':
                    mean_reversion_indicators['KC_UPPER'] = latest_data[key]
                elif key == 'KCBe_20_2':
                    mean_reversion_indicators['KC_MIDDLE'] = latest_data[key]
                elif key == 'KCLe_20_2':
                    mean_reversion_indicators['KC_LOWER'] = latest_data[key]
                elif key == 'DC_UPPER_20':
                    mean_reversion_indicators['DC_UPPER'] = latest_data[key]
                elif key == 'DC_MIDDLE_20':
                    mean_reversion_indicators['DC_MIDDLE'] = latest_data[key]
                elif key == 'DC_LOWER_20':
                    mean_reversion_indicators['DC_LOWER'] = latest_data[key]

        # Validate mean reversion indicators for logical consistency
        if 'BB_UPPER' in mean_reversion_indicators and 'BB_LOWER' in mean_reversion_indicators:
            if mean_reversion_indicators['BB_UPPER'] <= mean_reversion_indicators['BB_LOWER']:
                print(f"DataPipeline: Warning - Bollinger Bands values inconsistent for {symbol}")

        if 'KC_UPPER' in mean_reversion_indicators and 'KC_LOWER' in mean_reversion_indicators:
            if mean_reversion_indicators['KC_UPPER'] <= mean_reversion_indicators['KC_LOWER']:
                print(f"DataPipeline: Warning - Keltner Channels values inconsistent for {symbol}")

        if 'DC_UPPER' in mean_reversion_indicators and 'DC_LOWER' in mean_reversion_indicators:
            if mean_reversion_indicators['DC_UPPER'] <= mean_reversion_indicators['DC_LOWER']:
                print(f"DataPipeline: Warning - Donchian Channels values inconsistent for {symbol}")

        # Extract volatility indicators with validation
        volatility_indicators = {}
        volatility_keys = ['ATRr_14', 'HV_20', 'CHAIKIN_VOL_10_10']

        for key in volatility_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                # Simplify the key names for easier access
                if key == 'ATRr_14':
                    volatility_indicators['ATR'] = latest_data[key]
                elif key == 'HV_20':
                    volatility_indicators['HISTORICAL_VOLATILITY'] = latest_data[key]
                elif key == 'CHAIKIN_VOL_10_10':
                    volatility_indicators['CHAIKIN_VOLATILITY'] = latest_data[key]

        # Validate volatility indicators for logical consistency
        for indicator, value in volatility_indicators.items():
            metadata = get_indicator_metadata(indicator)
            if metadata:
                # Check for extreme values that might indicate calculation errors
                if indicator == 'ATR':
                    if value < 0:
                        print(f"DataPipeline: Warning - {indicator} value {value} is negative for {symbol}")
                elif indicator == 'HISTORICAL_VOLATILITY':
                    if value < 0:
                        print(f"DataPipeline: Warning - {indicator} value {value} is negative for {symbol}")
                    elif value > 5:  # Extremely high volatility (>500%)
                        print(f"DataPipeline: Warning - {indicator} value {value} extremely high for {symbol}")

        # Extract trend indicators with validation
        trend_indicators = {}
        trend_keys = []

        # Add EMA keys based on available data length
        if len(ohlcv_df) >= 200:
            trend_keys.extend(['EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200'])
        elif len(ohlcv_df) >= 50:
            trend_keys.extend(['EMA_5', 'EMA_10', 'EMA_20', 'EMA_50'])
        elif len(ohlcv_df) >= 20:
            trend_keys.extend(['EMA_5', 'EMA_10', 'EMA_20'])
        elif len(ohlcv_df) >= 10:
            trend_keys.extend(['EMA_5', 'EMA_10'])
        elif len(ohlcv_df) >= 5:
            trend_keys.append('EMA_5')

        # Add ADX keys if available
        if len(ohlcv_df) >= 14:
            trend_keys.extend(['ADX', 'DI_PLUS', 'DI_MINUS'])

        # Add Aroon keys if available
        if len(ohlcv_df) >= 14:
            trend_keys.extend(['AROON_UP', 'AROON_DOWN', 'AROON_OSC'])

        # Add PSAR key if available
        if len(ohlcv_df) >= 5:
            trend_keys.append('PSAR')

        for key in trend_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                trend_indicators[key] = latest_data[key]

        # Validate trend indicators for logical consistency
        for indicator, value in trend_indicators.items():
            metadata = get_indicator_metadata(indicator)
            if metadata:
                # Check for extreme values that might indicate calculation errors
                if indicator.startswith('EMA_'):
                    if value <= 0:
                        print(f"DataPipeline: Warning - {indicator} value {value} is non-positive for {symbol}")
                elif indicator == 'ADX':
                    if not (0 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 100] for {symbol}")
                elif indicator in ['DI_PLUS', 'DI_MINUS']:
                    if not (0 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 100] for {symbol}")
                elif indicator in ['AROON_UP', 'AROON_DOWN']:
                    if not (0 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 100] for {symbol}")
                elif indicator == 'AROON_OSC':
                    if not (-100 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [-100, 100] for {symbol}")
                elif indicator == 'PSAR':
                    if value <= 0:
                        print(f"DataPipeline: Warning - {indicator} value {value} is non-positive for {symbol}")

        # Extract volume indicators with validation
        volume_indicators = {}
        volume_keys = ['OBV', 'VWAP', 'MFI_14', 'AD']

        for key in volume_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                # Simplify the key names for easier access
                if key == 'OBV':
                    volume_indicators['OBV'] = latest_data[key]
                elif key == 'VWAP':
                    volume_indicators['VWAP'] = latest_data[key]
                elif key == 'MFI_14':
                    volume_indicators['MFI'] = latest_data[key]
                elif key == 'AD':
                    volume_indicators['ADL'] = latest_data[key]

        # Validate volume indicators for logical consistency
        for indicator, value in volume_indicators.items():
            metadata = get_indicator_metadata(indicator)
            if metadata:
                # Check for extreme values that might indicate calculation errors
                if indicator == 'MFI':
                    if not (0 <= value <= 100):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 100] for {symbol}")
                elif indicator == 'VWAP':
                    if value <= 0:
                        print(f"DataPipeline: Warning - {indicator} value {value} is non-positive for {symbol}")

        # Extract statistical indicators with validation
        statistical_indicators = {}
        statistical_keys = ['HURST', 'Z_SCORE_20', 'CORRELATION_20']

        for key in statistical_keys:
            if key in latest_data and not pd.isna(latest_data[key]):
                # Simplify the key names for easier access
                if key == 'HURST':
                    statistical_indicators['HURST'] = latest_data[key]
                elif key == 'Z_SCORE_20':
                    statistical_indicators['Z_SCORE'] = latest_data[key]
                elif key == 'CORRELATION_20':
                    statistical_indicators['CORRELATION'] = latest_data[key]

        # Validate statistical indicators for logical consistency
        for indicator, value in statistical_indicators.items():
            metadata = get_indicator_metadata(indicator)
            if metadata:
                # Check for extreme values that might indicate calculation errors
                if indicator == 'HURST':
                    if not (0 <= value <= 1):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [0, 1] for {symbol}")
                elif indicator == 'Z_SCORE':
                    if abs(value) > 5:  # Extremely high Z-score (>5 standard deviations)
                        print(f"DataPipeline: Warning - {indicator} value {value} extremely high for {symbol}")
                elif indicator == 'CORRELATION':
                    if not (-1 <= value <= 1):
                        print(f"DataPipeline: Warning - {indicator} value {value} outside expected range [-1, 1] for {symbol}")

        # Collect historical indicators
        historical_data = ohlcv_df.tail(historical_periods)
        historical_indicators = []
        historical_momentum = []
        historical_mean_reversion = []
        historical_volatility = []
        historical_trend = []
        historical_volume = []
        historical_statistical = []
        historical_ohlc = []
        for idx, row in historical_data.iterrows():
            indicators = {}
            for key in indicator_keys:
                if key in row and not pd.isna(row[key]):
                    simple_key = key.replace('_14', '').upper()
                    indicators[simple_key] = row[key]
            historical_indicators.append(indicators)

            # Collect historical momentum indicators
            momentum = {}
            for key in momentum_keys:
                if key in row and not pd.isna(row[key]):
                    # Simplify the key names for easier access
                    if key == 'STOCHk_14_3_3':
                        momentum['STOCH_K'] = row[key]
                    elif key == 'STOCHd_14_3_3':
                        momentum['STOCH_D'] = row[key]
                    elif key == 'WILLR_14':
                        momentum['WILLR'] = row[key]
                    elif key == 'CCI_20_0.015':
                        momentum['CCI'] = row[key]
            historical_momentum.append(momentum)

            # Collect historical mean reversion indicators
            mean_reversion = {}
            for key in mean_reversion_keys:
                if key in row and not pd.isna(row[key]):
                    # Simplify the key names for easier access
                    if key == 'BBU_20_2.0_2.0':
                        mean_reversion['BB_UPPER'] = row[key]
                    elif key == 'BBM_20_2.0_2.0':
                        mean_reversion['BB_MIDDLE'] = row[key]
                    elif key == 'BBL_20_2.0_2.0':
                        mean_reversion['BB_LOWER'] = row[key]
                    elif key == 'BBW_20_2.0_2.0':
                        mean_reversion['BB_WIDTH'] = row[key]
                    elif key == 'KCUe_20_2':
                        mean_reversion['KC_UPPER'] = row[key]
                    elif key == 'KCBe_20_2':
                        mean_reversion['KC_MIDDLE'] = row[key]
                    elif key == 'KCLe_20_2':
                        mean_reversion['KC_LOWER'] = row[key]
                    elif key == 'DC_UPPER_20':
                        mean_reversion['DC_UPPER'] = row[key]
                    elif key == 'DC_MIDDLE_20':
                        mean_reversion['DC_MIDDLE'] = row[key]
                    elif key == 'DC_LOWER_20':
                        mean_reversion['DC_LOWER'] = row[key]
            historical_mean_reversion.append(mean_reversion)

            # Collect historical volatility indicators
            volatility = {}
            for key in volatility_keys:
                if key in row and not pd.isna(row[key]):
                    # Simplify the key names for easier access
                    if key == 'ATRr_14':
                        volatility['ATR'] = row[key]
                    elif key == 'HV_20':
                        volatility['HISTORICAL_VOLATILITY'] = row[key]
                    elif key == 'CHAIKIN_VOL_10_10':
                        volatility['CHAIKIN_VOLATILITY'] = row[key]
            historical_volatility.append(volatility)

            # Collect historical trend indicators
            trend = {}
            for key in trend_keys:
                if key in row and not pd.isna(row[key]):
                    trend[key] = row[key]
            historical_trend.append(trend)

            # Collect historical volume indicators
            volume = {}
            for key in volume_keys:
                if key in row and not pd.isna(row[key]):
                    # Simplify the key names for easier access
                    if key == 'OBV':
                        volume['OBV'] = row[key]
                    elif key == 'VWAP':
                        volume['VWAP'] = row[key]
                    elif key == 'MFI_14':
                        volume['MFI'] = row[key]
                    elif key == 'AD':
                        volume['ADL'] = row[key]
            historical_volume.append(volume)

            # Collect historical statistical indicators
            statistical = {}
            for key in statistical_keys:
                if key in row and not pd.isna(row[key]):
                    # Simplify the key names for easier access
                    if key == 'HURST':
                        statistical['HURST'] = row[key]
                    elif key == 'Z_SCORE_20':
                        statistical['Z_SCORE'] = row[key]
                    elif key == 'CORRELATION_20':
                        statistical['CORRELATION'] = row[key]
            historical_statistical.append(statistical)

            # Collect historical OHLCV data for LocalSignalGenerator
            ohlc_row = {
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            }
            historical_ohlc.append(ohlc_row)

        market_data = MarketData(
            symbol=symbol,
            price=latest_data["Close"],
            volume=latest_data["Volume"],
            timestamp=latest_data.name.to_pydatetime(),
            ohlc={
                "Open": latest_data["Open"], "High": latest_data["High"],
                "Low": latest_data["Low"], "Close": latest_data["Close"],
            },
            technical_indicators=technical_indicators,
            momentum_indicators=momentum_indicators,
            mean_reversion_indicators=mean_reversion_indicators,
            volatility_indicators=volatility_indicators,
            trend_indicators=trend_indicators,
            volume_indicators=volume_indicators,
            historical_indicators=historical_indicators,
            historical_momentum=historical_momentum,
            historical_mean_reversion=historical_mean_reversion,
            historical_volatility=historical_volatility,
            historical_trend=historical_trend,
            historical_volume=historical_volume,
            statistical_indicators=statistical_indicators,
            historical_statistical=historical_statistical,
            historical_ohlc=historical_ohlc,
        )

        if self.cache:
            self.cache.set(cache_key, market_data, self.cache_ttl_seconds)

        return market_data
    async def fetch_and_process_multiple_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime, historical_periods: int = 10
    ) -> List[Optional[MarketData]]:
        """
        Fetches and processes data for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols.
            start_date: Start date for data.
            end_date: End date for data.
            historical_periods: Number of historical periods to include in historical_indicators.

        Returns:
            List of MarketData objects, one per symbol (None if failed).
        """
        tasks = [self.fetch_and_process_data(symbol, start_date, end_date, historical_periods) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _calculate_hurst_exponent(self, ts, min_window=10, max_window=100, num_windows=5):
        """
        Calculate the Hurst Exponent using rescaled range (R/S) analysis.

        Args:
            ts: Time series data (pandas Series)
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            num_windows: Number of window sizes to use

        Returns:
            Hurst Exponent value or None if calculation fails
        """
        try:
            # Convert to numpy array and remove NaN values
            ts_clean = ts.dropna().values
            n = len(ts_clean)

            if n < max_window:
                return None

            # Calculate log returns
            log_returns = np.diff(np.log(ts_clean))

            # Generate window sizes
            window_sizes = np.linspace(min_window, max_window, num_windows, dtype=int)

            # Calculate R/S values for each window size
            rs_values = []

            for window in window_sizes:
                if window >= n:
                    continue

                # Divide the series into sub-series of length window
                num_subseries = n // window

                if num_subseries < 2:
                    continue

                rs_values_window = []

                for i in range(num_subseries):
                    # Extract sub-series
                    subseries = log_returns[i*window:(i+1)*window]

                    if len(subseries) < 2:
                        continue

                    # Calculate mean and standard deviation
                    mean = np.mean(subseries)
                    std = np.std(subseries)

                    if std == 0:
                        continue

                    # Calculate cumulative deviation
                    cumdev = np.cumsum(subseries - mean)

                    # Calculate range
                    r = np.max(cumdev) - np.min(cumdev)

                    # Calculate rescaled range
                    rs = r / std
                    rs_values_window.append(rs)

                if rs_values_window:
                    rs_values.append(np.mean(rs_values_window))

            if len(rs_values) < 2:
                return None

            # Calculate Hurst Exponent using linear regression
            # log(R/S) = H * log(window) + constant
            log_window_sizes = np.log(window_sizes[:len(rs_values)])
            log_rs_values = np.log(rs_values)

            # Linear regression
            coeffs = np.polyfit(log_window_sizes, log_rs_values, 1)
            hurst = coeffs[0]

            return hurst

        except Exception as e:
            print(f"Error in Hurst Exponent calculation: {e}")
            return None
