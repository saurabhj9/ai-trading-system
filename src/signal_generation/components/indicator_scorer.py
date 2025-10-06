"""
Indicator Scorer component for the Local Signal Generation Framework.

This component converts raw indicator values to standardized scores (-1.0 to 1.0)
and determines the signal type suggested by each indicator.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import talib

from ..core import IndicatorScore, SignalType, MarketRegime


class IndicatorScorer:
    """
    Converts raw indicator values to standardized scores and signal types.

    This component is responsible for normalizing indicator values to a
    consistent scale (-1.0 to 1.0) and determining the signal type suggested
    by each indicator based on market regime and indicator characteristics.
    """

    def __init__(self, config: Dict):
        """
        Initialize the indicator scorer.

        Args:
            config: Configuration dictionary with scoring parameters
        """
        self.config = config

        # Indicator configuration
        self.indicator_weights = config.get("indicator_weights", {})
        self.regime_adjustments = config.get("regime_adjustments", {})

        # Scoring parameters
        self.overbought_thresholds = config.get("overbought_thresholds", {
            "RSI": 70.0,
            "STOCH": 80.0,
            "WILLR": -20.0,
        })

        self.oversold_thresholds = config.get("oversold_thresholds", {
            "RSI": 30.0,
            "STOCH": 20.0,
            "WILLR": -80.0,
        })

        # Neutral zones for oscillators
        self.neutral_zones = config.get("neutral_zones", {
            "RSI": (40.0, 60.0),
            "STOCH": (30.0, 70.0),
            "WILLR": (-60.0, -40.0),
        })

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all necessary technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict[str, float]: Dictionary of indicator values
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None

        indicators = {}

        # Momentum indicators
        indicators['RSI'] = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50.0

        # MACD - handle the return value properly
        macd_result = talib.MACD(close)
        if len(macd_result) >= 3 and len(close) >= 26:
            indicators['MACD'] = macd_result[0][-1] if not np.isnan(macd_result[0][-1]) else 0.0
            indicators['MACD_SIGNAL'] = macd_result[1][-1] if not np.isnan(macd_result[1][-1]) else 0.0
            indicators['MACD_HIST'] = macd_result[2][-1] if not np.isnan(macd_result[2][-1]) else 0.0
        else:
            indicators['MACD'] = 0.0
            indicators['MACD_SIGNAL'] = 0.0
            indicators['MACD_HIST'] = 0.0

        # Trend indicators
        indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0
        indicators['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0
        indicators['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0

        # Volatility indicators
        indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0
        indicators['NATR'] = talib.NATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0

        # Volume indicators (if volume data is available)
        if volume is not None and len(volume) >= 20:
            indicators['OBV'] = talib.OBV(close, volume)[-1]
            indicators['AD'] = talib.AD(high, low, close, volume)[-1]
            indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
        else:
            indicators['OBV'] = 0.0
            indicators['AD'] = 0.0
            indicators['MFI'] = 50.0

        # Oscillator indicators - handle the return value properly
        stoch_result = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        if len(stoch_result) >= 2 and len(close) >= 14:
            indicators['STOCH_K'] = stoch_result[0][-1] if not np.isnan(stoch_result[0][-1]) else 50.0
            indicators['STOCH_D'] = stoch_result[1][-1] if not np.isnan(stoch_result[1][-1]) else 50.0
        else:
            indicators['STOCH_K'] = 50.0
            indicators['STOCH_D'] = 50.0

        indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else -50.0
        indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0.0

        # Clean up NaN values
        for key, value in indicators.items():
            if np.isnan(value) or np.isinf(value):
                indicators[key] = 0.0

        return indicators

    def score_indicator(self, name: str, value: float, regime: MarketRegime) -> IndicatorScore:
        """
        Score a single indicator based on its value and the current market regime.

        Args:
            name: Name of the indicator
            value: Raw indicator value
            regime: Current market regime

        Returns:
            IndicatorScore: Scored indicator
        """
        # Get default weight if not specified
        weight = self.indicator_weights.get(name, 0.5)

        # Adjust weight based on regime
        regime_multiplier = self.regime_adjustments.get(regime.name, {}).get(name, 1.0)
        adjusted_weight = min(weight * regime_multiplier, 1.0)

        # Score based on indicator type
        if name in ['RSI', 'STOCH', 'WILLR', 'MFI', 'CCI']:
            score, signal_type, confidence = self._score_oscillator(name, value, regime)
        elif name in ['MACD', 'MACD_SIGNAL', 'MACD_HIST']:
            score, signal_type, confidence = self._score_macd(name, value, regime)
        elif name in ['ADX', 'PLUS_DI', 'MINUS_DI']:
            score, signal_type, confidence = self._score_trend_indicator(name, value, regime)
        elif name in ['ATR', 'NATR']:
            score, signal_type, confidence = self._score_volatility_indicator(name, value, regime)
        elif name in ['OBV', 'AD']:
            score, signal_type, confidence = self._score_volume_indicator(name, value, regime)
        else:
            # Default scoring
            score, signal_type, confidence = 0.0, SignalType.HOLD, 0.3

        return IndicatorScore(
            name=name,
            value=value,
            score=score,
            weight=adjusted_weight,
            signal_type=signal_type,
            confidence=confidence
        )

    def _score_oscillator(self, name: str, value: float, regime: MarketRegime) -> Tuple[float, SignalType, float]:
        """Score oscillator indicators like RSI, Stochastic, etc."""
        overbought = self.overbought_thresholds.get(name, 70.0)
        oversold = self.oversold_thresholds.get(name, 30.0)
        neutral_min, neutral_max = self.neutral_zones.get(name, (40.0, 60.0))

        confidence = 0.7

        if name == 'WILLR':  # WILLR is inverted (-100 to 0)
            overbought, oversold = -oversold, -overbought
            neutral_min, neutral_max = -neutral_max, -neutral_min

        if value >= overbought:
            return -0.8, SignalType.SELL, confidence
        elif value <= oversold:
            return 0.8, SignalType.BUY, confidence
        elif neutral_min <= value <= neutral_max:
            return 0.0, SignalType.HOLD, 0.3
        elif value > neutral_max:
            # Between neutral and overbought
            normalized = (value - neutral_max) / (overbought - neutral_max)
            return -normalized * 0.5, SignalType.SELL, confidence * 0.6
        else:
            # Between oversold and neutral
            normalized = (value - oversold) / (neutral_min - oversold)
            return normalized * 0.5, SignalType.BUY, confidence * 0.6

    def _score_macd(self, name: str, value: float, regime: MarketRegime) -> Tuple[float, SignalType, float]:
        """Score MACD indicators."""
        if name == 'MACD_HIST':
            if value > 0:
                score = min(value / 0.01, 1.0)  # Normalize, cap at 1.0
                return score, SignalType.BUY, 0.6
            else:
                score = max(value / 0.01, -1.0)  # Normalize, cap at -1.0
                return score, SignalType.SELL, 0.6
        elif name == 'MACD':
            # MACD line vs signal line comparison handled by MACD_HIST
            return 0.0, SignalType.HOLD, 0.3
        else:
            return 0.0, SignalType.HOLD, 0.3

    def _score_trend_indicator(self, name: str, value: float, regime: MarketRegime) -> Tuple[float, SignalType, float]:
        """Score trend indicators like ADX, DI."""
        if name == 'PLUS_DI':
            # Positive directional movement
            score = value / 50.0  # Normalize roughly 0-1
            return min(score, 1.0), SignalType.BUY, 0.5
        elif name == 'MINUS_DI':
            # Negative directional movement
            score = value / 50.0  # Normalize roughly 0-1
            return min(score, 1.0), SignalType.SELL, 0.5
        elif name == 'ADX':
            # ADX measures trend strength, not direction
            if value > 25:
                return 0.3, SignalType.HOLD, 0.4  # Strong trend
            elif value < 20:
                return -0.3, SignalType.HOLD, 0.4  # Weak trend/ranging
            else:
                return 0.0, SignalType.HOLD, 0.3
        else:
            return 0.0, SignalType.HOLD, 0.3

    def _score_volatility_indicator(self, name: str, value: float, regime: MarketRegime) -> Tuple[float, SignalType, float]:
        """Score volatility indicators."""
        if name in ['ATR', 'NATR']:
            # High volatility might suggest caution
            if regime == MarketRegime.VOLATILE:
                return -0.2, SignalType.HOLD, 0.4
            else:
                return 0.1, SignalType.HOLD, 0.3
        else:
            return 0.0, SignalType.HOLD, 0.3

    def _score_volume_indicator(self, name: str, value: float, regime: MarketRegime) -> Tuple[float, SignalType, float]:
        """Score volume indicators."""
        # Volume indicators are more about confirmation than direction
        if name == 'OBV' or name == 'AD':
            if value > 0:
                return 0.2, SignalType.BUY, 0.4
            else:
                return -0.2, SignalType.SELL, 0.4
        elif name == 'MFI':
            # Money Flow Index is like RSI but with volume
            return self._score_oscillator('MFI', value, regime)
        else:
            return 0.0, SignalType.HOLD, 0.3

    def score_all_indicators(self, df: pd.DataFrame, regime: MarketRegime) -> List[IndicatorScore]:
        """
        Score all indicators for the given market data and regime.

        Args:
            df: DataFrame with OHLCV data
            regime: Current market regime

        Returns:
            List[IndicatorScore]: List of scored indicators
        """
        indicator_values = self.calculate_indicators(df)

        scored_indicators = []
        for name, value in indicator_values.items():
            score = self.score_indicator(name, value, regime)
            scored_indicators.append(score)

        return scored_indicators
