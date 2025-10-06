"""
Market Regime Detection System.

This module provides the core logic for detecting market regimes (e.g., trending,
ranging, volatile) based on technical indicators. It is designed to be a
reusable component that can be integrated into trading agents or other
analytical modules.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib


class Regime(Enum):
    """Enumeration for different market regimes."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class MarketRegime:
    """Data class to store the detected market regime and related information."""
    timestamp: datetime
    regime: Regime
    confidence: float
    supporting_indicators: Dict[str, float]


class RegimeDetector:
    """
    Detects the current market regime based on technical indicators.
    """

    def __init__(self, config: Dict):
        """
        Initializes the RegimeDetector with configuration parameters.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.adx_period = config.get("adx_period", 14)
        self.atr_period = config.get("atr_period", 14)
        self.hurst_exponent_lag = config.get("hurst_exponent_lag", 100)
        self.trend_strength_threshold = config.get("trend_strength_threshold", 25)
        self.ranging_threshold = config.get("ranging_threshold", 20)
        self.volatility_threshold_percent = config.get("volatility_threshold_percent", 2.5)
        self.confirmation_periods = config.get("confirmation_periods", 3)

        self._regime_history: List[MarketRegime] = []
        self._pending_regime: Optional[Regime] = None
        self._pending_count = 0

    def _calculate_hurst_exponent(self, ts: np.ndarray) -> float:
        """
        Calculates the Hurst Exponent for a time series.

        Args:
            ts: A numpy array of time series data (e.g., closing prices).

        Returns:
            The Hurst Exponent.
        """
        lags = range(2, min(self.hurst_exponent_lag, len(ts) // 2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0

        return hurst

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates all necessary technical indicators for regime detection.

        Args:
            df: A pandas DataFrame with OHLC data.

        Returns:
            A dictionary of calculated indicator values.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # ADX and Directional Movement
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

        # ATR
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        normalized_atr = (atr[-1] / close[-1]) * 100

        # Hurst Exponent
        hurst = self._calculate_hurst_exponent(close)

        return {
            "adx": adx[-1] if not np.isnan(adx[-1]) else 0,
            "plus_di": plus_di[-1] if not np.isnan(plus_di[-1]) else 0,
            "minus_di": minus_di[-1] if not np.isnan(minus_di[-1]) else 0,
            "normalized_atr": normalized_atr if not np.isnan(normalized_atr) else 0,
            "hurst": hurst if not np.isnan(hurst) else 0.5,
        }

    def _determine_potential_regime(self, indicators: Dict[str, float]) -> Regime:
        """
        Determines the potential regime based on indicator values.

        Args:
            indicators: A dictionary of calculated indicator values.

        Returns:
            The determined potential regime.
        """
        # 1. Volatility Check
        if indicators["normalized_atr"] > self.volatility_threshold_percent:
            return Regime.VOLATILE

        # 2. Trend Strength Check
        if indicators["adx"] > self.trend_strength_threshold:
            if indicators["plus_di"] > indicators["minus_di"]:
                return Regime.TRENDING_UP
            else:
                return Regime.TRENDING_DOWN

        # 3. Ranging Check
        if indicators["adx"] < self.ranging_threshold and indicators["hurst"] < 0.5:
            return Regime.RANGING

        # 4. Fallback to UNCERTAIN if no other condition is met
        return Regime.UNCERTAIN

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detects the market regime based on the provided financial data.

        Args:
            df: A pandas DataFrame with a time series of OHLC data.

        Returns:
            A MarketRegime object with the detected regime and metadata.
        """
        if len(df) < self.hurst_exponent_lag:
            raise ValueError(f"Not enough data to calculate indicators. Need at least {self.hurst_exponent_lag} periods.")

        indicators = self._calculate_indicators(df)
        potential_regime = self._determine_potential_regime(indicators)

        # Regime confirmation logic
        last_confirmed_regime = self._regime_history[-1].regime if self._regime_history else Regime.UNCERTAIN

        if potential_regime == last_confirmed_regime:
            # Regime is stable, no need for confirmation
            confirmed_regime = potential_regime
            self._pending_regime = None
            self._pending_count = 0
        else:
            # New potential regime detected
            if potential_regime == self._pending_regime:
                self._pending_count += 1
                if self._pending_count >= self.confirmation_periods:
                    # Confirmed the new regime
                    confirmed_regime = potential_regime
                    self._pending_regime = None
                    self._pending_count = 0
                else:
                    # Still in confirmation period, stick to the last confirmed regime
                    confirmed_regime = last_confirmed_regime
            else:
                # New potential regime, reset confirmation counter
                self._pending_regime = potential_regime
                self._pending_count = 1
                confirmed_regime = last_confirmed_regime

        confidence = 0.8 if confirmed_regime != Regime.UNCERTAIN else 0.4

        market_regime = MarketRegime(
            timestamp=df.index[-1].to_pydatetime(),
            regime=confirmed_regime,
            confidence=confidence,
            supporting_indicators=indicators,
        )

        self._regime_history.append(market_regime)
        # Keep history size manageable
        if len(self._regime_history) > 200:
            self._regime_history.pop(0)

        return market_regime

    def get_current_regime(self) -> Optional[MarketRegime]:
        """Returns the most recently detected market regime."""
        return self._regime_history[-1] if self._regime_history else None

    def get_regime_history(self) -> List[MarketRegime]:
        """Returns the history of detected market regimes."""
        return self._regime_history
