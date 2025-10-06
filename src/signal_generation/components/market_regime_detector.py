"""
Market Regime Detector component for the Local Signal Generation Framework.

This component detects the current market regime based on technical indicators.
It wraps the existing market regime detection functionality and integrates it
with the signal generation framework.
"""

from typing import Dict, Optional, List, Any
import pandas as pd
from datetime import datetime

from ..core import MarketRegime
from ...analysis.market_regime import RegimeDetector as BaseRegimeDetector, Regime as BaseRegime


class MarketRegimeDetector:
    """
    Detects the current market regime based on technical indicators.

    This component wraps the existing RegimeDetector and provides an interface
    compatible with the signal generation framework.
    """

    def __init__(self, config: Dict):
        """
        Initialize the market regime detector.

        Args:
            config: Configuration dictionary with detector parameters
        """
        self.config = config
        self._detector = BaseRegimeDetector(config)
        self._current_regime: Optional[MarketRegime] = None
        self._regime_history: List[Dict[str, Any]] = []
        self._current_regime_data: Optional[Dict[str, Any]] = None

    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            MarketRegime: Detected market regime
        """
        # Use the existing detector
        base_regime = self._detector.detect(market_data)

        # Convert to our framework's MarketRegime
        regime = self._convert_regime(base_regime)

        # Store additional regime data
        regime_data = {
            "regime": regime,
            "confidence": base_regime.confidence,
            "supporting_indicators": base_regime.supporting_indicators,
            "timestamp": base_regime.timestamp,
        }

        # Update history
        self._current_regime = regime
        self._current_regime_data = regime_data
        self._regime_history.append(regime_data)

        # Keep history size manageable
        if len(self._regime_history) > 200:
            self._regime_history.pop(0)

        return regime

    def _convert_regime(self, base_regime) -> MarketRegime:
        """
        Convert from base Regime to framework MarketRegime.

        Args:
            base_regime: Regime from the base detector

        Returns:
            MarketRegime: Converted regime
        """
        # Map regime enums
        regime_mapping = {
            BaseRegime.TRENDING_UP: MarketRegime.TRENDING_UP,
            BaseRegime.TRENDING_DOWN: MarketRegime.TRENDING_DOWN,
            BaseRegime.RANGING: MarketRegime.RANGING,
            BaseRegime.VOLATILE: MarketRegime.VOLATILE,
            BaseRegime.UNCERTAIN: MarketRegime.UNCERTAIN,
        }

        return regime_mapping.get(base_regime.regime, MarketRegime.UNCERTAIN)

    def get_current_regime(self) -> Optional[MarketRegime]:
        """
        Get the most recently detected regime.

        Returns:
            MarketRegime: Current regime or None if no detection has occurred
        """
        return self._current_regime

    def get_regime_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently detected regime data.

        Returns:
            Dict: Current regime data or None if no detection has occurred
        """
        return self._current_regime_data

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of detected regimes.

        Returns:
            List[Dict]: History of regimes with additional data
        """
        return self._regime_history.copy()

    def is_regime_stable(self, periods: int = 3) -> bool:
        """
        Check if the current regime has been stable for a given number of periods.

        Args:
            periods: Number of periods to check for stability

        Returns:
            bool: True if regime is stable, False otherwise
        """
        if len(self._regime_history) < periods:
            return False

        recent_regimes = self._regime_history[-periods:]
        return all(r["regime"] == recent_regimes[0]["regime"] for r in recent_regimes)

    def get_regime_duration(self) -> int:
        """
        Get the duration of the current regime in periods.

        Returns:
            int: Number of periods the current regime has been active
        """
        if not self._regime_history:
            return 0

        current_regime = self._regime_history[-1]["regime"]
        duration = 0

        # Count backwards from the most recent regime
        for regime in reversed(self._regime_history):
            if regime["regime"] == current_regime:
                duration += 1
            else:
                break

        return duration
