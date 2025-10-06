"""
Signal Validator component for the Local Signal Generation Framework.

This component performs quality control checks on generated signals to ensure
they meet reliability standards before being used for trading decisions.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

from ..core import Signal, ValidationResult, SignalType, MarketRegime, IndicatorScore


class SignalValidator:
    """
    Validates generated signals to ensure they meet quality standards.

    This component performs multiple validation checks including confidence
    thresholds, indicator consistency, market regime compatibility, and
    historical performance analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize the signal validator.

        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config

        # Validation thresholds
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.5)
        self.max_risk_threshold = config.get("max_risk_threshold", 0.8)
        self.min_indicators_required = config.get("min_indicators_required", 3)

        # Consistency checks
        self.max_indicator_disagreement = config.get("max_indicator_disagreement", 0.7)
        self.regime_compatibility_weights = config.get("regime_compatibility_weights", {
            MarketRegime.TRENDING_UP: {SignalType.BUY: 1.0, SignalType.SELL: 0.3, SignalType.HOLD: 0.6},
            MarketRegime.TRENDING_DOWN: {SignalType.BUY: 0.3, SignalType.SELL: 1.0, SignalType.HOLD: 0.6},
            MarketRegime.RANGING: {SignalType.BUY: 0.6, SignalType.SELL: 0.6, SignalType.HOLD: 1.0},
            MarketRegime.VOLATILE: {SignalType.BUY: 0.4, SignalType.SELL: 0.4, SignalType.HOLD: 1.0},
            MarketRegime.UNCERTAIN: {SignalType.BUY: 0.3, SignalType.SELL: 0.3, SignalType.HOLD: 1.0},
        })

        # Historical validation
        self.enable_historical_validation = config.get("enable_historical_validation", True)
        self.historical_lookback_periods = config.get("historical_lookback_periods", 20)
        self.min_historical_accuracy = config.get("min_historical_accuracy", 0.6)

        # Validation history
        self.validation_history: List[Tuple[Signal, ValidationResult]] = []

    def validate_signal(self,
                       signal: Signal,
                       indicator_scores: List[IndicatorScore],
                       market_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a generated signal against multiple criteria.

        Args:
            signal: Signal to validate
            indicator_scores: List of indicator scores that contributed to the signal
            market_data: Market data used for signal generation

        Returns:
            ValidationResult: Result of validation with issues and recommendations
        """
        issues = []
        recommendations = []
        adjusted_confidence = signal.confidence

        # 1. Confidence threshold validation
        confidence_valid, confidence_adjustment = self._validate_confidence(signal)
        if not confidence_valid:
            issues.append(f"Signal confidence {signal.confidence:.2f} below threshold {self.min_confidence_threshold}")
            recommendations.append("Increase signal confidence through stronger indicator agreement")
        adjusted_confidence *= confidence_adjustment

        # 2. Indicator consistency validation
        consistency_valid, consistency_adjustment = self._validate_indicator_consistency(indicator_scores)
        if not consistency_valid:
            issues.append("High indicator disagreement detected")
            recommendations.append("Review indicator weights and scoring parameters")
        adjusted_confidence *= consistency_adjustment

        # 3. Market regime compatibility validation
        regime_valid, regime_adjustment = self._validate_regime_compatibility(signal, signal.regime)
        if not regime_valid:
            issues.append(f"Signal {signal.signal_type.value} not compatible with {signal.regime.value} regime")
            recommendations.append("Consider regime-specific signal adjustments")
        adjusted_confidence *= regime_adjustment

        # 4. Risk assessment validation
        risk_valid, risk_adjustment = self._validate_risk_assessment(signal, market_data)
        if not risk_valid:
            issues.append("Signal risk assessment exceeds acceptable levels")
            recommendations.append("Apply additional risk mitigation measures")
        adjusted_confidence *= risk_adjustment

        # 5. Historical performance validation
        if self.enable_historical_validation:
            historical_valid, historical_adjustment = self._validate_historical_performance(signal, market_data)
            if not historical_valid:
                issues.append("Historical performance for similar signals is poor")
                recommendations.append("Review historical signal performance patterns")
            adjusted_confidence *= historical_adjustment

        # 6. Technical validation
        technical_valid, technical_adjustment = self._validate_technical_criteria(signal, indicator_scores)
        if not technical_valid:
            issues.append("Signal fails technical validation criteria")
            recommendations.append("Review technical indicator calculations and thresholds")
        adjusted_confidence *= technical_adjustment

        # Determine overall validity
        is_valid = (
            confidence_valid and
            consistency_valid and
            regime_valid and
            risk_valid and
            (historical_valid if self.enable_historical_validation else True) and
            technical_valid
        )

        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            confidence=adjusted_confidence,
            issues=issues,
            recommendations=recommendations,
            adjusted_confidence=adjusted_confidence
        )

        # Store in history
        self.validation_history.append((signal, result))

        # Keep history manageable
        if len(self.validation_history) > 1000:
            self.validation_history.pop(0)

        return result

    def _validate_confidence(self, signal: Signal) -> Tuple[bool, float]:
        """Validate signal confidence against threshold."""
        if signal.confidence >= self.min_confidence_threshold:
            return True, 1.0
        else:
            # Adjust confidence based on how far below threshold
            adjustment = signal.confidence / self.min_confidence_threshold
            return False, adjustment

    def _validate_indicator_consistency(self, indicator_scores: List[IndicatorScore]) -> Tuple[bool, float]:
        """Validate indicator agreement levels."""
        if len(indicator_scores) < self.min_indicators_required:
            return False, 0.5

        # Calculate disagreement level
        signal_types = [score.signal_type for score in indicator_scores]
        unique_signals = len(set(signal_types))

        if unique_signals == 1:
            return True, 1.0
        elif unique_signals == 2:
            # Calculate agreement ratio
            most_common_signal = max(set(signal_types), key=signal_types.count)
            agreement_ratio = signal_types.count(most_common_signal) / len(signal_types)

            if agreement_ratio >= 0.7:
                return True, 0.9
            else:
                return False, 0.7
        else:
            # High disagreement
            return False, 0.5

    def _validate_regime_compatibility(self, signal: Signal, regime: MarketRegime) -> Tuple[bool, float]:
        """Validate signal compatibility with market regime."""
        compatibility_weights = self.regime_compatibility_weights.get(regime, {})
        compatibility = compatibility_weights.get(signal.signal_type, 0.5)

        if compatibility >= 0.8:
            return True, 1.0
        elif compatibility >= 0.6:
            return True, compatibility
        else:
            return False, compatibility

    def _validate_risk_assessment(self, signal: Signal, market_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate signal against risk criteria."""
        # Extract risk metrics from signal metadata or market data
        risk_level = signal.metadata.get("risk_level", 0.5)

        if risk_level <= self.max_risk_threshold:
            return True, 1.0
        else:
            # Apply penalty based on how much risk exceeds threshold
            excess_risk = risk_level - self.max_risk_threshold
            adjustment = max(0.3, 1.0 - excess_risk)
            return False, adjustment

    def _validate_historical_performance(self, signal: Signal, market_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate signal based on historical performance of similar signals."""
        # This is a simplified implementation
        # In a real system, you would query historical signal performance data

        # For now, return neutral validation
        # TODO: Implement actual historical performance analysis
        return True, 1.0

    def _validate_technical_criteria(self, signal: Signal, indicator_scores: List[IndicatorScore]) -> Tuple[bool, float]:
        """Validate signal against technical criteria."""
        if not indicator_scores:
            return False, 0.3

        # Check for extreme indicator values that might indicate anomalies
        extreme_values = 0
        for score in indicator_scores:
            if abs(score.score) > 0.95:
                extreme_values += 1

        # If too many extreme values, it might indicate an anomaly
        if extreme_values > len(indicator_scores) * 0.3:
            return False, 0.7

        # Check for sufficient indicator diversity
        indicator_categories = set()
        for score in indicator_scores:
            # Categorize indicators (simplified)
            if score.name in ['RSI', 'STOCH', 'WILLR']:
                indicator_categories.add('oscillator')
            elif score.name in ['MACD', 'ADX', 'PLUS_DI', 'MINUS_DI']:
                indicator_categories.add('trend')
            elif score.name in ['ATR', 'NATR']:
                indicator_categories.add('volatility')
            elif score.name in ['OBV', 'AD', 'MFI']:
                indicator_categories.add('volume')

        if len(indicator_categories) < 2:
            return False, 0.8

        return True, 1.0

    def get_validation_statistics(self) -> Dict[str, float]:
        """
        Get statistics about validation performance.

        Returns:
            Dict[str, float]: Validation statistics
        """
        if not self.validation_history:
            return {}

        total_validations = len(self.validation_history)
        valid_signals = sum(1 for _, result in self.validation_history if result.is_valid)

        avg_confidence = np.mean([result.confidence for _, result in self.validation_history])
        avg_adjusted_confidence = np.mean([result.adjusted_confidence for _, result in self.validation_history])

        avg_issues_per_signal = np.mean([len(result.issues) for _, result in self.validation_history])

        return {
            "total_validations": total_validations,
            "valid_signal_ratio": valid_signals / total_validations,
            "avg_confidence": avg_confidence,
            "avg_adjusted_confidence": avg_adjusted_confidence,
            "avg_issues_per_signal": avg_issues_per_signal,
        }

    def get_recent_validation_issues(self, limit: int = 10) -> List[str]:
        """
        Get recent validation issues.

        Args:
            limit: Maximum number of issues to return

        Returns:
            List[str]: Recent validation issues
        """
        recent_issues = []
        for signal, result in reversed(self.validation_history[-limit:]):
            for issue in result.issues:
                recent_issues.append(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M')}: {issue}")

        return recent_issues
