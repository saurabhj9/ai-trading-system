"""
Configuration for the Local Signal Generation Framework.

This module provides configuration classes for all components of the
signal generation system.
"""

from typing import Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class MarketRegimeConfig(BaseSettings):
    """Configuration for market regime detection."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_REGIME_')

    adx_period: int = 14
    atr_period: int = 14
    hurst_exponent_lag: int = 100
    trend_strength_threshold: float = 25.0
    ranging_threshold: float = 20.0
    volatility_threshold_percent: float = 2.5
    confirmation_periods: int = 3


class IndicatorScorerConfig(BaseSettings):
    """Configuration for indicator scoring."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_SCORER_')

    # Indicator weights
    indicator_weights: Dict[str, float] = {
        "RSI": 0.8,
        "MACD": 0.9,
        "ADX": 0.7,
        "PLUS_DI": 0.6,
        "MINUS_DI": 0.6,
        "ATR": 0.4,
        "OBV": 0.5,
        "STOCH": 0.7,
        "WILLR": 0.6,
        "MFI": 0.6,
        "CCI": 0.5,
        "NATR": 0.4,
        "AD": 0.5,
    }

    # Regime-specific adjustments
    regime_adjustments: Dict[str, Dict[str, float]] = {
        "TRENDING_UP": {
            "ADX": 1.2,
            "PLUS_DI": 1.3,
            "MINUS_DI": 0.7,
            "MACD": 1.1,
        },
        "TRENDING_DOWN": {
            "ADX": 1.2,
            "PLUS_DI": 0.7,
            "MINUS_DI": 1.3,
            "MACD": 1.1,
        },
        "RANGING": {
            "RSI": 1.2,
            "STOCH": 1.2,
            "WILLR": 1.1,
        },
        "VOLATILE": {
            "ATR": 1.3,
            "NATR": 1.3,
        },
    }

    # Overbought/oversold thresholds
    overbought_thresholds: Dict[str, float] = {
        "RSI": 70.0,
        "STOCH": 80.0,
        "WILLR": -20.0,
        "MFI": 80.0,
    }

    oversold_thresholds: Dict[str, float] = {
        "RSI": 30.0,
        "STOCH": 20.0,
        "WILLR": -80.0,
        "MFI": 20.0,
    }

    # Neutral zones
    neutral_zones: Dict[str, tuple] = {
        "RSI": (40.0, 60.0),
        "STOCH": (30.0, 70.0),
        "WILLR": (-60.0, -40.0),
        "MFI": (40.0, 60.0),
    }


class ConsensusCombinerConfig(BaseSettings):
    """Configuration for consensus signal combination."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_CONSENSUS_')

    consensus_threshold: float = 0.6
    min_indicators: int = 3
    confidence_weight_factor: float = 1.5

    # Signal strength thresholds
    strength_thresholds: Dict[str, float] = {
        "WEAK": 0.3,
        "MODERATE": 0.5,
        "STRONG": 0.7,
        "VERY_STRONG": 0.85,
    }


class DecisionTreeConfig(BaseSettings):
    """Configuration for hierarchical decision tree."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_TREE_')

    # Decision thresholds for each level
    decision_thresholds: Dict[str, float] = {
        "MARKET_REGIME": 0.7,
        "INDICATOR_ANALYSIS": 0.6,
        "SIGNAL_CONSENSUS": 0.65,
        "RISK_ASSESSMENT": 0.5,
        "FINAL_DECISION": 0.6,
    }

    # Risk parameters
    max_risk_level: float = 0.7
    volatility_threshold: float = 0.8


class SignalValidatorConfig(BaseSettings):
    """Configuration for signal validation."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_VALIDATOR_')

    # Validation thresholds
    min_confidence_threshold: float = 0.5
    max_risk_threshold: float = 0.8
    min_indicators_required: int = 3

    # Consistency checks
    max_indicator_disagreement: float = 0.7

    # Historical validation
    enable_historical_validation: bool = True
    historical_lookback_periods: int = 20
    min_historical_accuracy: float = 0.6


class ConflictDetectorConfig(BaseSettings):
    """Configuration for conflict detection."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_CONFLICT_')

    # Conflict detection thresholds
    direction_conflict_threshold: float = 0.4
    strength_conflict_threshold: float = 2
    regime_conflict_threshold: float = 0.6

    # Conflict severity weights
    severity_weights: Dict[str, float] = {
        "DIRECTION_CONFLICT": 0.8,
        "STRENGTH_CONFLICT": 0.5,
        "REGIME_CONFLICT": 0.7,
        "TEMPORAL_CONFLICT": 0.4,
        "INDICATOR_CONFLICT": 0.6,
    }


class EscalationLogicConfig(BaseSettings):
    """Configuration for LLM escalation logic."""
    model_config = SettingsConfigDict(env_prefix='SIGNAL_ESCALATION_')

    # Escalation triggers
    high_conflict_count: int = 3
    high_severity_conflicts: int = 1
    low_validation_confidence: float = 0.4
    uncertain_regime_duration: int = 5
    extreme_market_conditions: bool = True
    complex_indicator_patterns: bool = True

    # Escalation thresholds
    escalation_probability_threshold: float = 0.7
    max_escalations_per_hour: int = 5


class SignalGenerationConfig(BaseSettings):
    """Main configuration for the signal generation framework."""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Component configurations
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    indicator_scorer: IndicatorScorerConfig = IndicatorScorerConfig()
    consensus_combiner: ConsensusCombinerConfig = ConsensusCombinerConfig()
    decision_tree: DecisionTreeConfig = DecisionTreeConfig()
    signal_validator: SignalValidatorConfig = SignalValidatorConfig()
    conflict_detector: ConflictDetectorConfig = ConflictDetectorConfig()
    escalation_logic: EscalationLogicConfig = EscalationLogicConfig()

    # General settings
    max_history_size: int = 1000

    # Performance targets
    target_latency_ms: float = 100.0
    target_accuracy: float = 0.85
    target_cost_reduction: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "market_regime": self.market_regime.model_dump(),
            "indicator_scorer": self.indicator_scorer.model_dump(),
            "consensus_combiner": self.consensus_combiner.model_dump(),
            "decision_tree": self.decision_tree.model_dump(),
            "signal_validator": self.signal_validator.model_dump(),
            "conflict_detector": self.conflict_detector.model_dump(),
            "escalation_logic": self.escalation_logic.model_dump(),
            "max_history_size": self.max_history_size,
            "target_latency_ms": self.target_latency_ms,
            "target_accuracy": self.target_accuracy,
            "target_cost_reduction": self.target_cost_reduction,
        }


# Global configuration instance
signal_generation_config = SignalGenerationConfig()
