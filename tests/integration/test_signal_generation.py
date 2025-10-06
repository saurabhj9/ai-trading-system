"""
Integration tests for the Local Signal Generation Framework.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.signal_generation import LocalSignalGenerator
from src.signal_generation.core import SignalType, SignalStrength, MarketRegime


class TestLocalSignalGenerator:
    """Integration tests for the LocalSignalGenerator."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        # Generate 100 days of OHLCV data
        np.random.seed(42)  # For reproducible tests

        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Create a trending price series
        base_price = 100
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)  # Random noise
        close_prices = base_price + trend + noise

        # Generate OHLC data
        high = close_prices + np.random.uniform(0, 2, 100)
        low = close_prices - np.random.uniform(0, 2, 100)
        open_prices = low + np.random.uniform(0, high - low)
        volume = np.random.uniform(1000000, 5000000, 100)

        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close_prices,
            'volume': volume
        }, index=dates)

    @pytest.fixture
    def signal_generator(self):
        """Create a signal generator instance for testing."""
        config = {
            "market_regime": {
                "adx_period": 14,
                "atr_period": 14,
                "hurst_exponent_lag": 50,  # Reduced for test data
                "trend_strength_threshold": 25,
                "ranging_threshold": 20,
                "volatility_threshold_percent": 2.5,
                "confirmation_periods": 2,  # Reduced for testing
            },
            "indicator_scorer": {
                "indicator_weights": {
                    "RSI": 0.8,
                    "MACD": 0.9,
                    "ADX": 0.7,
                    "PLUS_DI": 0.6,
                    "MINUS_DI": 0.6,
                    "ATR": 0.4,
                    "OBV": 0.5,
                    "STOCH": 0.7,
                    "WILLR": 0.6,
                },
                "regime_adjustments": {
                    "TRENDING_UP": {
                        "ADX": 1.2,
                        "PLUS_DI": 1.3,
                        "MINUS_DI": 0.7,
                    },
                    "TRENDING_DOWN": {
                        "ADX": 1.2,
                        "PLUS_DI": 0.7,
                        "MINUS_DI": 1.3,
                    },
                },
            },
            "consensus_combiner": {
                "consensus_threshold": 0.6,
                "min_indicators": 3,
                "confidence_weight_factor": 1.5,
            },
            "decision_tree": {
                "decision_thresholds": {
                    "MARKET_REGIME": 0.7,
                    "INDICATOR_ANALYSIS": 0.6,
                    "SIGNAL_CONSENSUS": 0.65,
                    "RISK_ASSESSMENT": 0.5,
                    "FINAL_DECISION": 0.6,
                },
                "max_risk_level": 0.7,
                "volatility_threshold": 0.8,
            },
            "signal_validator": {
                "min_confidence_threshold": 0.3,  # Reduced for testing
                "max_risk_threshold": 0.8,
                "min_indicators_required": 2,  # Reduced for testing
                "max_indicator_disagreement": 0.8,  # Increased for testing
                "enable_historical_validation": False,  # Disabled for testing
            },
            "conflict_detector": {
                "direction_conflict_threshold": 0.3,  # Reduced for testing
                "strength_conflict_threshold": 2,
                "regime_conflict_threshold": 0.6,
            },
            "escalation_logic": {
                "high_conflict_count": 2,  # Reduced for testing
                "high_severity_conflicts": 1,
                "low_validation_confidence": 0.3,  # Reduced for testing
                "uncertain_regime_duration": 3,  # Reduced for testing
                "escalation_probability_threshold": 0.8,  # Increased for testing
                "max_escalations_per_hour": 10,  # Increased for testing
            },
            "max_history_size": 100,
        }

        return LocalSignalGenerator(config)

    def test_signal_generation(self, signal_generator, sample_market_data):
        """Test basic signal generation."""
        signal, metadata = signal_generator.generate_signal(sample_market_data, "TEST")

        # Check signal structure
        assert signal is not None
        assert signal.symbol == "TEST"
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MODERATE, SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.RANGING, MarketRegime.VOLATILE, MarketRegime.UNCERTAIN]
        assert signal.source == "LocalSignalGenerator"
        assert signal.reasoning is not None
        assert signal.price > 0

        # Check metadata structure
        assert metadata is not None
        assert "generation_time" in metadata
        assert "market_regime" in metadata
        assert "indicator_scores" in metadata
        assert "validation_result" in metadata
        assert "conflicts" in metadata
        assert "escalation" in metadata
        assert "performance_metrics" in metadata

        # Check generation time is reasonable
        assert metadata["generation_time"] < 1.0  # Should be fast

    def test_multiple_signal_generation(self, signal_generator, sample_market_data):
        """Test generating multiple signals over time."""
        signals = []

        # Generate signals for different time periods
        for i in range(5):
            # Use different subsets of data
            start_idx = i * 10
            end_idx = (i + 1) * 10 + 50  # Ensure enough data for indicators
            data_subset = sample_market_data.iloc[start_idx:end_idx]

            signal, metadata = signal_generator.generate_signal(data_subset, f"TEST_{i}")
            signals.append(signal)

        # Check that we got different signals
        assert len(signals) == 5

        # Check that signals are stored in history
        history = signal_generator.get_signal_history()
        assert len(history) >= 5

        # Check performance metrics
        metrics = signal_generator.get_performance_metrics()
        assert metrics["total_signals_generated"] >= 5
        assert metrics["avg_generation_time"] > 0
        assert metrics["avg_validation_confidence"] >= 0

    def test_component_status(self, signal_generator, sample_market_data):
        """Test component status reporting."""
        # Generate a signal first
        signal_generator.generate_signal(sample_market_data, "TEST")

        # Get component status
        status = signal_generator.get_component_status()

        # Check status structure
        assert "market_regime_detector" in status
        assert "signal_validator" in status
        assert "conflict_detector" in status
        assert "escalation_logic" in status

        # Check market regime detector status
        regime_status = status["market_regime_detector"]
        assert "current_regime" in regime_status
        assert "regime_duration" in regime_status
        assert "is_stable" in regime_status

    def test_signal_validation(self, signal_generator, sample_market_data):
        """Test signal validation functionality."""
        signal, metadata = signal_generator.generate_signal(sample_market_data, "TEST")

        # Check validation result
        validation_result = metadata["validation_result"]
        assert "is_valid" in validation_result
        assert "confidence" in validation_result
        assert "issues" in validation_result
        assert "recommendations" in validation_result
        assert "adjusted_confidence" in validation_result

        # Validation confidence should be reasonable
        assert 0.0 <= validation_result["confidence"] <= 1.0
        assert 0.0 <= validation_result["adjusted_confidence"] <= 1.0

    def test_conflict_detection(self, signal_generator, sample_market_data):
        """Test conflict detection functionality."""
        signal, metadata = signal_generator.generate_signal(sample_market_data, "TEST")

        # Check conflicts
        conflicts = metadata["conflicts"]
        assert isinstance(conflicts, list)

        # If there are conflicts, check their structure
        for conflict in conflicts:
            assert "conflict_type" in conflict
            assert "conflicting_signals" in conflict
            assert "severity" in conflict
            assert "description" in conflict
            assert "resolution_strategy" in conflict

    def test_escalation_logic(self, signal_generator, sample_market_data):
        """Test LLM escalation logic."""
        signal, metadata = signal_generator.generate_signal(sample_market_data, "TEST")

        # Check escalation information
        escalation = metadata["escalation"]
        assert "required" in escalation
        assert "probability" in escalation
        assert "reasoning" in escalation

        # Probability should be between 0 and 1
        assert 0.0 <= escalation["probability"] <= 1.0

    def test_error_handling(self, signal_generator):
        """Test error handling with invalid data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        signal, metadata = signal_generator.generate_signal(empty_df, "TEST")

        # Should return an error signal
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == SignalStrength.WEAK
        assert signal.confidence == 0.0
        assert "error" in metadata

        # Test with insufficient data
        insufficient_df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })

        signal, metadata = signal_generator.generate_signal(insufficient_df, "TEST")

        # Should handle gracefully
        assert signal is not None
        assert "error" in metadata or signal.confidence == 0.0

    def test_performance_targets(self, signal_generator, sample_market_data):
        """Test that performance targets are met."""
        signal, metadata = signal_generator.generate_signal(sample_market_data, "TEST")

        # Check latency target (< 100ms)
        generation_time = metadata["generation_time"]
        assert generation_time < 0.1, f"Generation time {generation_time}s exceeds 100ms target"

        # Check confidence target (≥ 85% for valid signals)
        if metadata["validation_result"]["is_valid"]:
            assert signal.confidence >= 0.85, f"Signal confidence {signal.confidence} below 85% target"

    def test_config_integration(self, signal_generator):
        """Test that configuration is properly integrated."""
        # Check that components are configured
        assert signal_generator.market_regime_detector is not None
        assert signal_generator.indicator_scorer is not None
        assert signal_generator.consensus_combiner is not None
        assert signal_generator.decision_tree is not None
        assert signal_generator.signal_validator is not None
        assert signal_generator.conflict_detector is not None
        assert signal_generator.escalation_logic is not None

        # Check that configuration values are used
        assert signal_generator.max_history_size == 100
