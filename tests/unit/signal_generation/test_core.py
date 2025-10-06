"""
Unit tests for core signal generation components.
"""

import pytest
from datetime import datetime
from src.signal_generation.core import (
    Signal, SignalType, SignalStrength, MarketRegime,
    IndicatorScore, ConflictInfo, ValidationResult
)


class TestSignal:
    """Test the Signal data class."""

    def test_signal_creation(self):
        """Test creating a signal with valid data."""
        signal = Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            symbol="AAPL",
            price=150.0,
            confidence=0.8,
            regime=MarketRegime.TRENDING_UP,
            source="test",
            reasoning="Test signal"
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.symbol == "AAPL"
        assert signal.price == 150.0
        assert signal.confidence == 0.8
        assert signal.regime == MarketRegime.TRENDING_UP
        assert signal.source == "test"
        assert signal.reasoning == "Test signal"
        assert signal.id is not None
        assert signal.timestamp is not None

    def test_signal_validation(self):
        """Test signal validation."""
        # Valid signal should not raise exception
        Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            symbol="AAPL",
            price=150.0,
            confidence=0.8,
            regime=MarketRegime.TRENDING_UP
        )

        # Invalid confidence should raise exception
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Signal(
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                symbol="AAPL",
                price=150.0,
                confidence=1.5,  # Invalid
                regime=MarketRegime.TRENDING_UP
            )

        # Negative price should raise exception
        with pytest.raises(ValueError, match="Price must be non-negative"):
            Signal(
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                symbol="AAPL",
                price=-10.0,  # Invalid
                confidence=0.8,
                regime=MarketRegime.TRENDING_UP
            )

    def test_signal_to_dict(self):
        """Test converting signal to dictionary."""
        signal = Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            symbol="AAPL",
            price=150.0,
            confidence=0.8,
            regime=MarketRegime.TRENDING_UP
        )

        signal_dict = signal.to_dict()

        assert signal_dict["signal_type"] == "BUY"
        assert signal_dict["strength"] == "STRONG"
        assert signal_dict["symbol"] == "AAPL"
        assert signal_dict["price"] == 150.0
        assert signal_dict["confidence"] == 0.8
        assert signal_dict["regime"] == "TRENDING_UP"
        assert "timestamp" in signal_dict
        assert "id" in signal_dict

    def test_signal_from_dict(self):
        """Test creating signal from dictionary."""
        signal_data = {
            "id": "test-id",
            "timestamp": datetime.now().isoformat(),
            "signal_type": "BUY",
            "strength": "STRONG",
            "symbol": "AAPL",
            "price": 150.0,
            "confidence": 0.8,
            "regime": "TRENDING_UP",
            "indicators": {"RSI": 30.0},
            "metadata": {},
            "source": "test",
            "reasoning": "Test signal"
        }

        signal = Signal.from_dict(signal_data)

        assert signal.id == "test-id"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.symbol == "AAPL"
        assert signal.price == 150.0
        assert signal.confidence == 0.8
        assert signal.regime == MarketRegime.TRENDING_UP
        assert signal.indicators == {"RSI": 30.0}


class TestIndicatorScore:
    """Test the IndicatorScore data class."""

    def test_indicator_score_creation(self):
        """Test creating an indicator score with valid data."""
        score = IndicatorScore(
            name="RSI",
            value=30.0,
            score=0.8,
            weight=0.7,
            signal_type=SignalType.BUY,
            confidence=0.9
        )

        assert score.name == "RSI"
        assert score.value == 30.0
        assert score.score == 0.8
        assert score.weight == 0.7
        assert score.signal_type == SignalType.BUY
        assert score.confidence == 0.9

    def test_indicator_score_validation(self):
        """Test indicator score validation."""
        # Valid score should not raise exception
        IndicatorScore(
            name="RSI",
            value=30.0,
            score=0.8,
            weight=0.7,
            signal_type=SignalType.BUY,
            confidence=0.9
        )

        # Invalid score should raise exception
        with pytest.raises(ValueError, match="Score must be between -1.0 and 1.0"):
            IndicatorScore(
                name="RSI",
                value=30.0,
                score=1.5,  # Invalid
                weight=0.7,
                signal_type=SignalType.BUY,
                confidence=0.9
            )

        # Invalid weight should raise exception
        with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
            IndicatorScore(
                name="RSI",
                value=30.0,
                score=0.8,
                weight=1.5,  # Invalid
                signal_type=SignalType.BUY,
                confidence=0.9
            )

        # Invalid confidence should raise exception
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            IndicatorScore(
                name="RSI",
                value=30.0,
                score=0.8,
                weight=0.7,
                signal_type=SignalType.BUY,
                confidence=1.5  # Invalid
            )


class TestConflictInfo:
    """Test the ConflictInfo data class."""

    def test_conflict_info_creation(self):
        """Test creating conflict information."""
        conflict = ConflictInfo(
            conflict_type="DIRECTION_CONFLICT",
            conflicting_signals=["signal1", "signal2"],
            severity="HIGH",
            description="Signals conflict in direction",
            resolution_strategy="prefer_higher_confidence"
        )

        assert conflict.conflict_type == "DIRECTION_CONFLICT"
        assert conflict.conflicting_signals == ["signal1", "signal2"]
        assert conflict.severity == "HIGH"
        assert conflict.description == "Signals conflict in direction"
        assert conflict.resolution_strategy == "prefer_higher_confidence"


class TestValidationResult:
    """Test the ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            issues=["Issue 1"],
            recommendations=["Recommendation 1"],
            adjusted_confidence=0.7
        )

        assert result.is_valid is True
        assert result.confidence == 0.8
        assert result.issues == ["Issue 1"]
        assert result.recommendations == ["Recommendation 1"]
        assert result.adjusted_confidence == 0.7

    def test_validation_result_default_adjusted_confidence(self):
        """Test default adjusted confidence behavior."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            issues=[],
            recommendations=[]
        )

        # Should default to confidence if not specified
        assert result.adjusted_confidence == 0.8
