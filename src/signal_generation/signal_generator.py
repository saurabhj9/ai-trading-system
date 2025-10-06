"""
Main Signal Generator class for the Local Signal Generation Framework.

This class orchestrates all components to generate trading signals without
relying on LLM calls, providing a complete signal generation pipeline.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
import time

from .core import Signal, SignalType, SignalStrength, MarketRegime, ValidationResult, ConflictInfo
from .components import (
    MarketRegimeDetector,
    IndicatorScorer,
    ConsensusSignalCombiner,
    HierarchicalDecisionTree,
    SignalValidator,
    ConflictDetector,
    LLMEscalationLogic,
)


class LocalSignalGenerator:
    """
    Main signal generator that orchestrates all components.

    This class provides a complete pipeline for generating trading signals
    without relying on LLM calls, including market regime detection,
    indicator scoring, consensus combination, validation, and conflict detection.
    """

    def __init__(self, config: Dict):
        """
        Initialize the local signal generator.

        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config

        # Initialize components
        self.market_regime_detector = MarketRegimeDetector(config.get("market_regime", {}))
        self.indicator_scorer = IndicatorScorer(config.get("indicator_scorer", {}))
        self.consensus_combiner = ConsensusSignalCombiner(config.get("consensus_combiner", {}))
        self.decision_tree = HierarchicalDecisionTree(config.get("decision_tree", {}))
        self.signal_validator = SignalValidator(config.get("signal_validator", {}))
        self.conflict_detector = ConflictDetector(config.get("conflict_detector", {}))
        self.escalation_logic = LLMEscalationLogic(config.get("escalation_logic", {}))

        # Signal history
        self.signal_history: List[Signal] = []
        self.max_history_size = config.get("max_history_size", 1000)

        # Performance tracking
        self.performance_metrics = {
            "total_signals_generated": 0,
            "total_escalations": 0,
            "avg_generation_time": 0.0,
            "avg_validation_confidence": 0.0,
        }

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Tuple[Signal, Dict[str, Any]]:
        """
        Generate a trading signal based on market data.

        Args:
            market_data: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            Tuple[Signal, Dict[str, Any]]: (Generated signal, metadata)
        """
        start_time = time.time()

        try:
            # 1. Detect market regime
            market_regime = self.market_regime_detector.detect_regime(market_data)
            regime_data = self.market_regime_detector.get_regime_data()

            # 2. Score indicators
            indicator_scores = self.indicator_scorer.score_all_indicators(market_data, market_regime)

            # 3. Combine signals using consensus
            consensus_signal, consensus_strength, consensus_confidence = self.consensus_combiner.combine_signals(indicator_scores)

            # 4. Make hierarchical decision
            decision_tree = self.decision_tree.make_decision(
                market_regime=market_regime,
                indicator_scores=indicator_scores,
                consensus_signal=consensus_signal,
                consensus_strength=consensus_strength,
                consensus_confidence=consensus_confidence,
                market_data={"symbol": symbol, "data_points": len(market_data)}
            )

            # 5. Create signal
            final_signal_type = decision_tree.decision
            final_confidence = decision_tree.confidence

            signal = Signal(
                signal_type=final_signal_type,
                strength=consensus_strength,
                symbol=symbol,
                price=market_data['close'].iloc[-1],
                confidence=final_confidence,
                regime=market_regime,
                indicators={score.name: score.value for score in indicator_scores},
                source="LocalSignalGenerator",
                reasoning=decision_tree.reasoning,
                metadata={
                    "decision_tree": self.decision_tree.export_decision_tree(),
                    "consensus_signal": consensus_signal.value,
                    "consensus_confidence": consensus_confidence,
                    "regime_confidence": regime_data.get("confidence", 0.0) if regime_data else 0.0,
                    "indicator_count": len(indicator_scores),
                }
            )

            # 6. Validate signal
            validation_result = self.signal_validator.validate_signal(
                signal, indicator_scores, {"symbol": symbol}
            )

            # 7. Detect conflicts
            conflicts = self.conflict_detector.detect_conflicts(
                signal, indicator_scores, self.get_recent_signals(5)
            )

            # 8. Check for escalation
            should_escalate, escalation_probability, escalation_reasoning = self.escalation_logic.should_escalate(
                signal, validation_result, conflicts,
                {"symbol": symbol, "volatility": self._calculate_volatility(market_data)},
                self.market_regime_detector.get_regime_duration()
            )

            # Update signal with validation and conflict information
            signal.metadata.update({
                "validation_result": validation_result.__dict__,
                "conflicts": [conflict.__dict__ for conflict in conflicts],
                "escalation_required": should_escalate,
                "escalation_probability": escalation_probability,
                "escalation_reasoning": escalation_reasoning,
            })

            # Adjust confidence based on validation
            signal.confidence = validation_result.adjusted_confidence

            # 9. Store signal in history
            self._store_signal(signal)

            # 10. Update performance metrics
            generation_time = time.time() - start_time
            self._update_performance_metrics(generation_time, validation_result, should_escalate)

            # 11. Prepare metadata
            metadata = {
                "generation_time": generation_time,
                "market_regime": {
                    "regime": market_regime.value,
                    "confidence": regime_data.get("confidence", 0.0) if regime_data else 0.0,
                    "supporting_indicators": regime_data.get("supporting_indicators", {}) if regime_data else {},
                },
                "indicator_scores": [score.__dict__ for score in indicator_scores],
                "validation_result": validation_result.__dict__,
                "conflicts": [conflict.__dict__ for conflict in conflicts],
                "escalation": {
                    "required": should_escalate,
                    "probability": escalation_probability,
                    "reasoning": escalation_reasoning,
                },
                "performance_metrics": self.get_performance_metrics(),
            }

            return signal, metadata

        except Exception as e:
            # Create error signal
            error_signal = Signal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                symbol=symbol,
                price=market_data['close'].iloc[-1] if not market_data.empty else 0.0,
                confidence=0.0,
                regime=MarketRegime.UNCERTAIN,
                source="LocalSignalGenerator",
                reasoning=f"Error generating signal: {str(e)}",
                metadata={"error": str(e)}
            )

            metadata = {
                "error": str(e),
                "generation_time": time.time() - start_time,
                "performance_metrics": self.get_performance_metrics(),
            }

            return error_signal, metadata

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility from market data."""
        if len(market_data) < 2:
            return 0.0

        returns = market_data['close'].pct_change().dropna()
        return returns.std() if not returns.empty else 0.0

    def _store_signal(self, signal: Signal):
        """Store signal in history."""
        self.signal_history.append(signal)

        # Keep history size manageable
        if len(self.signal_history) > self.max_history_size:
            self.signal_history.pop(0)

    def _update_performance_metrics(self, generation_time: float, validation_result: ValidationResult, should_escalate: bool):
        """Update performance metrics."""
        self.performance_metrics["total_signals_generated"] += 1

        if should_escalate:
            self.performance_metrics["total_escalations"] += 1

        # Update average generation time
        total_signals = self.performance_metrics["total_signals_generated"]
        current_avg = self.performance_metrics["avg_generation_time"]
        self.performance_metrics["avg_generation_time"] = (
            (current_avg * (total_signals - 1) + generation_time) / total_signals
        )

        # Update average validation confidence
        current_avg_conf = self.performance_metrics["avg_validation_confidence"]
        self.performance_metrics["avg_validation_confidence"] = (
            (current_avg_conf * (total_signals - 1) + validation_result.confidence) / total_signals
        )

    def get_recent_signals(self, limit: int = 10) -> List[Signal]:
        """Get recent signals from history."""
        return self.signal_history[-limit:] if self.signal_history else []

    def get_signal_history(self) -> List[Signal]:
        """Get complete signal history."""
        return self.signal_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()

        # Add calculated metrics
        if metrics["total_signals_generated"] > 0:
            metrics["escalation_rate"] = metrics["total_escalations"] / metrics["total_signals_generated"]
        else:
            metrics["escalation_rate"] = 0.0

        # Add component statistics
        metrics.update({
            "validation_stats": self.signal_validator.get_validation_statistics(),
            "conflict_stats": self.conflict_detector.get_conflict_statistics(),
            "escalation_stats": self.escalation_logic.get_escalation_statistics(),
        })

        return metrics

    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_signals_generated": 0,
            "total_escalations": 0,
            "avg_generation_time": 0.0,
            "avg_validation_confidence": 0.0,
        }

    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "market_regime_detector": {
                "current_regime": self.market_regime_detector.get_current_regime().__dict__ if self.market_regime_detector.get_current_regime() else None,
                "regime_duration": self.market_regime_detector.get_regime_duration(),
                "is_stable": self.market_regime_detector.is_regime_stable(),
            },
            "signal_validator": {
                "validation_count": len(self.signal_validator.validation_history),
                "recent_issues": self.signal_validator.get_recent_validation_issues(5),
            },
            "conflict_detector": {
                "conflict_count": len(self.conflict_detector.conflict_history),
                "recent_conflicts": len([c for c in self.conflict_detector.conflict_history if (datetime.now() - c.timestamp).total_seconds() < 3600]),
            },
            "escalation_logic": {
                "escalation_count": len(self.escalation_logic.escalation_history),
                "recent_escalations": len(self.escalation_logic.recent_escalations),
            },
        }
