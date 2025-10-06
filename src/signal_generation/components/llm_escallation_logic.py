"""
LLM Escalation Logic component for the Local Signal Generation Framework.

This component determines when to escalate to LLM-based analysis for complex
scenarios that cannot be reliably handled by the rule-based system.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

from ..core import Signal, ConflictInfo, ValidationResult, SignalType, MarketRegime


class LLMEscalationLogic:
    """
    Determines when to escalate to LLM-based analysis.

    This component evaluates various conditions to decide whether a signal
    requires LLM intervention for more sophisticated analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize the LLM escalation logic.

        Args:
            config: Configuration dictionary with escalation parameters
        """
        self.config = config

        # Escalation triggers
        self.escalation_triggers = {
            "high_conflict_count": config.get("high_conflict_count", 3),
            "high_severity_conflicts": config.get("high_severity_conflicts", 1),
            "low_validation_confidence": config.get("low_validation_confidence", 0.4),
            "uncertain_regime_duration": config.get("uncertain_regime_duration", 5),
            "extreme_market_conditions": config.get("extreme_market_conditions", True),
            "complex_indicator_patterns": config.get("complex_indicator_patterns", True),
        }

        # Escalation thresholds
        self.escalation_probability_threshold = config.get("escalation_probability_threshold", 0.7)
        self.max_escalations_per_hour = config.get("max_escalations_per_hour", 5)

        # Escalation history
        self.escalation_history: List[Dict[str, Any]] = []
        self.recent_escalations: List[datetime] = []

    def should_escalate(self,
                       signal: Signal,
                       validation_result: ValidationResult,
                       conflicts: List[ConflictInfo],
                       market_data: Dict[str, Any],
                       regime_duration: int) -> Tuple[bool, float, str]:
        """
        Determine whether to escalate to LLM analysis.

        Args:
            signal: The signal to evaluate for escalation
            validation_result: Result of signal validation
            conflicts: List of detected conflicts
            market_data: Market data used for signal generation
            regime_duration: Duration of current market regime in periods

        Returns:
            Tuple[bool, float, str]: (should_escalate, escalation_probability, reasoning)
        """
        escalation_factors = []
        escalation_probability = 0.0
        reasoning_parts = []

        # 1. Check conflict count and severity
        conflict_factor, conflict_reasoning = self._evaluate_conflicts(conflicts)
        if conflict_factor > 0:
            escalation_factors.append(("conflicts", conflict_factor))
            reasoning_parts.append(conflict_reasoning)

        # 2. Check validation confidence
        validation_factor, validation_reasoning = self._evaluate_validation(validation_result)
        if validation_factor > 0:
            escalation_factors.append(("validation", validation_factor))
            reasoning_parts.append(validation_reasoning)

        # 3. Check regime uncertainty
        regime_factor, regime_reasoning = self._evaluate_regime_uncertainty(signal.regime, regime_duration)
        if regime_factor > 0:
            escalation_factors.append(("regime", regime_factor))
            reasoning_parts.append(regime_reasoning)

        # 4. Check market conditions
        market_factor, market_reasoning = self._evaluate_market_conditions(market_data)
        if market_factor > 0:
            escalation_factors.append(("market", market_factor))
            reasoning_parts.append(market_reasoning)

        # 5. Check rate limiting
        rate_limit_factor, rate_limit_reasoning = self._check_rate_limiting()
        if rate_limit_factor > 0:
            escalation_factors.append(("rate_limit", rate_limit_factor))
            reasoning_parts.append(rate_limit_reasoning)

        # Calculate overall escalation probability
        if escalation_factors:
            # Weight factors by importance
            weights = {
                "conflicts": 0.3,
                "validation": 0.25,
                "regime": 0.2,
                "market": 0.15,
                "rate_limit": 0.1,
            }

            weighted_probability = 0.0
            for factor_name, factor_value in escalation_factors:
                weight = weights.get(factor_name, 0.1)
                weighted_probability += factor_value * weight

            escalation_probability = min(weighted_probability, 1.0)

        # Combine reasoning
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No escalation triggers detected"

        # Make final decision
        should_escalate = (
            escalation_probability >= self.escalation_probability_threshold and
            len(self.recent_escalations) < self.max_escalations_per_hour
        )

        # Record escalation decision
        self._record_escalation_decision(
            signal, should_escalate, escalation_probability, reasoning, escalation_factors
        )

        return should_escalate, escalation_probability, reasoning

    def _evaluate_conflicts(self, conflicts: List[ConflictInfo]) -> Tuple[float, str]:
        """Evaluate conflicts for escalation triggers."""
        if not conflicts:
            return 0.0, ""

        # Count conflicts by severity
        high_severity = sum(1 for c in conflicts if c.severity == "HIGH")
        medium_severity = sum(1 for c in conflicts if c.severity == "MEDIUM")
        total_conflicts = len(conflicts)

        # Calculate conflict factor
        conflict_factor = 0.0

        if high_severity >= self.escalation_triggers["high_severity_conflicts"]:
            conflict_factor += 0.8

        if total_conflicts >= self.escalation_triggers["high_conflict_count"]:
            conflict_factor += 0.6

        # Add contribution from medium severity conflicts
        conflict_factor += min(medium_severity * 0.2, 0.4)

        # Cap at 1.0
        conflict_factor = min(conflict_factor, 1.0)

        reasoning = f"Conflicts: {total_conflicts} total, {high_severity} high severity, {medium_severity} medium"

        return conflict_factor, reasoning

    def _evaluate_validation(self, validation_result: ValidationResult) -> Tuple[float, str]:
        """Evaluate validation result for escalation triggers."""
        if validation_result.is_valid and validation_result.confidence >= 0.7:
            return 0.0, ""

        validation_factor = 0.0

        # Low confidence trigger
        if validation_result.confidence < self.escalation_triggers["low_validation_confidence"]:
            validation_factor += 0.7

        # Multiple issues trigger
        if len(validation_result.issues) >= 3:
            validation_factor += 0.5
        elif len(validation_result.issues) >= 2:
            validation_factor += 0.3

        # Large confidence adjustment trigger
        confidence_drop = validation_result.confidence - validation_result.adjusted_confidence
        if confidence_drop > 0.3:
            validation_factor += 0.4

        # Cap at 1.0
        validation_factor = min(validation_factor, 1.0)

        reasoning = f"Validation: confidence {validation_result.confidence:.2f}, "
        reasoning += f"{len(validation_result.issues)} issues, "
        reasoning += f"confidence drop {confidence_drop:.2f}"

        return validation_factor, reasoning

    def _evaluate_regime_uncertainty(self, regime: MarketRegime, regime_duration: int) -> Tuple[float, str]:
        """Evaluate regime uncertainty for escalation triggers."""
        regime_factor = 0.0

        # Uncertain regime trigger
        if regime == MarketRegime.UNCERTAIN:
            if regime_duration >= self.escalation_triggers["uncertain_regime_duration"]:
                regime_factor += 0.6
            else:
                regime_factor += 0.3

        # Volatile regime trigger
        elif regime == MarketRegime.VOLATILE and regime_duration >= 3:
            regime_factor += 0.4

        # Long duration in any regime (potential regime change)
        elif regime_duration >= 10:
            regime_factor += 0.2

        reasoning = f"Regime: {regime.value} for {regime_duration} periods"

        return regime_factor, reasoning

    def _evaluate_market_conditions(self, market_data: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate market conditions for escalation triggers."""
        if not self.escalation_triggers["extreme_market_conditions"]:
            return 0.0, ""

        market_factor = 0.0
        extreme_conditions = []

        # Check for extreme volatility
        if "volatility" in market_data:
            volatility = market_data["volatility"]
            if volatility > 0.8:  # High volatility threshold
                market_factor += 0.3
                extreme_conditions.append(f"high volatility ({volatility:.2f})")

        # Check for unusual volume
        if "volume_ratio" in market_data:
            volume_ratio = market_data["volume_ratio"]
            if volume_ratio > 3.0 or volume_ratio < 0.3:  # Unusual volume
                market_factor += 0.2
                extreme_conditions.append(f"unusual volume ({volume_ratio:.2f})")

        # Check for price gaps
        if "price_gap" in market_data:
            price_gap = market_data["price_gap"]
            if abs(price_gap) > 0.05:  # 5% gap
                market_factor += 0.4
                extreme_conditions.append(f"price gap ({price_gap:.2%})")

        # Check for rapid price changes
        if "price_change_rate" in market_data:
            price_change_rate = market_data["price_change_rate"]
            if abs(price_change_rate) > 0.1:  # 10% rapid change
                market_factor += 0.3
                extreme_conditions.append(f"rapid price change ({price_change_rate:.2%})")

        # Cap at 1.0
        market_factor = min(market_factor, 1.0)

        reasoning = f"Market conditions: {', '.join(extreme_conditions)}" if extreme_conditions else ""

        return market_factor, reasoning

    def _check_rate_limiting(self) -> Tuple[float, str]:
        """Check if escalation rate limiting should prevent escalation."""
        current_time = datetime.now()

        # Clean old escalations (older than 1 hour)
        self.recent_escalations = [
            esc_time for esc_time in self.recent_escalations
            if current_time - esc_time < timedelta(hours=1)
        ]

        # Check if we're approaching the limit
        escalations_this_hour = len(self.recent_escalations)

        if escalations_this_hour >= self.max_escalations_per_hour:
            return 1.0, f"Rate limit reached: {escalations_this_hour} escalations this hour"
        elif escalations_this_hour >= self.max_escalations_per_hour * 0.8:
            return 0.5, f"Approaching rate limit: {escalations_this_hour}/{self.max_escalations_per_hour}"
        else:
            return 0.0, ""

    def _record_escalation_decision(self,
                                   signal: Signal,
                                   should_escalate: bool,
                                   probability: float,
                                   reasoning: str,
                                   factors: List[Tuple[str, float]]):
        """Record the escalation decision for analysis."""
        decision_record = {
            "timestamp": datetime.now(),
            "signal_id": signal.id,
            "signal_type": signal.signal_type.value,
            "should_escalate": should_escalate,
            "probability": probability,
            "reasoning": reasoning,
            "factors": factors,
        }

        self.escalation_history.append(decision_record)

        # Track escalations for rate limiting
        if should_escalate:
            self.recent_escalations.append(datetime.now())

        # Keep history manageable
        if len(self.escalation_history) > 1000:
            self.escalation_history = self.escalation_history[-1000:]

    def get_escalation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about escalation decisions.

        Returns:
            Dict[str, Any]: Escalation statistics
        """
        if not self.escalation_history:
            return {}

        total_decisions = len(self.escalation_history)
        escalations = sum(1 for record in self.escalation_history if record["should_escalate"])

        # Calculate escalation rate
        escalation_rate = escalations / total_decisions

        # Calculate average probability
        avg_probability = np.mean([record["probability"] for record in self.escalation_history])

        # Analyze factors
        factor_counts = {}
        for record in self.escalation_history:
            for factor_name, factor_value in record["factors"]:
                if factor_name not in factor_counts:
                    factor_counts[factor_name] = {"count": 0, "total_value": 0.0}
                factor_counts[factor_name]["count"] += 1
                factor_counts[factor_name]["total_value"] += factor_value

        factor_averages = {
            name: data["total_value"] / data["count"]
            for name, data in factor_counts.items()
        }

        return {
            "total_decisions": total_decisions,
            "total_escalations": escalations,
            "escalation_rate": escalation_rate,
            "avg_probability": avg_probability,
            "factor_contributions": factor_averages,
            "recent_escalations_hour": len(self.recent_escalations),
        }

    def create_escalation_request(self, signal: Signal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured request for LLM escalation.

        Args:
            signal: The signal to escalate
            context: Additional context for the LLM

        Returns:
            Dict[str, Any]: Structured escalation request
        """
        return {
            "request_type": "signal_analysis",
            "signal": signal.to_dict(),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "analysis_requirements": [
                "Evaluate signal validity in current market context",
                "Assess risk factors and potential outcomes",
                "Provide recommendation with confidence level",
                "Suggest alternative approaches if signal is questionable",
            ],
        }
