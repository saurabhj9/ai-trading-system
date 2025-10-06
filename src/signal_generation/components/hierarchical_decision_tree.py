"""
Hierarchical Decision Tree component for the Local Signal Generation Framework.

This component implements a transparent decision-making workflow that processes
signals through multiple levels of analysis, providing clear reasoning for
each decision.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core import Signal, SignalType, SignalStrength, MarketRegime, IndicatorScore


class DecisionLevel(Enum):
    """Enumeration for decision tree levels."""
    MARKET_REGIME = "MARKET_REGIME"
    INDICATOR_ANALYSIS = "INDICATOR_ANALYSIS"
    SIGNAL_CONSENSUS = "SIGNAL_CONSENSUS"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    FINAL_DECISION = "FINAL_DECISION"


@dataclass
class DecisionNode:
    """
    Represents a node in the hierarchical decision tree.

    Attributes:
        level: Decision level of this node
        decision: Decision made at this node
        confidence: Confidence in the decision
        reasoning: Human-readable explanation
        supporting_data: Data supporting the decision
        child_nodes: Child nodes in the tree
    """
    level: DecisionLevel
    decision: SignalType
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    child_nodes: List['DecisionNode'] = field(default_factory=list)


class HierarchicalDecisionTree:
    """
    Implements a hierarchical decision-making process for signal generation.

    This component processes signals through multiple levels of analysis,
    providing transparency and explainability in the decision-making process.
    """

    def __init__(self, config: Dict):
        """
        Initialize the hierarchical decision tree.

        Args:
            config: Configuration dictionary with decision tree parameters
        """
        self.config = config

        # Decision thresholds for each level
        self.thresholds = config.get("decision_thresholds", {
            DecisionLevel.MARKET_REGIME: 0.7,
            DecisionLevel.INDICATOR_ANALYSIS: 0.6,
            DecisionLevel.SIGNAL_CONSENSUS: 0.65,
            DecisionLevel.RISK_ASSESSMENT: 0.5,
            DecisionLevel.FINAL_DECISION: 0.6,
        })

        # Risk parameters
        self.max_risk_level = config.get("max_risk_level", 0.7)
        self.volatility_threshold = config.get("volatility_threshold", 0.8)

        # Decision tree structure
        self.root_node: Optional[DecisionNode] = None
        self.decision_history: List[DecisionNode] = []

    def make_decision(self,
                     market_regime: MarketRegime,
                     indicator_scores: List[IndicatorScore],
                     consensus_signal: SignalType,
                     consensus_strength: SignalStrength,
                     consensus_confidence: float,
                     market_data: Dict[str, Any]) -> DecisionNode:
        """
        Make a hierarchical decision based on all available information.

        Args:
            market_regime: Current market regime
            indicator_scores: List of scored indicators
            consensus_signal: Consensus signal from indicators
            consensus_strength: Strength of the consensus signal
            consensus_confidence: Confidence in the consensus signal
            market_data: Additional market data

        Returns:
            DecisionNode: Root node of the decision tree
        """
        # Level 1: Market Regime Assessment
        regime_node = self._assess_market_regime(market_regime, market_data)

        # Level 2: Indicator Analysis
        indicator_node = self._analyze_indicators(indicator_scores, market_regime)
        regime_node.child_nodes.append(indicator_node)

        # Level 3: Signal Consensus Evaluation
        consensus_node = self._evaluate_consensus(
            consensus_signal, consensus_strength, consensus_confidence, indicator_scores
        )
        indicator_node.child_nodes.append(consensus_node)

        # Level 4: Risk Assessment
        risk_node = self._assess_risk(market_regime, indicator_scores, market_data)
        consensus_node.child_nodes.append(risk_node)

        # Level 5: Final Decision
        final_node = self._make_final_decision(
            regime_node, indicator_node, consensus_node, risk_node
        )
        risk_node.child_nodes.append(final_node)

        # Store the decision tree
        self.root_node = regime_node
        self.decision_history.append(regime_node)

        # Keep history manageable
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)

        return regime_node

    def _assess_market_regime(self, regime: MarketRegime, market_data: Dict[str, Any]) -> DecisionNode:
        """Assess the market regime and its implications."""
        regime_preferences = {
            MarketRegime.TRENDING_UP: (SignalType.BUY, 0.8),
            MarketRegime.TRENDING_DOWN: (SignalType.SELL, 0.8),
            MarketRegime.RANGING: (SignalType.HOLD, 0.6),
            MarketRegime.VOLATILE: (SignalType.HOLD, 0.4),
            MarketRegime.UNCERTAIN: (SignalType.HOLD, 0.3),
        }

        preferred_signal, confidence = regime_preferences.get(regime, (SignalType.HOLD, 0.3))

        reasoning = f"Market regime is {regime.value}. "

        if regime == MarketRegime.TRENDING_UP:
            reasoning += "Strong uptrend suggests buying opportunities."
        elif regime == MarketRegime.TRENDING_DOWN:
            reasoning += "Strong downtrend suggests selling opportunities."
        elif regime == MarketRegime.RANGING:
            reasoning += "Ranging market suggests holding or range-bound strategies."
        elif regime == MarketRegime.VOLATILE:
            reasoning += "High volatility suggests caution and holding positions."
        else:
            reasoning += "Uncertain market conditions suggest holding."

        return DecisionNode(
            level=DecisionLevel.MARKET_REGIME,
            decision=preferred_signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "regime": regime.value,
                "regime_confidence": getattr(regime, 'confidence', 0.5),
            }
        )

    def _analyze_indicators(self, indicator_scores: List[IndicatorScore], regime: MarketRegime) -> DecisionNode:
        """Analyze individual indicators and their collective implications."""
        if not indicator_scores:
            return DecisionNode(
                level=DecisionLevel.INDICATOR_ANALYSIS,
                decision=SignalType.HOLD,
                confidence=0.0,
                reasoning="No indicator data available for analysis.",
                supporting_data={}
            )

        # Count signal types
        signal_counts = {}
        total_weight = 0
        weighted_scores = {SignalType.BUY: 0.0, SignalType.SELL: 0.0, SignalType.HOLD: 0.0}

        for score in indicator_scores:
            signal_counts[score.signal_type] = signal_counts.get(score.signal_type, 0) + 1
            total_weight += score.weight
            weighted_scores[score.signal_type] += score.score * score.weight

        # Determine dominant signal
        dominant_signal = max(weighted_scores.items(), key=lambda x: x[1])[0]
        confidence = min(weighted_scores[dominant_signal] / total_weight if total_weight > 0 else 0, 1.0)

        # Create reasoning
        top_indicators = sorted(indicator_scores, key=lambda x: abs(x.score), reverse=True)[:3]
        indicator_names = [ind.name for ind in top_indicators]

        reasoning = f"Indicator analysis shows preference for {dominant_signal.value}. "
        reasoning += f"Top contributing indicators: {', '.join(indicator_names)}. "
        reasoning += f"Signal distribution: {signal_counts}."

        return DecisionNode(
            level=DecisionLevel.INDICATOR_ANALYSIS,
            decision=dominant_signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "signal_counts": signal_counts,
                "weighted_scores": weighted_scores,
                "top_indicators": indicator_names,
                "num_indicators": len(indicator_scores),
            }
        )

    def _evaluate_consensus(self,
                           consensus_signal: SignalType,
                           consensus_strength: SignalStrength,
                           consensus_confidence: float,
                           indicator_scores: List[IndicatorScore]) -> DecisionNode:
        """Evaluate the consensus signal and its reliability."""

        # Adjust confidence based on signal strength
        strength_multipliers = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.7,
            SignalStrength.STRONG: 0.9,
            SignalStrength.VERY_STRONG: 1.0,
        }

        adjusted_confidence = consensus_confidence * strength_multipliers.get(consensus_strength, 0.5)

        reasoning = f"Consensus signal is {consensus_signal.value} with {consensus_strength.value} strength. "
        reasoning += f"Base confidence: {consensus_confidence:.2f}, adjusted for strength: {adjusted_confidence:.2f}. "

        # Add context about agreement level
        agreement_ratio = len([s for s in indicator_scores if s.signal_type == consensus_signal]) / len(indicator_scores)
        reasoning += f"Agreement level: {agreement_ratio:.2f}."

        return DecisionNode(
            level=DecisionLevel.SIGNAL_CONSENSUS,
            decision=consensus_signal,
            confidence=adjusted_confidence,
            reasoning=reasoning,
            supporting_data={
                "consensus_signal": consensus_signal.value,
                "consensus_strength": consensus_strength.value,
                "base_confidence": consensus_confidence,
                "adjusted_confidence": adjusted_confidence,
                "agreement_ratio": agreement_ratio,
            }
        )

    def _assess_risk(self,
                    market_regime: MarketRegime,
                    indicator_scores: List[IndicatorScore],
                    market_data: Dict[str, Any]) -> DecisionNode:
        """Assess risk factors and their impact on the decision."""
        risk_factors = []
        risk_level = 0.0

        # Regime-based risk
        if market_regime == MarketRegime.VOLATILE:
            risk_factors.append("High volatility regime")
            risk_level += 0.3
        elif market_regime == MarketRegime.UNCERTAIN:
            risk_factors.append("Uncertain market regime")
            risk_level += 0.2

        # Indicator disagreement risk
        signal_types = set(score.signal_type for score in indicator_scores)
        if len(signal_types) > 2:
            risk_factors.append("High indicator disagreement")
            risk_level += 0.2

        # Low confidence risk
        avg_confidence = sum(score.confidence for score in indicator_scores) / len(indicator_scores) if indicator_scores else 0
        if avg_confidence < 0.5:
            risk_factors.append("Low indicator confidence")
            risk_level += 0.2

        # Cap risk level
        risk_level = min(risk_level, 1.0)

        # Determine risk-adjusted decision
        if risk_level > self.max_risk_level:
            decision = SignalType.HOLD
            reasoning = f"High risk level ({risk_level:.2f}) detected. Risk factors: {', '.join(risk_factors)}. "
            reasoning += "Recommending HOLD due to elevated risk."
            confidence = 0.7
        else:
            decision = SignalType.HOLD  # Default, will be overridden by final decision
            reasoning = f"Risk level ({risk_level:.2f}) is acceptable. "
            if risk_factors:
                reasoning += "Identified risk factors: " + ", ".join(risk_factors) + ". "
            else:
                reasoning += "No significant risk factors identified. "
            confidence = 0.8

        return DecisionNode(
            level=DecisionLevel.RISK_ASSESSMENT,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "avg_indicator_confidence": avg_confidence,
            }
        )

    def _make_final_decision(self,
                           regime_node: DecisionNode,
                           indicator_node: DecisionNode,
                           consensus_node: DecisionNode,
                           risk_node: DecisionNode) -> DecisionNode:
        """Make the final decision based on all previous levels."""

        # If risk is too high, always hold
        if risk_node.decision == SignalType.HOLD and risk_node.confidence > 0.7:
            return DecisionNode(
                level=DecisionLevel.FINAL_DECISION,
                decision=SignalType.HOLD,
                confidence=risk_node.confidence,
                reasoning=f"Final decision overridden by risk assessment: {risk_node.reasoning}",
                supporting_data={"override_reason": "HIGH_RISK"}
            )

        # Weight each level's decision
        weights = {
            DecisionLevel.MARKET_REGIME: 0.25,
            DecisionLevel.INDICATOR_ANALYSIS: 0.25,
            DecisionLevel.SIGNAL_CONSENSUS: 0.35,
            DecisionLevel.RISK_ASSESSMENT: 0.15,
        }

        # Calculate weighted scores for each signal type
        signal_scores = {SignalType.BUY: 0.0, SignalType.SELL: 0.0, SignalType.HOLD: 0.0}

        for node, weight in [(regime_node, weights[DecisionLevel.MARKET_REGIME]),
                            (indicator_node, weights[DecisionLevel.INDICATOR_ANALYSIS]),
                            (consensus_node, weights[DecisionLevel.SIGNAL_CONSENSUS])]:
            signal_scores[node.decision] += weight * node.confidence

        # Add risk penalty for non-HOLD signals if risk is elevated
        if risk_node.supporting_data.get("risk_level", 0) > 0.5:
            risk_penalty = risk_node.supporting_data.get("risk_level", 0) * 0.3
            signal_scores[SignalType.BUY] *= (1 - risk_penalty)
            signal_scores[SignalType.SELL] *= (1 - risk_penalty)
            signal_scores[SignalType.HOLD] += risk_penalty

        # Determine final decision
        final_signal = max(signal_scores.items(), key=lambda x: x[1])[0]
        final_confidence = signal_scores[final_signal]

        reasoning = f"Final decision: {final_signal.value} (confidence: {final_confidence:.2f}). "
        reasoning += f"Weighted scores - BUY: {signal_scores[SignalType.BUY]:.2f}, "
        reasoning += f"SELL: {signal_scores[SignalType.SELL]:.2f}, HOLD: {signal_scores[SignalType.HOLD]:.2f}. "
        reasoning += f"Regime: {regime_node.decision.value}, Indicators: {indicator_node.decision.value}, "
        reasoning += f"Consensus: {consensus_node.decision.value}, Risk: {risk_node.supporting_data.get('risk_level', 0):.2f}"

        return DecisionNode(
            level=DecisionLevel.FINAL_DECISION,
            decision=final_signal,
            confidence=final_confidence,
            reasoning=reasoning,
            supporting_data={
                "weighted_scores": signal_scores,
                "regime_decision": regime_node.decision.value,
                "indicator_decision": indicator_node.decision.value,
                "consensus_decision": consensus_node.decision.value,
                "risk_level": risk_node.supporting_data.get("risk_level", 0),
            }
        )

    def get_decision_tree(self) -> Optional[DecisionNode]:
        """Get the most recent decision tree."""
        return self.root_node

    def get_decision_path(self) -> List[DecisionNode]:
        """Get the decision path from root to final decision."""
        if not self.root_node:
            return []

        path = [self.root_node]
        current = self.root_node

        while current.child_nodes:
            current = current.child_nodes[0]
            path.append(current)

        return path

    def export_decision_tree(self) -> Dict[str, Any]:
        """Export the decision tree as a dictionary for serialization."""
        if not self.root_node:
            return {}

        def node_to_dict(node: DecisionNode) -> Dict[str, Any]:
            return {
                "level": node.level.value,
                "decision": node.decision.value,
                "confidence": node.confidence,
                "reasoning": node.reasoning,
                "supporting_data": node.supporting_data,
                "child_nodes": [node_to_dict(child) for child in node.child_nodes],
            }

        return node_to_dict(self.root_node)
