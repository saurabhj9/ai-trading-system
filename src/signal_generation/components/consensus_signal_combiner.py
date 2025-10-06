"""
Consensus Signal Combiner component for the Local Signal Generation Framework.

This component combines multiple indicator signals using weighted voting to
produce a consensus signal with associated confidence.
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

from ..core import IndicatorScore, SignalType, SignalStrength


class ConsensusSignalCombiner:
    """
    Combines multiple indicator signals using weighted voting.

    This component implements a consensus mechanism that takes into account
    indicator weights, scores, and confidence levels to produce a final
    signal with an overall confidence score.
    """

    def __init__(self, config: Dict):
        """
        Initialize the consensus signal combiner.

        Args:
            config: Configuration dictionary with combining parameters
        """
        self.config = config

        # Voting thresholds
        self.consensus_threshold = config.get("consensus_threshold", 0.6)
        self.min_indicators = config.get("min_indicators", 3)

        # Weight adjustments based on indicator confidence
        self.confidence_weight_factor = config.get("confidence_weight_factor", 1.5)

        # Signal strength calculation parameters
        self.strength_thresholds = config.get("strength_thresholds", {
            "WEAK": 0.3,
            "MODERATE": 0.5,
            "STRONG": 0.7,
            "VERY_STRONG": 0.85,
        })

    def combine_signals(self, indicator_scores: List[IndicatorScore]) -> Tuple[SignalType, SignalStrength, float]:
        """
        Combine multiple indicator signals into a consensus signal.

        Args:
            indicator_scores: List of scored indicators

        Returns:
            Tuple[SignalType, SignalStrength, float]:
                (consensus signal type, signal strength, confidence)
        """
        if not indicator_scores:
            return SignalType.HOLD, SignalStrength.WEAK, 0.0

        if len(indicator_scores) < self.min_indicators:
            return SignalType.HOLD, SignalStrength.WEAK, 0.2

        # Calculate weighted votes
        weighted_votes = self._calculate_weighted_votes(indicator_scores)

        # Determine consensus signal
        consensus_signal = self._determine_consensus(weighted_votes)

        # Calculate overall confidence
        confidence = self._calculate_confidence(indicator_scores, consensus_signal)

        # Determine signal strength
        strength = self._determine_signal_strength(indicator_scores, consensus_signal)

        return consensus_signal, strength, confidence

    def _calculate_weighted_votes(self, indicator_scores: List[IndicatorScore]) -> Dict[SignalType, float]:
        """
        Calculate weighted votes for each signal type.

        Args:
            indicator_scores: List of scored indicators

        Returns:
            Dict[SignalType, float]: Weighted votes for each signal type
        """
        votes = {SignalType.BUY: 0.0, SignalType.SELL: 0.0, SignalType.HOLD: 0.0}

        for score in indicator_scores:
            # Adjust weight based on indicator confidence
            adjusted_weight = score.weight * (1 + (score.confidence - 0.5) * self.confidence_weight_factor)
            adjusted_weight = max(0.0, min(1.0, adjusted_weight))  # Clamp to [0, 1]

            # Apply score magnitude to weight
            effective_weight = adjusted_weight * abs(score.score)

            votes[score.signal_type] += effective_weight

        return votes

    def _determine_consensus(self, weighted_votes: Dict[SignalType, float]) -> SignalType:
        """
        Determine the consensus signal type from weighted votes.

        Args:
            weighted_votes: Weighted votes for each signal type

        Returns:
            SignalType: Consensus signal type
        """
        total_votes = sum(weighted_votes.values())

        if total_votes == 0:
            return SignalType.HOLD

        # Calculate percentages
        vote_percentages = {
            signal_type: votes / total_votes
            for signal_type, votes in weighted_votes.items()
        }

        # Find the signal with highest percentage
        max_signal = max(vote_percentages.items(), key=lambda x: x[1])

        # Check if it meets consensus threshold
        if max_signal[1] >= self.consensus_threshold:
            return max_signal[0]
        else:
            # No clear consensus, return HOLD
            return SignalType.HOLD

    def _calculate_confidence(self, indicator_scores: List[IndicatorScore], consensus_signal: SignalType) -> float:
        """
        Calculate overall confidence in the consensus signal.

        Args:
            indicator_scores: List of scored indicators
            consensus_signal: The determined consensus signal

        Returns:
            float: Overall confidence (0.0 to 1.0)
        """
        if not indicator_scores:
            return 0.0

        # Get indicators that agree with consensus
        agreeing_indicators = [
            score for score in indicator_scores
            if score.signal_type == consensus_signal
        ]

        if not agreeing_indicators:
            return 0.0

        # Calculate weighted average confidence
        total_weight = sum(score.weight for score in agreeing_indicators)
        if total_weight == 0:
            return 0.0

        weighted_confidence = sum(
            score.confidence * score.weight
            for score in agreeing_indicators
        ) / total_weight

        # Adjust based on agreement level
        agreement_ratio = len(agreeing_indicators) / len(indicator_scores)
        adjusted_confidence = weighted_confidence * agreement_ratio

        # Consider score magnitudes
        avg_score_magnitude = np.mean([abs(score.score) for score in agreeing_indicators])
        final_confidence = adjusted_confidence * (0.5 + 0.5 * avg_score_magnitude)

        return min(1.0, max(0.0, final_confidence))

    def _determine_signal_strength(self, indicator_scores: List[IndicatorScore], consensus_signal: SignalType) -> SignalStrength:
        """
        Determine the strength of the consensus signal.

        Args:
            indicator_scores: List of scored indicators
            consensus_signal: The determined consensus signal

        Returns:
            SignalStrength: Strength of the signal
        """
        # Get indicators that agree with consensus
        agreeing_indicators = [
            score for score in indicator_scores
            if score.signal_type == consensus_signal
        ]

        if not agreeing_indicators:
            return SignalStrength.WEAK

        # Calculate weighted average score magnitude
        total_weight = sum(score.weight for score in agreeing_indicators)
        if total_weight == 0:
            return SignalStrength.WEAK

        weighted_score = sum(
            abs(score.score) * score.weight
            for score in agreeing_indicators
        ) / total_weight

        # Determine strength based on thresholds
        if weighted_score >= self.strength_thresholds["VERY_STRONG"]:
            return SignalStrength.VERY_STRONG
        elif weighted_score >= self.strength_thresholds["STRONG"]:
            return SignalStrength.STRONG
        elif weighted_score >= self.strength_thresholds["MODERATE"]:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def get_signal_distribution(self, indicator_scores: List[IndicatorScore]) -> Dict[SignalType, float]:
        """
        Get the distribution of signals across all indicators.

        Args:
            indicator_scores: List of scored indicators

        Returns:
            Dict[SignalType, float]: Percentage distribution of signals
        """
        if not indicator_scores:
            return {SignalType.BUY: 0.0, SignalType.SELL: 0.0, SignalType.HOLD: 1.0}

        signal_counts = Counter(score.signal_type for score in indicator_scores)
        total_indicators = len(indicator_scores)

        return {
            signal_type: count / total_indicators
            for signal_type, count in signal_counts.items()
        }

    def get_disagreement_level(self, indicator_scores: List[IndicatorScore]) -> float:
        """
        Calculate the level of disagreement among indicators.

        Args:
            indicator_scores: List of scored indicators

        Returns:
            float: Disagreement level (0.0 to 1.0, where 1.0 is maximum disagreement)
        """
        if len(indicator_scores) < 2:
            return 0.0

        signal_distribution = self.get_signal_distribution(indicator_scores)

        # Calculate entropy as a measure of disagreement
        entropy = 0.0
        for percentage in signal_distribution.values():
            if percentage > 0:
                entropy -= percentage * np.log2(percentage)

        # Normalize entropy (maximum entropy for 3 signals is log2(3) ≈ 1.585)
        max_entropy = np.log2(3)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy
