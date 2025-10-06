"""
Conflict Detector component for the Local Signal Generation Framework.

This component identifies conflicts between signals and provides strategies
for resolving them to ensure consistent and reliable trading decisions.
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from ..core import Signal, ConflictInfo, SignalType, SignalStrength, IndicatorScore


class ConflictDetector:
    """
    Detects and analyzes conflicts between signals and indicators.

    This component identifies various types of conflicts including direction
    conflicts, strength conflicts, regime conflicts, and temporal conflicts.
    """

    def __init__(self, config: Dict):
        """
        Initialize the conflict detector.

        Args:
            config: Configuration dictionary with conflict detection parameters
        """
        self.config = config

        # Conflict detection thresholds
        self.direction_conflict_threshold = config.get("direction_conflict_threshold", 0.4)
        self.strength_conflict_threshold = config.get("strength_conflict_threshold", 2)
        self.regime_conflict_threshold = config.get("regime_conflict_threshold", 0.6)

        # Conflict severity weights
        self.severity_weights = config.get("severity_weights", {
            "DIRECTION_CONFLICT": 0.8,
            "STRENGTH_CONFLICT": 0.5,
            "REGIME_CONFLICT": 0.7,
            "TEMPORAL_CONFLICT": 0.4,
            "INDICATOR_CONFLICT": 0.6,
        })

        # Resolution strategies
        self.resolution_strategies = {
            "DIRECTION_CONFLICT": "prefer_higher_confidence",
            "STRENGTH_CONFLICT": "prefer_consensus",
            "REGIME_CONFLICT": "prefer_regime_compatible",
            "TEMPORAL_CONFLICT": "prefer_most_recent",
            "INDICATOR_CONFLICT": "recalculate_weights",
        }

        # Conflict history
        self.conflict_history: List[ConflictInfo] = []

    def detect_conflicts(self,
                        primary_signal: Signal,
                        indicator_scores: List[IndicatorScore],
                        recent_signals: Optional[List[Signal]] = None) -> List[ConflictInfo]:
        """
        Detect conflicts between the primary signal and other signals/indicators.

        Args:
            primary_signal: The primary signal to check for conflicts
            indicator_scores: List of indicator scores that contributed to the signal
            recent_signals: Optional list of recent signals for temporal conflict detection

        Returns:
            List[ConflictInfo]: List of detected conflicts
        """
        conflicts = []

        # 1. Detect direction conflicts with indicators
        direction_conflicts = self._detect_direction_conflicts(primary_signal, indicator_scores)
        conflicts.extend(direction_conflicts)

        # 2. Detect strength conflicts
        strength_conflicts = self._detect_strength_conflicts(primary_signal, indicator_scores)
        conflicts.extend(strength_conflicts)

        # 3. Detect regime conflicts
        regime_conflicts = self._detect_regime_conflicts(primary_signal)
        conflicts.extend(regime_conflicts)

        # 4. Detect temporal conflicts if recent signals are provided
        if recent_signals:
            temporal_conflicts = self._detect_temporal_conflicts(primary_signal, recent_signals)
            conflicts.extend(temporal_conflicts)

        # 5. Detect indicator conflicts
        indicator_conflicts = self._detect_indicator_conflicts(indicator_scores)
        conflicts.extend(indicator_conflicts)

        # Store conflicts in history
        self.conflict_history.extend(conflicts)

        # Keep history manageable
        if len(self.conflict_history) > 500:
            self.conflict_history = self.conflict_history[-500:]

        return conflicts

    def _detect_direction_conflicts(self, signal: Signal, indicator_scores: List[IndicatorScore]) -> List[ConflictInfo]:
        """Detect conflicts between signal direction and indicator directions."""
        if not indicator_scores:
            return []

        # Count indicators suggesting each direction
        direction_counts = defaultdict(int)
        conflicting_indicators = []

        for score in indicator_scores:
            direction_counts[score.signal_type] += 1
            if score.signal_type != signal.signal_type:
                conflicting_indicators.append(score.name)

        # Calculate agreement ratio
        total_indicators = len(indicator_scores)
        agreement_count = direction_counts[signal.signal_type]
        agreement_ratio = agreement_count / total_indicators

        # Check for direction conflict
        if agreement_ratio < self.direction_conflict_threshold:
            severity = self._calculate_severity("DIRECTION_CONFLICT", 1.0 - agreement_ratio)

            description = f"Signal direction ({signal.signal_type.value}) conflicts with "
            description += f"{total_indicators - agreement_count} out of {total_indicators} indicators. "
            description += f"Conflicting indicators: {', '.join(conflicting_indicators)}"

            return [ConflictInfo(
                conflict_type="DIRECTION_CONFLICT",
                conflicting_signals=[signal.id],
                severity=severity,
                description=description,
                resolution_strategy=self.resolution_strategies["DIRECTION_CONFLICT"]
            )]

        return []

    def _detect_strength_conflicts(self, signal: Signal, indicator_scores: List[IndicatorScore]) -> List[ConflictInfo]:
        """Detect conflicts between signal strength and indicator strength."""
        if not indicator_scores:
            return []

        # Calculate average indicator strength
        indicator_strengths = []
        for score in indicator_scores:
            if score.signal_type == signal.signal_type:
                indicator_strengths.append(abs(score.score))

        if not indicator_strengths:
            return []

        avg_indicator_strength = np.mean(indicator_strengths)

        # Map signal strength to numeric value
        signal_strength_values = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0,
        }

        signal_strength_value = signal_strength_values.get(signal.strength, 0.5)

        # Check for strength conflict
        strength_ratio = avg_indicator_strength / signal_strength_value if signal_strength_value > 0 else 1.0

        if strength_ratio < (1.0 / self.strength_conflict_threshold) or strength_ratio > self.strength_conflict_threshold:
            severity = self._calculate_severity("STRENGTH_CONFLICT", abs(1.0 - strength_ratio))

            if strength_ratio < 1.0:
                description = f"Signal strength ({signal.strength.value}) is stronger than "
                description += f"supporting indicators (avg: {avg_indicator_strength:.2f})"
            else:
                description = f"Signal strength ({signal.strength.value}) is weaker than "
                description += f"supporting indicators (avg: {avg_indicator_strength:.2f})"

            return [ConflictInfo(
                conflict_type="STRENGTH_CONFLICT",
                conflicting_signals=[signal.id],
                severity=severity,
                description=description,
                resolution_strategy=self.resolution_strategies["STRENGTH_CONFLICT"]
            )]

        return []

    def _detect_regime_conflicts(self, signal: Signal) -> List[ConflictInfo]:
        """Detect conflicts between signal and market regime."""
        # Define regime-signal compatibility
        incompatible_combinations = {
            "TRENDING_UP": ["SELL"],
            "TRENDING_DOWN": ["BUY"],
            "VOLATILE": ["BUY", "SELL"],
            "UNCERTAIN": ["BUY", "SELL"],
        }

        incompatible_signals = incompatible_combinations.get(signal.regime.value, [])

        if signal.signal_type.value in incompatible_signals:
            severity = self._calculate_severity("REGIME_CONFLICT", self.regime_conflict_threshold)

            description = f"Signal type ({signal.signal_type.value}) is incompatible with "
            description += f"current market regime ({signal.regime.value})"

            return [ConflictInfo(
                conflict_type="REGIME_CONFLICT",
                conflicting_signals=[signal.id],
                severity=severity,
                description=description,
                resolution_strategy=self.resolution_strategies["REGIME_CONFLICT"]
            )]

        return []

    def _detect_temporal_conflicts(self, signal: Signal, recent_signals: List[Signal]) -> List[ConflictInfo]:
        """Detect conflicts with recent signals."""
        if not recent_signals:
            return []

        # Look for recent signals with opposite directions
        conflicting_recent = []
        for recent_signal in recent_signals[-5:]:  # Check last 5 signals
            if recent_signal.signal_type != signal.signal_type:
                conflicting_recent.append(recent_signal.id)

        if conflicting_recent:
            severity = self._calculate_severity("TEMPORAL_CONFLICT", 0.5)

            description = f"Signal conflicts with {len(conflicting_recent)} recent signals. "
            description += f"Recent signal IDs: {', '.join(conflicting_recent[-3:])}"

            return [ConflictInfo(
                conflict_type="TEMPORAL_CONFLICT",
                conflicting_signals=[signal.id] + conflicting_recent,
                severity=severity,
                description=description,
                resolution_strategy=self.resolution_strategies["TEMPORAL_CONFLICT"]
            )]

        return []

    def _detect_indicator_conflicts(self, indicator_scores: List[IndicatorScore]) -> List[ConflictInfo]:
        """Detect conflicts between indicators themselves."""
        if len(indicator_scores) < 2:
            return []

        # Group indicators by type
        indicator_groups = defaultdict(list)
        for score in indicator_scores:
            # Simple categorization
            if score.name in ['RSI', 'STOCH', 'WILLR', 'MFI']:
                indicator_groups['oscillator'].append(score)
            elif score.name in ['MACD', 'ADX', 'PLUS_DI', 'MINUS_DI']:
                indicator_groups['trend'].append(score)
            elif score.name in ['OBV', 'AD']:
                indicator_groups['volume'].append(score)

        conflicts = []

        # Check for conflicts within each group
        for group_name, group_scores in indicator_groups.items():
            if len(group_scores) < 2:
                continue

            # Check for opposing signals within the group
            signal_types = [score.signal_type for score in group_scores]
            unique_signals = set(signal_types)

            if len(unique_signals) > 1 and (SignalType.BUY in unique_signals and SignalType.SELL in unique_signals):
                severity = self._calculate_severity("INDICATOR_CONFLICT", 0.6)

                conflicting_indicators = [score.name for score in group_scores]
                description = f"Conflicting signals within {group_name} indicators: "
                description += f"{', '.join(conflicting_indicators)}"

                conflicts.append(ConflictInfo(
                    conflict_type="INDICATOR_CONFLICT",
                    conflicting_signals=[],  # No signal IDs for indicator conflicts
                    severity=severity,
                    description=description,
                    resolution_strategy=self.resolution_strategies["INDICATOR_CONFLICT"]
                ))

        return conflicts

    def _calculate_severity(self, conflict_type: str, intensity: float) -> str:
        """Calculate conflict severity based on type and intensity."""
        base_weight = self.severity_weights.get(conflict_type, 0.5)
        weighted_intensity = intensity * base_weight

        if weighted_intensity >= 0.8:
            return "HIGH"
        elif weighted_intensity >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def resolve_conflicts(self, conflicts: List[ConflictInfo]) -> Dict[str, any]:
        """
        Provide resolution strategies for detected conflicts.

        Args:
            conflicts: List of detected conflicts

        Returns:
            Dict[str, any]: Resolution recommendations
        """
        if not conflicts:
            return {"has_conflicts": False, "resolutions": []}

        resolutions = []

        for conflict in conflicts:
            resolution = {
                "conflict_type": conflict.conflict_type,
                "severity": conflict.severity,
                "strategy": conflict.resolution_strategy,
                "description": conflict.description,
                "action": self._get_resolution_action(conflict)
            }
            resolutions.append(resolution)

        # Sort by severity
        resolutions.sort(key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x["severity"]], reverse=True)

        return {
            "has_conflicts": True,
            "num_conflicts": len(conflicts),
            "resolutions": resolutions,
            "primary_resolution": resolutions[0] if resolutions else None
        }

    def _get_resolution_action(self, conflict: ConflictInfo) -> str:
        """Get specific action for resolving a conflict."""
        actions = {
            "prefer_higher_confidence": "Prefer the signal or indicator with higher confidence score",
            "prefer_consensus": "Use the consensus signal from multiple indicators",
            "prefer_regime_compatible": "Adjust signal to be compatible with current market regime",
            "prefer_most_recent": "Give preference to the most recent signal",
            "recalculate_weights": "Recalculate indicator weights based on current conditions",
        }

        return actions.get(conflict.resolution_strategy, "Manual review required")

    def get_conflict_statistics(self) -> Dict[str, any]:
        """
        Get statistics about detected conflicts.

        Returns:
            Dict[str, any]: Conflict statistics
        """
        if not self.conflict_history:
            return {}

        total_conflicts = len(self.conflict_history)

        # Count by type
        conflict_types = defaultdict(int)
        severity_counts = defaultdict(int)

        for conflict in self.conflict_history:
            conflict_types[conflict.conflict_type] += 1
            severity_counts[conflict.severity] += 1

        # Calculate rates
        conflict_rates = {
            conflict_type: count / total_conflicts
            for conflict_type, count in conflict_types.items()
        }

        return {
            "total_conflicts": total_conflicts,
            "conflict_types": dict(conflict_types),
            "conflict_rates": conflict_rates,
            "severity_distribution": dict(severity_counts),
            "high_severity_ratio": severity_counts.get("HIGH", 0) / total_conflicts,
        }
