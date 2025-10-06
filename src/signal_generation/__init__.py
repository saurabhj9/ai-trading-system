"""
Local Signal Generation Framework.

This module provides a rule-based system for generating trading signals without
relying on LLM calls. It includes market regime detection, indicator scoring,
consensus signal combination, hierarchical decision trees, signal validation,
conflict detection, and LLM escalation logic.

The framework is designed to be:
- Fast: <100ms signal generation latency
- Cost-effective: >70% reduction in LLM costs
- Accurate: ≥85% signal accuracy
- Transparent: Clear decision-making process
"""

from .core import (
    Signal,
    SignalType,
    SignalStrength,
    SignalGenerator as BaseSignalGenerator,
)

from .signal_generator import LocalSignalGenerator

from .components import (
    MarketRegimeDetector,
    IndicatorScorer,
    ConsensusSignalCombiner,
    HierarchicalDecisionTree,
    SignalValidator,
    ConflictDetector,
    LLMEscalationLogic,
)

__all__ = [
    # Core classes
    "Signal",
    "SignalType",
    "SignalStrength",
    "BaseSignalGenerator",
    # Main signal generator
    "LocalSignalGenerator",
    # Component classes
    "MarketRegimeDetector",
    "IndicatorScorer",
    "ConsensusSignalCombiner",
    "HierarchicalDecisionTree",
    "SignalValidator",
    "ConflictDetector",
    "LLMEscalationLogic",
]

__version__ = "1.0.0"
