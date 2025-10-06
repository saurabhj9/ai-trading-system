"""
Components for the Local Signal Generation Framework.

This module contains the individual components that work together to generate
trading signals without relying on LLM calls.
"""

from .market_regime_detector import MarketRegimeDetector
from .indicator_scorer import IndicatorScorer
from .consensus_signal_combiner import ConsensusSignalCombiner
from .hierarchical_decision_tree import HierarchicalDecisionTree
from .signal_validator import SignalValidator
from .conflict_detector import ConflictDetector
from .llm_escallation_logic import LLMEscalationLogic

__all__ = [
    "MarketRegimeDetector",
    "IndicatorScorer",
    "ConsensusSignalCombiner",
    "HierarchicalDecisionTree",
    "SignalValidator",
    "ConflictDetector",
    "LLMEscalationLogic",
]
