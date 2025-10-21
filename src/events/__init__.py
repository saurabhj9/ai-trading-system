"""
Event-driven trigger detection system for real-time market analysis.

This package provides comprehensive trigger detection capabilities for various
market events including technical patterns, volatility changes, trend shifts,
and signal conflicts. It enables automated, real-time response to market
conditions without requiring manual analysis requests.
"""

from .trigger_detector import BaseTriggerDetector, TriggerEvent, TriggerType, TriggerSeverity
from .event_bus import EventBus, EventSubscriber
from .cooldown_manager import CooldownManager, DecisionTTL

__all__ = [
    "BaseTriggerDetector",
    "TriggerEvent",
    "TriggerType",
    "TriggerSeverity",
    "EventBus",
    "EventSubscriber",
    "CooldownManager",
    "DecisionTTL"
]
