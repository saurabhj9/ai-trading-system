"""
Base trigger detection framework for event-driven analysis.

This module provides the foundation for detecting various types of market
triggers that can automatically initiate analysis or trading actions.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers that can be detected."""
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    TREND = "trend"
    CONFLICT = "conflict"
    REGIME = "regime"
    PRICE = "price"
    VOLUME = "volume"


class TriggerSeverity(Enum):
    """Severity levels for triggers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TriggerEvent:
    """
    Represents a trigger detection event.
    """
    symbol: str
    trigger_type: TriggerType
    severity: TriggerSeverity
    timestamp: datetime
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger event to dictionary."""
        return {
            'symbol': self.symbol,
            'trigger_type': self.trigger_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'data': self.data,
            'confidence': self.confidence,
            'ttl': self.ttl.total_seconds() if self.ttl else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerEvent':
        """Create trigger event from dictionary."""
        return cls(
            symbol=data['symbol'],
            trigger_type=TriggerType(data['trigger_type']),
            severity=TriggerSeverity(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            description=data['description'],
            data=data.get('data', {}),
            confidence=data.get('confidence', 0.0),
            ttl=timedelta(seconds=data['ttl']) if data.get('ttl') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class TriggerConfig:
    """Configuration for trigger detection."""
    enabled: bool = True
    min_confidence: float = 0.7
    cooldown_seconds: int = 300  # 5 minutes default
    sensitivity: float = 0.5  # 0.0 to 1.0
    max_triggers_per_minute: int = 5
    required_data_points: int = 10
    custom_params: Dict[str, Any] = field(default_factory=dict)


class TriggerState:
    """Maintains state for trigger detection across multiple time periods."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_trigger_time: Optional[datetime] = None
        self.trigger_count = 0
        self.last_values: Dict[str, float] = {}
        self.trigger_history: List[TriggerEvent] = []
        self.state_data: Dict[str, Any] = {}

    def update_last_trigger(self):
        """Update the last trigger time."""
        self.last_trigger_time = datetime.now()
        self.trigger_count += 1

    def should_trigger(self, config: TriggerConfig) -> bool:
        """Check if trigger should fire based on cooldown."""
        if not config.enabled:
            return False

        if self.last_trigger_time is None:
            return True

        time_since_last = datetime.now() - self.last_trigger_time
        return time_since_last.total_seconds() >= config.cooldown_seconds

    def add_trigger_event(self, event: TriggerEvent):
        """Add a trigger event to history."""
        self.trigger_history.append(event)
        self.update_last_trigger()

        # Keep only last 100 events to prevent memory issues
        if len(self.trigger_history) > 100:
            self.trigger_history = self.trigger_history[-100:]

    def get_recent_triggers(self, minutes: int = 60) -> List[TriggerEvent]:
        """Get triggers from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [event for event in self.trigger_history if event.timestamp >= cutoff_time]


class BaseTriggerDetector(ABC):
    """
    Abstract base class for trigger detectors.

    Each detector type implements specific logic for identifying
    market conditions that should trigger analysis or actions.
    """

    def __init__(self, trigger_type: TriggerType, config: Optional[TriggerConfig] = None):
        """
        Initialize the trigger detector.

        Args:
            trigger_type: Type of trigger this detector handles
            config: Configuration for the trigger detector
        """
        self.trigger_type = trigger_type
        self.config = config or TriggerConfig()
        self.states: Dict[str, TriggerState] = {}
        self.is_running = False

        # Statistics
        self.stats = {
            'triggers_detected': 0,
            'false_positives': 0,
            'last_detection': None,
            'symbols_monitored': set()
        }

        logger.info(f"Initialized {trigger_type.value} trigger detector")

    @abstractmethod
    async def detect_triggers(self, symbol: str, market_data: Dict[str, Any]) -> List[TriggerEvent]:
        """
        Detect triggers for a symbol based on market data.

        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data including price, indicators, etc.

        Returns:
            List of trigger events detected
        """
        pass

    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """
        Get the list of required data fields for this detector.

        Returns:
            List of field names required for trigger detection
        """
        pass

    def get_state(self, symbol: str) -> TriggerState:
        """Get or create trigger state for a symbol."""
        if symbol not in self.states:
            self.states[symbol] = TriggerState(symbol)
            self.stats['symbols_monitored'].add(symbol)
        return self.states[symbol]

    def validate_trigger_confidence(self, confidence: float) -> bool:
        """Check if trigger confidence meets minimum threshold."""
        return confidence >= self.config.min_confidence

    def validate_trigger_frequency(self, symbol: str) -> bool:
        """Check if trigger frequency is within limits."""
        if not self.config.max_triggers_per_minute:
            return True

        state = self.get_state(symbol)
        recent_triggers = state.get_recent_triggers(1)  # Last minute
        return len(recent_triggers) < self.config.max_triggers_per_minute

    def calculate_trigger_severity(self, confidence: float, impact_score: float = 1.0) -> TriggerSeverity:
        """
        Calculate trigger severity based on confidence and impact.

        Args:
            confidence: Trigger confidence score (0.0 to 1.0)
            impact_score: Impact score (0.0 to 1.0)

        Returns:
            Trigger severity level
        """
        combined_score = confidence * impact_score

        if combined_score >= 0.9:
            return TriggerSeverity.CRITICAL
        elif combined_score >= 0.75:
            return TriggerSeverity.HIGH
        elif combined_score >= 0.6:
            return TriggerSeverity.MEDIUM
        else:
            return TriggerSeverity.LOW

    def is_trading_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if current time is within regular trading hours."""
        if timestamp is None:
            timestamp = datetime.now()

        # Simple check - can be enhanced with market calendar
        if timestamp.weekday() >= 5:  # Weekend
            return False

        # Trading hours (9:30 AM - 4:00 PM EST)
        time_est = timestamp.time()
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= time_est <= market_close

    def apply_sensitivity_filter(self, raw_value: float) -> float:
        """
        Apply sensitivity filtering to trigger values.

        Args:
            raw_value: Raw trigger value/difference

        Returns:
            Sensitivity-adjusted value
        """
        # Higher sensitivity means smaller values pass through
        threshold = 1.0 - self.config.sensitivity
        if abs(raw_value) < threshold:
            return 0.0
        return raw_value

    async def analyze_market_data(self, symbol: str, market_data: Dict[str, Any]) -> List[TriggerEvent]:
        """
        Analyze market data and return validated triggers.

        This is the main entry point for trigger detection that includes
        validation, filtering, and state management.

        Args:
            symbol: Stock symbol to analyze
            market_data: Market data dictionary

        Returns:
            List of validated trigger events
        """
        try:
            # Skip if detector is disabled
            if not self.config.enabled:
                return []

            # Validate required data fields
            required_fields = self.get_required_data_fields()
            missing_fields = [field for field in required_fields if field not in market_data]
            if missing_fields:
                logger.debug(f"Missing required fields for {symbol}: {missing_fields}")
                return []

            # Check if enough data points are available
            if 'historical_data' in market_data:
                data_points = len(market_data['historical_data'])
                if data_points < self.config.required_data_points:
                    logger.debug(f"Insufficient data points for {symbol}: {data_points} < {self.config.required_data_points}")
                    return []

            # Get trigger state and check cooldown
            state = self.get_state(symbol)
            if not state.should_trigger(self.config):
                return []

            # Detect triggers using specific implementation
            raw_triggers = await self.detect_triggers(symbol, market_data)

            # Apply filters and validation
            validated_triggers = []
            for trigger in raw_triggers:
                # Confidence filter
                if not self.validate_trigger_confidence(trigger.confidence):
                    continue

                # Frequency filter
                if not self.validate_trigger_frequency(trigger.symbol):
                    continue

                # Sensitivity filter
                if self.config.sensitivity < 1.0:
                    # Apply sensitivity to trigger data
                    trigger.data['confidence'] = trigger.confidence
                    trigger.confidence *= self.config.sensitivity

                # Set TTL if not specified
                if trigger.ttl is None:
                    trigger.ttl = timedelta(seconds=self.config.cooldown_seconds)

                validated_triggers.append(trigger)
                state.add_trigger_event(trigger)

                # Update statistics
                self.stats['triggers_detected'] += 1
                self.stats['last_detection'] = datetime.now()

            if validated_triggers:
                logger.info(f"Detected {len(validated_triggers)} {self.trigger_type.value} triggers for {symbol}")

            return validated_triggers

        except Exception as e:
            logger.error(f"Error analyzing market data for {symbol}: {e}")
            return []

    async def start(self):
        """Start the trigger detector."""
        self.is_running = True
        logger.info(f"Started {self.trigger_type.value} trigger detector")

    async def stop(self):
        """Stop the trigger detector."""
        self.is_running = False
        logger.info(f"Stopped {self.trigger_type.value} trigger detector")

    def update_config(self, config: TriggerConfig):
        """Update detector configuration."""
        self.config = config
        logger.info(f"Updated {self.trigger_type.value} trigger detector config")

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['symbols_monitored_count'] = len(self.stats['symbols_monitored'])
        stats['states_count'] = len(self.states)
        stats['trigger_type'] = self.trigger_type.value

        # Calculate trigger rate
        if self.stats['triggers_detected'] > 0:
            stats['avg_confidence'] = np.mean([
                event.confidence for state in self.states.values()
                for event in state.trigger_history[-10:]  # Last 10 events
            ]) if any(state.trigger_history for state in self.states.values()) else 0.0
        else:
            stats['avg_confidence'] = 0.0

        return stats

    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific symbol."""
        state = self.get_state(symbol)

        recent_triggers = state.get_recent_triggers(60)  # Last hour
        hour_triggers = state.get_recent_triggers(60 * 24)  # Last 24 hours

        return {
            'symbol': symbol,
            'total_triggers': state.trigger_count,
            'last_trigger_time': state.last_trigger_time.isoformat() if state.last_trigger_time else None,
            'triggers_last_hour': len(recent_triggers),
            'triggers_last_24h': len(hour_triggers),
            'last_10_triggers': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'severity': event.severity.value,
                    'confidence': event.confidence,
                    'description': event.description
                }
                for event in state.trigger_history[-10:]
            ]
        }

    def get_active_symbols(self) -> List[str]:
        """Get list of symbols currently being monitored."""
        return list(self.states.keys())

    def reset_symbol_state(self, symbol: str):
        """Reset trigger state for a symbol."""
        if symbol in self.states:
            del self.states[symbol]
            self.stats['symbols_monitored'].discard(symbol)
            logger.info(f"Reset trigger state for {symbol}")


class CompositeTriggerDetector:
    """
    Manages multiple trigger detectors and coordinates their execution.
    """

    def __init__(self, detectors: List[BaseTriggerDetector]):
        """
        Initialize composite detector.

        Args:
            detectors: List of trigger detectors to manage
        """
        self.detectors = {detector.trigger_type: detector for detector in detectors}
        self.is_running = False

        # Combined statistics
        self.stats = {
            'total_triggers': 0,
            'detectors_active': 0,
            'symbols_monitored': set(),
            'last_check_time': None
        }

        logger.info(f"Initialized composite trigger detector with {len(detectors)} detectors")

    async def detect_triggers(self, symbol: str, market_data: Dict[str, Any]) -> Dict[TriggerType, List[TriggerEvent]]:
        """
        Detect triggers using all active detectors.

        Args:
            symbol: Stock symbol to analyze
            market_data: Market data dictionary

        Returns:
            Dictionary mapping trigger types to lists of trigger events
        """
        all_triggers = {}

        for trigger_type, detector in self.detectors.items():
            if detector.is_running:
                try:
                    triggers = await detector.analyze_market_data(symbol, market_data)
                    if triggers:
                        all_triggers[trigger_type] = triggers
                        self.stats['total_triggers'] += len(triggers)
                except Exception as e:
                    logger.error(f"Error in {trigger_type.value} detector for {symbol}: {e}")

        self.stats['last_check_time'] = datetime.now()
        if all_triggers:
            total_trigger_count = sum(len(triggers) for triggers in all_triggers.values())
            logger.info(f"Total triggers detected for {symbol}: {total_trigger_count}")

        return all_triggers

    async def start(self):
        """Start all detectors."""
        for detector in self.detectors.values():
            await detector.start()

        self.is_running = True
        self.stats['detectors_active'] = len(self.detectors)
        logger.info("Started all trigger detectors")

    async def stop(self):
        """Stop all detectors."""
        for detector in self.detectors.values():
            await detector.stop()

        self.is_running = False
        self.stats['detectors_active'] = 0
        logger.info("Stopped all trigger detectors")

    def get_detector(self, trigger_type: TriggerType) -> Optional[BaseTriggerDetector]:
        """Get a specific detector by type."""
        return self.detectors.get(trigger_type)

    def add_detector(self, detector: BaseTriggerDetector):
        """Add a new detector."""
        self.detectors[detector.trigger_type] = detector
        if self.is_running:
            asyncio.create_task(detector.start())
        logger.info(f"Added {detector.trigger_type.value} detector")

    def remove_detector(self, trigger_type: TriggerType):
        """Remove a detector."""
        if trigger_type in self.detectors:
            detector = self.detectors[trigger_type]
            asyncio.create_task(detector.stop())
            del self.detectors[trigger_type]
            logger.info(f"Removed {trigger_type.value} detector")

    def get_composite_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all detectors."""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['detector_types'] = list(self.detectors.keys())

        # Aggregate statistics from individual detectors
        detector_stats = {}
        for trigger_type, detector in self.detectors.items():
            detector_stats[trigger_type.value] = detector.get_statistics()

        stats['detectors'] = detector_stats

        # Aggregate monitored symbols
        all_symbols = set()
        for detector in self.detectors.values():
            all_symbols.update(detector.stats['symbols_monitored'])

        stats['total_symbols_monitored'] = len(all_symbols)
        stats['symbols_monitored'] = list(all_symbols)[:50]  # Limit to first 50

        return stats
