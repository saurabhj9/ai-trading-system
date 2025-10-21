"""
Cooldown manager and decision TTL system for controlling trigger frequency.

This module manages rate limiting, cooldown periods, and decision time-to-live
to prevent excessive triggers and maintain system stability.
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .trigger_detector import TriggerType, TriggerSeverity

logger = logging.getLogger(__name__)


class CooldownType(Enum):
    """Types of cooldown periods."""
    SYMBOL = "symbol"           # Per-symbol cooldown
    TRIGGER_TYPE = "trigger_type"  # Per-trigger-type cooldown
    SEVERITY = "severity"       # Per-severity cooldown
    GLOBAL = "global"           # Global cooldown for all triggers
    TIME_BASED = "time_based"   # Time-based cooldown (e.g., N per hour)


@dataclass
class CooldownConfig:
    """Configuration for cooldown periods."""
    enabled: bool = True

    # Default cooldown periods in seconds
    symbol_cooldown_seconds: int = 300      # 5 minutes per symbol
    trigger_type_cooldown_seconds: int = 180  # 3 minutes per trigger type
    severity_cooldown_seconds: Dict[TriggerSeverity, int] = field(default_factory=lambda: {
        TriggerSeverity.LOW: 900,      # 15 minutes
        TriggerSeverity.MEDIUM: 600,   # 10 minutes
        TriggerSeverity.HIGH: 300,     # 5 minutes
        TriggerSeverity.CRITICAL: 60   # 1 minute
    })
    global_cooldown_seconds: int = 60      # 1 minute global

    # Frequency limits
    max_triggers_per_minute: int = 10
    max_triggers_per_hour: int = 100
    max_triggers_per_day: int = 500

    # Symbol-specific limits
    max_same_symbol_triggers_per_minute: int = 2
    max_same_trigger_type_per_hour: int = 50

    # Decision TTL settings
    default_decision_ttl_seconds: int = 1800  # 30 minutes
    critical_decision_ttl_seconds: int = 900   # 15 minutes
    low_decision_ttl_seconds: int = 3600       # 1 hour

    # Cache settings
    enable_cache_persistence: bool = True
    cache_cleanup_interval_seconds: int = 300   # 5 minutes


@dataclass
class TriggerRecord:
    """Record of a trigger occurrence."""
    symbol: str
    trigger_type: TriggerType
    severity: TriggerSeverity
    timestamp: datetime
    trigger_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cooldown_key(self, cooldown_type: CooldownType) -> str:
        """Get cooldown key for different cooldown types."""
        if cooldown_type == CooldownType.SYMBOL:
            return f"symbol:{self.symbol}"
        elif cooldown_type == CooldownType.TRIGGER_TYPE:
            return f"trigger_type:{self.trigger_type.value}"
        elif cooldown_type == CooldownType.SEVERITY:
            return f"severity:{self.severity.value}"
        elif cooldown_type == CooldownType.GLOBAL:
            return "global"
        else:
            return f"hash:{self._calculate_hash()}"

    def _calculate_hash(self) -> str:
        """Calculate hash for unique trigger identification."""
        data = f"{self.symbol}:{self.trigger_type.value}:{self.severity.value}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:8]


@dataclass
class DecisionRecord:
    """Record of a decision with TTL."""
    symbol: str
    decision_type: str
    decision: str
    timestamp: datetime
    ttl: timedelta
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if decision has expired."""
        return datetime.now() > self.timestamp + self.ttl

    def get_remaining_ttl_seconds(self) -> float:
        """Get remaining TTL in seconds."""
        expiry_time = self.timestamp + self.ttl
        remaining = expiry_time - datetime.now()
        return max(0, remaining.total_seconds())


class CooldownManager:
    """
    Manages cooldown periods and rate limiting for trigger events.

    Prevents trigger flooding and ensures system stability by enforcing
    various types of cooldowns and frequency limits.
    """

    def __init__(
        self,
        config: Optional[CooldownConfig] = None,
        cache_backend: Optional[Any] = None
    ):
        """
        Initialize cooldown manager.

        Args:
            config: Cooldown configuration
            cache_backend: Optional cache backend for persistence
        """
        self.config = config or CooldownConfig()
        self.cache = cache_backend

        # In-memory storage for trigger records (fallback if no cache)
        self.trigger_records: List[TriggerRecord] = []
        self.decision_records: Dict[str, DecisionRecord] = {}  # key: decisions

        # Statistics
        self.stats = {
            'triggers_checked': 0,
            'triggers_blocked': 0,
            'triggers_allowed': 0,
            'cooldowns_active': 0,
            'decisions_cached': 0,
            'decisions_expired': 0,
            'last_cleanup': datetime.now()
        }

        # Start cleanup task if enabled
        if self.config.enable_cache_persistence:
            import asyncio
            asyncio.create_task(self._periodic_cleanup())

        logger.info("Cooldown manager initialized")

    async def _periodic_cleanup(self):
        """Periodic cleanup task for expired decisions."""
        while True:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval_seconds)
                # This would clean up trigger records, but we'll keep it minimal
                # for now since we don't have the cleanup method here
                pass
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def check_trigger_cooldown(self, record: TriggerRecord) -> Tuple[bool, List[str]]:
        """
        Check if trigger should be blocked due to cooldown.

        Args:
            record: Trigger record to check

        Returns:
            Tuple of (should_block, reasons_for_blocking)
        """
        self.stats['triggers_checked'] += 1

        if not self.config.enabled:
            self.stats['triggers_allowed'] += 1
            return False, []

        blocked_reasons = []
        now = datetime.now()

        # Check all cooldown types
        cooldown_checks = [
            (CooldownType.SYMBOL, self.config.symbol_cooldown_seconds),
            (CooldownType.TRIGGER_TYPE, self.config.trigger_type_cooldown_seconds),
            (CooldownType.SEVERITY, self.config.severity_cooldown_seconds.get(record.severity, 300)),
            (CooldownType.GLOBAL, self.config.global_cooldown_seconds)
        ]

        for cooldown_type, cooldown_seconds in cooldown_checks:
            if await self._is_cooled_down(record, cooldown_type, cooldown_seconds):
                blocked_reasons.append(f"{cooldown_type.value}_cooldown")

        # Check frequency limits
        if await self._exceeds_frequency_limits(record):
            blocked_reasons.append("frequency_limit")

        # Symbol-specific limits
        if await self._exceeds_symbol_limits(record):
            blocked_reasons.append("symbol_limit")

        # Update statistics
        if blocked_reasons:
            self.stats['triggers_blocked'] += 1
            logger.debug(f"Trigger blocked for {record.symbol}: {', '.join(blocked_reasons)}")
            return True, blocked_reasons
        else:
            self.stats['triggers_allowed'] += 1
            # Record the trigger for future cooldown checks
            await self._record_trigger(record)
            return False, []

    async def _is_cooled_down(
        self,
        record: TriggerRecord,
        cooldown_type: CooldownType,
        cooldown_seconds: int
    ) -> bool:
        """Check if a specific cooldown type is still active."""
        if cooldown_seconds <= 0:
            return False

        cooldown_key = record.get_cooldown_key(cooldown_type)
        now = datetime.now()

        # Try cache first
        if self.cache:
            try:
                cached_time = await self.cache.get(f"cooldown:{cooldown_key}")
                if cached_time:
                    last_trigger_time = datetime.fromisoformat(cached_time)
                    if (now - last_trigger_time).total_seconds() < cooldown_seconds:
                        return True
            except Exception as e:
                logger.warning(f"Cache error for cooldown {cooldown_key}: {e}")

        # Fallback to in-memory check
        relevant_records = await self._get_recent_records(cooldown_type, record, cooldown_seconds)
        if relevant_records:
            return True

        return False

    async def _exceeds_frequency_limits(self, record: TriggerRecord) -> bool:
        """Check if trigger exceeds frequency limits."""
        now = datetime.now()

        # Check per-minute limit
        if self.config.max_triggers_per_minute > 0:
            minute_ago = now - timedelta(minutes=1)
            minute_triggers = [
                r for r in self.trigger_records
                if r.timestamp >= minute_ago
            ]
            if len(minute_triggers) >= self.config.max_triggers_per_minute:
                return True

        # Check per-hour limit
        if self.config.max_triggers_per_hour > 0:
            hour_ago = now - timedelta(hours=1)
            hour_triggers = [
                r for r in self.trigger_records
                if r.timestamp >= hour_ago
            ]
            if len(hour_triggers) >= self.config.max_triggers_per_hour:
                return True

        # Check per-day limit
        if self.config.max_triggers_per_day > 0:
            day_ago = now - timedelta(days=1)
            day_triggers = [
                r for r in self.trigger_records
                if r.timestamp >= day_ago
            ]
            if len(day_triggers) >= self.config.max_triggers_per_day:
                return True

        return False

    async def _exceeds_symbol_limits(self, record: TriggerRecord) -> bool:
        """Check if trigger exceeds symbol-specific limits."""
        now = datetime.now()

        # Check same symbol per-minute limit
        if self.config.max_same_symbol_triggers_per_minute > 0:
            minute_ago = now - timedelta(minutes=1)
            symbol_triggers = [
                r for r in self.trigger_records
                if r.symbol == record.symbol and r.timestamp >= minute_ago
            ]
            if len(symbol_triggers) >= self.config.max_same_symbol_triggers_per_minute:
                return True

        # Check same trigger type per-hour limit
        if self.config.max_same_trigger_type_per_hour > 0:
            hour_ago = now - timedelta(hours=1)
            type_triggers = [
                r for r in self.trigger_records
                if r.trigger_type == record.trigger_type and r.timestamp >= hour_ago
            ]
            if len(type_triggers) >= self.config.max_same_trigger_type_per_hour:
                return True

        return False

    async def _get_recent_records(
        self,
        cooldown_type: CooldownType,
        record: TriggerRecord,
        seconds: int
    ) -> List[TriggerRecord]:
        """Get recent trigger records for cooldown checking."""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=seconds)

        if cooldown_type == CooldownType.SYMBOL:
            return [r for r in self.trigger_records if r.symbol == record.symbol and r.timestamp >= cutoff_time]
        elif cooldown_type == CooldownType.TRIGGER_TYPE:
            return [r for r in self.trigger_records if r.trigger_type == record.trigger_type and r.timestamp >= cutoff_time]
        elif cooldown_type == CooldownType.SEVERITY:
            return [r for r in self.trigger_records if r.severity == record.severity and r.timestamp >= cutoff_time]
        else:
            return [r for r in self.trigger_records if r.timestamp >= cutoff_time]

    async def _record_trigger(self, record: TriggerRecord):
        """Record a trigger for future cooldown checks."""
        self.trigger_records.append(record)

        # Keep only last 10,000 records to prevent memory issues
        if len(self.trigger_records) > 10000:
            self.trigger_records = self.trigger_records[-1000:]

        # Update cache if available
        if self.cache:
            try:
                cooldown_types = [
                    (CooldownType.SYMBOL, self.config.symbol_cooldown_seconds),
                    (CooldownType.TRIGGER_TYPE, self.config.trigger_type_cooldown_seconds),
                    (CooldownType.SEVERITY, self.config.severity_cooldown_seconds.get(record.severity, 300)),
                    (CooldownType.GLOBAL, self.config.global_cooldown_seconds)
                ]

                for cooldown_type, _ in cooldown_types:
                    key = f"cooldown:{record.get_cooldown_key(cooldown_type)}"
                    await self.cache.set(key, record.timestamp.isoformat(), ttl=3600)  # 1 hour TTL

            except Exception as e:
                logger.warning(f"Failed to cache trigger cooldown: {e}")


class DecisionTTL:
    """
    Manager for decision time-to-live (TTL) functionality.

    Caches decisions with expiration times to prevent redundant
    analysis and provide consistency for similar market conditions.
    """

    def __init__(
        self,
        config: Optional[CooldownConfig] = None,
        cache_backend: Optional[Any] = None
    ):
        """
        Initialize decision TTL manager.

        Args:
            config: Cooldown configuration containing TTL settings
            cache_backend: Optional cache backend for persistence
        """
        self.config = config or CooldownConfig()
        self.cache = cache_backend
        self.decisions: Dict[str, DecisionRecord] = {}

        # Statistics
        self.stats = {
            'decisions_cached': 0,
            'decisions_retrieved': 0,
            'decisions_expired': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("Decision TTL manager initialized")

    async def cache_decision(
        self,
        symbol: str,
        decision_type: str,
        decision: str,
        severity: TriggerSeverity = TriggerSeverity.MEDIUM,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Cache a decision with appropriate TTL.

        Args:
            symbol: Stock symbol
            decision_type: Type of decision (e.g., "buy", "sell", "hold")
            decision: The decision content
            severity: Decision severity
            confidence: Decision confidence
            metadata: Additional metadata

        Returns:
            Decision key for retrieval
        """
        # Determine TTL based on severity
        if severity == TriggerSeverity.CRITICAL:
            ttl_seconds = self.config.critical_decision_ttl_seconds
        elif severity == TriggerSeverity.LOW:
            ttl_seconds = self.config.low_decision_ttl_seconds
        else:
            ttl_seconds = self.config.default_decision_ttl_seconds

        ttl = timedelta(seconds=ttl_seconds)

        # Create decision record
        decision_key = self._generate_decision_key(symbol, decision_type)
        decision_record = DecisionRecord(
            symbol=symbol,
            decision_type=decision_type,
            decision=decision,
            timestamp=datetime.now(),
            ttl=ttl,
            confidence=confidence,
            metadata=metadata or {}
        )

        # Store in memory
        self.decisions[decision_key] = decision_record

        # Store in cache if available
        if self.cache:
            try:
                cache_data = {
                    'symbol': symbol,
                    'decision_type': decision_type,
                    'decision': decision,
                    'timestamp': decision_record.timestamp.isoformat(),
                    'ttl': ttl.total_seconds(),
                    'confidence': confidence,
                    'metadata': metadata or {}
                }
                await self.cache.set(f"decision:{decision_key}", json.dumps(cache_data), ttl=ttl_seconds)
            except Exception as e:
                logger.warning(f"Failed to cache decision: {e}")

        self.stats['decisions_cached'] += 1
        logger.debug(f"Cached decision for {symbol}: {decision_type} (TTL: {ttl_seconds}s)")

        return decision_key

    async def _periodic_cleanup(self):
        """Periodic cleanup task for expired decisions."""
        while True:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval_seconds)
                await self.cleanup_expired_decisions()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def get_decision(self, symbol: str, decision_type: str) -> Optional[DecisionRecord]:
        """
        Retrieve a cached decision if it hasn't expired.

        Args:
            symbol: Stock symbol
            decision_type: Type of decision

        Returns:
            Decision record if found and not expired, None otherwise
        """
        decision_key = self._generate_decision_key(symbol, decision_type)

        # Check memory first
        if decision_key in self.decisions:
            decision_record = self.decisions[decision_key]
            if not decision_record.is_expired():
                self.stats['decisions_retrieved'] += 1
                self.stats['cache_hits'] += 1
                return decision_record
            else:
                # Remove expired decision
                del self.decisions[decision_key]
                self.stats['decisions_expired'] += 1

        # Check cache
        if self.cache:
            try:
                cached_data = await self.cache.get(f"decision:{decision_key}")
                if cached_data:
                    data = json.loads(cached_data)
                    decision_record = DecisionRecord(
                        symbol=data['symbol'],
                        decision_type=data['decision_type'],
                        decision=data['decision'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        ttl=timedelta(seconds=data['ttl']),
                        confidence=data.get('confidence', 0.0),
                        metadata=data.get('metadata', {})
                    )

                    if not decision_record.is_expired():
                        # Refresh in-memory cache
                        self.decisions[decision_key] = decision_record
                        self.stats['decisions_retrieved'] += 1
                        self.stats['cache_hits'] += 1
                        return decision_record
                    else:
                        # Remove from cache
                        await self.cache.delete(f"decision:{decision_key}")
                        self.stats['decisions_expired'] += 1

            except Exception as e:
                logger.warning(f"Failed to retrieve cached decision: {e}")

        self.stats['cache_misses'] += 1
        return None

    async def invalidate_decision(self, symbol: str, decision_type: str) -> bool:
        """
        Invalidate a cached decision.

        Args:
            symbol: Stock symbol
            decision_type: Type of decision

        Returns:
            True if decision was invalidated, False if not found
        """
        decision_key = self._generate_decision_key(symbol, decision_type)

        invalidated = False

        # Remove from memory
        if decision_key in self.decisions:
            del self.decisions[decision_key]
            invalidated = True

        # Remove from cache
        if self.cache:
            try:
                await self.cache.delete(f"decision:{decision_key}")
                invalidated = True
            except Exception as e:
                logger.warning(f"Failed to delete cached decision: {e}")

        return invalidated

    def _generate_decision_key(self, symbol: str, decision_type: str) -> str:
        """Generate a unique key for a decision."""
        timestamp_str = datetime.now().strftime("%Y%m%d")
        return f"{symbol}:{decision_type}:{timestamp_str}"

    async def cleanup_expired_decisions(self):
        """Clean up expired decisions from memory."""
        current_time = datetime.now()
        expired_keys = [
            key for key, record in self.decisions.items()
            if record.is_expired()
        ]

        for key in expired_keys:
            del self.decisions[key]
            self.stats['decisions_expired'] += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired decisions")

    def get_statistics(self) -> Dict[str, Any]:
        """Get TTL manager statistics."""
        stats = {
            **self.stats,
            'active_decisions': len(self.decisions),
            'cache_hit_rate': 0.0
        }

        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests

        return stats

    def get_active_decisions(self) -> List[Dict[str, Any]]:
        """Get list of active (non-expired) decisions."""
        active_decisions = []

        for decision_record in self.decisions.values():
            if not decision_record.is_expired():
                active_decisions.append({
                    'symbol': decision_record.symbol,
                    'decision_type': decision_record.decision_type,
                    'decision': decision_record.decision,
                    'timestamp': decision_record.timestamp.isoformat(),
                    'remaining_ttl_seconds': decision_record.get_remaining_ttl_seconds(),
                    'confidence': decision_record.confidence
                })

        return active_decisions


# Utility functions for integration
async def create_trigger_record(
    symbol: str,
    trigger_type: TriggerType,
    severity: TriggerSeverity,
    description: str,
    metadata: Optional[Dict[str, Any]] = None
) -> TriggerRecord:
    """Create a trigger record for cooldown checking."""
    return TriggerRecord(
        symbol=symbol,
        trigger_type=trigger_type,
        severity=severity,
        timestamp=datetime.now(),
        trigger_id=f"{symbol}_{trigger_type.value}_{datetime.now().timestamp()}",
        metadata=metadata or {}
    )


def get_default_cooldown_config() -> CooldownConfig:
    """Get default cooldown configuration."""
    return CooldownConfig(
        enabled=True,
        symbol_cooldown_seconds=300,      # 5 minutes
        trigger_type_cooldown_seconds=180,  # 3 minutes
        severity_cooldown_seconds={
            TriggerSeverity.LOW: 900,      # 15 minutes
            TriggerSeverity.MEDIUM: 600,   # 10 minutes
            TriggerSeverity.HIGH: 300,     # 5 minutes
            TriggerSeverity.CRITICAL: 60   # 1 minute
        },
        global_cooldown_seconds=60,        # 1 minute
        max_triggers_per_minute=10,
        max_triggers_per_hour=100,
        max_triggers_per_day=500,
        max_same_symbol_triggers_per_minute=2,
        max_same_trigger_type_per_hour=50,
        default_decision_ttl_seconds=1800,  # 30 minutes
        critical_decision_ttl_seconds=900,   # 15 minutes
        low_decision_ttl_seconds=3600,       # 1 hour
        enable_cache_persistence=True,
        cache_cleanup_interval_seconds=300   # 5 minutes
    )
