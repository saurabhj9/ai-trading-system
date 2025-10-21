"""
Real-time data polling infrastructure for event-driven trigger detection.

This module provides smart polling capabilities with market-hours awareness,
symbol prioritization, and efficient batching for near real-time data collection.
"""
import asyncio
import logging
from datetime import datetime, time, timezone, timedelta
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from src.data.providers.unified_batch_provider import UnifiedBatchProvider
from src.data.cache import CacheManager
from src.data.cache.cache_config import CacheKeyBuilder, CacheDataType

logger = logging.getLogger(__name__)


class MarketHoursStatus(Enum):
    """Market hours status enumeration."""
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    AFTER_HOURS = "after_hours"


class SymbolPriority(Enum):
    """Symbol polling priority levels."""
    HIGH = "high"          # 5-10 seconds
    MEDIUM = "medium"      # 15-30 seconds
    LOW = "low"           # 60-120 seconds
    INACTIVE = "inactive"  # No polling


@dataclass
class PollingSymbol:
    """Configuration for a symbol being polled."""
    symbol: str
    priority: SymbolPriority
    last_poll_time: Optional[datetime] = None
    last_updated_price: Optional[float] = None
    active: bool = True
    market_hours_only: bool = True


@dataclass
class PollingConfig:
    """Configuration for the real-time poller."""
    # Polling intervals in seconds
    high_priority_interval: int = 5
    medium_priority_interval: int = 15
    low_priority_interval: int = 60

    # Market hours
    market_open_time: time = time(9, 30)  # 9:30 AM EST
    market_close_time: time = time(16, 0)  # 4:00 PM EST
    timezone: str = "US/Eastern"

    # Performance settings
    max_concurrent_requests: int = 10
    batch_size: int = 20
    enable_market_hours_filter: bool = True

    # Cache settings
    enable_real_time_cache: bool = True
    real_time_cache_ttl: int = 5  # 5 seconds for real-time data


class MarketHoursManager:
    """Manages market hours detection and filtering."""

    def __init__(self, config: PollingConfig):
        self.config = config
        import pytz
        self.ny_tz = pytz.timezone(config.timezone)

    def get_market_status(self) -> MarketHoursStatus:
        """Get current market hours status."""
        now = datetime.now(self.ny_tz)
        now_time = now.time()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return MarketHoursStatus.CLOSED

        # Check market hours
        if self.config.market_open_time <= now_time <= self.config.market_close_time:
            return MarketHoursStatus.MARKET_OPEN
        elif now_time < self.config.market_open_time:
            return MarketHoursStatus.PRE_MARKET
        else:
            return MarketHoursStatus.AFTER_HOURS

    def should_poll_symbol(self, symbol: PollingSymbol) -> bool:
        """Determine if a symbol should be polled based on market hours."""
        if not self.config.enable_market_hours_filter:
            return symbol.active

        if not symbol.active:
            return False

        if not symbol.market_hours_only:
            return True

        status = self.get_market_status()
        return status == MarketHoursStatus.MARKET_OPEN

    def get_next_market_open(self) -> datetime:
        """Get the next market open time."""
        now = datetime.now(self.ny_tz)
        market_open = now.replace(
            hour=self.config.market_open_time.hour,
            minute=self.config.market_open_time.minute,
            second=0,
            microsecond=0
        )

        # If today is weekend or past market open, go to next weekday
        if now.weekday() >= 5 or now.time() > market_open.time():
            # Move to next Monday
            days_until_monday = (7 - now.weekday()) % 7 or 7
            market_open += timedelta(days=days_until_monday)

        return market_open.astimezone(timezone.utc)


class RealTimePoller:
    """
    Smart polling system for real-time data collection with market awareness.
    """

    def __init__(
        self,
        batch_provider: UnifiedBatchProvider,
        cache: Optional[CacheManager] = None,
        config: Optional[PollingConfig] = None,
        on_data_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize the real-time poller.

        Args:
            batch_provider: Unified batch provider for efficient data fetching
            cache: Cache manager for storing real-time data
            config: Polling configuration
            on_data_callback: Callback function called when new data arrives
        """
        self.batch_provider = batch_provider
        self.cache = cache
        self.config = config or PollingConfig()
        self.on_data_callback = on_data_callback

        # Market hours management
        self.market_hours = MarketHoursManager(self.config)

        # Symbol management
        self.symbols: Dict[str, PollingSymbol] = {}
        self.polling_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Statistics
        self.stats = {
            'total_polls': 0,
            'successful_polls': 0,
            'failed_polls': 0,
            'last_poll_time': None,
            'symbols_polled': set()
        }

        logger.info("Real-time poller initialized")

    def add_symbol(
        self,
        symbol: str,
        priority: SymbolPriority = SymbolPriority.MEDIUM,
        market_hours_only: bool = True
    ):
        """Add a symbol to the polling list."""
        self.symbols[symbol] = PollingSymbol(
            symbol=symbol,
            priority=priority,
            market_hours_only=market_hours_only
        )
        logger.info(f"Added symbol {symbol} with {priority.value} priority")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from the polling list."""
        if symbol in self.symbols:
            del self.symbols[symbol]
            logger.info(f"Removed symbol {symbol} from polling")

    def set_symbol_priority(self, symbol: str, priority: SymbolPriority):
        """Update the priority of a symbol."""
        if symbol in self.symbols:
            self.symbols[symbol].priority = priority
            logger.info(f"Updated {symbol} priority to {priority.value}")

    def get_symbols_by_priority(self, priority: SymbolPriority) -> List[str]:
        """Get all symbols with a specific priority that are active."""
        return [
            s.symbol for s in self.symbols.values()
            if s.priority == priority and s.active and
            self.market_hours.should_poll_symbol(s)
        ]

    def _get_poll_interval(self, priority: SymbolPriority) -> int:
        """Get polling interval for a priority level."""
        intervals = {
            SymbolPriority.HIGH: self.config.high_priority_interval,
            SymbolPriority.MEDIUM: self.config.medium_priority_interval,
            SymbolPriority.LOW: self.config.low_priority_interval,
            SymbolPriority.INACTIVE: 0
        }
        return intervals.get(priority, self.config.medium_priority_interval)

    async def _poll_symbols_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Poll a batch of symbols for their current prices."""
        if not symbols:
            return {}

        try:
            # Use batch provider to get multiple quotes efficiently
            quotes = await self.batch_provider.get_multiple_quotes(symbols)

            # Update statistics
            self.stats['total_polls'] += len(symbols)
            successful_quotes = sum(1 for price in quotes.values() if price is not None)
            self.stats['successful_polls'] += successful_quotes
            self.stats['failed_polls'] += len(symbols) - successful_quotes
            self.stats['last_poll_time'] = datetime.now()
            self.stats['symbols_polled'].update(symbols)

            return {symbol: price for symbol, price in quotes.items() if price is not None}

        except Exception as e:
            logger.error(f"Error polling batch {symbols}: {e}")
            self.stats['failed_polls'] += len(symbols)
            return {}

    async def _poll_by_priority(self, priority: SymbolPriority):
        """Poll symbols of a specific priority based on their interval."""
        symbols = self.get_symbols_by_priority(priority)
        if not symbols:
            return

        interval = self._get_poll_interval(priority)
        if interval <= 0:
            return

        # Check if symbols need polling based on their last poll time
        symbols_to_poll = []
        now = datetime.now()

        for symbol in symbols:
            if symbol not in self.symbols:
                continue

            polling_symbol = self.symbols[symbol]
            if (
                polling_symbol.last_poll_time is None or
                (now - polling_symbol.last_poll_time).total_seconds() >= interval
            ):
                symbols_to_poll.append(symbol)

        if symbols_to_poll:
            logger.debug(f"Polling {len(symbols_to_poll)} {priority.value} priority symbols")

            # Poll in batches
            batch_size = self.config.batch_size
            for i in range(0, len(symbols_to_poll), batch_size):
                batch = symbols_to_poll[i:i + batch_size]
                quotes = await self._poll_symbols_batch(batch)

                # Update symbol states and cache
                for symbol, price in quotes.items():
                    if symbol in self.symbols:
                        self.symbols[symbol].last_poll_time = datetime.now()
                        self.symbols[symbol].last_updated_price = price

                        # Store in real-time cache
                        if self.cache and self.config.enable_real_time_cache:
                            cache_key = CacheKeyBuilder.build_real_time_key(symbol)
                            cache_data = {
                                'price': price,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'real_time_poller'
                            }
                            await self.cache.set(cache_key, cache_data, ttl=self.config.real_time_cache_ttl)

                        # Call callback if provided
                        if self.on_data_callback:
                            try:
                                await self.on_data_callback(symbol, price)
                            except Exception as e:
                                logger.error(f"Error in callback for {symbol}: {e}")

    async def poll_loop(self):
        """Main polling loop that continuously polls symbols based on their priorities."""
        logger.info("Starting real-time polling loop")

        while self.is_running:
            try:
                # Check market status
                market_status = self.market_hours.get_market_status()
                if market_status != MarketHoursStatus.MARKET_OPEN:
                    if market_status == MarketHoursStatus.CLOSED:
                        # Wait for next market open
                        next_open = self.market_hours.get_next_market_open()
                        wait_seconds = (next_open - datetime.now(timezone.utc)).total_seconds()
                        logger.info(f"Market closed. Waiting {wait_seconds/3600:.1f} hours until market open")
                        await asyncio.sleep(min(wait_seconds, 3600))  # Wake up at most every hour
                        continue
                    else:
                        # Pre-market or after hours - poll less frequently
                        await asyncio.sleep(self.config.low_priority_interval)
                        continue

                # Poll each priority level
                await self._poll_by_priority(SymbolPriority.HIGH)
                await asyncio.sleep(1)  # Small delay between priority levels

                await self._poll_by_priority(SymbolPriority.MEDIUM)
                await asyncio.sleep(1)

                await self._poll_by_priority(SymbolPriority.LOW)

                # Brief sleep before next cycle
                await asyncio.sleep(self.config.high_priority_interval)

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def start(self):
        """Start the real-time polling."""
        if self.is_running:
            logger.warning("Poller is already running")
            return

        self.is_running = True
        self.polling_task = asyncio.create_task(self.poll_loop())
        logger.info("Real-time poller started")

    async def stop(self):
        """Stop the real-time polling."""
        if not self.is_running:
            return

        self.is_running = False

        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass

        logger.info("Real-time poller stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get polling statistics."""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['market_status'] = self.market_hours.get_market_status().value
        stats['active_symbols'] = len([s for s in self.symbols.values() if s.active])
        stats['total_symbols'] = len(self.symbols)

        # Add priority breakdown
        priority_counts = {}
        for symbol in self.symbols.values():
            if symbol.active:
                priority = symbol.priority.value
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
        stats['symbols_by_priority'] = priority_counts

        # Calculate success rate
        if stats['total_polls'] > 0:
            stats['success_rate'] = stats['successful_polls'] / stats['total_polls']
        else:
            stats['success_rate'] = 0.0

        return stats

    async def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol from cache or memory."""
        # Check polling state first
        if symbol in self.symbols and self.symbols[symbol].last_updated_price:
            time_diff = datetime.now() - self.symbols[symbol].last_poll_time
            if time_diff.total_seconds() < 30:  # Fresh if within 30 seconds
                return self.symbols[symbol].last_updated_price

        # Check cache
        if self.cache and self.config.enable_real_time_cache:
            cache_key = CacheKeyBuilder.build_real_time_key(symbol)
            cached_data = await self.cache.get(cache_key)
            if cached_data and 'price' in cached_data:
                timestamp = datetime.fromisoformat(cached_data['timestamp'])
                age_seconds = (datetime.now() - timestamp).total_seconds()
                if age_seconds < 60:  # Fresh if within 1 minute
                    return cached_data['price']

        # Fetch fresh data
        try:
            quotes = await self.batch_provider.get_multiple_quotes([symbol])
            price = quotes.get(symbol)
            if price and symbol in self.symbols:
                self.symbols[symbol].last_updated_price = price
                self.symbols[symbol].last_poll_time = datetime.now()
            return price
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return None
