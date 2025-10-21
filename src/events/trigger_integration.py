"""
Integration layer connecting event-driven triggers with the Local Signal Generation Framework.

This module orchestrates the interaction between real-time data polling, trigger detection,
and the existing signal generation system to enable automated, event-driven analysis.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from .trigger_detector import CompositeTriggerDetector, TriggerEvent, TriggerType, TriggerSeverity, TriggerConfig
from .technical_triggers import TechnicalTriggerDetector
from .event_bus import EventBus, TriggerHandlerSubscriber, SubscriberType, LoggingSubscriber
from .cooldown_manager import CooldownManager, DecisionTTL, CooldownConfig, TriggerRecord
from ..data.real_time_poller import RealTimePoller, PollingConfig, SymbolPriority
from ..data.pipeline import DataPipeline
from ..data.providers.unified_batch_provider import UnifiedBatchProvider
from ..data.cache import CacheManager
from ..signal_generation.signal_generator import LocalSignalGenerator
from ..communication.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class TriggerSystemConfig:
    """Configuration for the complete trigger system."""
    # Polling settings
    polling_config: PollingConfig

    # Detector settings
    technical_enabled: bool = True
    volatility_enabled: bool = False  # To be implemented
    trend_enabled: bool = False       # To be implemented
    conflict_enabled: bool = False    # To be implemented
    regime_enabled: bool = False      # To be implemented

    # Cooldown settings
    cooldown_config: CooldownConfig = None

    # Integration settings
    auto_signal_generation: bool = True
    cache_decisions: bool = True
    enable_logging: bool = True

    # Performance settings
    max_concurrent_analyses: int = 5
    analysis_timeout_seconds: int = 30

    def __post_init__(self):
        if self.cooldown_config is None:
            self.cooldown_config = CooldownConfig()


class TriggerAnalysisSubscriber(TriggerHandlerSubscriber):
    """
    Subscriber that analyzes triggers by invoking the signal generation framework.
    """

    def __init__(
        self,
        signal_generator: LocalSignalGenerator,
        orchestrator: Optional[Orchestrator] = None,
        max_concurrent: int = 5,
        timeout_seconds: int = 30
    ):
        super().__init__("trigger_analyzer", SubscriberType.ANALYZER)
        self.signal_generator = signal_generator
        self.orchestrator = orchestrator
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.active_analyses: Dict[str, asyncio.Task] = {}

    async def handle_event(self, event: TriggerEvent) -> bool:
        """Handle trigger event by generating signal analysis."""
        try:
            # Check if analysis is already in progress for this symbol
            if event.symbol in self.active_analyses:
                logger.debug(f"Analysis already in progress for {event.symbol}")
                return True

            # Create analysis task
            analysis_task = asyncio.create_task(
                self._perform_analysis_with_timeout(event)
            )
            self.active_analyses[event.symbol] = analysis_task

            # Clean up completed tasks
            analysis_task.add_done_callback(
                lambda task: self.active_analyses.pop(event.symbol, None)
            )

            logger.info(f"Started signal generation for {event.symbol} due to {event.trigger_type.value} trigger")
            return True

        except Exception as e:
            logger.error(f"Error handling trigger event for {event.symbol}: {e}")
            return False

    async def _perform_analysis_with_timeout(self, event: TriggerEvent) -> bool:
        """Perform signal generation with timeout."""
        try:
            await asyncio.wait_for(
                self._perform_analysis(event),
                timeout=self.timeout_seconds
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Signal generation timeout for {event.symbol}")
            return False
        except Exception as e:
            logger.error(f"Signal generation error for {event.symbol}: {e}")
            return False

    async def _perform_analysis(self, event: TriggerEvent) -> bool:
        """Perform the actual signal generation."""
        try:
            # Generate local signal
            signal_result = await self.signal_generator.generate_signal(
                symbol=event.symbol,
                trigger_context={
                    'trigger_type': event.trigger_type.value,
                    'severity': event.severity.value,
                    'confidence': event.confidence,
                    'trigger_data': event.data,
                    'timestamp': event.timestamp.isoformat()
                }
            )

            # If orchestrator is available, process through full pipeline
            if self.orchestrator and signal_result.get('success', False):
                full_analysis = await self.orchestrator.analyze_symbol(
                    symbol=event.symbol,
                    force_analysis=True,  # Force analysis due to trigger
                    trigger_event=event
                )

                logger.info(f"Full analysis completed for {event.symbol}: trigger={event.trigger_type.value}")
            else:
                logger.info(f"Local signal generation completed for {event.symbol}: signal={signal_result}")

            return True

        except Exception as e:
            logger.error(f"Error in signal generation for {event.symbol}: {e}")
            return False


class TriggerSystem:
    """
    Complete trigger system integrating polling, detection, and signal generation.

    This class orchestrates all components of the event-driven trigger system:
    - Real-time data polling
    - Trigger detection across multiple types
    - Event distribution and handling
    - Cooldown management and decision caching
    - Integration with local signal generation
    """

    def __init__(
        self,
        batch_provider: UnifiedBatchProvider,
        cache: Optional[CacheManager] = None,
        signal_generator: Optional[LocalSignalGenerator] = None,
        orchestrator: Optional[Orchestrator] = None,
        config: Optional[TriggerSystemConfig] = None
    ):
        """
        Initialize the complete trigger system.

        Args:
            batch_provider: Batch provider for data fetching
            cache: Cache manager for decision storage
            signal_generator: Local signal generator
            orchestrator: System orchestrator for full analysis
            config: System configuration
        """
        self.config = config or TriggerSystemConfig(polling_config=PollingConfig())
        self.batch_provider = batch_provider
        self.cache = cache
        self.signal_generator = signal_generator
        self.orchestrator = orchestrator

        # Initialize components
        self._initialize_components()

        # System state
        self.is_running = False
        self.monitored_symbols: Set[str] = set()

        # Statistics
        self.stats = {
            'system_started_at': None,
            'triggers_detected': 0,
            'analyses_triggered': 0,
            'analyses_completed': 0,
            'analyses_failed': 0,
            'last_trigger_time': None,
            'uptime_seconds': 0
        }

        logger.info("Trigger system initialized")

    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize real-time poller
        self.poller = RealTimePoller(
            batch_provider=self.batch_provider,
            cache=self.cache,
            config=self.config.polling_config
        )

        # Initialize trigger detectors
        detectors = []

        if self.config.technical_enabled:
            tech_config = TriggerConfig(enabled=True)
            detectors.append(TechnicalTriggerDetector(tech_config))

        # Other detectors can be added here when implemented
        # if self.config.volatility_enabled:
        #     detectors.append(VolatilityTriggerDetector())
        # if self.config.trend_enabled:
        #     detectors.append(TrendTriggerDetector())

        self.detector = CompositeTriggerDetector(detectors)

        # Initialize event bus
        self.event_bus = EventBus(
            max_queue_size=10000,
            worker_count=5
        )

        # Initialize cooldown manager
        self.cooldown_manager = CooldownManager(
            config=self.config.cooldown_config,
            cache_backend=self.cache
        )

        # Initialize decision TTL
        self.decision_ttl = DecisionTTL(
            config=self.config.cooldown_config,
            cache_backend=self.cache
        )

        # Initialize data pipeline (for market data processing)
        self.data_pipeline = DataPipeline(
            provider=self.batch_provider,
            cache=self.cache
        )

        # Set up event bus subscribers
        self._setup_event_subscribers()

    def _setup_event_subscribers(self):
        """Set up event bus subscribers."""
        # Analysis subscriber (main integration point)
        if self.config.auto_signal_generation and self.signal_generator:
            analysis_subscriber = TriggerAnalysisSubscriber(
                signal_generator=self.signal_generator,
                orchestrator=self.orchestrator,
                max_concurrent=self.config.max_concurrent_analyses,
                timeout_seconds=self.config.analysis_timeout_seconds
            )

            self.event_bus.subscribe(
                subscriber=analysis_subscriber,
                trigger_types=list(TriggerType),
                severity_filter=None,  # All severities
                symbol_filter=None    # All symbols
            )

        # Logging subscriber
        if self.config.enable_logging:
            logging_subscriber = LoggingSubscriber()
            self.event_bus.subscribe(
                subscriber=logging_subscriber,
                trigger_types=list(TriggerType),
                severity_filter=[TriggerSeverity.HIGH, TriggerSeverity.CRITICAL]
            )

    async def start(self):
        """Start the trigger system."""
        if self.is_running:
            logger.warning("Trigger system is already running")
            return

        try:
            # Start all components
            await self.poller.start()
            await self.detector.start()
            await self.event_bus.start()

            self.is_running = True
            self.stats['system_started_at'] = datetime.now()

            # Start main monitoring loop
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Trigger system started successfully")

        except Exception as e:
            logger.error(f"Error starting trigger system: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the trigger system."""
        if not self.is_running:
            return

        self.is_running = False

        try:
            # Cancel monitoring task
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            # Stop all components
            await self.poller.stop()
            await self.detector.stop()
            await self.event_bus.stop()

            if self.stats['system_started_at']:
                self.stats['uptime_seconds'] = (
                    datetime.now() - self.stats['system_started_at']
                ).total_seconds()

            logger.info("Trigger system stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping trigger system: {e}")

    async def add_symbol(
        self,
        symbol: str,
        priority: SymbolPriority = SymbolPriority.MEDIUM,
        market_hours_only: bool = True
    ):
        """Add a symbol to trigger monitoring."""
        # Add to poller
        self.poller.add_symbol(symbol, priority, market_hours_only)

        # Add to monitored set
        self.monitored_symbols.add(symbol)

        logger.info(f"Added {symbol} to trigger monitoring with {priority.value} priority")

    async def remove_symbol(self, symbol: str):
        """Remove a symbol from trigger monitoring."""
        # Remove from poller
        self.poller.remove_symbol(symbol)

        # Remove from monitored set
        self.monitored_symbols.discard(symbol)

        # Reset detector state for symbol
        self.detector.get_detector(TriggerType.TECHNICAL).reset_symbol_state(symbol)

        logger.info(f"Removed {symbol} from trigger monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop that polls data and detects triggers."""
        logger.info("Starting trigger monitoring loop")

        while self.is_running:
            try:
                # Check if we have symbols to monitor
                if not self.monitored_symbols:
                    await asyncio.sleep(5)
                    continue

                # Process each symbol
                monitoring_tasks = []
                for symbol in self.monitored_symbols:
                    task = asyncio.create_task(self._monitor_symbol(symbol))
                    monitoring_tasks.append(task)

                # Wait for all monitoring tasks to complete
                if monitoring_tasks:
                    await asyncio.gather(*monitoring_tasks, return_exceptions=True)

                # Brief sleep before next cycle
                await asyncio.sleep(2)  # Monitor every 2 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

        logger.info("Trigger monitoring loop stopped")

    async def _monitor_symbol(self, symbol: str):
        """Monitor a single symbol for triggers."""
        try:
            # Get current price (from poller or fresh data)
            current_price = await self.poller.get_real_time_price(symbol)
            if not current_price:
                return

            # Process market data for triggers
            market_data = await self._get_market_data_for_symbol(symbol, current_price)
            if not market_data:
                return

            # Detect triggers
            trigger_events = await self.detector.detect_triggers(symbol, market_data)

            if trigger_events:
                for trigger_type, events in trigger_events.items():
                    for event in events:
                        # Check cooldown
                        trigger_record = TriggerRecord(
                            symbol=event.symbol,
                            trigger_type=event.trigger_type,
                            severity=event.severity,
                            timestamp=event.timestamp,
                            trigger_id=event.data.get('trigger_id', f"{symbol}_{trigger_type.value}_{event.timestamp.timestamp()}"),
                            metadata=event.data
                        )

                        should_block, reasons = await self.cooldown_manager.check_trigger_cooldown(trigger_record)

                        if not should_block:
                            # Publish trigger event
                            await self.event_bus.publish(event)
                            self.stats['triggers_detected'] += 1
                            self.stats['last_trigger_time'] = datetime.now()
                            self.stats['analyses_triggered'] += 1
                        else:
                            logger.debug(f"Trigger blocked for {symbol}: {', '.join(reasons)}")

        except Exception as e:
            logger.error(f"Error monitoring symbol {symbol}: {e}")

    async def _get_market_data_for_symbol(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Get market data for trigger analysis."""
        try:
            # Get recent historical data for technical indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 60 days for indicators

            # Fetch and process data through data pipeline
            market_data_obj = await self.data_pipeline.fetch_and_process_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                historical_periods=50  # Enough for most indicators
            )

            if not market_data_obj:
                return None

            # Convert to trigger-friendly format
            market_data = {
                'symbol': symbol,
                'current_price': current_price,
                'prev_price': market_data_obj.historical_indicators.get('Close', [0])[-2] if len(market_data_obj.historical_indicators.get('Close', [])) > 1 else current_price,
                'indicators': market_data_obj.historical_indicators,
                'historical_data': market_data_obj.historical_indicators,
                'timestamp': datetime.now().isoformat()
            }

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'is_running': self.is_running,
            'monitored_symbols_count': len(self.monitored_symbols),
            'monitored_symbols': list(self.monitored_symbols),
            **self.stats
        }

        # Add component statistics
        if hasattr(self, 'poller'):
            stats['poller'] = self.poller.get_statistics()

        if hasattr(self, 'detector'):
            stats['detector'] = self.detector.get_composite_statistics()

        if hasattr(self, 'event_bus'):
            stats['event_bus'] = self.event_bus.get_statistics()

        if hasattr(self, 'cooldown_manager'):
            cooldown_stats = self.cooldown_manager.stats.copy()
            cooldown_stats['trigger_records_count'] = len(self.cooldown_manager.trigger_records)
            stats['cooldown_manager'] = cooldown_stats

        if hasattr(self, 'decision_ttl'):
            stats['decision_ttl'] = self.decision_ttl.get_statistics()

        # Calculate analysis success rate
        if stats['analyses_triggered'] > 0:
            stats['analysis_success_rate'] = stats['analyses_completed'] / stats['analyses_triggered']
        else:
            stats['analysis_success_rate'] = 0.0

        return stats

    async def force_trigger_check(self, symbol: str) -> Dict[str, Any]:
        """Force an immediate trigger check for a symbol."""
        try:
            if symbol not in self.monitored_symbols:
                return {'error': f'Symbol {symbol} is not monitored'}

            # Get current price
            current_price = await self.poller.get_real_time_price(symbol)
            if not current_price:
                return {'error': f'Could not get price for {symbol}'}

            # Get market data
            market_data = await self._get_market_data_for_symbol(symbol, current_price)
            if not market_data:
                return {'error': f'Could not get market data for {symbol}'}

            # Detect triggers
            trigger_results = await self.detector.detect_triggers(symbol, market_data)

            # Format results
            results = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'triggers_found': {}
            }

            for trigger_type, events in trigger_results.items():
                results['triggers_found'][trigger_type.value] = [
                    {
                        'severity': event.severity.value,
                        'description': event.description,
                        'confidence': event.confidence,
                        'data': event.data
                    }
                    for event in events
                ]

            return results

        except Exception as e:
            logger.error(f"Error in force trigger check for {symbol}: {e}")
            return {'error': str(e)}


# Factory function for easy initialization
async def create_trigger_system(
    batch_provider: UnifiedBatchProvider,
    cache: Optional[CacheManager] = None,
    signal_generator: Optional[LocalSignalGenerator] = None,
    orchestrator: Optional[Orchestrator] = None,
    symbols: Optional[List[str]] = None,
    config: Optional[TriggerSystemConfig] = None
) -> TriggerSystem:
    """
    Create and initialize a trigger system.

    Args:
        batch_provider: Batch provider for data fetching
        cache: Cache manager
        signal_generator: Local signal generator
        orchestrator: System orchestrator
        symbols: Initial symbols to monitor
        config: System configuration

    Returns:
        Initialized trigger system
    """
    system = TriggerSystem(
        batch_provider=batch_provider,
        cache=cache,
        signal_generator=signal_generator,
        orchestrator=orchestrator,
        config=config
    )

    # Add initial symbols
    if symbols:
        for symbol in symbols:
            await system.add_symbol(symbol)

    return system
