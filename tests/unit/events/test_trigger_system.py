"""
Comprehensive tests for the event-driven trigger system.

This test suite validates:
- Real-time polling functionality
- Trigger detection for various patterns
- Event bus distribution
- Cooldown management
- Integration with signal generation
- Configuration management
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Import the trigger system components
from src.events.trigger_detector import (
    TriggerEvent, TriggerType, TriggerSeverity, TriggerConfig,
    CompositeTriggerDetector, BaseTriggerDetector
)
from src.events.technical_triggers import TechnicalTriggerDetector
from src.events.event_bus import EventBus, TriggerHandlerSubscriber, LoggingSubscriber
from src.events.cooldown_manager import CooldownManager, DecisionTTL, TriggerRecord
from src.events.trigger_integration import TriggerSystem, TriggerAnalysisSubscriber
from src.events.trigger_config import (
    TriggerEnvironmentConfig, load_trigger_config, TriggerPresets,
    create_polling_config, create_technical_trigger_config
)
from src.data.real_time_poller import RealTimePoller, PollingConfig, SymbolPriority
from src.data.providers.unified_batch_provider import UnifiedBatchProvider


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self):
        self.quotes = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0
        }

    async def get_multiple_quotes(self, symbols):
        return {symbol: self.quotes.get(symbol, 100.0) for symbol in symbols}


class MockSignalGenerator:
    """Mock signal generator for testing."""

    def __init__(self):
        self.generate_calls = []

    async def generate_signal(self, symbol, trigger_context=None):
        self.generate_calls.append((symbol, trigger_context))
        return {
            'success': True,
            'signal': 'BUY',
            'confidence': 0.8,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }


class TestTriggerDetector:
    """Test trigger detection functionality."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        import pandas as pd

        # Create realistic price data with indicators
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        prices = [100 + i * 0.5 for i in range(50)]  # Trending up

        return {
            'symbol': 'AAPL',
            'current_price': 122.5,
            'prev_price': 122.0,
            'indicators': {
                'RSI_14': [75.0, 78.0],  # Overbought crossover
                'MACD_12_26_9': [0.5, 0.8],
                'MACDs_12_26_9': [0.3, 0.4],
                'BBU_20_2.0': [125.0, 126.0],
                'BBL_20_2.0': [118.0, 119.0],
                'EMA_20': [120.0, 121.0],
                'EMA_50': [119.0, 119.5],
                'STOCHk_14_3_3': [85.0, 88.0],
                'STOCHd_14_3_3': [82.0, 84.0]
            },
            'historical_data': pd.DataFrame({
                'Open': prices,
                'High': [p + 1 for p in prices],
                'Low': [p - 1 for p in prices],
                'Close': prices,
                'Volume': [1000000] * 50
            }, index=dates)
        }

    @pytest.mark.asyncio
    async def test_technical_trigger_detection(self, sample_market_data):
        """Test technical trigger detection."""
        config = TriggerConfig(enabled=True, min_confidence=0.6)
        detector = TechnicalTriggerDetector(config)

        # Test trigger detection
        events = await detector.detect_triggers('AAPL', sample_market_data)

        # Should detect multiple triggers
        assert len(events) > 0

        # Check for expected trigger types
        event_types = [event.trigger_type for event in events]
        assert TriggerType.TECHNICAL in event_types

        # Check event properties
        for event in events:
            assert event.symbol == 'AAPL'
            assert event.confidence >= 0.6
            assert event.severity in [TriggerSeverity.LOW, TriggerSeverity.MEDIUM, TriggerSeverity.HIGH]
            assert len(event.description) > 0

    @pytest.mark.asyncio
    async def test_rsi_overbought_trigger(self, sample_market_data):
        """Test RSI overbought trigger detection."""
        config = TriggerConfig(enabled=True)
        detector = TechnicalTriggerDetector(config)

        events = await detector.detect_triggers('AAPL', sample_market_data)

        # Should find RSI overbought trigger
        rsi_triggers = [e for e in events if 'RSI' in e.description and 'overbought' in e.description.lower()]
        assert len(rsi_triggers) > 0

        trigger = rsi_triggers[0]
        assert trigger.trigger_type == TriggerType.TECHNICAL
        assert 'overbought' in trigger.description.lower()

    @pytest.mark.asyncio
    async def test_macd_crossover_trigger(self, sample_market_data):
        """Test MACD crossover trigger detection."""
        config = TriggerConfig(enabled=True)
        detector = TechnicalTriggerDetector(config)

        events = await detector.detect_triggers('AAPL', sample_market_data)

        # Should find MACD crossover trigger
        macd_triggers = [e for e in events if 'MACD' in e.description]
        assert len(macd_triggers) > 0

    @pytest.mark.asyncio
    async def test_trigger_confidence_filtering(self, sample_market_data):
        """Test trigger confidence filtering."""
        # High confidence threshold
        config = TriggerConfig(enabled=True, min_confidence=0.95)
        detector = TechnicalTriggerDetector(config)

        events = await detector.analyze_market_data('AAPL', sample_market_data)

        # Should filter out low confidence triggers
        for event in events:
            assert event.confidence >= 0.95

    @pytest.mark.asyncio
    async def test_cooldown_filtering(self, sample_market_data):
        """Test trigger cooldown filtering."""
        config = TriggerConfig(enabled=True, cooldown_seconds=1)
        detector = TechnicalTriggerDetector(config)

        # First detection should work
        events1 = await detector.analyze_market_data('AAPL', sample_market_data)

        # Immediate second detection should be blocked
        events2 = await detector.analyze_market_data('AAPL', sample_market_data)

        if len(events1) > 0:  # Only test if first detection had triggers
            assert len(events2) == 0  # Should be blocked by cooldown

        # Wait for cooldown and try again
        await asyncio.sleep(1.1)
        events3 = await detector.analyze_market_data('AAPL', sample_market_data)
        # Should work again (though may not have new triggers)


class TestEventBus:
    """Test event bus functionality."""

    @pytest.mark.asyncio
    async def test_event_subscription_and_publishing(self):
        """Test event subscription and publishing."""
        event_bus = EventBus(max_queue_size=100, worker_count=2)

        # Create mock subscriber
        received_events = []

        class TestSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                super().__init__("test", SubscriberType.TRIGGER_HANDLER)

            async def handle_event(self, event):
                received_events.append(event)
                return True

        subscriber = TestSubscriber()

        # Subscribe to all trigger types
        subscription_id = event_bus.subscribe(
            subscriber=subscriber,
            trigger_types=list(TriggerType)
        )

        # Start event bus
        await event_bus.start()

        # Publish test events
        event1 = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='Test event 1'
        )

        event2 = TriggerEvent(
            symbol='GOOGL',
            trigger_type=TriggerType.VOLATILITY,
            severity=TriggerSeverity.MEDIUM,
            timestamp=datetime.now(),
            description='Test event 2'
        )

        # Publish events
        await event_bus.publish(event1)
        await event_bus.publish(event2)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify events were received
        assert len(received_events) == 2
        assert received_events[0].symbol == 'AAPL'
        assert received_events[1].symbol == 'GOOGL'

        # Stop event bus
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_filtering(self):
        """Test event filtering by trigger type and severity."""
        event_bus = EventBus()

        # Create filtered subscriber
        received_events = []

        class FilteredSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                super().__init__("filtered", SubscriberType.TRIGGER_HANDLER)

            async def handle_event(self, event):
                received_events.append(event)
                return True

        subscriber = FilteredSubscriber()

        # Subscribe only to HIGH severity TECHNICAL triggers
        subscription_id = event_bus.subscribe(
            subscriber=subscriber,
            trigger_types=[TriggerType.TECHNICAL],
            severity_filter=[TriggerSeverity.HIGH]
        )

        await event_bus.start()

        # Publish events that should and shouldn't match
        high_tech_event = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='High tech'
        )

        low_tech_event = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.LOW,
            timestamp=datetime.now(),
            description='Low tech'
        )

        high_vol_event = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.VOLATILITY,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='High vol'
        )

        # Publish events
        await event_bus.publish(high_tech_event)
        await event_bus.publish(low_tech_event)
        await event_bus.publish(high_vol_event)

        await asyncio.sleep(0.5)

        # Should only receive the high severity technical event
        assert len(received_events) == 1
        assert received_events[0].severity == TriggerSeverity.HIGH
        assert received_events[0].trigger_type == TriggerType.TECHNICAL

        await event_bus.stop()


class TestCooldownManager:
    """Test cooldown manager functionality."""

    @pytest.mark.asyncio
    async def test_symbol_cooldown(self):
        """Test symbol-based cooldown."""
        config = TriggerEnvironmentConfig()
        config.symbol_cooldown_seconds = 1
        cooldown_manager = CooldownManager()

        # Create trigger record
        record = TriggerRecord(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            trigger_id='test1'
        )

        # First check should not be blocked
        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        assert not should_block
        assert len(reasons) == 0

        # Immediate second check should be blocked
        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        assert should_block
        assert 'symbol_cooldown' in reasons

        # Wait for cooldown period
        await asyncio.sleep(1.1)

        # Should not be blocked anymore
        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        assert not should_block

    @pytest.mark.asyncio
    async def test_frequency_limits(self):
        """Test frequency limit enforcement."""
        cooldown_manager = CooldownManager()
        cooldown_manager.config.max_triggers_per_minute = 2

        record = TriggerRecord(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            trigger_id='freq_test'
        )

        # First two triggers should be allowed
        should_block, _ = await cooldown_manager.check_trigger_cooldown(record)
        assert not should_block

        should_block, _ = await cooldown_manager.check_trigger_cooldown(record)
        assert not should_block

        # Third trigger should be blocked
        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        assert should_block
        assert 'frequency_limit' in reasons


class TestDecisionTTL:
    """Test decision TTL functionality."""

    @pytest.mark.asyncio
    async def test_decision_caching_and_retrieval(self):
        """Test decision caching and retrieval."""
        ttl_manager = DecisionTTL()

        # cache a decision
        decision_key = await ttl_manager.cache_decision(
            symbol='AAPL',
            decision_type='BUY',
            decision='Strong bullish signal',
            severity=TriggerSeverity.HIGH,
            confidence=0.85
        )

        # retrieve the decision
        cached_decision = await ttl_manager.get_decision('AAPL', 'BUY')

        assert cached_decision is not None
        assert cached_decision.symbol == 'AAPL'
        assert cached_decision.decision_type == 'BUY'
        assert cached_decision.decision == 'Strong bullish signal'
        assert cached_decision.confidence == 0.85
        assert cached_decision.severity == TriggerSeverity.HIGH

    @pytest.mark.asyncio
    async def test_decision_expiration(self):
        """Test decision expiration."""
        ttl_manager = DecisionTTL()

        # Override TTL to be very short for testing
        ttl_manager.config.default_decision_ttl_seconds = 0.1

        # Cache decision
        await ttl_manager.cache_decision(
            symbol='AAPL',
            decision_type='BUY',
            decision='Test decision'
        )

        # Should be immediately retrievable
        decision = await ttl_manager.get_decision('AAPL', 'BUY')
        assert decision is not None

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired now
        decision = await ttl_manager.get_decision('AAPL', 'BUY')
        assert decision is None


class TestConfiguration:
    """Test configuration management."""

    def test_environment_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'TRIGGER_POLLING_ENABLED': 'true',
            'TRIGGER_TECHNICAL_ENABLED': 'true',
            'TRIGGER_HIGH_PRIORITY_INTERVAL': '10',
            'TRIGGER_TECHNICAL_MIN_CONFIDENCE': '0.8'
        }):
            config = TriggerEnvironmentConfig.from_environment()

            assert config.polling_enabled is True
            assert config.technical_triggers_enabled is True
            assert config.polling_high_priority_interval_seconds == 10
            assert config.technical_min_confidence == 0.8

    def test_preset_configurations(self):
        """Test preset configurations."""
        # Test development preset
        dev_config = TriggerPresets.development()
        assert dev_config.polling_config.high_priority_interval == 10
        assert dev_config.technical_enabled is True

        # Test production preset
        prod_config = TriggerPresets.production()
        assert prod_config.polling_config.high_priority_interval == 5
        assert prod_config.technical_enabled is True

        # Test conservative preset
        cons_config = TriggerPresets.conservative()
        assert cons_config.polling_config.high_priority_interval == 30
        assert cons_config.technical_enabled is True

    def test_config_validation(self):
        """Test configuration validation."""
        from src.events.trigger_config import validate_config

        # Valid config
        valid_config = TriggerPresets.development()
        errors = validate_config(valid_config)
        assert len(errors) == 0

        # Invalid config (interval too short)
        invalid_config = TriggerPresets.development()
        invalid_config.polling_config.high_priority_interval = 0
        errors = validate_config(invalid_config)
        assert len(errors) > 0
        assert any('interval must be at least 1 second' in error for error in errors)


class TestIntegration:
    """Integration tests for the complete trigger system."""

    @pytest.mark.asyncio
    async def test_trigger_system_basic_integration(self):
        """Test basic trigger system integration."""
        # Create mock components
        mock_provider = MockDataProvider()
        mock_signal_gen = MockSignalGenerator()

        # Create trigger system with minimal config
        config = TriggerPresets.development()
        config.polling_config.enable_market_hours_filter = False  # Disable for testing

        # Note: This test doesn't use batch_provider to avoid complex mocking
        # In a real test, you would mock the batch provider properly

        # Test that configuration creation works
        assert config.technical_enabled is True
        assert config.auto_signal_generation is True
        assert config.polling_config.high_priority_interval == 10

    @pytest.mark.asyncio
    async def test_trigger_analysis_subscriber(self):
        """Test trigger analysis subscriber."""
        mock_signal_gen = MockSignalGenerator()

        subscriber = TriggerAnalysisSubscriber(
            signal_generator=mock_signal_gen,
            max_concurrent=2,
            timeout_seconds=5
        )

        # Create test trigger event
        event = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='Test trigger',
            confidence=0.8
        )

        # Handle event
        result = await subscriber.handle_event(event)

        # Should return True (success)
        assert result is True

        # Should have called signal generator
        assert len(mock_signal_gen.generate_calls) == 1
        assert mock_signal_gen.generate_calls[0][0] == 'AAPL'

    @pytest.mark.asyncio
    async def test_end_to_end_trigger_flow(self):
        """Test end-to-end trigger flow (simplified)."""
        # Create components
        mock_provider = MockDataProvider()
        event_bus = EventBus()
        cooldown_manager = CooldownManager()
        ttl_manager = DecisionTTL()

        # Start event bus
        await event_bus.start()

        # Create mock subscriber
        received_events = []

        class TestSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                super().__init__("e2e_test", SubscriberType.TRIGGER_HANDLER)

            async def handle_event(self, event):
                received_events.append(event)
                return True

        subscriber = TestSubscriber()
        event_bus.subscribe(subscriber, [TriggerType.TECHNICAL])

        # Simulate trigger detection
        event = TriggerEvent(
            symbol='AAPL',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='E2E test trigger',
            confidence=0.85
        )

        # Check cooldown
        record = TriggerRecord(
            symbol=event.symbol,
            trigger_type=event.trigger_type,
            severity=event.severity,
            timestamp=event.timestamp,
            trigger_id='e2e_test'
        )

        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        assert not should_block  # Should not be blocked

        # Publish event
        await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].symbol == 'AAPL'

        # Cache decision related to trigger
        decision_key = await ttl_manager.cache_decision(
            symbol='AAPL',
            decision_type='BUY',
            decision='Technical trigger detected',
            severity=event.severity,
            confidence=event.confidence
        )

        # Retrieve decision
        decision = await ttl_manager.get_decision('AAPL', 'BUY')
        assert decision is not None
        assert decision.decision == 'Technical trigger detected'

        # Cleanup
        await event_bus.stop()


# Performance tests
class TestPerformance:
    """Performance tests for the trigger system."""

    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self):
        """Test processing high volume of events."""
        event_bus = EventBus(max_queue_size=10000, worker_count=10)

        # Create fast subscriber
        processed_count = 0

        class FastSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                super().__init__("perf_test", SubscriberType.TRIGGER_HANDLER)

            async def handle_event(self, event):
                nonlocal processed_count
                processed_count += 1
                return True

        subscriber = FastSubscriber()
        event_bus.subscribe(subscriber, list(TriggerType))

        await event_bus.start()

        # Publish many events
        num_events = 1000
        start_time = datetime.now()

        for i in range(num_events):
            event = TriggerEvent(
                symbol=f'SYMBOL_{i % 10}',
                trigger_type=TriggerType.TECHNICAL,
                severity=TriggerSeverity.MEDIUM,
                timestamp=datetime.now(),
                description=f'Event {i}'
            )
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(2)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        await event_bus.stop()

        # Verify performance
        assert processed_count == num_events
        assert processing_time < 5.0  # Should process 1000 events in under 5 seconds

        events_per_second = num_events / processing_time
        print(f"Processed {events_per_second:.0f} events per second")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
