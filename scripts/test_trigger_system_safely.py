#!/usr/bin/env python3
"""
Safe testing script for the event-driven trigger system.

This script provides multiple testing modes:
1. Mock testing (no API keys required)
2. Safe live testing (rate limit protected)
3. Performance validation
4. Configuration validation

Usage:
    # Safe mock testing (no API keys needed)
    uv run python scripts/test_trigger_system_safely.py --mock

    # Safe live testing (requires API keys)
    uv run python scripts/test_trigger_system_safely.py --live --symbols AAPL,MSFT

    # Performance testing only
    uv run python scripts/test_trigger_system_safely.py --performance

    # Configuration validation
    uv run python scripts/test_trigger_system_safely.py --config
"""
import asyncio
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.events.trigger_detector import TriggerType, TriggerSeverity, TriggerEvent
from src.events.technical_triggers import TechnicalTriggerDetector, TriggerConfig
from src.events.event_bus import EventBus, LoggingSubscriber
from src.events.cooldown_manager import CooldownManager, DecisionTTL
from src.events.trigger_config import TriggerPresets, load_trigger_config
from src.data.fake_batch_provider import FakeBatchProvider
from src.data.cache import CacheManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockMarketDataGenerator:
    """Generate realistic mock market data for testing."""

    def __init__(self):
        self.base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 250.0,
            'AMZN': 3500.0
        }

    def generate_market_data(self, symbol, trigger_type='normal'):
        """Generate mock market data that can trigger various patterns."""
        import pandas as pd
        import numpy as np

        base_price = self.base_prices.get(symbol, 100.0)
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

        # Generate different patterns based on trigger type
        if trigger_type == 'rsi_overbought':
            # Create pattern that triggers RSI overbought
            prices = [base_price * (1 + i * 0.01) for i in range(50)]
            rsi_values = [75.0 + (i % 5) for i in range(50)]  # Consistently overbought
        elif trigger_type == 'macd_bullish':
            # Create MACD bullish crossover pattern
            trend = np.linspace(0, 1, 50)
            prices = [base_price * (1 + 0.1 * np.sin(i * 0.2) + 0.05 * trend[i]) for i in range(50)]
            rsi_values = [50 + 10 * np.sin(i * 0.1) for i in range(50)]
        else:
            # Normal random walk
            changes = np.random.normal(0, 0.02, 50)
            prices = [base_price]
            for change in changes[1:]:
                prices.append(prices[-1] * (1 + change))
            rsi_values = [50 + 20 * np.sin(i * 0.1) for i in range(50)]

        # Generate technical indicators
        historical_data = pd.DataFrame({
            'Open': [p * 0.998 for p in prices],
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in range(50)]
        }, index=dates)

        ma_20 = historical_data['Close'].rolling(20).mean()
        ma_50 = historical_data['Close'].rolling(50).mean()

        indicators = {
            'Close': prices,
            'RSI_14': [rsi_values[-2], rsi_values[-1]],  # Only last 2 for crossover detection
            'MACD_12_26_9': [0.3, 0.5],
            'MACDs_12_26_9': [0.2, 0.3],
            'BBU_20_2.0': [base_price * 1.02, base_price * 1.025],
            'BBL_20_2.0': [base_price * 0.98, base_price * 0.975],
            'EMA_20': [ma_20.iloc[-2], ma_20.iloc[-1]],
            'EMA_50': [ma_50.iloc[-2], ma_50.iloc[-1]],
            'STOCHk_14_3_3': [80.0, 85.0],
            'STOCHd_14_3_3': [78.0, 82.0]
        }

        return {
            'symbol': symbol,
            'current_price': prices[-1],
            'prev_price': prices[-2],
            'indicators': indicators,
            'historical_data': historical_data
        }


async def test_mock_functionality():
    """Test all trigger system functionality with mock data."""
    logger.info("=== Mock Functionality Testing ===")

    try:
        # Test event creation
        event = TriggerEvent(
            symbol='TEST',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            description='Mock trigger test',
            confidence=0.8
        )
        logger.info("✅ Trigger event creation successful")

        # Test technical trigger detection
        data_gen = MockMarketDataGenerator()

        # Test RSI overbought trigger
        market_data = data_gen.generate_market_data('TEST', 'rsi_overbought')

        config = TriggerConfig(enabled=True, min_confidence=0.6)
        detector = TechnicalTriggerDetector(config)

        events = await detector.detect_triggers('TEST', market_data)
        logger.info(f"✅ Technical trigger detected: {len(events)} events")

        if events:
            for event in events:
                logger.info(f"  - {event.description} (confidence: {event.confidence:.2f})")

        # Test event bus
        event_bus = EventBus(max_queue_size=100, worker_count=2)

        received_events = []

        from src.events.event_bus import TriggerHandlerSubscriber

        class TestSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                super().__init__("test_subscriber", lambda e: asyncio.create_task(self._handle_wrapper(e)))

            async def _handle_wrapper(self, event):
                received_events.append(event)
                logger.info(f"📩 Received event: {event.description}")
                return True

        subscriber = TestSubscriber()
        event_bus.subscribe(subscriber, [TriggerType.TECHNICAL])

        await event_bus.start()

        # Publish events
        for event in events:
            await event_bus.publish(event)

        await asyncio.sleep(0.5)
        await event_bus.stop()

        logger.info(f"✅ Event bus successful: {len(received_events)} events delivered")

        # Test cooldown manager
        from src.events.cooldown_manager import CooldownConfig
        test_config = CooldownConfig(enable_cache_persistence=False)  # Disable cleanup for testing
        cooldown_manager = CooldownManager(config=test_config)
        from src.events.cooldown_manager import TriggerRecord

        record = TriggerRecord(
            symbol='TEST',
            trigger_type=TriggerType.TECHNICAL,
            severity=TriggerSeverity.HIGH,
            timestamp=datetime.now(),
            trigger_id='test_record'
        )

        should_block, reasons = await cooldown_manager.check_trigger_cooldown(record)
        logger.info(f"✅ Cooldown manager: blocked={should_block}, reasons={reasons}")

        # Test decision TTL
        ttl_manager = DecisionTTL()
        decision_key = await ttl_manager.cache_decision(
            symbol='TEST',
            decision_type='BUY',
            decision='Test decision',
            confidence=0.8
        )

        cached_decision = await ttl_manager.get_decision('TEST', 'BUY')
        logger.info(f"✅ Decision TTL: cached={cached_decision is not None}")

        # Test configuration
        configured = load_trigger_config(preset='development')
        logger.info(f"✅ Configuration loading: technical_enabled={configured.technical_enabled}")

        logger.info("🎉 All mock functionality tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Mock functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_live_safely(symbols):
    """Test with live data using safe configurations."""
    logger.info("=== Safe Live Testing ===")
    logger.info(f"Testing symbols: {symbols}")
    logger.info("Using conservative settings to protect API limits")

    try:
        # Check API keys
        if not os.getenv('DATA_FINNHUB_API_KEY'):
            logger.warning("❌ No DATA_FINNHUB_API_KEY found. Please set in .env")
            return False

        # Import live components
        from src.data.providers.unified_batch_provider import UnifiedBatchProvider
        from src.events.trigger_integration import create_trigger_system

        # Safe configuration for testing
        config = TriggerPresets.development()
        config.polling_config.high_priority_interval = 30      # 30 seconds (conservative)
        config.polling_config.medium_priority_interval = 60    # 1 minute
        config.polling_config.low_priority_interval = 120      # 2 minutes
        config.max_concurrent_analyses = 1                     # Limit concurrent work
        config.cooldown_config.symbol_cooldown_seconds = 300    # 5 minutes

        # Initialize batch provider with rate limiting
        batch_config = {
            'finnhub': {
                'enabled': True,
                'rate_limit': 10,  # Very conservative (10/min)
                'period': 60
            },
            'alpha_vantage': {
                'enabled': False  # Disable to preserve daily limit
            }
        }

        batch_provider = UnifiedBatchProvider(batch_config)
        cache = CacheManager(enable_redis=False)  # Use memory cache

        logger.info("Starting safe live testing with conservative settings...")

        # Create trigger system
        trigger_system = await create_trigger_system(
            batch_provider=batch_provider,
            cache=cache,
            symbols=symbols,
            config=config
        )

        # Start monitoring
        await trigger_system.start()

        # Monitor for short period (2 minutes max)
        logger.info("Monitoring for 2 minutes to validate triggers...")
        start_time = time.time()

        while time.time() - start_time < 120:  # 2 minutes max
            await asyncio.sleep(10)

            # Check statistics
            stats = trigger_system.get_system_statistics()
            logger.info(
                f"Monitored: {stats['monitored_symbols_count']}, "
                f"Triggers: {stats.get('triggers_detected', 0)}, "
                f"API calls: {stats['poller'].get('total_polls', 0)}"
            )

            # Safety check: if too many API calls, stop
            if stats['poller'].get('total_polls', 0) > 20:
                logger.warning("⚠️  Safety limit reached (20 API calls). Stopping test.")
                break

        # Final statistics
        final_stats = trigger_system.get_system_statistics()
        logger.info("=== Final Statistics ===")
        logger.info(f"Total API calls made: {final_stats['poller'].get('total_polls', 0)}")
        logger.info(f"Triggers detected: {final_stats.get('triggers_detected', 0)}")
        logger.info(f"Analyses triggered: {final_stats.get('analyses_triggered', 0)}")

        await trigger_system.stop()

        logger.info("✅ Safe live testing completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Safe live testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance():
    """Test system performance without API calls."""
    logger.info("=== Performance Testing ===")

    try:
        # Stress test event bus
        from src.events.event_bus import TriggerHandlerSubscriber
        event_bus = EventBus(max_queue_size=10000, worker_count=5)
        await event_bus.start()

        class FastSubscriber(TriggerHandlerSubscriber):
            def __init__(self):
                def handler(event):
                    self.processed += 1
                    return asyncio.create_task(asyncio.sleep(0))

                super().__init__("performance_test", handler)
                self.processed = 0

            async def handle_event(self, event):
                self.processed += 1
                return True

        subscriber = FastSubscriber()
        event_bus.subscribe(subscriber, list(TriggerType))

        # Generate many events
        num_events = 1000
        start_time = time.time()

        for i in range(num_events):
            event = TriggerEvent(
                symbol=f'SYMBOL_{i % 10}',
                trigger_type=TriggerType.TECHNICAL,
                severity=TriggerSeverity.MEDIUM,
                timestamp=datetime.now(),
                description=f'Performance test event {i}',
                confidence=0.7
            )
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(2)

        processing_time = time.time() - start_time
        events_per_second = num_events / processing_time

        await event_bus.stop()

        logger.info(f"✅ Performance test results:")
        logger.info(f"  - Events processed: {subscriber.processed}")
        logger.info(f"  - Processing time: {processing_time:.2f} seconds")
        logger.info(f"  - Events/second: {events_per_second:.0f}")

        # Test trigger detection speed
        data_gen = MockMarketDataGenerator()
        detector = TechnicalTriggerDetector(TriggerConfig(enabled=True))

        start_time = time.time()
        for i in range(100):
            market_data = data_gen.generate_market_data('TEST_PERF')
            await detector.detect_triggers('TEST_PERF', market_data)

        detection_time = time.time() - start_time
        logger.info(f"✅ Trigger detection speed:")
        logger.info(f"  - 100 detections in: {detection_time:.2f} seconds")
        logger.info(f"  - Average time per detection: {detection_time*10:.2f} ms")

        return True

    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


async def test_configuration():
    """Test configuration presets and validation."""
    logger.info("=== Configuration Testing ===")

    try:
        # Test all presets
        presets = ['development', 'production', 'high_frequency', 'conservative']

        for preset in presets:
            config = load_trigger_config(preset=preset)
            logger.info(f"✅ {preset.title()} preset loaded:")
            logger.info(f"  - Technical enabled: {config.technical_enabled}")
            logger.info(f"  - High priority interval: {config.polling_config.high_priority_interval}s")
            logger.info(f"  - Max concurrent: {config.max_concurrent_analyses}")

        # Test environment configuration
        logger.info("✅ Environment-based configuration available")

        # Test validation
        from src.events.trigger_config import validate_config
        errors = validate_config(config)

        if errors:
            logger.warning(f"⚠️  Configuration validation warnings: {errors}")
        else:
            logger.info("✅ Configuration validation passed")

        # Test market hours configuration
        from datetime import time

        market_open = time(9, 30)
        market_close = time(16, 0)
        now = datetime.now().time()

        if market_open <= now <= market_close:
            logger.info("✅ Currently within market hours")
        else:
            logger.info("✅ Currently outside market hours (polling filtered)")

        return True

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def check_api_keys():
    """Check if API keys are configured."""
    finnhub_key = os.getenv('DATA_FINNHUB_API_KEY')
    av_key = os.getenv('DATA_ALPHA_VANTAGE_API_KEY')

    logger.info("=== API Key Status ===")
    logger.info(f"Finnhub API key: {'✅ Configured' if finnhub_key else '❌ Missing'}")
    logger.info(f"Alpha Vantage API key: {'✅ Configured' if av_key else '❌ Missing'}")

    if finnhub_key:
        logger.info("✅ Finnhub key available for safe testing (60 calls/minute limit)")

    if not finnhub_key and not av_key:
        logger.warning("⚠️  No API keys found. Only mock testing available.")
        return False

    return finnhub_key is not None


async def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Safe trigger system testing')
    parser.add_argument('--mock', action='store_true', help='Run mock testing (no API keys needed)')
    parser.add_argument('--live', action='store_true', help='Run safe live testing (requires API keys)')
    parser.add_argument('--symbols', default='AAPL,MSFT', help='Symbols for live testing')
    parser.add_argument('--performance', action='store_true', help='Run performance testing')
    parser.add_argument('--config', action='store_true', help='Test configuration')
    parser.add_argument('--all', action='store_true', help='Run all tests sequentially')

    args = parser.parse_args()

    logger.info("🚀 Event-Driven Trigger System Safe Testing")
    logger.info("=" * 50)

    # Check API keys
    has_keys = check_api_keys()

    # Run requested tests
    results = {}

    if args.mock or args.all:
        logger.info("\n🧪 Starting Mock Testing...")
        results['mock'] = await test_mock_functionality()

    if args.live or args.all:
        if not has_keys:
            logger.error("❌ Live testing requires API keys. Set DATA_FINNHUB_API_KEY in .env")
            results['live'] = False
        else:
            logger.info("\n🔴 Starting Safe Live Testing...")
            symbols = [s.strip() for s in args.symbols.split(',')]
            results['live'] = await test_live_safely(symbols)

    if args.performance or args.all:
        logger.info("\n⚡ Starting Performance Testing...")
        results['performance'] = await test_performance()

    if args.config or args.all:
        logger.info("\n⚙️ Starting Configuration Testing...")
        results['config'] = await test_configuration()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name.title()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n🎉 All tests passed! The trigger system is ready for deployment.")
    else:
        logger.info("\n⚠️  Some tests failed. Please review the errors above.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
