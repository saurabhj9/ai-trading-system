#!/usr/bin/env python3
"""
Demonstration script for the event-driven trigger system.

This script shows how to:
1. Initialize the trigger system
2. Add symbols for monitoring
3. Detect and handle trigger events
4. Integrate with signal generation
5. Monitor system performance

Usage: python scripts/trigger_system_demo.py
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.providers.unified_batch_provider import UnifiedBatchProvider
from src.data.cache import CacheManager
from src.events.trigger_integration import create_trigger_system, TriggerSystemConfig
from src.events.trigger_config import TriggerPresets, load_trigger_config
from src.signal_generation.signal_generator import LocalSignalGenerator
from src.communication.orchestrator import Orchestrator
from src.data.pipeline import DataPipeline
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.risk import RiskManagementAgent
from src.agents.portfolio import PortfolioManagementAgent
from src.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStateManager:
    """Mock state manager for demonstration."""
    def get_portfolio_state(self):
        return {
            "cash": 10000,
            "positions": {},
            "equity": 10000
        }


async def demo_basic_trigger_system():
    """Demonstrate basic trigger system functionality."""
    logger.info("=== Basic Trigger System Demo ===")

    # Initialize batch provider (using environment config)
    batch_config = {}
    if hasattr(settings, 'DATA_PROVIDERS'):
        batch_config.update(settings.DATA_PROVIDERS)

    batch_provider = UnifiedBatchProvider(batch_config)
    cache = CacheManager(enable_redis=False)  # Use memory cache for demo

    # Create trigger system with development config
    config = TriggerPresets.development()
    config.technical_enabled = True
    config.auto_signal_generation = True

    # Create trigger system (without signal generator for basic demo)
    trigger_system = await create_trigger_system(
        batch_provider=batch_provider,
        cache=cache,
        symbols=['AAPL', 'GOOGL'],
        config=config
    )

    try:
        # Start the trigger system
        logger.info("Starting trigger system...")
        await trigger_system.start()

        # Monitor for a short period
        logger.info("Monitoring symbols for triggers for 30 seconds...")
        await asyncio.sleep(30)

        # Check statistics
        stats = trigger_system.get_system_statistics()
        logger.info(f"System statistics: {stats}")

        # Force trigger check for a symbol
        logger.info("Force checking triggers for AAPL...")
        trigger_check = await trigger_system.force_trigger_check('AAPL')
        logger.info(f"Trigger check results: {trigger_check}")

    finally:
        # Stop trigger system
        logger.info("Stopping trigger system...")
        await trigger_system.stop()
        logger.info("Trigger system stopped")


async def demo_with_signal_generation():
    """Demonstrate trigger system with signal generation integration."""
    logger.info("=== Trigger System with Signal Generation Demo ===")

    # Initialize components
    batch_config = {}
    if hasattr(settings, 'DATA_PROVIDERS'):
        batch_config.update(settings.DATA_PROVIDERS)

    batch_provider = UnifiedBatchProvider(batch_config)
    cache = CacheManager(enable_redis=False)

    # Create data pipeline
    data_pipeline = DataPipeline(provider=batch_provider, cache=cache)

    # Create agents (using existing agent configurations)
    technical_agent = TechnicalAnalysisAgent()
    sentiment_agent = SentimentAnalysisAgent()
    risk_agent = RiskManagementAgent()
    portfolio_agent = PortfolioManagementAgent()

    # Create signal generator
    signal_generator = LocalSignalGenerator()

    # Create orchestrator
    mock_state_manager = MockStateManager()
    orchestrator = Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=technical_agent,
        sentiment_agent=sentiment_agent,
        risk_agent=risk_agent,
        portfolio_agent=portfolio_agent,
        state_manager=mock_state_manager
    )

    # Create trigger system with signal generation
    config = TriggerPresets.development()
    config.technical_enabled = True
    config.auto_signal_generation = True

    trigger_system = await create_trigger_system(
        batch_provider=batch_provider,
        cache=cache,
        signal_generator=signal_generator,
        orchestrator=orchestrator,
        symbols=['MSFT', 'TSLA'],
        config=config
    )

    try:
        # Start the trigger system
        logger.info("Starting enhanced trigger system...")
        await trigger_system.start()

        # Monitor for triggers
        logger.info("Monitoring for triggers with signal generation for 45 seconds...")
        await asyncio.sleep(45)

        # Check final statistics
        stats = trigger_system.get_system_statistics()
        logger.info(f"Enhanced system statistics: {stats}")

        # Test trigger-driven analysis
        if stats['triggers_detected'] > 0:
            logger.info("Testing trigger-driven analysis...")
            # This would normally be triggered automatically, but we'll test it manually
            from src.events.trigger_detector import TriggerEvent, TriggerType, TriggerSeverity

            test_event = TriggerEvent(
                symbol='MSFT',
                trigger_type=TriggerType.TECHNICAL,
                severity=TriggerSeverity.HIGH,
                timestamp=datetime.now(),
                description="Demo trigger for testing",
                confidence=0.8
            )

            # Analyze with trigger event
            analysis_result = await orchestrator.analyze_symbol('MSFT', test_event, force_analysis=True)
            logger.info(f"Trigger-driven analysis result: {analysis_result}")

    finally:
        # Stop trigger system
        logger.info("Stopping enhanced trigger system...")
        await trigger_system.stop()
        logger.info("Enhanced trigger system stopped")


async def demo_configuration_management():
    """Demonstrate configuration management."""
    logger.info("=== Configuration Management Demo ===")

    # Load environment-based configuration
    env_config = load_trigger_config(use_environment=True)
    logger.info(f"Environment-based config loaded")
    logger.info(f"- Technical triggers enabled: {env_config.technical_enabled}")
    logger.info(f"- Polling enabled: {env_config.polling_config.enable_market_hours_filter}")
    logger.info(f"- High priority interval: {env_config.polling_config.high_priority_interval}s")

    # Test preset configurations
    presets = ['development', 'production', 'conservative', 'high_frequency']
    for preset_name in presets:
        preset_config = load_trigger_config(preset=preset_name)
        logger.info(f"Presets demo - {preset_name}:")
        logger.info(f"  - Technical enabled: {preset_config.technical_enabled}")
        logger.info(f"  - Polling interval: {preset_config.polling_config.high_priority_interval}s")
        logger.info(f"  - Max concurrent analyses: {preset_config.max_concurrent_analyses}")

    # Validate configuration
    from src.events.trigger_config import validate_config
    errors = validate_config(env_config)
    if errors:
        logger.warning(f"Configuration validation errors: {errors}")
    else:
        logger.info("Configuration validation passed")


async def demo_performance_testing():
    """Demonstrate performance testing capabilities."""
    logger.info("=== Performance Testing Demo ===")

    # Create high-frequency configuration
    config = TriggerPresets.high_frequency()
    config.polling_config.high_priority_interval = 1  # Very fast for testing

    batch_provider = UnifiedBatchProvider()
    cache = CacheManager(enable_redis=False)

    trigger_system = await create_trigger_system(
        batch_provider=batch_provider,
        cache=cache,
        symbols=['AAPL'],  # Single symbol for focused testing
        config=config
    )

    try:
        logger.info("Starting performance test...")
        await trigger_system.start()

        # Monitor performance
        start_time = datetime.now()
        test_duration = 20  # 20-second test

        logger.info(f"Running performance test for {test_duration} seconds...")
        await asyncio.sleep(test_duration)

        end_time = datetime.now()
        test_time = (end_time - start_time).total_seconds()

        # Get performance statistics
        stats = trigger_system.get_system_statistics()

        logger.info("Performance Test Results:")
        logger.info(f"  - Test duration: {test_time:.2f} seconds")
        logger.info(f"  - Triggers detected: {stats.get('triggers_detected', 0)}")
        logger.info(f"  - Analyses triggered: {stats.get('analyses_triggered', 0)}")
        logger.info(f"  - Analyses completed: {stats.get('analyses_completed', 0)}")
        logger.info(f"  - Event bus stats: {stats.get('event_bus', {})}")

        # Calculate rates
        if test_time > 0:
            trigger_rate = stats.get('triggers_detected', 0) / test_time
            logger.info(f"  - Trigger detection rate: {trigger_rate:.2f} triggers/second")

    finally:
        await trigger_system.stop()
        logger.info("Performance test completed")


async def main():
    """Main demonstration function."""
    logger.info("Event-Driven Trigger System Demonstration")
    logger.info("=" * 50)

    try:
        # Check if environment is configured
        if not os.getenv('DATA_ALPHA_VANTAGE_API_KEY') and not os.getenv('DATA_FINNHUB_API_KEY'):
            logger.warning("No API keys configured. Using mock data for demonstration.")
            logger.info("Set DATA_ALPHA_VANTAGE_API_KEY or DATA_FINNHUB_API_KEY for real data.")

        # Run demonstrations
        await demo_configuration_management()
        print("\n" + "="*50 + "\n")

        await demo_basic_trigger_system()
        print("\n" + "="*50 + "\n")

        await demo_with_signal_generation()
        print("\n" + "="*50 + "\n")

        await demo_performance_testing()

        logger.info("All demonstrations completed successfully!")

    except KeyboardInterrupt:
        logger.info("Demonstrations interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
