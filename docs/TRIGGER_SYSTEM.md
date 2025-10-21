# Event-Driven Trigger System

This document describes the event-driven trigger system that enables real-time, automated market analysis and signal generation without requiring manual requests.

## Overview

The event-driven trigger system transforms your AI trading system from reactive (on-demand analysis) to proactive (automatic detection and response to market events). It continuously monitors market data, detects predefined patterns and conditions, and automatically initiates analysis when significant events occur.

## Architecture

### Core Components

1. **Real-Time Poller** - Continuously fetches market data with market-hours awareness
2. **Trigger Detectors** - Analyze data for specific patterns (RSI crossovers, volume spikes, etc.)
3. **Event Bus** - Distributes trigger events to registered handlers
4. **Cooldown Manager** - Prevents excessive triggers and enforces rate limits
5. **Decision TTL** - Caches analysis results with expiration to prevent redundant processing
6. **Integration Layer** - Connects with existing Local Signal Generation Framework

### Data Flow

```
Market Data → Real-Time Poller → Trigger Detectors → Event Bus → Signal Generation → Analysis Decision
     ↓                    ↓                    ↓               ↓              ↓
   Cache             Cooldown           Cooldown      Decision TTL    Orchestrator
  Management          Manager            Manager        Manager          API
```

## Features

### Technical Triggers

The system currently implements comprehensive technical analysis triggers:

- **RSI Analysis**: Overbought/oversold conditions and centerline crossovers
- **MACD Analysis**: Line crossovers and signal line crosses
- **Bollinger Bands**: Upper/lower band breaches and squeeze detection
- **Moving Averages**: EMA crossovers and price vs 200-day SMA
- **Stochastic**: Overbought/oversold and %K/%D crossovers
- **Price Patterns**: Volume spikes, gaps, support/resistance breakouts

### Smart Polling

- **Market Hours Awareness**: Automatically adjusts polling during market hours
- **Symbol Prioritization**: High/medium/low priority with configurable intervals
- **Batch Processing**: Efficient API usage through batched requests
- **Rate Limiting**: Respects API provider limits (Finnhub: 60/min, Alpha Vantage: 25/day)

### Performance Features

- **Cooldown Mechanisms**: Prevents excessive triggers (symbol-based, type-based, severity-based)
- **Decision Caching**: Caches analysis results with configurable TTL
- **Event Bus**: High-performance async event distribution
- **Resource Management**: Configurable concurrency and limits

## Configuration

### Environment Variables

Configure the trigger system using these environment variables:

```bash
# Enable/disable components
TRIGGER_POLLING_ENABLED=false
TRIGGER_TECHNICAL_ENABLED=false
TRIGGER_AUTO_SIGNAL_GENERATION=false

# Polling intervals (in seconds)
TRIGGER_HIGH_PRIORITY_INTERVAL=5
TRIGGER_MEDIUM_PRIORITY_INTERVAL=15
TRIGGER_LOW_PRIORITY_INTERVAL=60

# Market hours
TRIGGER_MARKET_HOURS_FILTER=true
TRIGGER_MARKET_OPEN_HOUR=9
TRIGGER_MARKET_OPEN_MINUTE=30
TRIGGER_MARKET_CLOSE_HOUR=16
TRIGGER_MARKET_CLOSE_MINUTE=0

# Technical trigger settings
TRIGGER_TECHNICAL_MIN_CONFIDENCE=0.7
TRIGGER_TECHNICAL_COOLDOWN_SECONDS=300

# Event processing
TRIGGER_EVENT_BUS_WORKERS=5
TRIGGER_MAX_EVENTS_PER_MINUTE=50

# Performance
TRIGGER_MAX_CONCURRENT_ANALYSES=3
TRIGGER_ANALYSIS_TIMEOUT_SECONDS=30
TRIGGER_ENABLE_REAL_TIME_CACHE=true
```

### Preset Configurations

The system includes preset configurations for different use cases:

- **Development**: Relaxed limits for testing and development
- **Production**: Balanced settings for live trading
- **High-Frequency**: Fast polling, high concurrency for active trading
- **Conservative**: Longer cooldowns, higher confidence requirements

## Usage

### Basic Setup

```python
from src.events.trigger_integration import create_trigger_system
from src.events.trigger_config import TriggerPresets
from src.data.providers.unified_batch_provider import UnifiedBatchProvider
from src.data.cache import CacheManager

# Initialize components
batch_provider = UnifiedBatchProvider()
cache = CacheManager()

# Create trigger system
trigger_system = await create_trigger_system(
    batch_provider=batch_provider,
    cache=cache,
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    config=TriggerPresets.production()
)

# Start monitoring
await trigger_system.start()

# Add symbols
await trigger_system.add_symbol('TSLA', priority=SymbolPriority.HIGH)

# Monitor statistics
stats = trigger_system.get_system_statistics()
print(f"Monitored: {stats['monitored_symbols_count']}")
print(f"Triggers: {stats['triggers_detected']}")

# Stop when done
await trigger_system.stop()
```

### Integration with Existing System

The trigger system integrates seamlessly with your existing components:

```python
from src.signal_generation.signal_generator import LocalSignalGenerator
from src.communication.orchestrator import Orchestrator

# Create with full integration
trigger_system = await create_trigger_system(
    batch_provider=batch_provider,
    cache=cache,
    signal_generator=local_signal_generator,  # For local-first analysis
    orchestrator=orchestrator,                # For full agent analysis
    symbols=['AAPL'],
    config=config
)
```

### Custom Event Handlers

```python
from src.events.event_bus import TriggerHandlerSubscriber, SubscriberType

class CustomTradingHandler(TriggerHandlerSubscriber):
    def __init__(self):
        super().__init__("trading_bot", SubscriberType.TRADER)

    async def handle_event(self, event):
        # Custom trading logic here
        if event.severity == TriggerSeverity.CRITICAL:
            await self.place_order(event.symbol, "BUY")
        return True

# Register with event bus
event_bus.subscribe(
    subscriber=CustomTradingHandler(),
    trigger_types=[TriggerType.TECHNICAL],
    severity_filter=[TriggerSeverity.HIGH, TriggerSeverity.CRITICAL]
)
```

## Monitoring and Statistics

### System Statistics

```python
stats = trigger_system.get_system_statistics()

# Key metrics
print(f"Triggers detected: {stats['triggers_detected']}")
print(f"Analyses triggered: {stats['analyses_triggered']}")
print(f"Analysis success rate: {stats['analysis_success_rate']:.2%}")
print(f"System uptime: {stats['uptime_seconds']:.0f} seconds")

# Component details
print(f"Event bus delivery rate: {stats['event_bus']['delivery_rate']:.2%}")
print(f"Cooldown block rate: {stats['cooldown_manager']['triggers_blocked']}")
print(f"Decision cache hit rate: {stats['decision_ttl']['cache_hit_rate']:.2%}")
```

### Performance Monitoring

```python
# Monitor specific symbol statistics
symbol_stats = trigger_system.detector.get_detector(TriggerType.TECHNICAL).get_symbol_statistics('AAPL')
print(f"AAPL triggers today: {symbol_stats['triggers_last_24h']}")

# Get active decisions
active_decisions = trigger_system.decision_ttl.get_active_decisions()
for decision in active_decisions:
    print(f"{decision['symbol']}: {decision['decision']} (TTL: {decision['remaining_ttl_seconds']:.0f}s)")
```

## Testing

Run the demonstration script to see the system in action:

```bash
python scripts/trigger_system_demo.py
```

Run the test suite:

```bash
# Run trigger system tests
python -m pytest tests/unit/events/test_trigger_system.py -v

# Run integration tests
python -m pytest tests/integration/test_trigger_integration.py -v
```

## Performance Considerations

### API Rate Limits

- **Finnhub**: 60 calls/minute (free tier)
- **Alpha Vantage**: 25 calls/day (free tier)
- **YFinance**: No official limits but requires throttling

### Scaling Recommendations

1. **API Optimization**: Enable caching and batch processing
2. **Symbol Limiting**: Start with 5-10 symbols, monitor performance
3. **Configuration Tuning**: Adjust intervals based on trading frequency needs
4. **Resource Management**: Monitor CPU/memory usage with high-frequency polling

### Expected Performance

- **Trigger Detection**: <50ms per symbol
- **Event Distribution**: <10ms per event
- **Signal Generation**: <100ms for local analysis
- **API Efficiency**: 60-80% reduction in calls through batching and caching

## Troubleshooting

### Common Issues

1. **No Triggers Detected**
   - Check API keys are configured
   - Verify symbols are valid and actively traded
   - Lower confidence thresholds temporarily

2. **Too Many Triggers**
   - Increase cooldown periods
   - Raise minimum confidence levels
   - Adjust polling intervals

3. **High Memory Usage**
   - Reduce history lookback periods
   - Lower event queue sizes
   - Enable Redis for external caching

### Debug Logging

Enable debug logging for detailed monitoring:

```python
import logging
logging.getLogger('src.events').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned additional trigger types:

- **Volatility Triggers**: ATR spikes, volatility regime changes
- **Trend Triggers**: ADX crossovers, Aroon signals
- **Conflict Triggers**: Divergent signals across indicators
- **Regime Triggers**: Market structure breaks

Planned improvements:

- **WebSocket Streaming**: Real-time data for sub-second latency
- **Machine Learning Triggers**: Pattern recognition beyond traditional technicals
- **Multi-Symbol Correlation**: Cross-trading opportunities
- **Risk-Adjusted Triggers**: Position-size aware triggering

## API Reference

### Core Classes

- `TriggerSystem`: Main system orchestrator
- `RealTimePoller`: Market data polling with awareness
- `TechnicalTriggerDetector`: Technical analysis triggers
- `EventBus`: Async event distribution system
- `CooldownManager`: Rate limiting and cooldowns
- `DecisionTTL`: Decision caching with expiration

### Configuration Classes

- `TriggerSystemConfig`: Complete system configuration
- `PollingConfig`: Real-time polling settings
- `TriggerConfig`: Individual detector configuration
- `CooldownConfig`: Rate limiting settings

### Event Types

- `TriggerEvent`: Trigger event container
- `TriggerType`: Types of triggers (TECHNICAL, VOLATILITY, etc.)
- `TriggerSeverity`: Severity levels (LOW, MEDIUM, HIGH, CRITICAL)

For detailed API documentation, see the inline docstrings and type hints in the source code.
