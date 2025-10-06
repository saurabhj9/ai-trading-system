"""
Example usage of the Local Signal Generation Framework.

This script demonstrates how to use the signal generation framework
to produce trading signals without relying on LLM calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_generation import LocalSignalGenerator
from config.signal_generation import signal_generation_config


def generate_sample_market_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Generate sample market data for demonstration.

    Args:
        symbol: Symbol name for the data
        days: Number of days of data to generate

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating {days} days of sample market data for {symbol}...")

    # Set seed for reproducible results
    np.random.seed(42)

    # Generate date range
    dates = pd.date_range(start=datetime.now() - timedelta(days=days),
                         periods=days, freq='D')

    # Generate price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 20, days)  # Upward trend
    noise = np.random.normal(0, 2, days)  # Random noise
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, days))  # Seasonal component

    close_prices = base_price + trend + noise + seasonal

    # Generate OHLC data
    high = close_prices + np.random.uniform(0, 3, days)
    low = close_prices - np.random.uniform(0, 3, days)
    open_prices = low + np.random.uniform(0, high - low)

    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_prices, close_prices))
    low = np.minimum(low, np.minimum(open_prices, close_prices))

    # Generate volume data
    volume = np.random.uniform(1000000, 5000000, days)

    # Create DataFrame
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return market_data


def demonstrate_signal_generation():
    """Demonstrate the signal generation framework."""
    print("=" * 60)
    print("Local Signal Generation Framework - Example Usage")
    print("=" * 60)

    # 1. Initialize the signal generator
    print("\n1. Initializing Signal Generator...")
    config = signal_generation_config.to_dict()
    signal_generator = LocalSignalGenerator(config)
    print("✅ Signal generator initialized successfully")

    # 2. Generate sample market data
    print("\n2. Generating Sample Market Data...")
    market_data = generate_sample_market_data("AAPL", days=100)
    print(f"✅ Generated {len(market_data)} days of market data")
    print(f"   Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")

    # 3. Generate a signal
    print("\n3. Generating Trading Signal...")
    signal, metadata = signal_generator.generate_signal(market_data, "AAPL")

    # 4. Display signal information
    print("\n4. Signal Generated:")
    print(f"   Symbol: {signal.symbol}")
    print(f"   Signal Type: {signal.signal_type.value}")
    print(f"   Signal Strength: {signal.strength.value}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Price: ${signal.price:.2f}")
    print(f"   Market Regime: {signal.regime.value}")
    print(f"   Source: {signal.source}")
    print(f"   Timestamp: {signal.timestamp}")

    # 5. Display reasoning
    print("\n5. Signal Reasoning:")
    print(f"   {signal.reasoning}")

    # 6. Display market regime information
    print("\n6. Market Regime Information:")
    regime_info = metadata['market_regime']
    print(f"   Regime: {regime_info['regime']}")
    print(f"   Confidence: {regime_info['confidence']:.2f}")
    print(f"   Supporting Indicators: {regime_info['supporting_indicators']}")

    # 7. Display top indicators
    print("\n7. Top Contributing Indicators:")
    indicator_scores = metadata['indicator_scores']
    # Sort by absolute score
    top_indicators = sorted(indicator_scores,
                           key=lambda x: abs(x['score']),
                           reverse=True)[:5]
    for indicator in top_indicators:
        print(f"   {indicator['name']}: {indicator['value']:.2f} "
              f"(score: {indicator['score']:.2f}, "
              f"signal: {indicator['signal_type']}, "
              f"confidence: {indicator['confidence']:.2f})")

    # 8. Display validation result
    print("\n8. Signal Validation:")
    validation = metadata['validation_result']
    print(f"   Valid: {validation['is_valid']}")
    print(f"   Confidence: {validation['confidence']:.2f}")
    print(f"   Adjusted Confidence: {validation['adjusted_confidence']:.2f}")
    if validation['issues']:
        print("   Issues:")
        for issue in validation['issues']:
            print(f"     - {issue}")
    if validation['recommendations']:
        print("   Recommendations:")
        for rec in validation['recommendations']:
            print(f"     - {rec}")

    # 9. Display conflicts
    print("\n9. Conflict Detection:")
    conflicts = metadata['conflicts']
    if conflicts:
        print(f"   Found {len(conflicts)} conflicts:")
        for conflict in conflicts:
            print(f"     - {conflict['conflict_type']}: {conflict['description']}")
    else:
        print("   No conflicts detected")

    # 10. Display escalation information
    print("\n10. LLM Escalation:")
    escalation = metadata['escalation']
    print(f"    Escalation Required: {escalation['required']}")
    print(f"    Escalation Probability: {escalation['probability']:.2f}")
    if escalation['reasoning']:
        print(f"    Reasoning: {escalation['reasoning']}")

    # 11. Display performance metrics
    print("\n11. Performance Metrics:")
    performance = metadata['performance_metrics']
    print(f"    Generation Time: {metadata['generation_time']:.3f} seconds")
    print(f"    Total Signals Generated: {performance['total_signals_generated']}")
    print(f"    Average Generation Time: {performance['avg_generation_time']:.3f} seconds")
    print(f"    Average Validation Confidence: {performance['avg_validation_confidence']:.2f}")
    print(f"    Escalation Rate: {performance.get('escalation_rate', 0):.2%}")

    # 12. Generate multiple signals to show consistency
    print("\n12. Generating Multiple Signals for Consistency Check...")
    signals = []
    for i in range(5):
        # Use different subsets of data
        start_idx = i * 10
        end_idx = (i + 1) * 10 + 50
        data_subset = market_data.iloc[start_idx:end_idx]

        signal, _ = signal_generator.generate_signal(data_subset, f"AAPL_{i}")
        signals.append(signal)
        print(f"    Signal {i+1}: {signal.signal_type.value} "
              f"(confidence: {signal.confidence:.2f})")

    # 13. Display component status
    print("\n13. Component Status:")
    status = signal_generator.get_component_status()
    for component, info in status.items():
        print(f"    {component}:")
        for key, value in info.items():
            print(f"      {key}: {value}")

    # 14. Display signal history
    print("\n14. Signal History:")
    history = signal_generator.get_signal_history()
    print(f"    Total signals in history: {len(history)}")
    if history:
        recent = history[-3:]  # Last 3 signals
        print("    Recent signals:")
        for i, sig in enumerate(recent):
            print(f"      {i+1}. {sig.signal_type.value} at ${sig.price:.2f} "
                  f"(confidence: {sig.confidence:.2f})")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("Configuration Management Example")
    print("=" * 60)

    # Display current configuration
    print("\n1. Current Configuration:")
    config_dict = signal_generation_config.to_dict()

    # Display key configuration sections
    for section, values in config_dict.items():
        if isinstance(values, dict):
            print(f"\n   {section.upper()}:")
            for key, value in list(values.items())[:3]:  # Show first 3 items
                print(f"      {key}: {value}")
            if len(values) > 3:
                print(f"      ... and {len(values) - 3} more settings")

    print("\n2. Performance Targets:")
    print(f"   Target Latency: {signal_generation_config.target_latency_ms}ms")
    print(f"   Target Accuracy: {signal_generation_config.target_accuracy:.0%}")
    print(f"   Target Cost Reduction: {signal_generation_config.target_cost_reduction:.0%}")

    print("\n3. Configuration can be customized via environment variables:")
    print("   Example: SIGNAL_SCORER_CONSENSUS_THRESHOLD=0.7 python script.py")


if __name__ == "__main__":
    try:
        # Run the demonstration
        demonstrate_signal_generation()
        demonstrate_configuration()

        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Integrate with your existing trading system")
        print("2. Customize configuration for your specific needs")
        print("3. Run comprehensive backtesting")
        print("4. Monitor performance in production")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
