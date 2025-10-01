# Local Signal Generation Framework Design

This document outlines the design for a sophisticated local signal generation framework that will serve as the foundation for the AI Trading System's local-first analysis approach.

## Overview

The Local Signal Generation Framework (LSGF) is designed to replace LLM-dependent technical analysis with a rule-based system that can make high-quality trading decisions locally. The framework combines multiple indicators, detects market regimes, and generates consensus-based signals with confidence scores.

## Architecture

### Core Components

#### 1. SignalGenerator Class
The main orchestrator that coordinates all signal generation activities.

```python
class SignalGenerator:
    """
    Main signal generation engine that combines indicators, detects regimes,
    and produces consensus-based trading signals.
    """

    def __init__(self, config: SignalGeneratorConfig):
        self.indicator_weights = config.indicator_weights
        self.regime_detector = MarketRegimeDetector(config.regime_config)
        self.signal_combiner = SignalCombiner(config.combiner_config)
        self.confidence_calculator = ConfidenceCalculator(config.confidence_config)
        self.trigger_detector = TriggerDetector(config.trigger_config)
        self.decision_cache = DecisionCache(config.cache_config)
```

#### 2. MarketRegimeDetector
Identifies current market conditions to adjust indicator weights and strategies.

```python
class MarketRegimeDetector:
    """
    Detects market regimes (trending, ranging, volatile) to adapt
    signal generation strategies accordingly.
    """

    def detect_regime(self, market_data: MarketData) -> MarketRegime:
        """
        Analyzes market data to determine current regime.

        Returns:
            MarketRegime: Trending, Ranging, Volatile, or Transitional
        """

    def get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Returns indicator weights optimized for the current regime.
        """
```

#### 3. SignalCombiner
Combines individual indicator signals into a consensus decision.

```python
class SignalCombiner:
    """
    Combines signals from multiple indicators using weighted voting
    and consensus mechanisms.
    """

    def combine_signals(self,
                       indicator_signals: Dict[str, Signal],
                       weights: Dict[str, float]) -> ConsensusSignal:
        """
        Combines individual indicator signals into a consensus.

        Args:
            indicator_signals: Dictionary of indicator signals
            weights: Indicator weights for the current regime

        Returns:
            ConsensusSignal: Combined signal with strength and agreement
        """
```

#### 4. ConfidenceCalculator
Calculates confidence scores based on signal agreement and historical performance.

```python
class ConfidenceCalculator:
    """
    Calculates confidence scores for trading signals based on
    indicator agreement, historical performance, and market conditions.
    """

    def calculate_confidence(self,
                           consensus_signal: ConsensusSignal,
                           regime: MarketRegime,
                           historical_performance: Dict) -> float:
        """
        Calculates confidence score (0.0 to 1.0) for the signal.
        """
```

#### 5. TriggerDetector
Identifies specific market events that should trigger immediate analysis.

```python
class TriggerDetector:
    """
    Detects market events and triggers that require immediate attention
    or potential LLM escalation.
    """

    def detect_triggers(self, market_data: MarketData) -> List[Trigger]:
        """
        Scans market data for significant events and triggers.

        Returns:
            List[Trigger]: List of detected triggers
        """
```

## Signal Generation Process

### Step 1: Market Regime Detection
1. **Analyze Market Structure**: Use ADX, moving averages, and volatility
2. **Classify Regime**: Trending, Ranging, Volatile, or Transitional
3. **Adjust Strategy**: Select appropriate indicator weights and thresholds

### Step 2: Individual Signal Generation
1. **Calculate Indicators**: Generate all technical indicators
2. **Generate Raw Signals**: Convert indicator values to BUY/SELL/HOLD signals
3. **Apply Filters**: Remove weak signals based on thresholds

### Step 3: Signal Combination
1. **Apply Regime Weights**: Weight indicators based on regime effectiveness
2. **Consensus Voting**: Combine signals using weighted voting
3. **Agreement Measurement**: Calculate how much indicators agree
4. **Strength Calculation**: Determine signal strength based on consensus

### Step 4: Confidence Assessment
1. **Historical Performance**: Check how similar signals performed
2. **Agreement Bonus**: Increase confidence for high agreement
3. **Regime Adjustment**: Adjust confidence based on regime predictability
4. **Final Score**: Generate final confidence score (0.0 to 1.0)

### Step 5: Trigger Detection
1. **Event Scanning**: Look for significant market events
2. **Conflict Detection**: Identify conflicting signals
3. **Escalation Decision**: Determine if LLM escalation is needed

## Market Regime Classification

### Trending Market
**Characteristics:**
- ADX > 25
- Price above/below key moving averages
- Unidirectional price movement

**Strategy:**
- Emphasize trend-following indicators (EMAs, MACD, Parabolic SAR)
- Reduce weight on mean reversion indicators
- Higher confidence in trend-aligned signals

**Indicator Weights:**
- Trend: 50%
- Momentum: 30%
- Volume: 15%
- Mean Reversion: 5%

### Ranging Market
**Characteristics:**
- ADX < 20
- Price oscillating around moving averages
- Clear support/resistance levels

**Strategy:**
- Emphasize mean reversion indicators (Bollinger Bands, Stochastic)
- Reduce weight on trend indicators
- Higher confidence in mean reversion signals

**Indicator Weights:**
- Mean Reversion: 50%
- Momentum: 25%
- Volatility: 15%
- Trend: 10%

### Volatile Market
**Characteristics:**
- High ATR relative to price
- Large price swings
- Increased volume

**Strategy:**
- Emphasize volatility indicators (ATR, Bollinger Band width)
- Use wider thresholds for signals
- Higher confidence in volatility-adjusted signals

**Indicator Weights:**
- Volatility: 40%
- Volume: 25%
- Momentum: 20%
- Trend: 15%

### Transitional Market
**Characteristics:**
- Regime change in progress
- Mixed signals across indicators
- Unclear direction

**Strategy:**
- Use balanced indicator weights
- Lower confidence thresholds
- Higher likelihood of LLM escalation

**Indicator Weights:**
- Balanced across all categories (20-25% each)

## Signal Combination Logic

### Weighted Voting System
```python
def weighted_voting(signals: Dict[str, Signal], weights: Dict[str, float]) -> Dict[str, float]:
    """
    Combines signals using weighted voting.

    Returns:
        Dict with BUY, SELL, HOLD scores
    """
    scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    for indicator, signal in signals.items():
        weight = weights.get(indicator, 0.0)
        scores[signal.value] += weight * signal.strength

    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        for key in scores:
            scores[key] /= total

    return scores
```

### Consensus Measurement
```python
def calculate_consensus(scores: Dict[str, float]) -> float:
    """
    Measures the level of agreement among indicators.

    Returns:
        Consensus score (0.0 to 1.0)
    """
    max_score = max(scores.values())
    second_max = sorted(scores.values())[-2]

    # Higher consensus when top signal is much stronger than second
    consensus = (max_score - second_max) / max_score
    return min(consensus, 1.0)
```

### Signal Strength Calculation
```python
def calculate_signal_strength(scores: Dict[str, float], consensus: float) -> Signal:
    """
    Determines the final signal and its strength.
    """
    top_signal = max(scores, key=scores.get)
    strength = scores[top_signal] * consensus

    # Apply minimum strength threshold
    if strength < 0.3:
        return Signal("HOLD", strength)

    return Signal(top_signal, strength)
```

## Confidence Calculation

### Base Confidence Factors
1. **Signal Strength**: Stronger signals get higher confidence
2. **Consensus Level**: Higher agreement increases confidence
3. **Regime Reliability**: Some regimes are more predictable
4. **Historical Performance**: Past success rates influence confidence

### Dynamic Confidence Adjustment
```python
def calculate_dynamic_confidence(
    base_confidence: float,
    regime: MarketRegime,
    historical_performance: Dict,
    market_conditions: Dict
) -> float:
    """
    Adjusts confidence based on multiple factors.
    """

    # Regime adjustment
    regime_multiplier = {
        MarketRegime.TRENDING: 1.2,
        MarketRegime.RANGING: 1.1,
        MarketRegime.VOLATILE: 0.9,
        MarketRegime.TRANSITIONAL: 0.7
    }

    # Historical performance adjustment
    success_rate = historical_performance.get("success_rate", 0.5)
    performance_multiplier = 0.5 + success_rate

    # Market conditions adjustment
    volatility_factor = max(0.7, min(1.3, 1.0 / (1.0 + market_conditions.get("volatility", 0.0))))

    # Combine all factors
    adjusted_confidence = (
        base_confidence *
        regime_multiplier[regime] *
        performance_multiplier *
        volatility_factor
    )

    return min(max(adjusted_confidence, 0.0), 1.0)
```

## Trigger Detection System

### Trigger Types
1. **Technical Triggers**: Indicator crossovers, threshold breaches
2. **Volatility Triggers**: Sudden volatility increases
3. **Volume Triggers**: Unusual volume spikes
4. **Pattern Triggers**: Chart pattern formations
5. **Conflict Triggers**: Divergent signals across categories

### Escalation Criteria
```python
def should_escalate_to_llm(
    consensus_signal: ConsensusSignal,
    triggers: List[Trigger],
    confidence: float,
    regime: MarketRegime
) -> bool:
    """
    Determines if LLM escalation is warranted.
    """

    # Low confidence always escalates
    if confidence < 0.4:
        return True

    # High-priority triggers escalate
    high_priority_triggers = [t for t in triggers if t.priority == TriggerPriority.HIGH]
    if high_priority_triggers:
        return True

    # Conflicting signals escalate
    if consensus_signal.agreement < 0.3:
        return True

    # Transitional regime escalates more often
    if regime == MarketRegime.TRANSITIONAL and confidence < 0.7:
        return True

    return False
```

## Decision Caching

### Cache Key Generation
```python
def generate_cache_key(market_data: MarketData, regime: MarketRegime) -> str:
    """
    Generates a unique key for caching decisions.
    """
    # Quantize market data to reduce cache size
    quantized_data = quantize_market_data(market_data)

    # Create hash
    data_string = f"{market_data.symbol}_{regime.value}_{quantized_data}"
    return hashlib.md5(data_string.encode()).hexdigest()
```

### Cache Strategy
1. **TTL Management**: Decisions expire after configurable time
2. **Regime Awareness**: Different cache per regime
3. **Similarity Matching**: Use fuzzy matching for similar market conditions
4. **Performance Tracking**: Monitor cache hit rates and effectiveness

## Configuration System

### Signal Generator Configuration
```python
@dataclass
class SignalGeneratorConfig:
    indicator_weights: Dict[str, Dict[str, float]]
    regime_config: RegimeDetectorConfig
    combiner_config: SignalCombinerConfig
    confidence_config: ConfidenceCalculatorConfig
    trigger_config: TriggerDetectorConfig
    cache_config: DecisionCacheConfig

    # Global settings
    min_confidence_threshold: float = 0.3
    max_cache_age_seconds: int = 300
    enable_llm_escalation: bool = True
```

### Regime-Specific Weights
```python
REGIME_WEIGHTS = {
    MarketRegime.TRENDING: {
        "trend": 0.5, "momentum": 0.3, "volume": 0.15, "mean_reversion": 0.05
    },
    MarketRegime.RANGING: {
        "mean_reversion": 0.5, "momentum": 0.25, "volatility": 0.15, "trend": 0.1
    },
    MarketRegime.VOLATILE: {
        "volatility": 0.4, "volume": 0.25, "momentum": 0.2, "trend": 0.15
    },
    MarketRegime.TRANSITIONAL: {
        "trend": 0.25, "momentum": 0.25, "mean_reversion": 0.25, "volatility": 0.25
    }
}
```

## Performance Optimization

### Vectorization
- Use numpy/pandas for indicator calculations
- Vectorize signal combination logic
- Batch process multiple symbols

### Caching Strategy
- Cache indicator calculations
- Cache regime detection results
- Cache signal combinations
- Implement hierarchical caching

### Memory Management
- Limit historical data retention
- Use efficient data structures
- Implement garbage collection for old cache entries

## Testing Strategy

### Unit Tests
- Test each component independently
- Mock market data for reproducible tests
- Test edge cases and boundary conditions

### Integration Tests
- Test full signal generation pipeline
- Test with real market data
- Test performance under load

### Backtesting
- Compare local signals to LLM signals
- Test performance across different market conditions
- Validate confidence calibration

## Monitoring Metrics

### Signal Quality Metrics
- Signal accuracy by regime
- Confidence calibration
- Signal distribution (BUY/SELL/HOLD ratios)
- Cache hit rates

### Performance Metrics
- Signal generation latency
- Memory usage
- Cache effectiveness
- LLM escalation frequency

This framework provides a robust foundation for local signal generation that can handle complex market conditions while maintaining high performance and reducing reliance on expensive LLM calls.
