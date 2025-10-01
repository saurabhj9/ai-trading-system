# Technical Indicators Implementation Plan

This document outlines the detailed implementation plan for expanding the technical indicators in the AI Trading System as part of Phase 6.

## Overview

The goal is to expand from the current basic indicators (RSI, MACD) to a comprehensive suite of indicators organized by category, enabling sophisticated local signal generation and reducing reliance on LLM calls for routine technical analysis.

## Indicator Categories & Implementation Details

### 1. Momentum Indicators

#### 1.1 Stochastic Oscillator
- **Formula**: %K = 100 * (Close - Low14) / (High14 - Low14)
- **Parameters**: k_period=14, d_period=3, smoothing=3
- **Signals**:
  - Overbought > 80, Oversold < 20
  - %K crossing above %D = bullish
  - %K crossing below %D = bearish
- **Implementation**: Use pandas_ta `stoch` function
- **Data Required**: OHLC data

#### 1.2 Williams %R
- **Formula**: %R = -100 * (High14 - Close) / (High14 - Low14)
- **Parameters**: period=14
- **Signals**:
  - Overbought > -20, Oversold < -80
  - Divergence with price
- **Implementation**: Use pandas_ta `willr` function
- **Data Required**: OHLC data

#### 1.3 Commodity Channel Index (CCI)
- **Formula**: CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
- **Parameters**: period=20
- **Signals**:
  - Overbought > 100, Oversold < -100
  - Zero line crossovers
- **Implementation**: Use pandas_ta `cci` function
- **Data Required**: OHLC data

### 2. Mean Reversion Indicators

#### 2.1 Bollinger Bands
- **Formula**:
  - Middle Band = SMA(Close, 20)
  - Upper Band = SMA + 2 * StdDev
  - Lower Band = SMA - 2 * StdDev
- **Parameters**: period=20, std_dev=2
- **Signals**:
  - Price touching upper band = overbought
  - Price touching lower band = oversold
  - Band width indicates volatility
- **Implementation**: Use pandas_ta `bbands` function
- **Data Required**: OHLC data

#### 2.2 Keltner Channels
- **Formula**:
  - Middle Line = EMA(Close, 20)
  - Upper Channel = EMA + 2 * ATR
  - Lower Channel = EMA - 2 * ATR
- **Parameters**: ema_period=20, atr_period=10, multiplier=2
- **Signals**: Similar to Bollinger Bands but volatility-adjusted
- **Implementation**: Use pandas_ta `kc` function
- **Data Required**: OHLC data

#### 2.3 Donchian Channels
- **Formula**:
  - Upper Channel = Highest(High, 20)
  - Lower Channel = Lowest(Low, 20)
  - Middle Channel = (Upper + Lower) / 2
- **Parameters**: period=20
- **Signals**: Breakouts above/below channels
- **Implementation**: Custom calculation using pandas
- **Data Required**: OHLC data

### 3. Volatility Indicators

#### 3.1 Average True Range (ATR)
- **Formula**: ATR = EMA(True Range, 14)
- **True Range**: max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
- **Parameters**: period=14
- **Signals**:
  - High ATR = high volatility
  - Low ATR = low volatility
  - Can be used for position sizing
- **Implementation**: Use pandas_ta `atr` function
- **Data Required**: OHLC data

#### 3.2 Historical Volatility
- **Formula**: StdDev(log returns) * sqrt(252)
- **Parameters**: period=20
- **Signals**:
  - Compare current volatility to historical average
  - Volatility regime detection
- **Implementation**: Custom calculation
- **Data Required**: Close prices

#### 3.3 Chaikin Volatility
- **Formula**: CV = EMA(High-Low, 10) - EMA(High-Low, 10) of 10 periods ago
- **Parameters**: ema_period=10, roc_period=10
- **Signals**:
  - Positive values = increasing volatility
  - Negative values = decreasing volatility
- **Implementation**: Custom calculation
- **Data Required**: OHLC data

### 4. Trend Indicators

#### 4.1 Exponential Moving Averages (EMAs)
- **Formula**: EMA = (Close * α) + (Previous EMA * (1-α))
- **Parameters**: Multiple periods (5, 10, 20, 50, 200)
- **Signals**:
  - Price above EMA = bullish
  - Price below EMA = bearish
  - EMA crossovers
- **Implementation**: Use pandas_ta `ema` function
- **Data Required**: Close prices

#### 4.2 Average Directional Index (ADX)
- **Formula**: Complex calculation involving +DI, -DI, and True Range
- **Parameters**: period=14
- **Signals**:
  - ADX > 25 = trending market
  - ADX < 20 = ranging market
  - +DI > -DI = bullish trend
  - -DI > +DI = bearish trend
- **Implementation**: Use pandas_ta `adx` function
- **Data Required**: OHLC data

#### 4.3 Aroon Indicator
- **Formula**:
  - Aroon Up = ((Period - Periods Since Highest High) / Period) * 100
  - Aroon Down = ((Period - Periods Since Lowest Low) / Period) * 100
- **Parameters**: period=14
- **Signals**:
  - Aroon Up > 70 = bullish strength
  - Aroon Down > 70 = bearish strength
  - Crossovers indicate trend changes
- **Implementation**: Use pandas_ta `aroon` function
- **Data Required**: OHLC data

#### 4.4 Parabolic SAR
- **Formula**: Complex iterative calculation
- **Parameters**: acceleration=0.02, maximum=0.2
- **Signals**:
  - Price above SAR = bullish
  - Price below SAR = bearish
  - SAR flips indicate trend reversal
- **Implementation**: Use pandas_ta `psar` function
- **Data Required**: OHLC data

### 5. Volume Indicators

#### 5.1 On-Balance Volume (OBV)
- **Formula**: OBV = Previous OBV + Volume (if price up) or - Volume (if price down)
- **Parameters**: None
- **Signals**:
  - OBV trending up = accumulation
  - OBV trending down = distribution
  - Divergence with price signals reversal
- **Implementation**: Use pandas_ta `obv` function
- **Data Required**: Close and Volume

#### 5.2 Volume Weighted Average Price (VWAP)
- **Formula**: VWAP = Σ(Price * Volume) / Σ(Volume)
- **Parameters**: None (reset daily)
- **Signals**:
  - Price above VWAP = bullish
  - Price below VWAP = bearish
  - VWAP as support/resistance
- **Implementation**: Custom calculation
- **Data Required**: OHLCV data

#### 5.3 Money Flow Index (MFI)
- **Formula**: Similar to RSI but uses volume-weighted price
- **Parameters**: period=14
- **Signals**:
  - Overbought > 80, Oversold < 20
  - Divergence with price
- **Implementation**: Use pandas_ta `mfi` function
- **Data Required**: OHLCV data

#### 5.4 Accumulation/Distribution Line
- **Formula**: A/D = Previous A/D + ((Close - Low) - (High - Close)) / (High - Low) * Volume
- **Parameters**: None
- **Signals**:
  - Rising A/D = accumulation
  - Falling A/D = distribution
  - Divergence with price
- **Implementation**: Use pandas_ta `ad` function
- **Data Required**: OHLCV data

### 6. Statistical Indicators

#### 6.1 Hurst Exponent
- **Formula**: Measures long-term memory of time series
- **Parameters**: Various window sizes
- **Signals**:
  - H > 0.5 = trending series
  - H < 0.5 = mean-reverting series
  - H = 0.5 = random walk
- **Implementation**: Custom calculation using rescaled range analysis
- **Data Required**: Close prices

#### 6.2 Z-Score
- **Formula**: Z = (Price - Mean) / StdDev
- **Parameters**: period=20
- **Signals**:
  - Z > 2 = overbought
  - Z < -2 = oversold
  - Mean reversion expectations
- **Implementation**: Custom calculation
- **Data Required**: Close prices

#### 6.3 Correlation Analysis
- **Formula**: Pearson correlation coefficient
- **Parameters**: period=20
- **Signals**:
  - High correlation with market index = systematic risk
  - Low correlation = diversification benefit
- **Implementation**: Custom calculation
- **Data Required**: Close prices of multiple symbols

## Implementation Strategy

### Phase 1: Core Indicators (Week 1)
1. **Momentum Indicators**: Stochastic, Williams %R, CCI
2. **Mean Reversion Indicators**: Bollinger Bands, Keltner Channels
3. **Basic Trend Indicators**: EMAs (5, 10, 20, 50, 200)

### Phase 2: Advanced Indicators (Week 2)
1. **Advanced Trend Indicators**: ADX, Aroon, Parabolic SAR
2. **Volatility Indicators**: ATR, Historical Volatility
3. **Volume Indicators**: OBV, VWAP, MFI

### Phase 3: Statistical Indicators (Week 3)
1. **Statistical Analysis**: Hurst Exponent, Z-Score
2. **Advanced Volume**: Accumulation/Distribution Line
3. **Correlation Analysis**: Multi-symbol correlation

## Data Structure Changes

### New Indicator Categories
```python
class IndicatorCategory(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    STATISTICAL = "statistical"
```

### Enhanced MarketData Structure
```python
@dataclass
class MarketData:
    # ... existing fields ...

    # Categorized indicators
    momentum_indicators: Dict[str, float] = field(default_factory=dict)
    mean_reversion_indicators: Dict[str, float] = field(default_factory=dict)
    volatility_indicators: Dict[str, float] = field(default_factory=dict)
    trend_indicators: Dict[str, float] = field(default_factory=dict)
    volume_indicators: Dict[str, float] = field(default_factory=dict)
    statistical_indicators: Dict[str, float] = field(default_factory=dict)

    # Historical data by category
    historical_momentum: List[Dict[str, float]] = field(default_factory=list)
    historical_mean_reversion: List[Dict[str, float]] = field(default_factory=list)
    # ... other categories
```

## Performance Considerations

### Vectorization
- Use pandas_ta functions which are optimized
- Implement custom indicators using numpy/pandas vectorization
- Avoid loops where possible

### Caching Strategy
- Cache calculated indicators by symbol and date range
- Implement incremental updates for new data
- Use Redis for persistence across runs

### Memory Management
- Limit historical data to required periods
- Use efficient data types (float32 instead of float64)
- Implement data cleanup for old indicators

## Testing Strategy

### Unit Tests
- Test each indicator calculation against known values
- Test edge cases (empty data, single values)
- Test parameter variations

### Integration Tests
- Test indicator calculation pipeline end-to-end
- Test data provider integration
- Test caching mechanisms

### Performance Tests
- Benchmark calculation times for each indicator
- Test memory usage with multiple symbols
- Test concurrent calculation for multiple symbols

## Configuration

### Indicator Parameters
```python
INDICATOR_CONFIG = {
    "momentum": {
        "stochastic": {"k_period": 14, "d_period": 3, "smoothing": 3},
        "williams_r": {"period": 14},
        "cci": {"period": 20}
    },
    "mean_reversion": {
        "bollinger_bands": {"period": 20, "std_dev": 2},
        "keltner_channels": {"ema_period": 20, "atr_period": 10, "multiplier": 2},
        "donchian_channels": {"period": 20}
    },
    # ... other categories
}
```

### Signal Thresholds
```python
SIGNAL_THRESHOLDS = {
    "momentum": {
        "stochastic_overbought": 80,
        "stochastic_oversold": 20,
        "williams_r_overbought": -20,
        "williams_r_oversold": -80,
        "cci_overbought": 100,
        "cci_oversold": -100
    },
    # ... other categories
}
```

This implementation plan provides a comprehensive foundation for sophisticated technical analysis that will enable the local-first signal generation system to make high-quality decisions without relying on LLM calls for routine technical analysis.
