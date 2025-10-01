"""
Metadata for technical indicators used in the AI Trading System.

This module contains metadata about various technical indicators including
their categories, reliability ratings, typical usage patterns, and signal thresholds.
"""

from typing import Dict, Any
from enum import Enum


class IndicatorCategory(Enum):
    """Categories of technical indicators."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    STATISTICAL = "statistical"


class IndicatorReliability(Enum):
    """Reliability rating for indicators."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Metadata for momentum indicators
MOMENTUM_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "STOCH_K": {
        "name": "Stochastic Oscillator %K",
        "category": IndicatorCategory.MOMENTUM,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the current close price relative to the high-low range over a period",
        "parameters": {
            "k_period": 14,
            "d_period": 3,
            "smoothing": 3
        },
        "signal_thresholds": {
            "overbought": 80,
            "oversold": 20
        },
        "typical_usage": [
            "Identify overbought/oversold conditions",
            "Generate signals when %K crosses %D",
            "Confirm price reversals"
        ],
        "data_requirements": ["OHLC"]
    },
    "STOCH_D": {
        "name": "Stochastic Oscillator %D",
        "category": IndicatorCategory.MOMENTUM,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "A moving average of %K, used to smooth the Stochastic Oscillator",
        "parameters": {
            "k_period": 14,
            "d_period": 3,
            "smoothing": 3
        },
        "signal_thresholds": {
            "overbought": 80,
            "oversold": 20
        },
        "typical_usage": [
            "Smooth %K signals",
            "Generate crossover signals with %K",
            "Confirm trend direction"
        ],
        "data_requirements": ["OHLC"]
    },
    "WILLR": {
        "name": "Williams %R",
        "category": IndicatorCategory.MOMENTUM,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures momentum by determining overbought/oversold levels",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "overbought": -20,
            "oversold": -80
        },
        "typical_usage": [
            "Identify overbought/oversold conditions",
            "Spot potential reversals",
            "Confirm price movements"
        ],
        "data_requirements": ["OHLC"]
    },
    "CCI": {
        "name": "Commodity Channel Index",
        "category": IndicatorCategory.MOMENTUM,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Identifies cyclical trends and measures the deviation of price from its statistical mean",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "overbought": 100,
            "oversold": -100
        },
        "typical_usage": [
            "Identify new trends",
            "Detect overbought/oversold conditions",
            "Spot divergences with price"
        ],
        "data_requirements": ["OHLC"]
    }
}

# Metadata for mean reversion indicators
MEAN_REVERSION_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "BB_UPPER": {
        "name": "Bollinger Bands Upper",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.HIGH,
        "description": "Upper band of Bollinger Bands, typically 2 standard deviations above the SMA",
        "parameters": {
            "period": 20,
            "std_dev": 2
        },
        "signal_thresholds": {
            "overbought": "price touches or exceeds upper band"
        },
        "typical_usage": [
            "Identify overbought conditions",
            "Signal potential reversals when price touches upper band",
            "Measure volatility through band width"
        ],
        "data_requirements": ["OHLC"]
    },
    "BB_MIDDLE": {
        "name": "Bollinger Bands Middle (SMA)",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.HIGH,
        "description": "Middle band of Bollinger Bands, which is a simple moving average",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "support_resistance": "price crossing the middle band"
        },
        "typical_usage": [
            "Identify trend direction",
            "Act as support/resistance level",
            "Reference for mean reversion"
        ],
        "data_requirements": ["OHLC"]
    },
    "BB_LOWER": {
        "name": "Bollinger Bands Lower",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.HIGH,
        "description": "Lower band of Bollinger Bands, typically 2 standard deviations below the SMA",
        "parameters": {
            "period": 20,
            "std_dev": 2
        },
        "signal_thresholds": {
            "oversold": "price touches or falls below lower band"
        },
        "typical_usage": [
            "Identify oversold conditions",
            "Signal potential reversals when price touches lower band",
            "Measure volatility through band width"
        ],
        "data_requirements": ["OHLC"]
    },
    "BB_WIDTH": {
        "name": "Bollinger Bands Width",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Distance between upper and lower Bollinger Bands, normalized by the middle band",
        "parameters": {
            "period": 20,
            "std_dev": 2
        },
        "signal_thresholds": {
            "low_volatility": "width < 0.04",
            "high_volatility": "width > 0.10"
        },
        "typical_usage": [
            "Identify volatility squeeze conditions",
            "Predict potential breakouts",
            "Measure market volatility"
        ],
        "data_requirements": ["OHLC"]
    },
    "KC_UPPER": {
        "name": "Keltner Channels Upper",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Upper channel of Keltner Channels, based on EMA plus ATR multiplier",
        "parameters": {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2
        },
        "signal_thresholds": {
            "overbought": "price touches or exceeds upper channel"
        },
        "typical_usage": [
            "Identify overbought conditions",
            "Signal potential reversals",
            "Trend following when price stays outside channel"
        ],
        "data_requirements": ["OHLC"]
    },
    "KC_MIDDLE": {
        "name": "Keltner Channels Middle (EMA)",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Middle channel of Keltner Channels, which is an exponential moving average",
        "parameters": {
            "ema_period": 20
        },
        "signal_thresholds": {
            "support_resistance": "price crossing the middle channel"
        },
        "typical_usage": [
            "Identify trend direction",
            "Act as support/resistance level",
            "Reference for mean reversion"
        ],
        "data_requirements": ["OHLC"]
    },
    "KC_LOWER": {
        "name": "Keltner Channels Lower",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Lower channel of Keltner Channels, based on EMA minus ATR multiplier",
        "parameters": {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2
        },
        "signal_thresholds": {
            "oversold": "price touches or falls below lower channel"
        },
        "typical_usage": [
            "Identify oversold conditions",
            "Signal potential reversals",
            "Trend following when price stays outside channel"
        ],
        "data_requirements": ["OHLC"]
    },
    "DC_UPPER": {
        "name": "Donchian Channels Upper",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Upper channel of Donchian Channels, based on the highest high over a period",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "breakout": "price exceeds upper channel"
        },
        "typical_usage": [
            "Identify breakout opportunities",
            "Set stop-loss levels",
            "Determine support/resistance"
        ],
        "data_requirements": ["OHLC"]
    },
    "DC_MIDDLE": {
        "name": "Donchian Channels Middle",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Middle channel of Donchian Channels, the average of upper and lower channels",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "support_resistance": "price crossing the middle channel"
        },
        "typical_usage": [
            "Identify trend direction",
            "Act as support/resistance level",
            "Reference for mean reversion"
        ],
        "data_requirements": ["OHLC"]
    },
    "DC_LOWER": {
        "name": "Donchian Channels Lower",
        "category": IndicatorCategory.MEAN_REVERSION,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Lower channel of Donchian Channels, based on the lowest low over a period",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "breakdown": "price falls below lower channel"
        },
        "typical_usage": [
            "Identify breakdown opportunities",
            "Set stop-loss levels",
            "Determine support/resistance"
        ],
        "data_requirements": ["OHLC"]
    }
}

# Metadata for volatility indicators
VOLATILITY_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "ATR": {
        "name": "Average True Range",
        "category": IndicatorCategory.VOLATILITY,
        "reliability": IndicatorReliability.HIGH,
        "description": "Measures market volatility by averaging the true range over a period",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "high_volatility": "ATR > 1.5 * 20-day SMA of ATR",
            "low_volatility": "ATR < 0.5 * 20-day SMA of ATR"
        },
        "typical_usage": [
            "Measure market volatility",
            "Set stop-loss levels based on volatility",
            "Position sizing adjustments",
            "Identify volatility breakouts"
        ],
        "data_requirements": ["OHLC"]
    },
    "HISTORICAL_VOLATILITY": {
        "name": "Historical Volatility",
        "category": IndicatorCategory.VOLATILITY,
        "reliability": IndicatorReliability.HIGH,
        "description": "Annualized standard deviation of logarithmic returns",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "high_volatility": "HV > 0.30 (30%)",
            "low_volatility": "HV < 0.15 (15%)"
        },
        "typical_usage": [
            "Options pricing models",
            "Risk management",
            "Volatility regime detection",
            "Compare implied vs historical volatility"
        ],
        "data_requirements": ["Close"]
    },
    "CHAIKIN_VOLATILITY": {
        "name": "Chaikin Volatility",
        "category": IndicatorCategory.VOLATILITY,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the rate of change of the spread between high and low prices",
        "parameters": {
            "ema_period": 10,
            "roc_period": 10
        },
        "signal_thresholds": {
            "increasing_volatility": "CV > 0",
            "decreasing_volatility": "CV < 0"
        },
        "typical_usage": [
            "Identify volatility direction",
            "Confirm price breakouts",
            "Detect volatility exhaustion"
        ],
        "data_requirements": ["OHLC"]
    }
}

# Metadata for trend indicators
TREND_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "EMA_5": {
        "name": "5-period Exponential Moving Average",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "5-period EMA provides short-term trend direction",
        "parameters": {
            "period": 5
        },
        "signal_thresholds": {
            "bullish": "price > EMA",
            "bearish": "price < EMA"
        },
        "typical_usage": [
            "Short-term trend identification",
            "Crossover signals with longer EMAs",
            "Support/resistance levels"
        ],
        "data_requirements": ["Close"]
    },
    "EMA_10": {
        "name": "10-period Exponential Moving Average",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "10-period EMA provides short to medium-term trend direction",
        "parameters": {
            "period": 10
        },
        "signal_thresholds": {
            "bullish": "price > EMA",
            "bearish": "price < EMA"
        },
        "typical_usage": [
            "Short to medium-term trend identification",
            "Crossover signals with other EMAs",
            "Support/resistance levels"
        ],
        "data_requirements": ["Close"]
    },
    "EMA_20": {
        "name": "20-period Exponential Moving Average",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "20-period EMA provides medium-term trend direction",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "bullish": "price > EMA",
            "bearish": "price < EMA"
        },
        "typical_usage": [
            "Medium-term trend identification",
            "Crossover signals with other EMAs",
            "Support/resistance levels"
        ],
        "data_requirements": ["Close"]
    },
    "EMA_50": {
        "name": "50-period Exponential Moving Average",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "50-period EMA provides medium to long-term trend direction",
        "parameters": {
            "period": 50
        },
        "signal_thresholds": {
            "bullish": "price > EMA",
            "bearish": "price < EMA"
        },
        "typical_usage": [
            "Medium to long-term trend identification",
            "Crossover signals with other EMAs",
            "Support/resistance levels"
        ],
        "data_requirements": ["Close"]
    },
    "EMA_200": {
        "name": "200-period Exponential Moving Average",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "200-period EMA provides long-term trend direction",
        "parameters": {
            "period": 200
        },
        "signal_thresholds": {
            "bullish": "price > EMA",
            "bearish": "price < EMA"
        },
        "typical_usage": [
            "Long-term trend identification",
            "Major support/resistance levels",
            "Bull/bear market determination"
        ],
        "data_requirements": ["Close"]
    },
    "ADX": {
        "name": "Average Directional Index",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "Measures trend strength regardless of direction",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "strong_trend": "ADX > 25",
            "weak_trend": "ADX < 20",
            "trending": "20 < ADX < 25"
        },
        "typical_usage": [
            "Determine if market is trending or ranging",
            "Confirm trend strength",
            "Filter for trend-following strategies"
        ],
        "data_requirements": ["OHLC"]
    },
    "DI_PLUS": {
        "name": "Positive Directional Indicator (+DI)",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures upward trend strength",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "bullish": "+DI > -DI",
            "bearish": "+DI < -DI"
        },
        "typical_usage": [
            "Identify upward trend strength",
            "Generate signals with -DI crossovers",
            "Confirm uptrends"
        ],
        "data_requirements": ["OHLC"]
    },
    "DI_MINUS": {
        "name": "Negative Directional Indicator (-DI)",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures downward trend strength",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "bearish": "-DI > +DI",
            "bullish": "-DI < +DI"
        },
        "typical_usage": [
            "Identify downward trend strength",
            "Generate signals with +DI crossovers",
            "Confirm downtrends"
        ],
        "data_requirements": ["OHLC"]
    },
    "AROON_UP": {
        "name": "Aroon Up",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the strength of an uptrend",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "strong_uptrend": "Aroon Up > 70",
            "weak_uptrend": "Aroon Up < 30"
        },
        "typical_usage": [
            "Identify new uptrends",
            "Measure uptrend strength",
            "Generate signals with Aroon Down crossovers"
        ],
        "data_requirements": ["OHLC"]
    },
    "AROON_DOWN": {
        "name": "Aroon Down",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the strength of a downtrend",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "strong_downtrend": "Aroon Down > 70",
            "weak_downtrend": "Aroon Down < 30"
        },
        "typical_usage": [
            "Identify new downtrends",
            "Measure downtrend strength",
            "Generate signals with Aroon Up crossovers"
        ],
        "data_requirements": ["OHLC"]
    },
    "AROON_OSC": {
        "name": "Aroon Oscillator",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Difference between Aroon Up and Aroon Down",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "bullish": "Oscillator > 0",
            "bearish": "Oscillator < 0",
            "strong_bullish": "Oscillator > 50",
            "strong_bearish": "Oscillator < -50"
        },
        "typical_usage": [
            "Identify trend direction",
            "Measure trend strength",
            "Generate reversal signals"
        ],
        "data_requirements": ["OHLC"]
    },
    "PSAR": {
        "name": "Parabolic SAR",
        "category": IndicatorCategory.TREND,
        "reliability": IndicatorReliability.HIGH,
        "description": "Follows price action and provides reversal points",
        "parameters": {
            "acceleration": 0.02,
            "maximum": 0.2
        },
        "signal_thresholds": {
            "bullish": "price > PSAR",
            "bearish": "price < PSAR"
        },
        "typical_usage": [
            "Identify trend direction",
            "Set trailing stop-loss levels",
            "Generate reversal signals"
        ],
        "data_requirements": ["OHLC"]
    }
}

# Metadata for volume indicators
VOLUME_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "OBV": {
        "name": "On-Balance Volume",
        "category": IndicatorCategory.VOLUME,
        "reliability": IndicatorReliability.HIGH,
        "description": "Cumulative volume indicator that adds volume on up days and subtracts on down days",
        "parameters": {},
        "signal_thresholds": {
            "trending_up": "OBV making higher highs",
            "trending_down": "OBV making lower lows",
            "divergence": "Price moving opposite to OBV"
        },
        "typical_usage": [
            "Confirm price trends",
            "Identify divergences between price and volume",
            "Detect accumulation/distribution patterns"
        ],
        "data_requirements": ["Close", "Volume"]
    },
    "VWAP": {
        "name": "Volume Weighted Average Price",
        "category": IndicatorCategory.VOLUME,
        "reliability": IndicatorReliability.HIGH,
        "description": "Average price weighted by volume, reset daily",
        "parameters": {},
        "signal_thresholds": {
            "bullish": "price > VWAP",
            "bearish": "price < VWAP"
        },
        "typical_usage": [
            "Identify intraday trend direction",
            "Find support and resistance levels",
            "Institutional trading benchmark"
        ],
        "data_requirements": ["OHLCV"]
    },
    "MFI": {
        "name": "Money Flow Index",
        "category": IndicatorCategory.VOLUME,
        "reliability": IndicatorReliability.HIGH,
        "description": "Volume-weighted RSI that measures buying and selling pressure",
        "parameters": {
            "period": 14
        },
        "signal_thresholds": {
            "overbought": 80,
            "oversold": 20
        },
        "typical_usage": [
            "Identify overbought/oversold conditions",
            "Confirm price trends with volume",
            "Spot divergences between price and money flow"
        ],
        "data_requirements": ["OHLCV"]
    },
    "ADL": {
        "name": "Accumulation/Distribution Line",
        "category": IndicatorCategory.VOLUME,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the cumulative flow of money into and out of a security",
        "parameters": {},
        "signal_thresholds": {
            "accumulation": "ADL trending upward",
            "distribution": "ADL trending downward",
            "divergence": "Price moving opposite to ADL"
        },
        "typical_usage": [
            "Confirm price trends",
            "Identify divergences between price and volume flow",
            "Detect accumulation/distribution patterns"
        ],
        "data_requirements": ["OHLCV"]
    }
}

# Metadata for statistical indicators
STATISTICAL_INDICATORS_METADATA: Dict[str, Dict[str, Any]] = {
    "HURST": {
        "name": "Hurst Exponent",
        "category": IndicatorCategory.STATISTICAL,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the long-term memory of a time series, indicating whether it is trending, mean-reverting, or random",
        "parameters": {
            "min_window": 10,
            "max_window": 100,
            "num_windows": 5
        },
        "signal_thresholds": {
            "trending": "H > 0.5",
            "mean_reverting": "H < 0.5",
            "random_walk": "H = 0.5"
        },
        "typical_usage": [
            "Identify market regime (trending vs mean-reverting)",
            "Adapt strategy based on market characteristics",
            "Risk management and position sizing"
        ],
        "data_requirements": ["Close"]
    },
    "Z_SCORE": {
        "name": "Z-Score",
        "category": IndicatorCategory.STATISTICAL,
        "reliability": IndicatorReliability.HIGH,
        "description": "Standardized score that measures how many standard deviations an element is from the mean",
        "parameters": {
            "period": 20
        },
        "signal_thresholds": {
            "overbought": 2.0,
            "oversold": -2.0,
            "extreme": 3.0
        },
        "typical_usage": [
            "Identify overbought/oversold conditions",
            "Mean reversion trading signals",
            "Statistical arbitrage opportunities"
        ],
        "data_requirements": ["Close"]
    },
    "CORRELATION": {
        "name": "Pearson Correlation",
        "category": IndicatorCategory.STATISTICAL,
        "reliability": IndicatorReliability.MEDIUM,
        "description": "Measures the linear correlation between two variables, typically a stock and a market index",
        "parameters": {
            "period": 20,
            "reference_symbol": "SPY"
        },
        "signal_thresholds": {
            "high_correlation": 0.7,
            "low_correlation": 0.3,
            "negative_correlation": -0.3
        },
        "typical_usage": [
            "Measure systematic risk",
            "Identify diversification benefits",
            "Pairs trading strategies"
        ],
        "data_requirements": ["Close", "Reference Symbol Close"]
    }
}


def get_indicator_metadata(indicator_name: str) -> Dict[str, Any]:
    """
    Get metadata for a specific indicator.

    Args:
        indicator_name: The name of the indicator

    Returns:
        Dictionary containing the indicator metadata
    """
    # Check in momentum indicators first
    metadata = MOMENTUM_INDICATORS_METADATA.get(indicator_name, {})
    if not metadata:
        # Check in mean reversion indicators
        metadata = MEAN_REVERSION_INDICATORS_METADATA.get(indicator_name, {})
    if not metadata:
        # Check in volatility indicators
        metadata = VOLATILITY_INDICATORS_METADATA.get(indicator_name, {})
    if not metadata:
        # Check in trend indicators
        metadata = TREND_INDICATORS_METADATA.get(indicator_name, {})
    if not metadata:
        # Check in volume indicators
        metadata = VOLUME_INDICATORS_METADATA.get(indicator_name, {})
    if not metadata:
        # Check in statistical indicators
        metadata = STATISTICAL_INDICATORS_METADATA.get(indicator_name, {})
    return metadata


def get_all_momentum_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all momentum indicators.

    Returns:
        Dictionary containing all momentum indicators metadata
    """
    return MOMENTUM_INDICATORS_METADATA.copy()


def get_all_mean_reversion_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all mean reversion indicators.

    Returns:
        Dictionary containing all mean reversion indicators metadata
    """
    return MEAN_REVERSION_INDICATORS_METADATA.copy()


def get_all_volatility_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all volatility indicators.

    Returns:
        Dictionary containing all volatility indicators metadata
    """
    return VOLATILITY_INDICATORS_METADATA.copy()


def get_all_trend_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all trend indicators.

    Returns:
        Dictionary containing all trend indicators metadata
    """
    return TREND_INDICATORS_METADATA.copy()


def get_all_volume_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all volume indicators.

    Returns:
        Dictionary containing all volume indicators metadata
    """
    return VOLUME_INDICATORS_METADATA.copy()


def get_all_statistical_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all statistical indicators.

    Returns:
        Dictionary containing all statistical indicators metadata
    """
    return STATISTICAL_INDICATORS_METADATA.copy()
