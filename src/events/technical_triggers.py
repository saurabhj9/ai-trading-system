"""
Technical trigger detector for identifying chart patterns and indicator crossovers.

This module detects various technical analysis patterns such as RSI/MACD crossovers,
Bollinger Band breaches, Stochastic extremes, and other traditional chart signals.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .trigger_detector import BaseTriggerDetector, TriggerType, TriggerSeverity, TriggerEvent, TriggerConfig

logger = logging.getLogger(__name__)


class TechnicalTriggerDetector(BaseTriggerDetector):
    """
    Detects technical analysis triggers from market data.

    This detector monitors for:
    - RSI overbought/oversold conditions and crossovers
    - MACD line crossovers and signal line crosses
    - Bollinger Band breaches
    - Moving average crossovers
    - Stochastic oscillator extremes
    - Price pattern breakouts
    """

    def __init__(self, config: Optional[TriggerConfig] = None):
        """Initialize technical trigger detector."""
        super().__init__(TriggerType.TECHNICAL, config)

        # Default technical analysis parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.ma_fast = 20
        self.ma_slow = 50
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20

        logger.info("Technical trigger detector initialized")

    def get_required_data_fields(self) -> List[str]:
        """Get required data fields for technical analysis."""
        return ['current_price', 'historical_data', 'indicators']

    async def detect_triggers(self, symbol: str, market_data: Dict[str, Any]) -> List[TriggerEvent]:
        """Detect technical triggers for a symbol."""
        triggers = []

        try:
            # Extract required data
            historical_data = market_data.get('historical_data', pd.DataFrame())
            indicators = market_data.get('indicators', {})

            if not historical_data or len(historical_data) < 2:
                return triggers

            # Get current and previous values
            current_price = market_data.get('current_price', 0)
            if not current_price:
                current_price = historical_data['Close'].iloc[-1]

            # Detect various technical patterns
            triggers.extend(await self._detect_rsi_triggers(symbol, indicators, current_price))
            triggers.extend(await self._detect_macd_triggers(symbol, indicators))
            triggers.extend(await self._detect_bollinger_band_triggers(symbol, indicators, current_price))
            triggers.extend(await self._detect_ma_crossover_triggers(symbol, indicators))
            triggers.extend(await self._detect_stochastic_triggers(symbol, indicators))
            triggers.extend(await self._detect_price_pattern_triggers(symbol, historical_data, current_price))

        except Exception as e:
            logger.error(f"Error detecting technical triggers for {symbol}: {e}")

        return triggers

    async def _detect_rsi_triggers(self, symbol: str, indicators: Dict[str, Any], current_price: float) -> List[TriggerEvent]:
        """Detect RSI-based triggers."""
        triggers = []

        try:
            rsi_values = indicators.get('RSI_14', [])
            if not rsi_values or len(rsi_values) < 2:
                return triggers

            current_rsi = rsi_values[-1]
            previous_rsi = rsi_values[-2]

            # RSI overbought/oversold conditions
            if current_rsi > self.rsi_overbought and previous_rsi <= self.rsi_overbought:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=self.calculate_trigger_severity(0.8, 0.9),
                    timestamp=datetime.now(),
                    description=f"RSI entered overbought zone ({current_rsi:.1f} > {self.rsi_overbought})",
                    data={
                        'indicator': 'RSI',
                        'current_value': current_rsi,
                        'previous_value': previous_rsi,
                        'threshold': self.rsi_overbought,
                        'signal': 'overbought'
                    },
                    confidence=min(0.9, (current_rsi - self.rsi_overbought) / 10)
                )
                triggers.append(trigger)

            elif current_rsi < self.rsi_oversold and previous_rsi >= self.rsi_oversold:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=self.calculate_trigger_severity(0.8, 0.9),
                    timestamp=datetime.now(),
                    description=f"RSI entered oversold zone ({current_rsi:.1f} < {self.rsi_oversold})",
                    data={
                        'indicator': 'RSI',
                        'current_value': current_rsi,
                        'previous_value': previous_rsi,
                        'threshold': self.rsi_oversold,
                        'signal': 'oversold'
                    },
                    confidence=min(0.9, (self.rsi_oversold - current_rsi) / 10)
                )
                triggers.append(trigger)

            # RSI center line crossover (50)
            if current_rsi > 50 and previous_rsi <= 50:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"RSI crossed above center line (50)",
                    data={
                        'indicator': 'RSI',
                        'current_value': current_rsi,
                        'previous_value': previous_rsi,
                        'signal': 'bullish_crossover'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

            elif current_rsi < 50 and previous_rsi >= 50:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"RSI crossed below center line (50)",
                    data={
                        'indicator': 'RSI',
                        'current_value': current_rsi,
                        'previous_value': previous_rsi,
                        'signal': 'bearish_crossover'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting RSI triggers for {symbol}: {e}")

        return triggers

    async def _detect_macd_triggers(self, symbol: str, indicators: Dict[str, Any]) -> List[TriggerEvent]:
        """Detect MACD-based triggers."""
        triggers = []

        try:
            macd_line = indicators.get('MACD_12_26_9', [])
            macd_signal = indicators.get('MACDs_12_26_9', [])
            macd_histogram = indicators.get('MACDh_12_26_9', [])

            if not macd_line or not macd_signal or len(macd_line) < 2 or len(macd_signal) < 2:
                return triggers

            current_macd = macd_line[-1]
            previous_macd = macd_line[-2]
            current_signal = macd_signal[-1]
            previous_signal = macd_signal[-2]

            # MACD line crossing signal line
            if current_macd > current_signal and previous_macd <= previous_signal:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description="MACD bullish crossover (MACD line crossed above signal line)",
                    data={
                        'indicator': 'MACD',
                        'current_macd': current_macd,
                        'current_signal': current_signal,
                        'previous_macd': previous_macd,
                        'previous_signal': previous_signal,
                        'signal': 'bullish_crossover'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

            elif current_macd < current_signal and previous_macd >= previous_signal:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description="MACD bearish crossover (MACD line crossed below signal line)",
                    data={
                        'indicator': 'MACD',
                        'current_macd': current_macd,
                        'current_signal': current_signal,
                        'previous_macd': previous_macd,
                        'previous_signal': previous_signal,
                        'signal': 'bearish_crossover'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

            # MACD zero line crossover
            if current_macd > 0 and previous_macd <= 0:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description="MACD crossed above zero line",
                    data={
                        'indicator': 'MACD',
                        'current_value': current_macd,
                        'signal': 'zero_cross_bullish'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

            elif current_macd < 0 and previous_macd >= 0:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description="MACD crossed below zero line",
                    data={
                        'indicator': 'MACD',
                        'current_value': current_macd,
                        'signal': 'zero_cross_bearish'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting MACD triggers for {symbol}: {e}")

        return triggers

    async def _detect_bollinger_band_triggers(self, symbol: str, indicators: Dict[str, Any], current_price: float) -> List[TriggerEvent]:
        """Detect Bollinger Band breach triggers."""
        triggers = []

        try:
            bb_upper = indicators.get('BBU_20_2.0', [])
            bb_lower = indicators.get('BBL_20_2.0', [])
            bb_middle = indicators.get('BBM_20_2.0', [])

            if not bb_upper or not bb_lower or len(bb_upper) < 2:
                return triggers

            current_upper = bb_upper[-1]
            current_lower = bb_lower[-1]
            current_middle = bb_middle[-1] if bb_middle else (current_upper + current_lower) / 2

            previous_upper = bb_upper[-2]
            previous_lower = bb_lower[-2]

            # Upper band breach
            if current_price > current_upper:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=self.calculate_trigger_severity(0.8, 0.8),
                    timestamp=datetime.now(),
                    description=f"Price breached upper Bollinger Band (${current_price:.2f} > ${current_upper:.2f})",
                    data={
                        'indicator': 'Bollinger_Bands',
                        'current_price': current_price,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'middle_band': current_middle,
                        'signal': 'upper_breach'
                    },
                    confidence=min(0.9, (current_price - current_upper) / current_upper * 10)
                )
                triggers.append(trigger)

            # Lower band breach
            elif current_price < current_lower:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=self.calculate_trigger_severity(0.8, 0.8),
                    timestamp=datetime.now(),
                    description=f"Price breached lower Bollinger Band (${current_price:.2f} < ${current_lower:.2f})",
                    data={
                        'indicator': 'Bollinger_Bands',
                        'current_price': current_price,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'middle_band': current_middle,
                        'signal': 'lower_breach'
                    },
                    confidence=min(0.9, (current_lower - current_price) / current_lower * 10)
                )
                triggers.append(trigger)

            # Squeeze identification (bands getting narrow)
            if len(bb_upper) >= 20:
                band_width_20 = (bb_upper[-20] - bb_lower[-20]) / current_middle * 100
                band_width_current = (current_upper - current_lower) / current_middle * 100

                if band_width_current < band_width_20 * 0.7:  # Width reduced by 30%
                    trigger = TriggerEvent(
                        symbol=symbol,
                        trigger_type=self.trigger_type,
                        severity=TriggerSeverity.MEDIUM,
                        timestamp=datetime.now(),
                        description="Bollinger Band squeeze detected (volatility contraction)",
                        data={
                            'indicator': 'Bollinger_Bands',
                            'current_band_width': band_width_current,
                            'historical_band_width': band_width_20,
                            'signal': 'squeeze'
                        },
                        confidence=0.6
                    )
                    triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting Bollinger Band triggers for {symbol}: {e}")

        return triggers

    async def _detect_ma_crossover_triggers(self, symbol: str, indicators: Dict[str, Any]) -> List[TriggerEvent]:
        """Detect moving average crossover triggers."""
        triggers = []

        try:
            # Get various moving averages
            ema_20 = indicators.get('EMA_20', [])
            ema_50 = indicators.get('EMA_50', [])
            sma_20 = indicators.get('SMA_20', [])
            sma_50 = indicators.get('SMA_50', [])
            sma_200 = indicators.get('SMA_200', [])

            # EMA 20/50 crossover
            if ema_20 and ema_50 and len(ema_20) >= 2 and len(ema_50) >= 2:
                if ema_20[-1] > ema_50[-1] and ema_20[-2] <= ema_50[-2]:
                    trigger = TriggerEvent(
                        symbol=symbol,
                        trigger_type=self.trigger_type,
                        severity=TriggerSeverity.HIGH,
                        timestamp=datetime.now(),
                        description="EMA 20 crossed above EMA 50 (bullish signal)",
                        data={
                            'indicator': 'MA_Crossover',
                            'fast_ma': 'EMA_20',
                            'slow_ma': 'EMA_50',
                            'fast_value': ema_20[-1],
                            'slow_value': ema_50[-1],
                            'signal': 'bullish_ema_crossover'
                        },
                        confidence=0.8
                    )
                    triggers.append(trigger)

                elif ema_20[-1] < ema_50[-1] and ema_20[-2] >= ema_50[-2]:
                    trigger = TriggerEvent(
                        symbol=symbol,
                        trigger_type=self.trigger_type,
                        severity=TriggerSeverity.HIGH,
                        timestamp=datetime.now(),
                        description="EMA 20 crossed below EMA 50 (bearish signal)",
                        data={
                            'indicator': 'MA_Crossover',
                            'fast_ma': 'EMA_20',
                            'slow_ma': 'EMA_50',
                            'fast_value': ema_20[-1],
                            'slow_value': ema_50[-1],
                            'signal': 'bearish_ema_crossover'
                        },
                        confidence=0.8
                    )
                    triggers.append(trigger)

            # Price vs 200-day SMA
            if sma_200 and len(sma_200) >= 1:
                current_price = indicators.get('current_price', 0)
                if current_price and len(sma_200) >= 2:
                    if current_price > sma_200[-1] and sma_200[-2] > 0:
                        # Check if this is a recent cross above
                        if 'prev_price' in indicators:
                            prev_price = indicators['prev_price']
                            if prev_price <= sma_200[-2]:
                                trigger = TriggerEvent(
                                    symbol=symbol,
                                    trigger_type=self.trigger_type,
                                    severity=TriggerSeverity.MEDIUM,
                                    timestamp=datetime.now(),
                                    description=f"Price crossed above 200-day SMA (${current_price:.2f} > ${sma_200[-1]:.2f})",
                                    data={
                                        'indicator': 'Price_vs_SMA',
                                        'current_price': current_price,
                                        'sma_200': sma_200[-1],
                                        'signal': 'price_above_sma200'
                                    },
                                    confidence=0.7
                                )
                                triggers.append(trigger)
                    elif current_price < sma_200[-1] and len(sma_200) >= 2:
                        if 'prev_price' in indicators:
                            prev_price = indicators['prev_price']
                            if prev_price >= sma_200[-2]:
                                trigger = TriggerEvent(
                                    symbol=symbol,
                                    trigger_type=self.trigger_type,
                                    severity=TriggerSeverity.MEDIUM,
                                    timestamp=datetime.now(),
                                    description=f"Price crossed below 200-day SMA (${current_price:.2f} < ${sma_200[-1]:.2f})",
                                    data={
                                        'indicator': 'Price_vs_SMA',
                                        'current_price': current_price,
                                        'sma_200': sma_200[-1],
                                        'signal': 'price_below_sma200'
                                    },
                                    confidence=0.7
                                )
                                triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting MA crossover triggers for {symbol}: {e}")

        return triggers

    async def _detect_stochastic_triggers(self, symbol: str, indicators: Dict[str, Any]) -> List[TriggerEvent]:
        """Detect Stochastic oscillator triggers."""
        triggers = []

        try:
            stoch_k = indicators.get('STOCHk_14_3_3', [])
            stoch_d = indicators.get('STOCHd_14_3_3', [])

            if not stoch_k or not stoch_d or len(stoch_k) < 2 or len(stoch_d) < 2:
                return triggers

            current_k = stoch_k[-1]
            previous_k = stoch_k[-2]
            current_d = stoch_d[-1]
            previous_d = stoch_d[-2]

            # Overbought/Oversold conditions
            if current_k > self.stoch_overbought and previous_k <= self.stoch_overbought:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"Stochastic entered overbought zone ({current_k:.1f} > {self.stoch_overbought})",
                    data={
                        'indicator': 'Stochastic',
                        'current_k': current_k,
                        'current_d': current_d,
                        'signal': 'overbought'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

            elif current_k < self.stoch_oversold and previous_k >= self.stoch_oversold:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"Stochastic entered oversold zone ({current_k:.1f} < {self.stoch_oversold})",
                    data={
                        'indicator': 'Stochastic',
                        'current_k': current_k,
                        'current_d': current_d,
                        'signal': 'oversold'
                    },
                    confidence=0.7
                )
                triggers.append(trigger)

            # Stochastic crossover (K crossing D)
            if current_k > current_d and previous_k <= previous_d and current_k < self.stoch_overbought:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description="Stochastic bullish crossover (%K crossed above %D)",
                    data={
                        'indicator': 'Stochastic',
                        'current_k': current_k,
                        'current_d': current_d,
                        'previous_k': previous_k,
                        'previous_d': previous_d,
                        'signal': 'bullish_crossover'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

            elif current_k < current_d and previous_k >= previous_d and current_k > self.stoch_oversold:
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description="Stochastic bearish crossover (%K crossed below %D)",
                    data={
                        'indicator': 'Stochastic',
                        'current_k': current_k,
                        'current_d': current_d,
                        'previous_k': previous_k,
                        'previous_d': previous_d,
                        'signal': 'bearish_crossover'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting Stochastic triggers for {symbol}: {e}")

        return triggers

    async def _detect_price_pattern_triggers(self, symbol: str, historical_data: pd.DataFrame, current_price: float) -> List[TriggerEvent]:
        """Detect direct price pattern triggers."""
        triggers = []

        try:
            if len(historical_data) < 10:
                return triggers

            # Recent price action analysis
            if len(historical_data) < 10:
                return triggers

            recent_prices = historical_data['Close'].tail(10)
            volumes = historical_data['Volume'].tail(10)

            # Volume spike detection
            avg_volume = volumes.mean()
            current_volume = volumes.iloc[-1]

            if current_volume > avg_volume * 3:  # 3x volume spike
                price_change_pct = (current_price - recent_prices.iloc[-2]) / recent_prices.iloc[-2] * 100

                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=self.calculate_trigger_severity(0.7, abs(price_change_pct) / 10),
                    timestamp=datetime.now(),
                    description=f"High volume spike detected ({current_volume:.0f} vs avg {avg_volume:.0f})",
                    data={
                        'indicator': 'Volume_Spike',
                        'current_volume': current_volume,
                        'avg_volume': avg_volume,
                        'volume_ratio': current_volume / avg_volume,
                        'price_change_pct': price_change_pct,
                        'signal': 'volume_spike'
                    },
                    confidence=min(0.9, current_volume / avg_volume / 5)
                )
                triggers.append(trigger)

            # Gap detection
            previous_close = recent_prices.iloc[-2]
            gap_pct = (current_price - previous_close) / previous_close * 100

            if abs(gap_pct) > 2:  # 2%+ gap
                direction = 'up' if gap_pct > 0 else 'down'

                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description=f"Price gap {direction} ({abs(gap_pct):.1f}%) detected",
                    data={
                        'indicator': 'Price_Gap',
                        'previous_close': previous_close,
                        'current_price': current_price,
                        'gap_pct': gap_pct,
                        'signal': f'gap_{direction}'
                    },
                    confidence=min(0.9, abs(gap_pct) / 5)
                )
                triggers.append(trigger)

            # Support/Resistance breakout
            highs = historical_data['High'].tail(20)
            lows = historical_data['Low'].tail(20)

            resistance_level = highs.max()
            support_level = lows.min()

            if current_price > resistance_level * 1.02:  # 2% above resistance
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description=f"Price broke through resistance (${current_price:.2f} > ${resistance_level:.2f})",
                    data={
                        'indicator': 'Resistance_Breakout',
                        'current_price': current_price,
                        'resistance_level': resistance_level,
                        'signal': 'resistance_breakout'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

            elif current_price < support_level * 0.98:  # 2% below support
                trigger = TriggerEvent(
                    symbol=symbol,
                    trigger_type=self.trigger_type,
                    severity=TriggerSeverity.HIGH,
                    timestamp=datetime.now(),
                    description=f"Price broke below support (${current_price:.2f} < ${support_level:.2f})",
                    data={
                        'indicator': 'Support_Breakdown',
                        'current_price': current_price,
                        'support_level': support_level,
                        'signal': 'support_breakdown'
                    },
                    confidence=0.8
                )
                triggers.append(trigger)

        except Exception as e:
            logger.error(f"Error detecting price pattern triggers for {symbol}: {e}")

        return triggers
