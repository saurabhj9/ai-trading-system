"""
Implements the Technical Analysis Agent.

This agent analyzes market data from a quantitative perspective, focusing on
price patterns, technical indicators, and other statistical measures.
"""
import json
import pandas as pd
import time
import random
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from ..analysis.market_regime import RegimeDetector
from ..config.settings import settings
from ..signal_generation.signal_generator import LocalSignalGenerator
from ..signal_generation.core import Signal, SignalType
from ..config.signal_generation import signal_generation_config
from .base import BaseAgent
from .data_structures import AgentDecision, MarketData


class TechnicalAnalysisAgent(BaseAgent):
    """
    An agent specialized in technical analysis of financial markets.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the TechnicalAnalysisAgent.
        """
        super().__init__(*args, **kwargs)
        if settings.regime_detection.ENABLED:
            self.regime_detector = RegimeDetector(
                config={
                    "adx_period": settings.regime_detection.ADX_PERIOD,
                    "atr_period": settings.regime_detection.ATR_PERIOD,
                    "hurst_exponent_lag": settings.regime_detection.HURST_EXPONENT_LAG,
                    "trend_strength_threshold": settings.regime_detection.TREND_STRENGTH_THRESHOLD,
                    "ranging_threshold": settings.regime_detection.RANGING_THRESHOLD,
                    "volatility_threshold_percent": settings.regime_detection.VOLATILITY_THRESHOLD_PERCENT,
                    "confirmation_periods": settings.regime_detection.CONFIRMATION_PERIODS,
                }
            )
        else:
            self.regime_detector = None

        # Initialize LocalSignalGenerator if enabled
        self.local_signal_generator = None
        if settings.signal_generation.LOCAL_SIGNAL_GENERATION_ENABLED:
            try:
                self.local_signal_generator = LocalSignalGenerator(
                    config=signal_generation_config.to_dict()
                )
            except Exception as e:
                print(f"Failed to initialize LocalSignalGenerator: {e}")
                if not settings.signal_generation.FALLBACK_TO_LLM_ON_ERROR:
                    raise

        # Performance tracking for comparison metrics
        self.performance_metrics = {
            "local_signals": 0,
            "llm_signals": 0,
            "local_avg_time": 0.0,
            "llm_avg_time": 0.0,
            "local_errors": 0,
            "llm_errors": 0,
            "escalations": 0,
            "comparisons": 0,
        }

    def get_system_prompt(self, market_regime: Optional[str] = None) -> str:
        """
        Returns the system prompt for the technical analysis LLM.

        Args:
            market_regime: The current market regime, if detected.
        """
        regime_instruction = ""
        if market_regime:
            regime_instruction = (
                f"The current market regime is {market_regime}. "
                "Adapt your analysis based on this regime: "
                "- In a TRENDING_UP market, prioritize strategies that capture upward momentum and be cautious against shorting. "
                "- In a TRENDING_DOWN market, prioritize shorting or defensive strategies and be cautious against buying. "
                "- In a RANGING market, consider mean-reversion strategies like buying near support and selling near resistance. "
                "- In a VOLATILE market, be cautious of false signals and consider wider stop-losses. "
                "- In an UNCERTAIN market, signal confidence may be lower, and a HOLD signal is often safer. "
            )

        return (
            "You are a specialized AI assistant for financial technical analysis. "
            "Your goal is to analyze market data and technical indicators to determine a trading signal (BUY, SELL, or HOLD). "
            "Provide a confidence score (0.0 to 1.0) and a brief, data-driven reasoning. "
            f"{regime_instruction}"
            "Key Indicator Interpretations: "
            "- RSI (Relative Strength Index): An RSI below 30 is generally considered oversold and a potential BUY signal. An RSI above 70 is overbought and a potential SELL signal. "
            "- MACD (Moving Average Convergence Divergence): A MACD line crossing above the signal line is a bullish (BUY) signal. A MACD line crossing below the signal line is a bearish (SELL) signal. "
            "Analyze all provided indicators together for a comprehensive view. "
            "When historical data is provided, look for trends (e.g., RSI improving from oversold levels) and divergences (e.g., price making new lows while RSI makes higher lows, indicating weakening downward momentum). "
            "Your final output must be a single JSON object with three keys: 'signal', 'confidence', and 'reasoning'."
        )

    async def get_user_prompt(self, market_data: MarketData) -> str:
        """
        Generates the user prompt for technical analysis.
        """
        historical_str = ""
        if market_data.historical_indicators:
            historical_str = f"- Historical Indicators (last {len(market_data.historical_indicators)} periods): {market_data.historical_indicators}\n"

        return (
            f"Analyze the following market data for {market_data.symbol}:\n"
            f"- Current Price: {market_data.price}\n"
            f"- Trading Volume: {market_data.volume}\n"
            f"- OHLC: {market_data.ohlc}\n"
            f"- Technical Indicators: {market_data.technical_indicators}\n"
            f"{historical_str}\n"
            "Based on this data, provide your trading signal, confidence, and reasoning "
            "as a single JSON object."
        )

    def create_decision(self, market_data: MarketData, response: str) -> AgentDecision:
        """
        Creates an AgentDecision from the LLM response for technical analysis.
        """
        try:
            decision_json = json.loads(response)
            signal = decision_json.get("signal", "HOLD")
            confidence = float(decision_json.get("confidence", 0.0))
            reasoning = decision_json.get("reasoning", "No reasoning provided.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, create a default decision with an error message.
            signal = "ERROR"
            confidence = 0.0
            reasoning = f"Failed to parse LLM response: {e}. Raw response: {response}"

        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "llm_response": response,
                "market_data_used": market_data.__dict__,
                "signal_source": "LLM",
            },
        )

    def _should_use_local_generation(self, market_data: MarketData) -> bool:
        """
        Determine if local signal generation should be used based on configuration.

        Args:
            market_data: The market data to analyze

        Returns:
            bool: True if local generation should be used
        """
        # Check if local generation is enabled
        if not settings.signal_generation.LOCAL_SIGNAL_GENERATION_ENABLED:
            return False

        # Check if LocalSignalGenerator is available
        if not self.local_signal_generator:
            return False

        # Check if symbol is in enabled list (if list is not empty)
        if settings.signal_generation.ENABLED_SYMBOLS:
            if market_data.symbol not in settings.signal_generation.ENABLED_SYMBOLS:
                return False

        # Check rollout percentage
        if settings.signal_generation.ROLLOUT_PERCENTAGE < 1.0:
            if random.random() > settings.signal_generation.ROLLOUT_PERCENTAGE:
                return False

        return True

    def _should_enable_comparison(self) -> bool:
        """
        Determine if side-by-side comparison should be enabled for this request.

        Returns:
            bool: True if comparison should be enabled
        """
        if not settings.signal_generation.ENABLE_SIDE_BY_SIDE_COMPARISON:
            return False

        # Sample based on comparison rate
        if settings.signal_generation.COMPARISON_SAMPLE_RATE < 1.0:
            return random.random() <= settings.signal_generation.COMPARISON_SAMPLE_RATE

        return True

    def _convert_market_data_to_dataframe(self, market_data: MarketData) -> pd.DataFrame:
        """
        Convert MarketData object to pandas DataFrame for LocalSignalGenerator.

        Args:
            market_data: The market data to convert

        Returns:
            pd.DataFrame: OHLCV data in DataFrame format
        """
        # Create DataFrame from historical data if available
        if market_data.historical_ohlc:
            df = pd.DataFrame(market_data.historical_ohlc)
            df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
            df.index = pd.to_datetime(df.index)
            return df

        # If no historical data, create a single-row DataFrame with current data
        current_data = {
            'open': [market_data.ohlc.get('open', market_data.price)],
            'high': [market_data.ohlc.get('high', market_data.price)],
            'low': [market_data.ohlc.get('low', market_data.price)],
            'close': [market_data.price],
            'volume': [market_data.volume],
        }
        df = pd.DataFrame(current_data)
        df.index = pd.to_datetime([market_data.timestamp])
        return df

    def _convert_signal_to_decision(self, signal: Signal, market_data: MarketData, source: str) -> AgentDecision:
        """
        Convert Signal object to AgentDecision.

        Args:
            signal: The signal from LocalSignalGenerator
            market_data: The original market data
            source: Source identifier ("LOCAL" or "LLM")

        Returns:
            AgentDecision: Converted decision object
        """
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal.signal_type.value,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            supporting_data={
                "signal_source": source,
                "signal_strength": signal.strength.value,
                "market_regime": signal.regime.value,
                "indicators": signal.indicators,
                "signal_metadata": signal.metadata,
                "market_data_used": market_data.__dict__,
            },
        )

    def _update_performance_metrics(self, source: str, generation_time: float, is_error: bool = False):
        """
        Update performance metrics for tracking.

        Args:
            source: "LOCAL" or "LLM"
            generation_time: Time taken to generate signal
            is_error: Whether an error occurred
        """
        if source == "LOCAL":
            self.performance_metrics["local_signals"] += 1
            if is_error:
                self.performance_metrics["local_errors"] += 1
            else:
                # Update average time
                total = self.performance_metrics["local_signals"]
                current_avg = self.performance_metrics["local_avg_time"]
                self.performance_metrics["local_avg_time"] = (
                    (current_avg * (total - 1) + generation_time) / total
                )
        elif source == "LLM":
            self.performance_metrics["llm_signals"] += 1
            if is_error:
                self.performance_metrics["llm_errors"] += 1
            else:
                # Update average time
                total = self.performance_metrics["llm_signals"]
                current_avg = self.performance_metrics["llm_avg_time"]
                self.performance_metrics["llm_avg_time"] = (
                    (current_avg * (total - 1) + generation_time) / total
                )

    async def analyze(self, market_data: MarketData, **kwargs) -> AgentDecision:
        """
        Performs technical analysis on the given market data using either
        local signal generation or LLM-based analysis based on configuration.
        """
        start_time = time.time()

        # Determine if we should use local generation
        use_local = self._should_use_local_generation(market_data)
        enable_comparison = self._should_enable_comparison()

        # Handle comparison mode (generate both signals)
        if enable_comparison and use_local and self.local_signal_generator:
            try:
                # Generate both signals for comparison
                local_decision = await self._generate_local_signal(market_data)
                llm_decision = await self._generate_llm_signal(market_data)

                # Use local signal as primary but include comparison data
                local_decision.supporting_data["comparison"] = {
                    "llm_signal": llm_decision.signal,
                    "llm_confidence": llm_decision.confidence,
                    "llm_reasoning": llm_decision.reasoning,
                }

                self.performance_metrics["comparisons"] += 1
                return local_decision
            except Exception as e:
                print(f"Error in comparison mode: {e}")
                # Fall back to LLM if comparison fails
                return await self._generate_llm_signal(market_data)

        # Handle hybrid mode
        elif settings.signal_generation.HYBRID_MODE_ENABLED and use_local and self.local_signal_generator:
            try:
                # Try local generation first
                local_decision = await self._generate_local_signal(market_data)

                # Check if we should escalate to LLM based on confidence or conflicts
                should_escalate = False
                escalation_reason = ""

                # Check confidence threshold
                if local_decision.confidence < settings.signal_generation.ESCALATION_CONFIDENCE_THRESHOLD:
                    should_escalate = True
                    escalation_reason = f"Low confidence ({local_decision.confidence:.2f} < {settings.signal_generation.ESCALATION_CONFIDENCE_THRESHOLD})"

                # Check for conflicts in metadata
                if "conflicts" in local_decision.supporting_data.get("signal_metadata", {}):
                    conflicts = local_decision.supporting_data["signal_metadata"]["conflicts"]
                    if len(conflicts) >= settings.signal_generation.ESCALATION_CONFLICT_THRESHOLD:
                        should_escalate = True
                        escalation_reason = f"Too many conflicts ({len(conflicts)} >= {settings.signal_generation.ESCALATION_CONFLICT_THRESHOLD})"

                # Escalate if needed
                if should_escalate and settings.signal_generation.ESCALATION_ENABLED:
                    self.performance_metrics["escalations"] += 1
                    llm_decision = await self._generate_llm_signal(market_data)

                    # Add escalation info to LLM decision
                    llm_decision.supporting_data["escalation_info"] = {
                        "escalated_from": "LOCAL",
                        "escalation_reason": escalation_reason,
                        "local_signal": local_decision.signal,
                        "local_confidence": local_decision.confidence,
                    }

                    generation_time = time.time() - start_time
                    self._update_performance_metrics("LLM", generation_time)
                    return llm_decision

                # Return local decision if no escalation
                generation_time = time.time() - start_time
                self._update_performance_metrics("LOCAL", generation_time)
                return local_decision

            except Exception as e:
                print(f"Error in local signal generation: {e}")
                if settings.signal_generation.FALLBACK_TO_LLM_ON_ERROR:
                    return await self._generate_llm_signal(market_data)
                else:
                    # Return error decision
                    return self._create_error_decision(market_data, str(e))

        # Handle local-only mode
        elif use_local and self.local_signal_generator:
            try:
                local_decision = await self._generate_local_signal(market_data)
                generation_time = time.time() - start_time
                self._update_performance_metrics("LOCAL", generation_time)
                return local_decision
            except Exception as e:
                print(f"Error in local signal generation: {e}")
                if settings.signal_generation.FALLBACK_TO_LLM_ON_ERROR:
                    return await self._generate_llm_signal(market_data)
                else:
                    return self._create_error_decision(market_data, str(e))

        # Default to LLM-based analysis
        else:
            llm_decision = await self._generate_llm_signal(market_data)
            generation_time = time.time() - start_time
            self._update_performance_metrics("LLM", generation_time)
            return llm_decision

    async def _generate_local_signal(self, market_data: MarketData) -> AgentDecision:
        """
        Generate a signal using the LocalSignalGenerator.

        Args:
            market_data: The market data to analyze

        Returns:
            AgentDecision: Decision from local signal generation
        """
        # Convert market data to DataFrame format
        df = self._convert_market_data_to_dataframe(market_data)

        # Generate signal using LocalSignalGenerator
        signal, metadata = self.local_signal_generator.generate_signal(df, market_data.symbol)

        # Convert to AgentDecision
        decision = self._convert_signal_to_decision(signal, market_data, "LOCAL")

        # Add metadata
        decision.supporting_data["local_generation_metadata"] = metadata

        return decision

    async def _generate_llm_signal(self, market_data: MarketData) -> AgentDecision:
        """
        Generate a signal using the LLM (original method).

        Args:
            market_data: The market data to analyze

        Returns:
            AgentDecision: Decision from LLM analysis
        """
        market_regime_str = None
        if self.regime_detector and market_data.historical_ohlc:
            try:
                df = pd.DataFrame(market_data.historical_ohlc)
                df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
                df.index = pd.to_datetime(df.index)
                regime = self.regime_detector.detect(df)
                market_regime_str = regime.regime.value
            except Exception as e:
                # Log the error and continue without regime detection
                print(f"Error during regime detection: {e}")

        user_prompt = await self.get_user_prompt(market_data)
        system_prompt = self.get_system_prompt(market_regime_str)
        llm_response = await self.make_llm_call(user_prompt, system_prompt=system_prompt)
        return self.create_decision(market_data, llm_response)

    def _create_error_decision(self, market_data: MarketData, error_message: str) -> AgentDecision:
        """
        Create an error decision when signal generation fails.

        Args:
            market_data: The market data that failed to analyze
            error_message: The error message

        Returns:
            AgentDecision: Error decision
        """
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal="ERROR",
            confidence=0.0,
            reasoning=f"Signal generation failed: {error_message}",
            supporting_data={
                "signal_source": "ERROR",
                "error_message": error_message,
                "market_data_used": market_data.__dict__,
            },
        )


    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.

        Returns:
            Dict[str, Any]: Performance metrics including local vs LLM comparison
        """
        metrics = self.performance_metrics.copy()

        # Add calculated metrics
        total_signals = metrics["local_signals"] + metrics["llm_signals"]
        if total_signals > 0:
            metrics["local_signal_percentage"] = metrics["local_signals"] / total_signals * 100
            metrics["llm_signal_percentage"] = metrics["llm_signals"] / total_signals * 100

        # Add error rates
        if metrics["local_signals"] > 0:
            metrics["local_error_rate"] = metrics["local_errors"] / metrics["local_signals"] * 100
        else:
            metrics["local_error_rate"] = 0.0

        if metrics["llm_signals"] > 0:
            metrics["llm_error_rate"] = metrics["llm_errors"] / metrics["llm_signals"] * 100
        else:
            metrics["llm_error_rate"] = 0.0

        # Add escalation rate
        if metrics["local_signals"] > 0:
            metrics["escalation_rate"] = metrics["escalations"] / metrics["local_signals"] * 100
        else:
            metrics["escalation_rate"] = 0.0

        # Add performance comparison
        if metrics["local_avg_time"] > 0 and metrics["llm_avg_time"] > 0:
            metrics["performance_improvement"] = (
                (metrics["llm_avg_time"] - metrics["local_avg_time"]) / metrics["llm_avg_time"] * 100
            )
        else:
            metrics["performance_improvement"] = 0.0

        # Add LocalSignalGenerator metrics if available
        if self.local_signal_generator:
            metrics["local_generator_metrics"] = self.local_signal_generator.get_performance_metrics()

        return metrics

    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            "local_signals": 0,
            "llm_signals": 0,
            "local_avg_time": 0.0,
            "llm_avg_time": 0.0,
            "local_errors": 0,
            "llm_errors": 0,
            "escalations": 0,
            "comparisons": 0,
        }

        # Reset LocalSignalGenerator metrics if available
        if self.local_signal_generator:
            self.local_signal_generator.reset_metrics()

    async def batch_analyze(self, market_data_list: List[MarketData], **kwargs) -> List[AgentDecision]:
        """
        Analyzes multiple market data instances in batch mode for optimized processing.

        This method supports both local and LLM-based signal generation in batch mode,
        with performance optimization for local generation.

        Args:
            market_data_list: List of market data to analyze.
            **kwargs: Additional arguments specific to the agent type.

        Returns:
            List[AgentDecision]: Decisions corresponding to each market data input.
        """
        if not market_data_list:
            return []

        start_time = time.time()
        decisions = []

        # Check if we should use local generation for the batch
        use_local = self._should_use_local_generation(market_data_list[0])  # Use first item as reference

        if use_local and self.local_signal_generator:
            # Process batch with local signal generation
            try:
                for market_data in market_data_list:
                    decision = await self._generate_local_signal(market_data)
                    decisions.append(decision)

                # Update performance metrics for batch
                batch_time = time.time() - start_time
                avg_time_per_signal = batch_time / len(market_data_list)
                for _ in market_data_list:
                    self._update_performance_metrics("LOCAL", avg_time_per_signal)

                return decisions

            except Exception as e:
                print(f"Error in batch local signal generation: {e}")
                if not settings.signal_generation.FALLBACK_TO_LLM_ON_ERROR:
                    # Return error decisions for all items
                    return [self._create_error_decision(md, str(e)) for md in market_data_list]

        # Fall back to individual LLM calls for batch
        # Note: We could optimize this with actual LLM batch processing in the future
        for market_data in market_data_list:
            try:
                decision = await self._generate_llm_signal(market_data)
                decisions.append(decision)
            except Exception as e:
                print(f"Error in batch LLM signal generation: {e}")
                error_decision = self._create_error_decision(market_data, str(e))
                decisions.append(error_decision)

        # Update performance metrics for batch
        batch_time = time.time() - start_time
        avg_time_per_signal = batch_time / len(market_data_list)
        for _ in market_data_list:
            self._update_performance_metrics("LLM", avg_time_per_signal)

        return decisions
