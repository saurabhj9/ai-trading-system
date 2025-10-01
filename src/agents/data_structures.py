"""
Core data structures for the AI Trading System.

This module defines the fundamental data classes used for communication and state
management between the various components of the system, including agents,
the data pipeline, and the orchestration layer.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config.settings import settings


@dataclass
class AgentConfig:
    """
    Configuration for a trading agent.

    Attributes:
        name: The unique name of the agent.
        model_name: The name of the language model to use for analysis.
        temperature: The sampling temperature for the language model.
        max_tokens: The maximum number of tokens to generate.
        timeout: The timeout in seconds for LLM API calls.
        retry_attempts: The number of times to retry a failed API call.
    """
    name: str
    model_name: str = settings.llm.DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 30.0
    retry_attempts: int = 3

@dataclass
class MarketData:
    """
    Represents a snapshot of market data for a specific financial instrument.

    Attributes:
        symbol: The stock ticker or symbol.
        price: The current market price.
        volume: The trading volume for the current period.
        timestamp: The timestamp of when the data was captured.
        ohlc: A dictionary containing the Open, High, Low, and Close prices.
        technical_indicators: A dictionary of calculated technical indicators.
        momentum_indicators: A dictionary of calculated momentum indicators.
        mean_reversion_indicators: A dictionary of calculated mean reversion indicators.
        historical_indicators: A list of dictionaries containing historical technical indicators.
        historical_momentum: A list of dictionaries containing historical momentum indicators.
        historical_mean_reversion: A list of dictionaries containing historical mean reversion indicators.
        fundamental_data: Optional dictionary of fundamental metrics.
    """
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    ohlc: Dict[str, float]
    technical_indicators: Dict[str, float]
    momentum_indicators: Dict[str, float] = field(default_factory=dict)
    mean_reversion_indicators: Dict[str, float] = field(default_factory=dict)
    volatility_indicators: Dict[str, float] = field(default_factory=dict)
    trend_indicators: Dict[str, float] = field(default_factory=dict)
    volume_indicators: Dict[str, float] = field(default_factory=dict)
    statistical_indicators: Dict[str, float] = field(default_factory=dict)
    historical_indicators: List[Dict[str, float]] = field(default_factory=list)
    historical_momentum: List[Dict[str, float]] = field(default_factory=list)
    historical_mean_reversion: List[Dict[str, float]] = field(default_factory=list)
    historical_volatility: List[Dict[str, float]] = field(default_factory=list)
    historical_trend: List[Dict[str, float]] = field(default_factory=list)
    historical_volume: List[Dict[str, float]] = field(default_factory=list)
    historical_statistical: List[Dict[str, float]] = field(default_factory=list)
    fundamental_data: Optional[Dict[str, Any]] = None

@dataclass
class AgentDecision:
    """
    Represents the output of a trading agent's analysis.

    Attributes:
        agent_name: The name of the agent that produced the decision.
        symbol: The stock ticker or symbol analyzed.
        signal: The trading signal (e.g., 'BUY', 'SELL', 'HOLD').
        confidence: The agent's confidence in the signal (from 0.0 to 1.0).
        reasoning: The textual explanation for the decision from the LLM.
        supporting_data: A dictionary of data used to make the decision.
        timestamp: The timestamp of when the decision was made.
    """
    agent_name: str
    symbol: str
    signal: str
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
