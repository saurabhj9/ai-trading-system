"""
Deterministic, rule-based versions of agents for fast backtesting.
"""
from typing import Dict, Any

from src.agents.data_structures import AgentDecision, MarketData


class DeterministicTechnicalAgent:
    """Rule-based technical analysis."""

    def analyze(self, market_data: MarketData) -> AgentDecision:
        rsi = market_data.technical_indicators.get('RSI', 50)
        signal = "HOLD"
        if rsi < 30:
            signal = "BUY"
        elif rsi > 70:
            signal = "SELL"
        return AgentDecision(
            agent_name="deterministic_technical",
            symbol=market_data.symbol,
            signal=signal,
            confidence=0.8,
            reasoning=f"RSI {rsi}: {signal}",
            timestamp=market_data.timestamp
        )


class DeterministicSentimentAgent:
    """Mock sentiment analysis."""

    def analyze(self, market_data: MarketData) -> AgentDecision:
        return AgentDecision(
            agent_name="deterministic_sentiment",
            symbol=market_data.symbol,
            signal="NEUTRAL",
            confidence=0.5,
            reasoning="Mock neutral sentiment",
            timestamp=market_data.timestamp
        )


class DeterministicRiskAgent:
    """Always approves."""

    def analyze(self, market_data: MarketData, decisions: Dict[str, AgentDecision], portfolio: Dict[str, Any]) -> AgentDecision:
        return AgentDecision(
            agent_name="deterministic_risk",
            symbol=market_data.symbol,
            signal="APPROVE",
            confidence=1.0,
            reasoning="Always approved",
            timestamp=market_data.timestamp
        )


class DeterministicPortfolioAgent:
    """Decides based on technical signal."""

    def analyze(self, market_data: MarketData, decisions: Dict[str, AgentDecision], portfolio: Dict[str, Any]) -> AgentDecision:
        tech_decision = decisions.get("technical")
        signal = tech_decision.signal if tech_decision else "HOLD"
        return AgentDecision(
            agent_name="deterministic_portfolio",
            symbol=market_data.symbol,
            signal=signal,
            confidence=0.9,
            reasoning=f"Following technical: {signal}",
            timestamp=market_data.timestamp
        )
