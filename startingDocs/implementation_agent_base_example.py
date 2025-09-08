from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import openai

@dataclass
class AgentConfig:
    name: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 30.0

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    ohlc: Dict[str, float]
    technical_indicators: Dict[str, float]

@dataclass
class AgentDecision:
    agent_name: str
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    supporting_data: Dict[str, Any]
    timestamp: datetime

class BaseAgent(ABC):
    def __init__(self, config: AgentConfig, llm_client, message_bus, state_manager):
        self.config = config
        self.llm_client = llm_client
        self.message_bus = message_bus
        self.state_manager = state_manager

    @abstractmethod
    async def analyze(self, market_data: MarketData) -> AgentDecision:
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    async def make_llm_call(self, user_prompt: str) -> str:
        system_prompt = self.get_system_prompt()

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM call failed for {self.config.name}: {e}")

# Example Technical Analysis Agent
class TechnicalAnalysisAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """You are a professional technical analyst with 20+ years of experience.
        Analyze market data using technical indicators and chart patterns.
        Provide clear BUY/SELL/HOLD signals with confidence levels and reasoning.
        Focus on: trend analysis, momentum indicators, volume analysis, support/resistance."""

    async def analyze(self, market_data: MarketData) -> AgentDecision:
        # Build analysis prompt with market data
        prompt = f"""
        Analyze {market_data.symbol}:
        Price: ${market_data.price}
        Volume: {market_data.volume:,}
        Technical Indicators:
        - RSI: {market_data.technical_indicators.get('rsi', 'N/A')}
        - MACD: {market_data.technical_indicators.get('macd', 'N/A')}
        - SMA 20: {market_data.technical_indicators.get('sma_20', 'N/A')}
        - SMA 50: {market_data.technical_indicators.get('sma_50', 'N/A')}

        Provide your analysis in JSON format:
        {{"signal": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "reasoning": "explanation"}}
        """

        # Get LLM analysis
        analysis = await self.make_llm_call(prompt)

        # Parse response (simplified - add proper JSON parsing)
        # For demo, return mock decision
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal="BUY",
            confidence=0.8,
            reasoning="Strong technical setup with RSI oversold",
            supporting_data=market_data.technical_indicators,
            timestamp=datetime.now()
        )
