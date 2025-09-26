"""
Implements the Technical Analysis Agent.

This agent analyzes market data from a quantitative perspective, focusing on
price patterns, technical indicators, and other statistical measures.
"""
import json

from .base import BaseAgent
from .data_structures import AgentDecision, MarketData


class TechnicalAnalysisAgent(BaseAgent):
    """
    An agent specialized in technical analysis of financial markets.
    """

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the technical analysis LLM.
        """
        return (
            "You are a specialized AI assistant for financial technical analysis. "
            "Your goal is to analyze the provided market data and technical indicators. "
            "Determine a trading signal (BUY, SELL, or HOLD) and a confidence score (0.0 to 1.0). "
            "You must provide your reasoning in a brief, data-driven explanation. "
            "Your final output must be a single JSON object with three keys: "
            "'signal', 'confidence', and 'reasoning'."
        )

    async def analyze(self, market_data: MarketData, **kwargs) -> AgentDecision:
        """
        Performs technical analysis on the given market data using an LLM.
        """
        # 1. Format the market data into a user prompt for the LLM.
        user_prompt = (
            f"Analyze the following market data for {market_data.symbol}:\n"
            f"- Current Price: {market_data.price}\n"
            f"- Trading Volume: {market_data.volume}\n"
            f"- OHLC: {market_data.ohlc}\n"
            f"- Technical Indicators: {market_data.technical_indicators}\n\n"
            "Based on this data, provide your trading signal, confidence, and reasoning "
            "as a single JSON object."
        )

        # 2. Make the LLM call.
        llm_response = await self.make_llm_call(user_prompt)

        # 3. Parse the LLM's JSON response to create an AgentDecision.
        try:
            decision_json = json.loads(llm_response)
            signal = decision_json.get("signal", "HOLD")
            confidence = float(decision_json.get("confidence", 0.0))
            reasoning = decision_json.get("reasoning", "No reasoning provided.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, create a default decision with an error message.
            signal = "ERROR"
            confidence = 0.0
            reasoning = f"Failed to parse LLM response: {e}. Raw response: {llm_response}"

        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "llm_response": llm_response,
                "market_data_used": market_data.__dict__,
            },
        )
