"""
Implements the Technical Analysis Agent.

This agent analyzes market data from a quantitative perspective, focusing on
price patterns, technical indicators, and other statistical measures.
"""
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
            "Your goal is to analyze the provided market data, including technical "
            "indicators, and determine a trading signal (BUY, SELL, or HOLD). "
            "Provide a confidence score for your signal and a brief, data-driven "
            "reasoning for your decision. Focus only on the quantitative data provided."
        )

    async def analyze(self, market_data: MarketData) -> AgentDecision:
        """
        Performs technical analysis on the given market data.

        For this initial implementation, it returns a mock decision.
        """
        # 1. Format the market data into a user prompt for the LLM.
        # For now, we'll use a placeholder prompt.
        user_prompt = (
            f"Analyze the following market data for {market_data.symbol}:\n"
            f"Price: {market_data.price}\n"
            f"Volume: {market_data.volume}\n"
            f"OHLC: {market_data.ohlc}\n"
            f"Indicators: {market_data.technical_indicators}\n\n"
            "Provide your signal, confidence, and reasoning."
        )

        # 2. Make the LLM call.
        llm_response = await self.make_llm_call(user_prompt)

        # 3. Parse the LLM response to create an AgentDecision.
        # For this placeholder, we will generate a mock decision.
        # TODO: Implement actual parsing of the llm_response.
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal="HOLD",
            confidence=0.5,
            reasoning="This is a mock analysis based on placeholder logic.",
            supporting_data={
                "llm_response": llm_response,
                "market_data_used": market_data.__dict__
            }
        )
