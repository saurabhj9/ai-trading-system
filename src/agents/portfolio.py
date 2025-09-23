"""
Implements the Portfolio Management Agent.

This agent synthesizes analyses from all other agents, considers the overall
portfolio state, and makes the final call on whether to execute a trade.
"""
from typing import Dict, Any

from .base import BaseAgent
from .data_structures import AgentDecision, MarketData


class PortfolioManagementAgent(BaseAgent):
    """
    An agent specialized in portfolio management and final trade decisions.
    """

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the portfolio management LLM.
        """
        return (
            "You are a specialized AI assistant for portfolio management. "
            "Your goal is to synthesize inputs from technical, sentiment, and risk "
            "analyses, consider the current portfolio state, and make a final "
            "trading decision (BUY, SELL, or HOLD). Provide confidence and reasoning "
            "based on all available data."
        )

    async def analyze(self, market_data: MarketData, agent_decisions: Dict[str, AgentDecision], portfolio_state: Dict[str, Any]) -> AgentDecision:
        """
        Makes the final portfolio management decision.

        Args:
            market_data: Current market data.
            agent_decisions: Decisions from all other agents.
            portfolio_state: Current portfolio state.

        Returns:
            The final trade decision.
        """
        # Format the data
        decisions_summary = {name: dec.signal for name, dec in agent_decisions.items()}
        user_prompt = (
            f"Synthesize the following for {market_data.symbol}:\n"
            f"Market Data: Price {market_data.price}\n"
            f"Agent Decisions: {decisions_summary}\n"
            f"Portfolio State: {portfolio_state}\n\n"
            "Make a final decision: BUY, SELL, or HOLD."
        )

        # LLM call
        llm_response = await self.make_llm_call(user_prompt)

        # Mock decision
        # TODO: Parse llm_response
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal="BUY",  # Mock
            confidence=0.8,
            reasoning="Mock portfolio decision: Synthesized inputs suggest buying.",
            supporting_data={
                "llm_response": llm_response,
                "agent_decisions": agent_decisions,
                "portfolio_state": portfolio_state
            }
        )
