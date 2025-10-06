"""
Implements the Portfolio Management Agent.

This agent synthesizes analyses from all other agents, considers the overall
portfolio state, and makes the final call on whether to execute a trade.
"""
import json
from typing import Any, Dict

from .base import BaseAgent, clean_json_response
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
            "You are a specialized AI assistant for portfolio management. Your goal is to "
            "synthesize inputs from technical, sentiment, and risk analysis agents. "
            "Consider the current portfolio state and make a final trading decision "
            "(BUY, SELL, or HOLD). Provide a confidence score (0.0 to 1.0) and "
            "data-driven reasoning. Your final output must be a single JSON object with "
            "three keys: 'signal', 'confidence', and 'reasoning'."
        )

    async def get_user_prompt(self, market_data: MarketData, **kwargs) -> str:
        """
        Generates the user prompt for portfolio management.
        """
        agent_decisions = kwargs.get("agent_decisions", {})
        portfolio_state = kwargs.get("portfolio_state", {})

        # Format the agent decisions and portfolio state into a user prompt.
        decisions_summary = {
            name: {"signal": dec.signal, "confidence": dec.confidence}
            for name, dec in agent_decisions.items()
        }
        return (
            f"Synthesize the following analyses for {market_data.symbol} to make a final trade decision:\n"
            f"- Market Data: Current Price=${market_data.price}\n"
            f"- Agent Decisions: {json.dumps(decisions_summary, indent=2)}\n"
            f"- Current Portfolio: {portfolio_state}\n\n"
            "Based on all available data, provide your final decision (BUY, SELL, or HOLD), "
            "confidence, and reasoning as a single JSON object."
        )

    def create_decision(self, market_data: MarketData, response: str, **kwargs) -> AgentDecision:
        """
        Creates an AgentDecision from the LLM response for portfolio management.
        """
        agent_decisions = kwargs.get("agent_decisions", {})
        portfolio_state = kwargs.get("portfolio_state", {})

        # Parse the LLM's JSON response to create an AgentDecision.
        try:
            # Clean the response to handle control characters
            cleaned_response = clean_json_response(response)
            decision_json = json.loads(cleaned_response)
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
                "agent_decisions": {
                    k: v.__dict__ for k, v in agent_decisions.items()
                },
                "portfolio_state": portfolio_state,
            },
        )

    async def analyze(
        self,
        market_data: MarketData,
        **kwargs
    ) -> AgentDecision:
        """
        Makes the final portfolio management decision by synthesizing agent inputs.
        """
        user_prompt = await self.get_user_prompt(market_data, **kwargs)
        llm_response = await self.make_llm_call(user_prompt)
        return self.create_decision(market_data, llm_response, **kwargs)
