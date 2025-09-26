"""
Implements the Portfolio Management Agent.

This agent synthesizes analyses from all other agents, considers the overall
portfolio state, and makes the final call on whether to execute a trade.
"""
import json
from typing import Any, Dict

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
            "You are a specialized AI assistant for portfolio management. Your goal is to "
            "synthesize inputs from technical, sentiment, and risk analysis agents. "
            "Consider the current portfolio state and make a final trading decision "
            "(BUY, SELL, or HOLD). Provide a confidence score (0.0 to 1.0) and "
            "data-driven reasoning. Your final output must be a single JSON object with "
            "three keys: 'signal', 'confidence', and 'reasoning'."
        )

    async def analyze(
        self,
        market_data: MarketData,
        agent_decisions: Dict[str, AgentDecision],
        portfolio_state: Dict[str, Any],
    ) -> AgentDecision:
        """
        Makes the final portfolio management decision by synthesizing agent inputs.
        """
        # Format the agent decisions and portfolio state into a user prompt.
        decisions_summary = {
            name: {"signal": dec.signal, "confidence": dec.confidence}
            for name, dec in agent_decisions.items()
        }
        user_prompt = (
            f"Synthesize the following analyses for {market_data.symbol} to make a final trade decision:\n"
            f"- Market Data: Current Price=${market_data.price}\n"
            f"- Agent Decisions: {json.dumps(decisions_summary, indent=2)}\n"
            f"- Current Portfolio: {portfolio_state}\n\n"
            "Based on all available data, provide your final decision (BUY, SELL, or HOLD), "
            "confidence, and reasoning as a single JSON object."
        )

        # Make the LLM call.
        llm_response = await self.make_llm_call(user_prompt)

        # Parse the LLM's JSON response to create an AgentDecision.
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
                "agent_decisions": {
                    k: v.model_dump() for k, v in agent_decisions.items()
                },
                "portfolio_state": portfolio_state,
            },
        )
