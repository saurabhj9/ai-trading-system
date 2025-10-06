"""
Implements the Risk Management Agent.

This agent assesses risk for proposed trades and the overall portfolio,
calculating metrics like Value at Risk (VaR) and position sizing.
"""
import json
from typing import Any, Dict

from .base import BaseAgent
from .data_structures import AgentDecision, MarketData


class RiskManagementAgent(BaseAgent):
    """
    An agent specialized in risk assessment and management.
    """

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the risk management LLM.
        """
        return (
            "You are a specialized AI assistant for financial risk management. "
            "Your goal is to assess the risk of a proposed trade. Analyze the "
            "provided market data, portfolio state, and proposed agent decisions. "
            "Determine a risk assessment signal (APPROVE or REJECT), a confidence score (0.0 to 1.0), "
            "and provide data-driven reasoning. Your final output must be a single JSON object with three keys: "
            "'signal', 'confidence', and 'reasoning'."
        )

    async def get_user_prompt(self, market_data: MarketData, **kwargs) -> str:
        """
        Generates the user prompt for risk assessment.
        """
        proposed_decisions = kwargs.get("proposed_decisions", {})
        portfolio_state = kwargs.get("portfolio_state", {})

        return (
            f"Assess the risk for a trade in {market_data.symbol} given the following:\n"
            f"- Market Data: Price=${market_data.price}, Volume={market_data.volume}\n"
            f"- Current Portfolio: {portfolio_state}\n"
            f"- Proposed Agent Decisions: { {k: v.signal for k, v in proposed_decisions.items()} }\n\n"
            "Based on this data, provide your risk assessment (APPROVE or REJECT), "
            "confidence, and reasoning as a single JSON object."
        )

    def create_decision(self, market_data: MarketData, response: str, **kwargs) -> AgentDecision:
        """
        Creates an AgentDecision from the LLM response for risk assessment.
        """
        proposed_decisions = kwargs.get("proposed_decisions", {})
        portfolio_state = kwargs.get("portfolio_state", {})

        try:
            decision_json = json.loads(response)
            signal = decision_json.get("signal", "REJECT")
            confidence = float(decision_json.get("confidence", 0.0))
            reasoning = decision_json.get("reasoning", "No reasoning provided.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, create a default decision with an error message.
            signal = "ERROR"
            confidence = 0.0
            reasoning = f"Failed to parse LLM response: {e}. Raw response: {response}"

        # Perform quantitative risk calculations
        try:
            # Calculate portfolio equity from cash and positions
            cash = portfolio_state.get("cash", 0.0)
            positions = portfolio_state.get("positions", {})
            positions_value = sum(
                pos.get("quantity", 0) * pos.get("current_price", 0)
                for pos in positions.values()
            )
            portfolio_equity = cash + positions_value

            position_size = self._calculate_position_size(
                portfolio_equity,
                market_data.price
            )
            # Add quantitative metrics to the reasoning
            quantitative_reasoning = (
                f" Quantitative Risk Check: Position size calculated at {position_size:.4f} shares. "
                f"This is based on a 1% portfolio risk tolerance and assumes a 5% stop-loss."
            )
            reasoning += quantitative_reasoning
            supporting_data = {
                "llm_response": response,
                "proposed_decisions": {k: v.__dict__ for k, v in proposed_decisions.items()},
                "portfolio_state": portfolio_state,
                "calculated_position_size": position_size,
            }
        except ValueError as e:
            signal = "REJECT"
            reasoning = f"Risk calculation failed: {e}"
            supporting_data = {
                "llm_response": response,
                "error": str(e)
            }

        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data=supporting_data,
        )

    async def analyze(
        self,
        market_data: MarketData,
        **kwargs
    ) -> AgentDecision:
        """
        Performs risk assessment on proposed decisions and the portfolio.

        TODO: Implement actual risk calculations (e.g., VaR) instead of relying solely on the LLM.
        """
        user_prompt = await self.get_user_prompt(market_data, **kwargs)
        llm_response = await self.make_llm_call(user_prompt)
        return self.create_decision(market_data, llm_response, **kwargs)

    def _calculate_position_size(
        self, portfolio_equity: float, current_price: float, risk_per_trade: float = 0.01, stop_loss_pct: float = 0.05
    ) -> float:
        """
        Calculates the position size based on a fixed fractional risk strategy.

        Args:
            portfolio_equity: The total equity of the portfolio.
            current_price: The current market price of the asset.
            risk_per_trade: The fraction of the portfolio to risk on a single trade.
            stop_loss_pct: The percentage of the price to set the stop-loss.

        Returns:
            The number of shares to buy.

        Raises:
            ValueError: If inputs are invalid for calculation.
        """
        if portfolio_equity <= 0 or current_price <= 0:
            raise ValueError("Portfolio equity and current price must be positive.")

        risk_amount = portfolio_equity * risk_per_trade
        stop_loss_price = current_price * (1 - stop_loss_pct)
        risk_per_share = current_price - stop_loss_price

        if risk_per_share <= 0:
            raise ValueError("Stop-loss percentage results in zero or negative risk per share.")

        return risk_amount / risk_per_share
