"""
Orchestrates the workflow of the trading agents using LangGraph.
"""
from datetime import datetime
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.data_structures import AgentDecision, MarketData
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.config.settings import settings
from src.data.pipeline import DataPipeline


class AgentState(dict):
    """
    Represents the state of the agentic graph. Inherits from dict.
    """
    market_data: MarketData
    decisions: Dict[str, AgentDecision]
    portfolio_state: Dict[str, Any]
    final_decision: AgentDecision
    error: str = ""


class Orchestrator:
    """
    Manages the agentic workflow, starting from data fetching.
    """

    def __init__(
        self,
        data_pipeline: DataPipeline,
        technical_agent: TechnicalAnalysisAgent,
        sentiment_agent: SentimentAnalysisAgent,
        risk_agent: RiskManagementAgent,
        portfolio_agent: PortfolioManagementAgent,
        state_manager: Any,  # For portfolio state
    ):
        self.data_pipeline = data_pipeline
        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.risk_agent = risk_agent
        self.portfolio_agent = portfolio_agent
        self.state_manager = state_manager
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("technical_analysis", self.run_technical_analysis)
        workflow.add_node("sentiment_analysis", self.run_sentiment_analysis)
        workflow.add_node("risk_management", self.run_risk_management)
        workflow.add_node("portfolio_management", self.run_portfolio_management)
        # Run technical and sentiment in parallel since they are independent
        workflow.set_entry_point(["technical_analysis", "sentiment_analysis"])
        workflow.add_edge("technical_analysis", "risk_management")
        workflow.add_edge("sentiment_analysis", "risk_management")
        workflow.add_edge("risk_management", "portfolio_management")
        workflow.add_edge("portfolio_management", END)
        return workflow.compile()

    async def run_technical_analysis(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found"}

        decision = await self.technical_agent.analyze(market_data)
        decisions = state.get("decisions", {})
        decisions["technical"] = decision
        return {"decisions": decisions}

    async def run_sentiment_analysis(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found"}

        decision = await self.sentiment_agent.analyze(market_data)
        decisions = state.get("decisions", {})
        decisions["sentiment"] = decision
        return {"decisions": decisions}

    async def run_risk_management(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        decisions = state.get("decisions", {})
        portfolio_state = state.get("portfolio_state", {})
        if not market_data:
            return {"error": "Market data not found"}

        decision = await self.risk_agent.analyze(
            market_data, decisions, portfolio_state
        )
        decisions["risk"] = decision
        return {"decisions": decisions}

    async def run_portfolio_management(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        decisions = state.get("decisions", {})
        portfolio_state = state.get("portfolio_state", {})
        if not market_data:
            return {"error": "Market data not found"}

        final_decision = await self.portfolio_agent.analyze(
            market_data, decisions, portfolio_state
        )
        return {"final_decision": final_decision}

    async def run(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> AgentState:
        market_data = await self.data_pipeline.fetch_and_process_data(
            symbol, start_date, end_date
        )
        if not market_data:
            return AgentState(error=f"Failed to fetch data for {symbol}")

        portfolio_state = self.state_manager.get_portfolio_state() or {
            "cash": settings.portfolio.STARTING_CASH,
            "positions": {},
        }
        initial_state = AgentState(
            market_data=market_data,
            decisions={},
            portfolio_state=portfolio_state
        )

        final_state_dict = await self.workflow.ainvoke(initial_state)
        return AgentState(final_state_dict)
