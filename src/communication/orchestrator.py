"""
Orchestrates the workflow of the trading agents using LangGraph.
"""
from datetime import datetime
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.data_structures import AgentDecision, MarketData
from src.agents.technical import TechnicalAnalysisAgent
from src.data.pipeline import DataPipeline


class AgentState(dict):
    """
    Represents the state of the agentic graph. Inherits from dict.
    """
    market_data: MarketData
    decision: AgentDecision
    error: str = ""


class Orchestrator:
    """
    Manages the agentic workflow, starting from data fetching.
    """

    def __init__(self, data_pipeline: DataPipeline, technical_agent: TechnicalAnalysisAgent):
        self.data_pipeline = data_pipeline
        self.technical_agent = technical_agent
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("technical_analysis", self.run_technical_analysis)
        workflow.set_entry_point("technical_analysis")
        workflow.add_edge("technical_analysis", END)
        return workflow.compile()

    async def run_technical_analysis(self, state: AgentState) -> dict:
        """
        Runs the technical analysis agent on the market data in the state.
        """
        market_data = state.get("market_data")
        if not market_data:
            error_message = "Market data not found in state for technical analysis."
            print(f"Orchestrator Error: {error_message}")
            return {"error": error_message}

        decision = await self.technical_agent.analyze(market_data)
        return {"decision": decision}

    async def run(self, symbol: str, start_date: datetime, end_date: datetime) -> AgentState:
        """
        Executes the full workflow from data fetching to agent analysis.
        """
        market_data = await self.data_pipeline.fetch_and_process_data(symbol, start_date, end_date)
        if not market_data:
            error_message = f"Failed to fetch or process data for {symbol}"
            print(f"Orchestrator Error: {error_message}")
            return AgentState(error=error_message)

        initial_state = AgentState(market_data=market_data)

        # LangGraph returns a dict, so we cast it back to AgentState
        final_state_dict = await self.workflow.ainvoke(initial_state)
        return AgentState(final_state_dict)
