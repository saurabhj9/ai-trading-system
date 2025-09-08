"""
Orchestrates the workflow of the trading agents using LangGraph.

This module defines the agentic graph that controls the flow of data and
execution between the different specialized agents.
"""
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.data_structures import AgentDecision, MarketData
from src.agents.technical import TechnicalAnalysisAgent


class AgentState(Dict[str, Any]):
    """
    Represents the state of the agentic graph.

    Attributes:
        market_data: The input market data for the analysis.
        decision: The final decision made by the agent workflow.
        error: A potential error message if the workflow fails.
    """
    market_data: MarketData
    decision: AgentDecision
    error: str = ""


class Orchestrator:
    """
    Manages the agentic workflow using a state graph.
    """

    def __init__(self, technical_agent: TechnicalAnalysisAgent):
        """
        Initializes the Orchestrator.

        Args:
            technical_agent: An instance of the TechnicalAnalysisAgent.
        """
        self.technical_agent = technical_agent
        self.workflow = self._build_graph()

    def _build_graph(self):
        """
        Builds the LangGraph workflow.
        """
        workflow = StateGraph(AgentState)

        # Add the technical analysis node
        workflow.add_node("technical_analysis", self.run_technical_analysis)

        # Set the entry point
        workflow.set_entry_point("technical_analysis")

        # All paths lead to the end for now
        workflow.add_edge("technical_analysis", END)

        return workflow.compile()

    async def run_technical_analysis(self, state: AgentState) -> AgentState:
        """
        Runs the technical analysis agent.

        Args:
            state: The current state of the graph.

        Returns:
            The updated state with the agent's decision.
        """
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found in state."}

        decision = await self.technical_agent.analyze(market_data)
        return {"decision": decision}

    async def run(self, market_data: MarketData) -> AgentState:
        """
        Executes the agentic workflow.

        Args:
            market_data: The initial market data to start the workflow.

        Returns:
            The final state of the graph after execution.
        """
        initial_state = AgentState(market_data=market_data)
        return await self.workflow.ainvoke(initial_state)
