"""
Integration tests for the Orchestrator.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.agents.data_structures import AgentConfig
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.orchestrator import Orchestrator
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider


@pytest.fixture
def technical_agent():
    """Provides a TechnicalAnalysisAgent with mock dependencies."""
    config = AgentConfig(name="TestTechnicalAgent")
    return TechnicalAnalysisAgent(
        config=config,
        llm_client=MagicMock(),
        message_bus=MagicMock(),
        state_manager=MagicMock(),
    )

@pytest.fixture
def data_pipeline():
    """Provides a DataPipeline instance with a real YFinanceProvider."""
    provider = YFinanceProvider()
    # No cache for this test to ensure we hit the provider
    return DataPipeline(provider=provider, cache=None)

@pytest.fixture
def orchestrator(data_pipeline, technical_agent):
    """Provides an Orchestrator instance with a real pipeline and a mock agent."""
    return Orchestrator(data_pipeline=data_pipeline, technical_agent=technical_agent)


@pytest.mark.asyncio
async def test_orchestrator_run(orchestrator):
    """
    Tests the full run of the orchestrator, ensuring it fetches data
    and executes the agent workflow.
    """
    # Arrange
    symbol = "MSFT"  # Using a real, reliable ticker
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Act
    final_state = await orchestrator.run(symbol, start_date, end_date)

    # Assert
    assert final_state.get("error") is None or final_state.get("error") == ""
    assert "decision" in final_state
    decision = final_state["decision"]
    assert decision.agent_name == "TestTechnicalAgent"
    assert decision.symbol == symbol
    assert decision.signal == "HOLD"  # The placeholder agent returns HOLD
