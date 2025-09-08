"""
Integration tests for the Orchestrator.
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.agents.data_structures import AgentConfig, MarketData
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.orchestrator import Orchestrator, AgentState


@pytest.fixture
def technical_agent():
    """Provides a TechnicalAnalysisAgent with mock dependencies."""
    config = AgentConfig(name="TestTechnicalAgent")
    # For this integration test, we don't need real clients/buses
    return TechnicalAnalysisAgent(
        config=config,
        llm_client=MagicMock(),
        message_bus=MagicMock(),
        state_manager=MagicMock(),
    )


@pytest.fixture
def orchestrator(technical_agent):
    """Provides an Orchestrator instance."""
    return Orchestrator(technical_agent=technical_agent)


@pytest.fixture
def sample_market_data():
    """Provides a sample MarketData object."""
    return MarketData(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        timestamp=datetime.now(),
        ohlc={"open": 149.0, "high": 151.0, "low": 148.0, "close": 150.5},
        technical_indicators={"RSI": 55.0, "MACD": 0.5},
    )


@pytest.mark.asyncio
async def test_orchestrator_run(orchestrator, sample_market_data):
    """
    Tests the full run of the orchestrator with a technical analysis agent.
    """
    # Act
    final_state = await orchestrator.run(sample_market_data)

    # Assert
    assert isinstance(final_state, dict)
    assert "decision" in final_state
    decision = final_state["decision"]
    assert decision.agent_name == "TestTechnicalAgent"
    assert decision.symbol == "AAPL"
    assert decision.signal == "HOLD"  # Based on the mock implementation
