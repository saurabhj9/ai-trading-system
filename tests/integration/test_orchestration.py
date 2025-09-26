"""
Integration tests for the Orchestrator.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.agents.data_structures import AgentConfig, AgentDecision
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.orchestrator import Orchestrator
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider


@pytest.fixture
def mock_decision():
    """Provides a mock AgentDecision."""
    return AgentDecision(
        agent_name="mock",
        symbol="MSFT",
        signal="HOLD",
        confidence=0.5,
        reasoning="Mocked decision.",
    )


@pytest.fixture
def technical_agent(mocker, mock_decision):
    """Provides a mocked TechnicalAnalysisAgent."""
    mock = mocker.MagicMock(spec=TechnicalAnalysisAgent)
    mock_decision.agent_name = "technical"
    mock.analyze.return_value = mock_decision
    return mock


@pytest.fixture
def sentiment_agent(mocker, mock_decision):
    """Provides a mocked SentimentAnalysisAgent."""
    mock = mocker.MagicMock(spec=SentimentAnalysisAgent)
    mock_decision.agent_name = "sentiment"
    mock.analyze.return_value = mock_decision
    return mock


@pytest.fixture
def risk_agent(mocker, mock_decision):
    """Provides a mocked RiskManagementAgent."""
    mock = mocker.MagicMock(spec=RiskManagementAgent)
    mock_decision.agent_name = "risk"
    mock.analyze.return_value = mock_decision
    return mock


@pytest.fixture
def portfolio_agent(mocker, mock_decision):
    """Provides a mocked PortfolioManagementAgent."""
    mock = mocker.MagicMock(spec=PortfolioManagementAgent)
    mock_decision.agent_name = "portfolio"
    mock.analyze.return_value = mock_decision
    return mock


@pytest.fixture
def data_pipeline():
    """Provides a DataPipeline instance with a real YFinanceProvider."""
    provider = YFinanceProvider()
    return DataPipeline(provider=provider, cache=None)


@pytest.fixture
def orchestrator(
    data_pipeline,
    technical_agent,
    sentiment_agent,
    risk_agent,
    portfolio_agent,
    mocker,
):
    """Provides an Orchestrator instance with a real pipeline and mock agents."""
    state_manager = mocker.MagicMock()
    state_manager.get_portfolio_state.return_value = None

    return Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=technical_agent,
        sentiment_agent=sentiment_agent,
        risk_agent=risk_agent,
        portfolio_agent=portfolio_agent,
        state_manager=state_manager,
    )


@pytest.mark.asyncio
async def test_orchestrator_run(orchestrator):
    """
    Tests the full run of the orchestrator, ensuring it fetches data
    and executes the agent workflow, producing a final decision.
    """
    # Arrange
    symbol = "MSFT"  # Using a real, reliable ticker
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Act
    final_state = await orchestrator.run(symbol, start_date, end_date)

    # Assert
    assert not final_state.get("error")
    assert "final_decision" in final_state
    decision = final_state["final_decision"]
    assert decision.agent_name == "portfolio"
    assert decision.symbol == symbol
