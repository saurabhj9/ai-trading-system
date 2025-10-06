"""
Unit tests for the TechnicalAnalysisAgent.
"""
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

import pytest

from src.agents.data_structures import AgentConfig, MarketData, AgentDecision
from src.agents.technical import TechnicalAnalysisAgent


@pytest.fixture
def agent_config():
    """Provides a default AgentConfig for tests."""
    return AgentConfig(name="TestTechnicalAgent")


@pytest.fixture
def mock_llm_client():
    """Provides a mock LLM client."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value='{"signal": "BUY", "confidence": 0.75, "reasoning": "Test reasoning"}')
    return mock


@pytest.fixture
def mock_message_bus():
    """Provides a mock message bus."""
    return MagicMock()


@pytest.fixture
def mock_state_manager():
    """Provides a mock state manager."""
    return MagicMock()


@pytest.fixture
def technical_agent(agent_config, mock_llm_client, mock_message_bus, mock_state_manager):
    """Provides an instance of the TechnicalAnalysisAgent with mock dependencies."""
    return TechnicalAnalysisAgent(
        config=agent_config,
        llm_client=mock_llm_client,
        message_bus=mock_message_bus,
        state_manager=mock_state_manager,
    )


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analyze_returns_agent_decision(technical_agent, sample_market_data):
    """
    Tests that the analyze method returns a valid AgentDecision object.
    """
    # Act
    decision = await technical_agent.analyze(sample_market_data)

    # Assert
    assert isinstance(decision, AgentDecision)
    assert decision.agent_name == technical_agent.config.name
    assert decision.symbol == sample_market_data.symbol
    assert decision.signal in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.reasoning, str)


@pytest.mark.unit
def test_get_system_prompt_returns_string(technical_agent):
    """
    Tests that get_system_prompt returns a non-empty string.
    """
    # Act
    prompt = technical_agent.get_system_prompt()

    # Assert
    assert isinstance(prompt, str)
    assert len(prompt) > 0
