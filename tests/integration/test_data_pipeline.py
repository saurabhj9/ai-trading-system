"""
Integration test for the full data pipeline and orchestration flow.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.agents.data_structures import AgentConfig, AgentDecision
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.orchestrator import Orchestrator, AgentState
from src.data.cache import CacheManager
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider


@pytest.mark.asyncio
async def test_end_to_end_data_flow_with_caching():
    """
    Tests the full data pipeline from the orchestrator down to the data provider,
    including caching functionality.
    """
    # 1. Instantiate all components
    provider = YFinanceProvider()
    cache = CacheManager()
    pipeline = DataPipeline(provider=provider, cache=cache, cache_ttl_seconds=60)

    # Create mock dependencies for the agent
    agent_config = AgentConfig(name="TestIntegrationAgent")
    agent = TechnicalAnalysisAgent(
        config=agent_config,
        llm_client=MagicMock(),
        message_bus=MagicMock(),
        state_manager=MagicMock()
    )

    orchestrator = Orchestrator(data_pipeline=pipeline, technical_agent=agent)

    # 2. Define parameters for the test run
    symbol = "SPY"  # Use a common ETF to ensure data availability
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # 3. First run (should be a cache miss)
    print("\n--- First run (expecting cache miss) ---")
    initial_state = await orchestrator.run(symbol=symbol, start_date=start_date, end_date=end_date)

    # 4. Assertions for the first run
    assert isinstance(initial_state, AgentState)
    assert initial_state.get("error") is None or initial_state.get("error") == ""
    assert "decision" in initial_state
    assert isinstance(initial_state["decision"], AgentDecision)

    decision = initial_state["decision"]
    assert decision.agent_name == "TestIntegrationAgent"
    assert decision.symbol == symbol
    assert decision.signal is not None

    assert "market_data" in initial_state
    market_data = initial_state["market_data"]
    assert market_data.symbol == symbol
    assert "RSI" in market_data.technical_indicators
    assert "MACD" in market_data.technical_indicators

    # 5. Second run (should be a cache hit)
    print("\n--- Second run (expecting cache hit) ---")
    cached_state = await orchestrator.run(symbol=symbol, start_date=start_date, end_date=end_date)

    # 6. Assertions for the cached run
    assert isinstance(cached_state, AgentState)
    # The result from the cache should be identical to the first run
    assert cached_state["market_data"].timestamp == initial_state["market_data"].timestamp
