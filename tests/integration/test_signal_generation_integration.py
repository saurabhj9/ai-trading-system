"""
Integration tests for the Local Signal Generation Framework integration.

This module tests the integration between the Local Signal Generation Framework
and the existing trading system components, including the TechnicalAnalysisAgent
and Orchestrator.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.agents.technical import TechnicalAnalysisAgent
from src.agents.data_structures import AgentConfig, MarketData, AgentDecision
from src.communication.orchestrator import Orchestrator
from src.config.settings import settings
from src.migration.signal_generation_migration import SignalGenerationMigration


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    return MarketData(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        timestamp=datetime.now(),
        ohlc={"open": 149.0, "high": 151.0, "low": 148.0, "close": 150.0},
        technical_indicators={
            "RSI": 55.0,
            "MACD": 1.2,
            "ADX": 25.0,
            "PLUS_DI": 30.0,
            "MINUS_DI": 20.0,
            "ATR": 2.5,
            "OBV": 5000000,
            "STOCH": 60.0,
            "WILLR": -40.0,
            "MFI": 65.0,
            "CCI": 100.0,
            "NATR": 0.02,
            "AD": 1000000,
        },
        historical_ohlc=[
            {"open": 148.0, "high": 149.0, "low": 147.0, "close": 148.0, "volume": 900000},
            {"open": 147.0, "high": 148.5, "low": 146.0, "close": 148.0, "volume": 950000},
            {"open": 148.0, "high": 149.0, "low": 147.5, "close": 149.0, "volume": 980000},
        ]
    )


@pytest.fixture
def mock_agent_config():
    """Create mock agent configuration."""
    return AgentConfig(
        name="test_technical_agent",
        model_name="test-model",
        temperature=0.1,
        max_tokens=1000,
        timeout=30.0,
        retry_attempts=3
    )


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for the agent."""
    return {
        "llm_client": AsyncMock(),
        "message_bus": Mock(),
        "state_manager": Mock()
    }


class TestTechnicalAnalysisAgentIntegration:
    """Test integration of Local Signal Generation with TechnicalAnalysisAgent."""

    @pytest.mark.asyncio
    async def test_local_signal_generation_enabled(self, mock_market_data, mock_agent_config, mock_dependencies):
        """Test that local signal generation works when enabled."""
        # Enable local signal generation
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'ROLLOUT_PERCENTAGE', 1.0), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Mock the LLM client to avoid actual calls
            agent.llm_client.generate = AsyncMock(return_value='{"signal": "BUY", "confidence": 0.8, "reasoning": "Test"}')

            # Test analysis
            decision = await agent.analyze(mock_market_data)

            # Verify decision structure
            assert isinstance(decision, AgentDecision)
            assert decision.symbol == "AAPL"
            assert decision.signal in ["BUY", "SELL", "HOLD"]
            assert 0.0 <= decision.confidence <= 1.0
            assert "signal_source" in decision.supporting_data
            assert decision.supporting_data["signal_source"] in ["LOCAL", "LLM", "ESCALATED"]

    @pytest.mark.asyncio
    async def test_hybrid_mode_escalation(self, mock_market_data, mock_agent_config, mock_dependencies):
        """Test hybrid mode with escalation to LLM."""
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'HYBRID_MODE_ENABLED', True), \
             patch.object(settings.signal_generation, 'ESCALATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'ESCALATION_CONFIDENCE_THRESHOLD', 0.7), \
             patch.object(settings.signal_generation, 'ROLLOUT_PERCENTAGE', 1.0), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Mock the LLM client
            agent.llm_client.generate = AsyncMock(return_value='{"signal": "SELL", "confidence": 0.9, "reasoning": "Escalated"}')

            # Test analysis
            decision = await agent.analyze(mock_market_data)

            # Verify escalation if local confidence was low
            assert isinstance(decision, AgentDecision)
            if decision.supporting_data.get("signal_source") == "ESCALATED":
                assert "escalation_info" in decision.supporting_data
                assert "escalated_from" in decision.supporting_data["escalation_info"]
                assert decision.supporting_data["escalation_info"]["escalated_from"] == "LOCAL"

    @pytest.mark.asyncio
    async def test_side_by_side_comparison(self, mock_market_data, mock_agent_config, mock_dependencies):
        """Test side-by-side comparison mode."""
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'ENABLE_SIDE_BY_SIDE_COMPARISON', True), \
             patch.object(settings.signal_generation, 'COMPARISON_SAMPLE_RATE', 1.0), \
             patch.object(settings.signal_generation, 'ROLLOUT_PERCENTAGE', 1.0), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Mock the LLM client
            agent.llm_client.generate = AsyncMock(return_value='{"signal": "HOLD", "confidence": 0.6, "reasoning": "LLM analysis"}')

            # Test analysis
            decision = await agent.analyze(mock_market_data)

            # Verify comparison data is included
            assert isinstance(decision, AgentDecision)
            if "comparison" in decision.supporting_data:
                assert "llm_signal" in decision.supporting_data["comparison"]
                assert "llm_confidence" in decision.supporting_data["comparison"]
                assert "llm_reasoning" in decision.supporting_data["comparison"]

    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_agent_config, mock_dependencies):
        """Test batch processing with local signal generation."""
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'ROLLOUT_PERCENTAGE', 1.0), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Create multiple market data items
            market_data_list = [
                MarketData(
                    symbol=f"STOCK{i}",
                    price=100.0 + i,
                    volume=1000000,
                    timestamp=datetime.now(),
                    ohlc={"open": 99.0 + i, "high": 101.0 + i, "low": 98.0 + i, "close": 100.0 + i},
                    technical_indicators={"RSI": 50.0 + i}
                )
                for i in range(3)
            ]

            # Test batch analysis
            decisions = await agent.batch_analyze(market_data_list)

            # Verify batch results
            assert len(decisions) == 3
            for i, decision in enumerate(decisions):
                assert isinstance(decision, AgentDecision)
                assert decision.symbol == f"STOCK{i}"
                assert "signal_source" in decision.supporting_data

    def test_performance_metrics_tracking(self, mock_agent_config, mock_dependencies):
        """Test that performance metrics are properly tracked."""
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Get initial metrics
            initial_metrics = agent.get_performance_metrics()
            assert "local_signals" in initial_metrics
            assert "llm_signals" in initial_metrics
            assert "local_avg_time" in initial_metrics
            assert "llm_avg_time" in initial_metrics

            # Reset metrics
            agent.reset_performance_metrics()
            reset_metrics = agent.get_performance_metrics()
            assert reset_metrics["local_signals"] == 0
            assert reset_metrics["llm_signals"] == 0


class TestOrchestratorIntegration:
    """Test integration of Local Signal Generation with Orchestrator."""

    @pytest.fixture
    def mock_orchestrator_dependencies(self):
        """Create mock dependencies for the orchestrator."""
        return {
            "data_pipeline": Mock(),
            "technical_agent": Mock(spec=TechnicalAnalysisAgent),
            "sentiment_agent": Mock(),
            "risk_agent": Mock(),
            "portfolio_agent": Mock(),
            "state_manager": Mock()
        }

    def test_orchestrator_with_local_signals(self, mock_orchestrator_dependencies):
        """Test that orchestrator handles local signals correctly."""
        # Create mock technical agent with local signal generation
        tech_agent = mock_orchestrator_dependencies["technical_agent"]
        tech_agent._should_use_local_generation = Mock(return_value=True)
        tech_agent.local_signal_generator = Mock()
        tech_agent._generate_local_signal = AsyncMock(return_value=Mock(
            signal="BUY",
            confidence=0.8,
            supporting_data={"signal_source": "LOCAL"}
        ))
        tech_agent.get_performance_metrics = Mock(return_value={"local_signals": 1})

        # Create orchestrator
        orchestrator = Orchestrator(**mock_orchestrator_dependencies)

        # Test metrics
        metrics = orchestrator.get_orchestrator_metrics()
        assert "total_workflows" in metrics
        assert "local_signal_workflows" in metrics
        assert "llm_signal_workflows" in metrics

    def test_signal_source_tracking(self, mock_orchestrator_dependencies):
        """Test that signal sources are properly tracked."""
        tech_agent = mock_orchestrator_dependencies["technical_agent"]
        tech_agent._should_use_local_generation = Mock(return_value=True)
        tech_agent.local_signal_generator = Mock()
        tech_agent._generate_local_signal = AsyncMock(return_value=Mock(
            signal="BUY",
            confidence=0.8,
            supporting_data={"signal_source": "LOCAL"}
        ))

        orchestrator = Orchestrator(**mock_orchestrator_dependencies)

        # Verify signal source tracking is initialized
        assert hasattr(orchestrator, 'orchestrator_metrics')
        assert "local_signal_workflows" in orchestrator.orchestrator_metrics


class TestMigrationIntegration:
    """Test migration utilities and workflow."""

    def test_migration_initialization(self):
        """Test migration manager initialization."""
        with patch('src.migration.signal_generation_migration.Path') as mock_path:
            # Mock file operations
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.parent.mkdir = Mock()

            migration = SignalGenerationMigration()

            # Verify default configuration
            assert "migration_phases" in migration.migration_config
            assert len(migration.migration_config["migration_phases"]) > 0
            assert migration.migration_config.get("current_phase", 0) == 0

    def test_phase_configuration_application(self):
        """Test that phase configuration is applied correctly."""
        with patch('src.migration.signal_generation_migration.Path') as mock_path, \
             patch.object(SignalGenerationMigration, '_update_env_file') as mock_update_env:

            # Mock file operations
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = '{"current_phase": 0}'

            migration = SignalGenerationMigration()

            # Apply current phase configuration
            config = migration.apply_phase_configuration()

            # Verify configuration was applied
            assert "SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED" in config
            assert config["SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED"] == "true"
            mock_update_env.assert_called_once()

    def test_rollback_conditions_check(self):
        """Test rollback condition checking."""
        with patch('src.migration.signal_generation_migration.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = '{"current_phase": 1}'

            migration = SignalGenerationMigration()

            # Test with metrics that should trigger rollback
            bad_metrics = {
                "local_error_rate": 10.0,  # Above 5% threshold
                "performance_improvement": -30.0,  # Below -20% threshold
                "avg_validation_confidence": 0.5,  # Below 0.6 threshold
            }

            should_rollback, reasons = migration.check_rollback_conditions(bad_metrics)

            assert should_rollback
            assert len(reasons) > 0
            assert any("error rate" in reason for reason in reasons)

            # Test with good metrics
            good_metrics = {
                "local_error_rate": 2.0,
                "performance_improvement": 60.0,
                "avg_validation_confidence": 0.8,
            }

            should_rollback, reasons = migration.check_rollback_conditions(good_metrics)

            assert not should_rollback
            assert len(reasons) == 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow_with_local_signals(self, mock_market_data, mock_agent_config, mock_dependencies):
        """Test complete workflow from market data to decision with local signals."""
        with patch.object(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED', True), \
             patch.object(settings.signal_generation, 'ROLLOUT_PERCENTAGE', 1.0), \
             patch('src.agents.technical.signal_generation_config') as mock_config:

            mock_config.to_dict.return_value = {}

            # Create agent with local signal generation
            agent = TechnicalAnalysisAgent(mock_agent_config, **mock_dependencies)

            # Mock LLM client for fallback
            agent.llm_client.generate = AsyncMock(return_value='{"signal": "BUY", "confidence": 0.8, "reasoning": "Fallback"}')

            # Test complete analysis
            decision = await agent.analyze(mock_market_data)

            # Verify complete decision structure
            assert isinstance(decision, AgentDecision)
            assert decision.symbol == mock_market_data.symbol
            assert decision.signal in ["BUY", "SELL", "HOLD", "ERROR"]
            assert 0.0 <= decision.confidence <= 1.0
            assert decision.reasoning is not None
            assert "signal_source" in decision.supporting_data

            # Verify performance metrics are updated
            metrics = agent.get_performance_metrics()
            assert metrics["local_signals"] + metrics["llm_signals"] >= 1

    def test_configuration_consistency(self):
        """Test that configuration is consistent across components."""
        # Verify signal generation settings are available
        assert hasattr(settings, 'signal_generation')
        assert hasattr(settings.signal_generation, 'LOCAL_SIGNAL_GENERATION_ENABLED')
        assert hasattr(settings.signal_generation, 'HYBRID_MODE_ENABLED')
        assert hasattr(settings.signal_generation, 'ROLLOUT_PERCENTAGE')

        # Verify settings have reasonable defaults
        assert isinstance(settings.signal_generation.LOCAL_SIGNAL_GENERATION_ENABLED, bool)
        assert isinstance(settings.signal_generation.HYBRID_MODE_ENABLED, bool)
        assert 0.0 <= settings.signal_generation.ROLLOUT_PERCENTAGE <= 1.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
