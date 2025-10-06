"""
Pytest configuration and shared fixtures for the AI Trading System test suite.

This module provides common fixtures and configuration for all test categories,
ensuring consistent test data and setup across the entire test suite.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.data_structures import (
    AgentConfig, MarketData, AgentDecision
)
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.risk import RiskManagementAgent
from src.agents.portfolio import PortfolioManagementAgent
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.cache import CacheManager
from src.llm.client import LLMClient
from src.communication.message_bus import MessageBus
from src.communication.state_manager import StateManager
from src.communication.orchestrator import Orchestrator
from src.signal_generation.signal_generator import LocalSignalGenerator
from src.config.settings import settings
from src.config.signal_generation import signal_generation_config


# ==============================
# Pytest Configuration
# ==============================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and profiling tests"
    )
    config.addinivalue_line(
        "markers", "validation: Validation and quality tests"
    )
    config.addinivalue_line(
        "markers", "comparison: Comparison tests between approaches"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )


# ==============================
# Market Data Fixtures
# ==============================

@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible tests

    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create a trending price series
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.normal(0, 2, 100)  # Random noise
    close_prices = base_price + trend + noise

    # Generate OHLC data
    high = close_prices + np.random.uniform(0, 2, 100)
    low = close_prices - np.random.uniform(0, 2, 100)
    open_prices = low + np.random.uniform(0, high - low)
    volume = np.random.uniform(1000000, 5000000, 100)

    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_market_data() -> MarketData:
    """Create sample market data for testing."""
    return MarketData(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        timestamp=datetime.now(),
        ohlc={
            "open": 149.0,
            "high": 151.0,
            "low": 148.0,
            "close": 150.0
        },
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


@pytest.fixture(params=["bullish", "bearish", "neutral"])
def market_data_scenarios(request) -> MarketData:
    """Create market data for different market scenarios."""
    scenario = request.param

    if scenario == "bullish":
        ohlc = {
            "open": 149.0,
            "high": 152.0,
            "low": 148.0,
            "close": 151.0
        }
        indicators = {
            "RSI": 65.0,
            "MACD": 1.5,
            "ADX": 30.0,
            "PLUS_DI": 35.0,
            "MINUS_DI": 15.0,
        }
    elif scenario == "bearish":
        ohlc = {
            "open": 151.0,
            "high": 152.0,
            "low": 147.0,
            "close": 148.0
        }
        indicators = {
            "RSI": 35.0,
            "MACD": -1.2,
            "ADX": 28.0,
            "PLUS_DI": 18.0,
            "MINUS_DI": 32.0,
        }
    else:  # neutral
        ohlc = {
            "open": 149.5,
            "high": 151.0,
            "low": 148.0,
            "close": 150.0
        }
        indicators = {
            "RSI": 50.0,
            "MACD": 0.1,
            "ADX": 20.0,
            "PLUS_DI": 22.0,
            "MINUS_DI": 23.0,
        }

    return MarketData(
        symbol="TEST",
        price=150.0,
        volume=1000000,
        timestamp=datetime.now(),
        ohlc=ohlc,
        technical_indicators=indicators
    )


# ==============================
# Agent Configuration Fixtures
# ==============================

@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a standard agent configuration for testing."""
    return AgentConfig(
        name="test_agent",
        model_name="test-model",
        temperature=0.1,
        max_tokens=1000,
        timeout=30.0,
        retry_attempts=3
    )


@pytest.fixture
def agent_configs() -> Dict[str, AgentConfig]:
    """Create configurations for all agent types."""
    return {
        "technical": AgentConfig(
            name="technical_agent",
            model_name="test-model",
            temperature=0.1,
            max_tokens=500
        ),
        "sentiment": AgentConfig(
            name="sentiment_agent",
            model_name="test-model",
            temperature=0.2,
            max_tokens=600
        ),
        "risk": AgentConfig(
            name="risk_agent",
            model_name="test-model",
            temperature=0.1,
            max_tokens=400
        ),
        "portfolio": AgentConfig(
            name="portfolio_agent",
            model_name="test-model",
            temperature=0.1,
            max_tokens=500
        )
    }


# ==============================
# Mock Dependencies Fixtures
# ==============================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = '{"signal": "BUY", "confidence": 0.8, "reasoning": "Test response"}'
    mock_client.last_usage = Mock()
    mock_client.last_usage.prompt_tokens = 100
    mock_client.last_usage.completion_tokens = 50
    mock_client.last_usage.total_tokens = 150
    return mock_client


@pytest.fixture
def mock_message_bus():
    """Create a mock message bus for testing."""
    return Mock()


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    mock_sm = Mock()
    mock_sm.get_portfolio_state.return_value = {
        "equity": 100000.0,
        "cash": 50000.0,
        "positions": {}
    }
    return mock_sm


@pytest.fixture
def mock_dependencies(mock_llm_client, mock_message_bus, mock_state_manager):
    """Bundle all mock dependencies for easy injection."""
    return {
        "llm_client": mock_llm_client,
        "message_bus": mock_message_bus,
        "state_manager": mock_state_manager
    }


# ==============================
# Agent Fixtures
# ==============================

@pytest.fixture
def technical_agent(agent_config, mock_dependencies):
    """Create a technical analysis agent for testing."""
    return TechnicalAnalysisAgent(
        config=agent_config,
        **mock_dependencies
    )


@pytest.fixture
def sentiment_agent(agent_config, mock_dependencies):
    """Create a sentiment analysis agent for testing."""
    return SentimentAnalysisAgent(
        config=agent_config,
        **mock_dependencies
    )


@pytest.fixture
def risk_agent(agent_config, mock_dependencies):
    """Create a risk management agent for testing."""
    return RiskManagementAgent(
        config=agent_config,
        **mock_dependencies
    )


@pytest.fixture
def portfolio_agent(agent_config, mock_dependencies):
    """Create a portfolio management agent for testing."""
    return PortfolioManagementAgent(
        config=agent_config,
        **mock_dependencies
    )


@pytest.fixture
def all_agents(agent_configs, mock_dependencies):
    """Create all agent types for testing."""
    return {
        "technical": TechnicalAnalysisAgent(
            config=agent_configs["technical"],
            **mock_dependencies
        ),
        "sentiment": SentimentAnalysisAgent(
            config=agent_configs["sentiment"],
            **mock_dependencies
        ),
        "risk": RiskManagementAgent(
            config=agent_configs["risk"],
            **mock_dependencies
        ),
        "portfolio": PortfolioManagementAgent(
            config=agent_configs["portfolio"],
            **mock_dependencies
        )
    }


# ==============================
# Data Pipeline Fixtures
# ==============================

@pytest.fixture
def data_provider():
    """Create a data provider for testing."""
    return YFinanceProvider()


@pytest.fixture
def cache_manager():
    """Create a cache manager for testing."""
    return CacheManager()


@pytest.fixture
def data_pipeline(data_provider, cache_manager):
    """Create a data pipeline for testing."""
    return DataPipeline(provider=data_provider, cache=cache_manager)


# ==============================
# Signal Generation Fixtures
# ==============================

@pytest.fixture
def signal_generator_config():
    """Create configuration for local signal generator."""
    return {
        "market_regime": {
            "adx_period": 14,
            "atr_period": 14,
            "hurst_exponent_lag": 50,
            "trend_strength_threshold": 25,
            "ranging_threshold": 20,
            "volatility_threshold_percent": 2.5,
            "confirmation_periods": 2,
        },
        "indicator_scorer": {
            "indicator_weights": {
                "RSI": 0.8,
                "MACD": 0.9,
                "ADX": 0.7,
                "PLUS_DI": 0.6,
                "MINUS_DI": 0.6,
                "ATR": 0.4,
                "OBV": 0.5,
                "STOCH": 0.7,
                "WILLR": 0.6,
            },
            "regime_adjustments": {
                "TRENDING_UP": {
                    "ADX": 1.2,
                    "PLUS_DI": 1.3,
                    "MINUS_DI": 0.7,
                },
                "TRENDING_DOWN": {
                    "ADX": 1.2,
                    "PLUS_DI": 0.7,
                    "MINUS_DI": 1.3,
                },
            },
        },
        "consensus_combiner": {
            "consensus_threshold": 0.6,
            "min_indicators": 3,
            "confidence_weight_factor": 1.5,
        },
        "decision_tree": {
            "decision_thresholds": {
                "MARKET_REGIME": 0.7,
                "INDICATOR_ANALYSIS": 0.6,
                "SIGNAL_CONSENSUS": 0.65,
                "RISK_ASSESSMENT": 0.5,
                "FINAL_DECISION": 0.6,
            },
            "max_risk_level": 0.7,
            "volatility_threshold": 0.8,
        },
        "signal_validator": {
            "min_confidence_threshold": 0.3,
            "max_risk_threshold": 0.8,
            "min_indicators_required": 2,
            "max_indicator_disagreement": 0.8,
            "enable_historical_validation": False,
        },
        "conflict_detector": {
            "direction_conflict_threshold": 0.3,
            "strength_conflict_threshold": 2,
            "regime_conflict_threshold": 0.6,
        },
        "escalation_logic": {
            "high_conflict_count": 2,
            "high_severity_conflicts": 1,
            "low_validation_confidence": 0.3,
            "uncertain_regime_duration": 3,
            "escalation_probability_threshold": 0.8,
            "max_escalations_per_hour": 10,
        },
        "max_history_size": 100,
    }


@pytest.fixture
def local_signal_generator(signal_generator_config):
    """Create a local signal generator for testing."""
    return LocalSignalGenerator(config=signal_generator_config)


# ==============================
# Orchestrator Fixture
# ==============================

@pytest.fixture
def orchestrator(all_agents, data_pipeline, mock_state_manager):
    """Create an orchestrator for testing."""
    return Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=all_agents["technical"],
        sentiment_agent=all_agents["sentiment"],
        risk_agent=all_agents["risk"],
        portfolio_agent=all_agents["portfolio"],
        state_manager=mock_state_manager
    )


# ==============================
# Test Data Fixtures
# ==============================

@pytest.fixture
def test_symbols():
    """Common symbols used in testing."""
    return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]


@pytest.fixture
def date_range():
    """Standard date range for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    return start_date, end_date


@pytest.fixture
def sample_news_headlines():
    """Sample news headlines for sentiment testing."""
    return [
        "Apple reports record quarterly earnings",
        "Analysts upgrade AAPL stock rating",
        "Strong demand for iPhone drives Apple shares higher",
        "Tesla faces production delays",
        "Competition intensifies in EV market",
        "Google announces new AI features",
        "Alphabet quarterly results meet expectations"
    ]


# ==============================
# Performance Testing Fixtures
# ==============================

@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "iterations": 10,
        "timeout": 60.0,
        "latency_threshold_ms": 100.0,
        "accuracy_threshold": 0.85,
        "cost_reduction_threshold": 0.70
    }


# ==============================
# Utility Functions
# ==============================

def create_agent_decision(
    signal: str = "BUY",
    confidence: float = 0.8,
    reasoning: str = "Test reasoning",
    agent_name: str = "test_agent",
    symbol: str = "TEST"
) -> AgentDecision:
    """Utility function to create agent decisions for testing."""
    return AgentDecision(
        agent_name=agent_name,
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        reasoning=reasoning,
        timestamp=datetime.now()
    )


def create_test_market_data_with_indicators(
    symbol: str = "TEST",
    price: float = 150.0,
    indicators: Optional[Dict[str, float]] = None
) -> MarketData:
    """Utility function to create market data with custom indicators."""
    if indicators is None:
        indicators = {
            "RSI": 50.0,
            "MACD": 0.1,
            "ADX": 25.0,
        }

    return MarketData(
        symbol=symbol,
        price=price,
        volume=1000000,
        timestamp=datetime.now(),
        ohlc={"open": price * 0.99, "high": price * 1.01, "low": price * 0.98, "close": price},
        technical_indicators=indicators
    )


# ==============================
# Async Test Support
# ==============================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==============================
# Environment Setup
# ==============================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up environment variables for testing."""
    # Disable actual API calls during testing
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATA_PROVIDER_CACHE_ENABLED", "false")

    # Set reasonable defaults for testing
    monkeypatch.setattr(settings.data, "CACHE_ENABLED", False)
    monkeypatch.setattr(settings.signal_generation, "LOCAL_SIGNAL_GENERATION_ENABLED", True)
    monkeypatch.setattr(settings.signal_generation, "ROLLOUT_PERCENTAGE", 1.0)
