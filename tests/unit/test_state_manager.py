"""
Unit tests for the StateManager.
"""
import pytest

from src.communication.state_manager import StateManager


@pytest.fixture
def state_manager():
    """Provides a fresh StateManager for each test."""
    return StateManager()


def test_initial_state_is_empty(state_manager):
    """Tests that the initial state is empty."""
    assert state_manager.get("some_key") is None


def test_set_and_get(state_manager):
    """Tests the basic set and get functionality."""
    state_manager.set("my_key", "my_value")
    assert state_manager.get("my_key") == "my_value"


def test_get_nonexistent_key_returns_none(state_manager):
    """Tests that getting a non-existent key returns None."""
    assert state_manager.get("non_existent_key") is None


def test_set_and_get_portfolio_state(state_manager):
    """Tests the portfolio-specific getter and setter."""
    portfolio = {"cash": 100000, "positions": {"AAPL": 100}}
    state_manager.set_portfolio_state(portfolio)
    assert state_manager.get_portfolio_state() == portfolio


def test_set_and_get_agent_decision(state_manager):
    """Tests the agent decision-specific getter and setter."""
    decision = {"signal": "BUY", "confidence": 0.8}
    state_manager.set_agent_decision("TestAgent", decision)
    assert state_manager.get_agent_decision("TestAgent") == decision
    assert state_manager.get_agent_decision("OtherAgent") is None
