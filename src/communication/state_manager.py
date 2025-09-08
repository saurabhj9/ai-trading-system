"""
Manages the shared state of the AI Trading System.

This module provides a StateManager class that serves as a centralized store
for all shared data, such as portfolio status, market data, and agent
decisions. This implementation uses a simple in-memory dictionary, making it
suitable for single-process applications and testing.
"""
from typing import Any, Dict, Optional


class StateManager:
    """
    A simple in-memory state manager.

    This class uses a dictionary to store the system's state. It is not
    suitable for multi-process or distributed applications, but serves as a
    foundational implementation.
    """

    def __init__(self):
        """Initializes the StateManager with an empty state dictionary."""
        self._state: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the state by key.

        Args:
            key: The key of the value to retrieve.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        return self._state.get(key)

    def set(self, key: str, value: Any):
        """
        Sets a value in the state.

        Args:
            key: The key of the value to set.
            value: The value to store.
        """
        self._state[key] = value

    def get_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """
        Retrieries the current state of the portfolio.

        Returns:
            A dictionary representing the portfolio state, or None if not set.
        """
        return self.get("portfolio_state")

    def set_portfolio_state(self, portfolio_state: Dict[str, Any]):
        """
        Updates the state of the portfolio.

        Args:
            portfolio_state: A dictionary representing the new portfolio state.
        """
        self.set("portfolio_state", portfolio_state)

    def get_agent_decision(self, agent_name: str) -> Optional[Any]:
        """
        Retrieves the latest decision from a specific agent.

        Args:
            agent_name: The name of the agent whose decision to retrieve.

        Returns:
            The agent's decision object, or None if not found.
        """
        return self.get(f"agent_decision_{agent_name}")

    def set_agent_decision(self, agent_name: str, decision: Any):
        """
        Stores the latest decision from a specific agent.

        Args:
            agent_name: The name of the agent.
            decision: The decision object to store.
        """
        self.set(f"agent_decision_{agent_name}", decision)
