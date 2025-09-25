"""
Defines the base class for all trading agents.

This module provides the abstract `BaseAgent` class that all specialized agents
(e.g., TechnicalAnalysisAgent, SentimentAnalysisAgent) must inherit from.
It establishes a common interface for agent configuration, analysis, and
interaction with the broader system.
"""
from abc import ABC, abstractmethod
from typing import Any

from .data_structures import AgentConfig, AgentDecision, MarketData


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    This class defines the common structure and methods that every agent must
    implement. It ensures that agents can be used interchangeably within the
    orchestration framework.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        message_bus: Any,
        state_manager: Any,
    ):
        """
        Initializes the BaseAgent.

        Args:
            config: The configuration object for the agent.
            llm_client: An instance of a client for interacting with a large language model.
            message_bus: An instance of the system's message bus for communication.
            state_manager: An instance of the system's state manager.
        """
        self.config = config
        self.llm_client = llm_client
        self.message_bus = message_bus
        self.state_manager = state_manager

    @abstractmethod
    async def analyze(self, market_data: MarketData) -> AgentDecision:
        """
        Analyzes the given market data to produce a trading decision.

        This is the core method of any agent. It takes market data as input and
        should return an AgentDecision object.

        Args:
            market_data: The market data to be analyzed.

        Returns:
            An AgentDecision object containing the agent's analysis and signal.
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the agent's LLM.

        This prompt provides the context and instructions for the language model
        that the agent will use for its analysis.

        Returns:
            A string containing the system prompt.
        """
        pass

    async def make_llm_call(self, user_prompt: str) -> str:
        """
        Makes a call to the language model with a given user prompt.

        This method will handle the interaction with the LLM, including formatting
        the request, handling retries, and managing errors.

        Args:
            user_prompt: The user-level prompt or query for the LLM.

        Returns:
            The textual response from the language model.
        """
        return await self.llm_client.generate(
            model=self.config.model_name,
            prompt=user_prompt,
            system_prompt=self.get_system_prompt(),
        )
