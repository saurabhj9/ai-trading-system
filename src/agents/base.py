"""
Defines the base class for all trading agents.

This module provides the abstract `BaseAgent` class that all specialized agents
(e.g., TechnicalAnalysisAgent, SentimentAnalysisAgent) must inherit from.
It establishes a common interface for agent configuration, analysis, and
interaction with the broader system.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import re

from .data_structures import AgentConfig, AgentDecision, MarketData


def clean_json_response(response: str) -> str:
    """
    Clean up JSON response from LLM to handle control characters.

    LLMs sometimes return JSON with unescaped newlines or other control
    characters that cause JSON parsing to fail. This function cleans them up.

    Args:
        response: Raw JSON string from LLM

    Returns:
        Cleaned JSON string safe for parsing
    """
    # Remove any leading/trailing whitespace
    response = response.strip()

    # Try to extract JSON object if wrapped in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Replace actual newlines within string values with escaped newlines
    # This is a simple heuristic - finds content between quotes and escapes newlines
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape newlines and other control characters
        content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'"{content}"'

    # Match string values in JSON (quoted content)
    response = re.sub(r'"([^"]*)"', escape_newlines_in_strings, response, flags=re.DOTALL)

    return response


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
        **kwargs,
    ):
        """
        Initializes the BaseAgent.

        Args:
            config: The configuration object for the agent.
            llm_client: An instance of a client for interacting with a large language model.
            message_bus: An instance of the system's message bus for communication.
            state_manager: An instance of the system's state manager.
            **kwargs: Additional keyword arguments for agent-specific dependencies.
        """
        self.config = config
        self.llm_client = llm_client
        self.message_bus = message_bus
        self.state_manager = state_manager
        # Capture any extra dependencies
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    async def analyze(self, market_data: MarketData, **kwargs) -> AgentDecision:
        """
        Analyzes the given market data to produce a trading decision.

        This is the core method of any agent. It takes market data as input and
        should return an AgentDecision object. Additional keyword arguments may
        be provided for agents that require more context (e.g., other agent decisions,
        portfolio state).

        Args:
            market_data: The market data to be analyzed.
            **kwargs: Additional arguments specific to the agent type.

        Returns:
            An AgentDecision object containing the agent's analysis and signal.
        """
        pass

    @abstractmethod
    async def get_user_prompt(self, market_data: MarketData) -> str:
        """
        Generates the user prompt for the given market data.

        This method constructs the prompt that will be sent to the LLM for analysis.
        May involve async operations like fetching external data.

        Args:
            market_data: The market data to generate a prompt for.

        Returns:
            A string containing the user prompt.
        """
        pass

    @abstractmethod
    def create_decision(self, market_data: MarketData, response: str) -> AgentDecision:
        """
        Creates an AgentDecision from the LLM response.

        This method parses the LLM response and constructs a structured decision.

        Args:
            market_data: The market data that was analyzed.
            response: The response from the LLM.

        Returns:
            An AgentDecision object.
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

    async def make_llm_call(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Makes a call to the language model with a given user prompt.

        This method will handle the interaction with the LLM, including formatting
        the request, handling retries, and managing errors.

        Args:
            user_prompt: The user-level prompt or query for the LLM.
            system_prompt: An optional system prompt to override the default.

        Returns:
            The textual response from the language model.
        """
        effective_system_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        return await self.llm_client.generate(
            model=self.config.model_name,
            prompt=user_prompt,
            system_prompt=effective_system_prompt,
        )

    async def batch_analyze(self, market_data_list: List[MarketData], **kwargs) -> List[AgentDecision]:
        """
        Analyzes multiple market data instances in batch mode for optimized processing.

        This method allows agents to process multiple market data points simultaneously
        using batched LLM calls, reducing latency. Subclasses should override this
        method to implement batch processing logic.

        Args:
            market_data_list: List of market data to analyze.
            **kwargs: Additional arguments specific to the agent type.

        Returns:
            List of AgentDecision objects corresponding to each market data input.

        Raises:
            NotImplementedError: If the agent does not support batch analysis.
        """
        raise NotImplementedError("Batch analysis not implemented for this agent")
