"""
Implements the Sentiment Analysis Agent.

This agent analyzes market sentiment by processing qualitative data from
news articles, social media, and financial news sources.
"""
import json
from typing import List, Dict, Any

from .base import BaseAgent, clean_json_response
from .data_structures import AgentDecision, MarketData
from ..config.settings import settings


class SentimentAnalysisAgent(BaseAgent):
    """
    An agent specialized in sentiment analysis of financial markets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Require Alpha Vantage API key for sentiment analysis
        if not settings.data.ALPHA_VANTAGE_API_KEY:
            raise ValueError(
                "DATA_ALPHA_VANTAGE_API_KEY is required for sentiment analysis. "
                "Please set it in your .env file to fetch real news data. "
                "Get your free API key at: https://www.alphavantage.co/support/#api-key"
            )
        from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
        self.news_provider = AlphaVantageProvider(settings.data.ALPHA_VANTAGE_API_KEY)

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the sentiment analysis LLM.
        """
        return (
            "You are a specialized AI assistant for financial sentiment analysis. "
            "Your goal is to analyze news headlines related to a stock and determine "
            "the market sentiment (BULLISH, BEARISH, or NEUTRAL). Provide a confidence "
            "score (0.0 to 1.0) and a brief, data-driven reasoning. Your final output "
            "must be a single JSON object with three keys: 'signal', 'confidence', "
            "and 'reasoning'."
        )

    async def get_user_prompt(self, market_data: MarketData) -> str:
        """
        Generates the user prompt for sentiment analysis by fetching news.
        """
        # Fetch news from the provider
        if not hasattr(self, "news_provider"):
            raise ValueError("News provider is not configured.")

        news_articles = await self.news_provider.fetch_news_sentiment(market_data.symbol)
        if not news_articles:
            # Return None to signal that we should use a default neutral decision
            return None

        # Format the news into a user prompt for the LLM.
        headlines = [article.get("title", "") for article in news_articles]
        return (
            f"Analyze the sentiment from the following news headlines for {market_data.symbol}:\n"
            + "\n".join(f"- {headline}" for headline in headlines)
            + "\n\nBased on these headlines, provide your sentiment (BULLISH, BEARISH, or NEUTRAL), "
            "confidence, and reasoning as a single JSON object."
        )

    def create_decision(self, market_data: MarketData, response: str) -> AgentDecision:
        """
        Creates an AgentDecision from the LLM response for sentiment analysis.
        """
        try:
            # Clean the response to handle control characters
            cleaned_response = clean_json_response(response)
            decision_json = json.loads(cleaned_response)
            signal = decision_json.get("signal", "NEUTRAL")
            confidence = float(decision_json.get("confidence", 0.0))
            reasoning = decision_json.get("reasoning", "No reasoning provided.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, create a default decision with an error message.
            signal = "ERROR"
            confidence = 0.0
            reasoning = f"Failed to parse LLM response: {e}. Raw response: {response}"

        # For supporting_data, we need headlines, but since get_user_prompt has them, perhaps store them.
        # For simplicity, assume we can extract or store.
        # Since create_decision doesn't have headlines, perhaps modify to take additional data.
        # For now, put response in supporting_data.
        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "llm_response": response,
                "market_data_used": market_data.__dict__,
            },
        )

    async def analyze(self, market_data: MarketData, **kwargs) -> AgentDecision:
        """
        Performs sentiment analysis on news data using an LLM.
        """
        try:
            user_prompt = await self.get_user_prompt(market_data)

            # If no news available (API rate limit or fetch failure), return neutral decision
            if user_prompt is None:
                return AgentDecision(
                    agent_name=self.config.name,
                    symbol=market_data.symbol,
                    signal="NEUTRAL",
                    confidence=0.5,
                    reasoning="No news articles available (API rate limit or fetch failure). Defaulting to NEUTRAL sentiment.",
                    supporting_data={"news_unavailable": True}
                )

            llm_response = await self.make_llm_call(user_prompt)
            return self.create_decision(market_data, llm_response)
        except ValueError as e:
            return self._error_decision(market_data.symbol, str(e))

    def _error_decision(self, symbol: str, reason: str) -> AgentDecision:
        """Creates a default error decision."""
        return AgentDecision(
            agent_name=self.config.name,
            symbol=symbol,
            signal="ERROR",
            confidence=0.0,
            reasoning=reason,
        )
