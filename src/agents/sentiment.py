"""
Implements the Sentiment Analysis Agent.

This agent analyzes market sentiment by processing qualitative data from
news articles, social media, and financial news sources.
"""
import json

from .base import BaseAgent
from .data_structures import AgentDecision, MarketData


class SentimentAnalysisAgent(BaseAgent):
    """
    An agent specialized in sentiment analysis of financial markets.
    """

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

    async def analyze(self, market_data: MarketData) -> AgentDecision:
        """
        Performs sentiment analysis on news data using an LLM.
        """
        # Fetch news from the provider
        if not hasattr(self, "news_provider"):
            return self._error_decision(market_data.symbol,"News provider is not configured.")

        news_articles = await self.news_provider.fetch_news_sentiment(market_data.symbol)
        if not news_articles:
            return self._error_decision(market_data.symbol, "No news articles found or failed to fetch news.")

        # Format the news into a user prompt for the LLM.
        headlines = [article.get("title", "") for article in news_articles]
        user_prompt = (
            f"Analyze the sentiment from the following news headlines for {market_data.symbol}:\n"
            + "\n".join(f"- {headline}" for headline in headlines)
            + "\n\nBased on these headlines, provide your sentiment (BULLISH, BEARISH, or NEUTRAL), "
            "confidence, and reasoning as a single JSON object."
        )

        # Make the LLM call.
        llm_response = await self.make_llm_call(user_prompt)

        # Parse the LLM's JSON response to create an AgentDecision.
        try:
            decision_json = json.loads(llm_response)
            signal = decision_json.get("signal", "NEUTRAL")
            confidence = float(decision_json.get("confidence", 0.0))
            reasoning = decision_json.get("reasoning", "No reasoning provided.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, create a default decision with an error message.
            signal = "ERROR"
            confidence = 0.0
            reasoning = f"Failed to parse LLM response: {e}. Raw response: {llm_response}"

        return AgentDecision(
            agent_name=self.config.name,
            symbol=market_data.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                "llm_response": llm_response,
                "news_analyzed": headlines,
                "market_data_used": market_data.model_dump(),
            },
        )

    def _error_decision(self, symbol: str, reason: str) -> AgentDecision:
        """Creates a default error decision."""
        return AgentDecision(
            agent_name=self.config.name,
            symbol=symbol,
            signal="ERROR",
            confidence=0.0,
            reasoning=reason,
        )
