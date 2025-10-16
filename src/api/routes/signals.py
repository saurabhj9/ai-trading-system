"""
API routes for trade signals.
"""
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from src.agents.data_structures import AgentConfig, AgentDecision
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.message_bus import MessageBus
from src.communication.orchestrator import Orchestrator
from src.communication.state_manager import StateManager
from src.data.cache import CacheManager
from src.data.pipeline import DataPipeline
from src.data.providers.composite_news_provider import CompositeNewsProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.llm.client import LLMClient
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


# In production, use a dependency injection framework like FastAPI's `Depends`
def get_orchestrator():
    """Factory function to create an orchestrator with dependencies."""
    # Core components
    llm_client = LLMClient()
    message_bus = MessageBus()
    state_manager = StateManager()

    # Initialize data providers
    yfinance_provider = YFinanceProvider(rate_limit=10, period=60.0)
    alpha_vantage_api_key = os.getenv("DATA_ALPHA_VANTAGE_API_KEY")
    marketaux_api_key = os.getenv("DATA_MARKETAUX_API_KEY")
    if not alpha_vantage_api_key and not marketaux_api_key:
        raise ValueError(
            "At least one of DATA_ALPHA_VANTAGE_API_KEY or DATA_MARKETAUX_API_KEY environment variables is required for sentiment analysis. "
            "Please set them in your .env file. "
            "Get your free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key"
        )
    composite_news_provider = CompositeNewsProvider(
        alpha_vantage_api_key=alpha_vantage_api_key,
        marketaux_api_key=marketaux_api_key
    )

    # Initialize data pipeline (uses yfinance for market data)
    data_pipeline = DataPipeline(provider=yfinance_provider, cache=CacheManager())

    # Initialize agents
    agent_dependencies = {
        "llm_client": llm_client,
        "message_bus": message_bus,
        "state_manager": state_manager,
    }
    technical_agent = TechnicalAnalysisAgent(
        config=AgentConfig(name="technical"), **agent_dependencies
    )
    sentiment_agent = SentimentAnalysisAgent(
        config=AgentConfig(name="sentiment"),
        news_provider=composite_news_provider,
        **agent_dependencies,
    )
    risk_agent = RiskManagementAgent(
        config=AgentConfig(name="risk"), **agent_dependencies
    )
    portfolio_agent = PortfolioManagementAgent(
        config=AgentConfig(name="portfolio"), **agent_dependencies
    )

    return Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=technical_agent,
        sentiment_agent=sentiment_agent,
        risk_agent=risk_agent,
        portfolio_agent=portfolio_agent,
        state_manager=state_manager,
    )

@router.get("/signals/{symbol}", response_model=Dict[str, Any])
async def get_trade_signals(
    symbol: str,
    days: int = Query(30, description="Number of days of historical data to analyze", ge=1, le=365)
):
    """
    Get trade signals for a given symbol by running the full agent workflow.
    """
    logger.info("Generating trade signals", symbol=symbol, days=days)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get orchestrator instance
        orchestrator = get_orchestrator()
        result = await orchestrator.run(symbol, start_date, end_date)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Format response
        response = {
            "symbol": symbol,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "final_decision": {
                "signal": result["final_decision"].signal,
                "confidence": result["final_decision"].confidence,
                "reasoning": result["final_decision"].reasoning,
                "timestamp": result["final_decision"].timestamp.isoformat()
            },
            "agent_decisions": {}
        }

        # Add individual agent decisions
        for agent_name, decision in result.get("decisions", {}).items():
            response["agent_decisions"][agent_name] = {
                "signal": decision.signal,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "timestamp": decision.timestamp.isoformat()
            }

        logger.info("Trade signals generated successfully", symbol=symbol, signal=response["final_decision"]["signal"])
        return response

    except Exception as e:
        logger.error("Error generating trade signals", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error generating signals: {str(e)}")
