"""
API routes for trade signals.
"""
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query

from src.agents.data_structures import AgentDecision, AgentConfig
from src.communication.orchestrator import Orchestrator
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.cache import CacheManager
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.risk import RiskManagementAgent
from src.agents.portfolio import PortfolioManagementAgent
from src.communication.state_manager import StateManager
from src.communication.message_bus import MessageBus
from src.llm.client import LLMClient
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Initialize components (in production, use dependency injection)
def get_orchestrator():
    """Factory function to create orchestrator with proper dependencies."""
    # Core components
    llm_client = LLMClient()
    message_bus = MessageBus()
    state_manager = StateManager()

    # Initialize data provider
    provider = YFinanceProvider(rate_limit=10, period=60.0)
    
    # Initialize cache manager (optional)
    cache_manager = CacheManager()
    
    # Initialize data pipeline
    data_pipeline = DataPipeline(provider=provider, cache=cache_manager)
    
    # Initialize agents with all dependencies
    agent_dependencies = {
        "llm_client": llm_client,
        "message_bus": message_bus,
        "state_manager": state_manager,
    }
    technical_agent = TechnicalAnalysisAgent(config=AgentConfig(name="technical"), **agent_dependencies)
    sentiment_agent = SentimentAnalysisAgent(config=AgentConfig(name="sentiment"), **agent_dependencies)
    risk_agent = RiskManagementAgent(config=AgentConfig(name="risk"), **agent_dependencies)
    portfolio_agent = PortfolioManagementAgent(config=AgentConfig(name="portfolio"), **agent_dependencies)
    
    return Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=technical_agent,
        sentiment_agent=sentiment_agent,
        risk_agent=risk_agent,
        portfolio_agent=portfolio_agent,
        state_manager=state_manager
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