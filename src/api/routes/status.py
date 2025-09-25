"""
API routes for system status and health checks.
"""
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

from src.communication.state_manager import StateManager
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

state_manager = StateManager()

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check endpoint.
    """
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Trading System API"
    }

@router.get("/status", response_model=Dict[str, Any])
async def system_status():
    """
    Get overall system status including portfolio state.
    """
    logger.info("System status requested")
    try:
        portfolio_state = state_manager.get_portfolio_state() or {"cash": 100000, "positions": {}}

        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio_state,
            "agents": {
                "technical": "available",
                "sentiment": "available (mock)",
                "risk": "available (mock)",
                "portfolio": "available (mock)"
            },
            "data_sources": {
                "yfinance": "available",
                "alpha_vantage": "available"
            },
            "caching": {
                "redis": "available"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }