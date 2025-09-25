"""
API routes for monitoring and metrics.
"""
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Simple metrics storage (in production, use Prometheus)
_metrics = {
    "requests_total": 0,
    "requests_duration_sum": 0.0,
    "errors_total": 0,
    "start_time": time.time()
}

def increment_requests():
    """Increment request counter."""
    _metrics["requests_total"] += 1

def record_request_duration(duration: float):
    """Record request duration."""
    _metrics["requests_duration_sum"] += duration

def increment_errors():
    """Increment error counter."""
    _metrics["errors_total"] += 1

@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """
    Get basic application metrics.
    """
    try:
        uptime = time.time() - _metrics["start_time"]
        avg_duration = (
            _metrics["requests_duration_sum"] / _metrics["requests_total"]
            if _metrics["requests_total"] > 0 else 0
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "requests_total": _metrics["requests_total"],
            "errors_total": _metrics["errors_total"],
            "average_request_duration_seconds": avg_duration,
            "error_rate": (
                _metrics["errors_total"] / _metrics["requests_total"]
                if _metrics["requests_total"] > 0 else 0
            )
        }
    except Exception as e:
        logger.error("Error retrieving metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving metrics")

@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for load balancer health checks.
    """
    return {"status": "pong", "timestamp": datetime.now().isoformat()}