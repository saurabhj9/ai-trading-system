"""
API routes for monitoring and metrics.
"""
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query

from src.utils.logging import get_logger
from src.utils.performance import profiler, get_performance_summary, log_performance_report

router = APIRouter()
logger = get_logger(__name__)

# Simple metrics storage (in production, use Prometheus)
_metrics = {
    "requests_total": 0,
    "requests_duration_sum": 0.0,
    "errors_total": 0,
    "start_time": time.time()
}

# Performance metrics storage
_performance_metrics = {
    "llm_calls_total": 0,
    "llm_calls_duration_sum": 0.0,
    "data_operations_total": 0,
    "data_operations_duration_sum": 0.0,
    "agent_operations_total": 0,
    "agent_operations_duration_sum": 0.0,
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

def record_llm_call(duration: float):
    """Record LLM call metrics."""
    _performance_metrics["llm_calls_total"] += 1
    _performance_metrics["llm_calls_duration_sum"] += duration

def record_data_operation(duration: float):
    """Record data operation metrics."""
    _performance_metrics["data_operations_total"] += 1
    _performance_metrics["data_operations_duration_sum"] += duration

def record_agent_operation(duration: float):
    """Record agent operation metrics."""
    _performance_metrics["agent_operations_total"] += 1
    _performance_metrics["agent_operations_duration_sum"] += duration

@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(include_performance: bool = Query(True, description="Include detailed performance metrics")):
    """
    Get comprehensive application metrics including performance data.
    """
    try:
        uptime = time.time() - _metrics["start_time"]
        avg_request_duration = (
            _metrics["requests_duration_sum"] / _metrics["requests_total"]
            if _metrics["requests_total"] > 0 else 0
        )

        base_metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "requests_total": _metrics["requests_total"],
            "errors_total": _metrics["errors_total"],
            "average_request_duration_seconds": avg_request_duration,
            "error_rate": (
                _metrics["errors_total"] / _metrics["requests_total"]
                if _metrics["requests_total"] > 0 else 0
            )
        }

        if include_performance:
            # Add performance metrics
            avg_llm_duration = (
                _performance_metrics["llm_calls_duration_sum"] / _performance_metrics["llm_calls_total"]
                if _performance_metrics["llm_calls_total"] > 0 else 0
            )
            avg_data_duration = (
                _performance_metrics["data_operations_duration_sum"] / _performance_metrics["data_operations_total"]
                if _performance_metrics["data_operations_total"] > 0 else 0
            )
            avg_agent_duration = (
                _performance_metrics["agent_operations_duration_sum"] / _performance_metrics["agent_operations_total"]
                if _performance_metrics["agent_operations_total"] > 0 else 0
            )

            performance_metrics = {
                "llm_calls_total": _performance_metrics["llm_calls_total"],
                "llm_calls_avg_duration_seconds": avg_llm_duration,
                "data_operations_total": _performance_metrics["data_operations_total"],
                "data_operations_avg_duration_seconds": avg_data_duration,
                "agent_operations_total": _performance_metrics["agent_operations_total"],
                "agent_operations_avg_duration_seconds": avg_agent_duration,
            }

            # Add detailed profiler metrics
            profiler_summary = get_performance_summary()
            performance_metrics["detailed_operations"] = profiler_summary

            base_metrics["performance"] = performance_metrics

        return base_metrics
    except Exception as e:
        logger.error("Error retrieving metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving metrics")

@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for load balancer health checks.
    """
    return {"status": "pong", "timestamp": datetime.now().isoformat()}

@router.get("/performance/summary")
async def get_performance_summary_endpoint():
    """
    Get detailed performance profiling summary.
    """
    try:
        summary = get_performance_summary()
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
    except Exception as e:
        logger.error("Error retrieving performance summary", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving performance summary")

@router.post("/performance/log-report")
async def log_performance_report_endpoint():
    """
    Trigger logging of a comprehensive performance report.
    """
    try:
        log_performance_report()
        return {"status": "Performance report logged", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error("Error logging performance report", error=str(e))
        raise HTTPException(status_code=500, detail="Error logging performance report")

@router.post("/performance/reset")
async def reset_performance_metrics_endpoint():
    """
    Reset all performance metrics.
    """
    try:
        from src.utils.performance import reset_performance_metrics
        reset_performance_metrics()
        return {"status": "Performance metrics reset", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error("Error resetting performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error resetting performance metrics")

@router.get("/performance/operations/{operation_name}")
async def get_operation_metrics(operation_name: str):
    """
    Get detailed metrics for a specific operation.
    """
    try:
        summary = profiler.get_metrics_summary(operation_name)
        if summary["count"] == 0:
            raise HTTPException(status_code=404, detail=f"No metrics found for operation: {operation_name}")

        return {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "metrics": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving operation metrics", operation=operation_name, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving operation metrics")
