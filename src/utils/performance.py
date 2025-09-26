"""
Performance profiling utilities for the AI Trading System.

This module provides timing decorators, context managers, and performance
measurement tools to profile and optimize system performance.
"""
import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def complete(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Mark the operation as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


class PerformanceProfiler:
    """Central profiler for collecting and analyzing performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self.active_operations: Dict[str, PerformanceMetrics] = {}

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a performance operation."""
        operation_id = f"{operation_name}_{time.time()}_{id(self)}"
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.active_operations[operation_id] = metrics
        return operation_id

    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None) -> None:
        """End tracking a performance operation."""
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            metrics.complete(success, error_message)
            self.metrics[metrics.operation_name].append(metrics)
            del self.active_operations[operation_id]

    def get_metrics_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for operations."""
        if operation_name:
            operations = self.metrics.get(operation_name, [])
        else:
            operations = [metric for metrics_list in self.metrics.values() for metric in metrics_list]

        if not operations:
            return {"count": 0, "avg_duration": 0, "min_duration": 0, "max_duration": 0}

        durations = [op.duration for op in operations if op.duration is not None]
        successful_ops = [op for op in operations if op.success]

        return {
            "count": len(operations),
            "successful_count": len(successful_ops),
            "success_rate": len(successful_ops) / len(operations) if operations else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "total_duration": sum(durations) if durations else 0,
        }

    def get_all_operation_names(self) -> list:
        """Get list of all tracked operation names."""
        return list(self.metrics.keys())

    def clear_metrics(self, operation_name: Optional[str] = None) -> None:
        """Clear collected metrics."""
        if operation_name:
            self.metrics[operation_name].clear()
        else:
            self.metrics.clear()
            self.active_operations.clear()


# Global profiler instance
profiler = PerformanceProfiler()


def time_function(operation_name: Optional[str] = None, log_result: bool = True):
    """
    Decorator to time function execution.

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        log_result: Whether to log the timing result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = profiler.start_operation(name)

            try:
                result = await func(*args, **kwargs)
                profiler.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                profiler.end_operation(operation_id, success=False, error_message=str(e))
                raise
            finally:
                if log_result:
                    summary = profiler.get_metrics_summary(name)
                    if summary["count"] > 0:
                        logger.info(
                            f"Performance: {name}",
                            duration=summary["avg_duration"],
                            count=summary["count"],
                            success_rate=summary["success_rate"]
                        )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = profiler.start_operation(name)

            try:
                result = func(*args, **kwargs)
                profiler.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                profiler.end_operation(operation_id, success=False, error_message=str(e))
                raise
            finally:
                if log_result:
                    summary = profiler.get_metrics_summary(name)
                    if summary["count"] > 0:
                        logger.info(
                            f"Performance: {name}",
                            duration=summary["avg_duration"],
                            count=summary["count"],
                            success_rate=summary["success_rate"]
                        )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def time_block(operation_name: str, metadata: Optional[Dict[str, Any]] = None, log_result: bool = True):
    """
    Context manager to time a block of code.

    Args:
        operation_name: Name for the timed operation
        metadata: Additional metadata to store with the metrics
        log_result: Whether to log the timing result
    """
    operation_id = profiler.start_operation(operation_name, metadata)

    try:
        yield
        profiler.end_operation(operation_id, success=True)
    except Exception as e:
        profiler.end_operation(operation_id, success=False, error_message=str(e))
        raise
    finally:
        if log_result:
            summary = profiler.get_metrics_summary(operation_name)
            if summary["count"] > 0:
                logger.info(
                    f"Performance block: {operation_name}",
                    duration=summary["avg_duration"],
                    count=summary["count"],
                    success_rate=summary["success_rate"]
                )


class AsyncTimer:
    """Async context manager for timing operations with more control."""

    def __init__(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.metadata = metadata
        self.operation_id: Optional[str] = None

    async def __aenter__(self):
        self.operation_id = profiler.start_operation(self.operation_name, self.metadata)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            profiler.end_operation(self.operation_id, success, error_message)


def measure_llm_call(model_name: str, prompt_length: Optional[int] = None):
    """
    Decorator specifically for LLM calls with additional metadata.

    Args:
        model_name: Name of the LLM model being used
        prompt_length: Length of the prompt (will be calculated if not provided)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract prompt length from arguments if not provided
            actual_prompt_length = prompt_length
            if actual_prompt_length is None and len(args) > 1:
                # Assume prompt is the second argument after self
                prompt_arg = args[1] if len(args) > 1 else kwargs.get('prompt', '')
                if isinstance(prompt_arg, str):
                    actual_prompt_length = len(prompt_arg)

            metadata = {
                "model": model_name,
                "prompt_length": actual_prompt_length,
                "call_type": "llm"
            }

            async with AsyncTimer(f"llm_call_{model_name}", metadata):
                return await func(*args, **kwargs)

        return async_wrapper

    return decorator


def measure_data_operation(operation_type: str, data_size: Optional[int] = None):
    """
    Decorator for data operations (fetching, processing, caching).

    Args:
        operation_type: Type of data operation (fetch, process, cache)
        data_size: Size of data being processed (optional)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            metadata = {
                "operation_type": operation_type,
                "data_size": data_size,
                "component": "data_pipeline"
            }

            async with AsyncTimer(f"data_{operation_type}", metadata):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            metadata = {
                "operation_type": operation_type,
                "data_size": data_size,
                "component": "data_pipeline"
            }

            with time_block(f"data_{operation_type}", metadata):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience functions for common use cases
def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of all performance metrics."""
    all_operations = profiler.get_all_operation_names()
    summary = {}

    for operation in all_operations:
        summary[operation] = profiler.get_metrics_summary(operation)

    return summary


def log_performance_report() -> None:
    """Log a comprehensive performance report."""
    summary = get_performance_summary()

    logger.info("Performance Report", total_operations=len(summary))

    for operation, metrics in summary.items():
        logger.info(
            f"Operation: {operation}",
            count=metrics["count"],
            avg_duration=metrics["avg_duration"],
            success_rate=metrics["success_rate"]
        )


def reset_performance_metrics() -> None:
    """Reset all performance metrics."""
    profiler.clear_metrics()
    logger.info("Performance metrics reset")
