#!/usr/bin/env python3
"""
Data Pipeline Performance Profiling Script.

This script benchmarks the data pipeline components including:
- Data fetching from providers
- Technical indicator calculations
- Caching performance
- Overall pipeline throughput
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from src.config.settings import settings
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.cache import CacheManager
from src.utils.performance import (
    time_function, time_block, measure_data_operation,
    get_performance_summary, log_performance_report, reset_performance_metrics
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@measure_data_operation("fetch", data_size=365)
async def benchmark_data_fetching(pipeline: DataPipeline, symbols: List[str], days: int = 365) -> Dict[str, Any]:
    """
    Benchmark data fetching performance for multiple symbols.

    Args:
        pipeline: Data pipeline instance
        symbols: List of stock symbols to fetch
        days: Number of days of historical data

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    for symbol in symbols:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        with time_block(f"fetch_{symbol}", {"symbol": symbol, "days": days}):
            data = await pipeline.fetch_and_process_data(symbol, start_date, end_date)

        results[symbol] = {
            "success": data is not None,
            "has_data": data is not None,
            "price": data.price if data else None,
            "volume": data.volume if data else None,
            "indicators_count": len(data.technical_indicators) if data else 0
        }

    return results


@measure_data_operation("indicator_calculation")
async def benchmark_indicator_calculation(pipeline: DataPipeline, symbol: str = "AAPL", days: int = 365) -> Dict[str, Any]:
    """
    Benchmark technical indicator calculation performance.

    Args:
        pipeline: Data pipeline instance
        symbol: Stock symbol to use for testing
        days: Number of days of data

    Returns:
        Dictionary with benchmark results
    """
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    # Fetch data multiple times to test indicator calculation performance
    iterations = 5
    results = []

    for i in range(iterations):
        with time_block(f"indicators_{symbol}_iter_{i}", {"iteration": i}):
            data = await pipeline.fetch_and_process_data(symbol, start_date, end_date)

        if data:
            results.append({
                "iteration": i,
                "indicators": data.technical_indicators,
                "indicator_count": len(data.technical_indicators)
            })

    return {
        "iterations": iterations,
        "successful_iterations": len([r for r in results if r["indicator_count"] > 0]),
        "avg_indicators_per_run": sum(r["indicator_count"] for r in results) / len(results) if results else 0,
        "results": results
    }


@measure_data_operation("cache_performance")
async def benchmark_cache_performance(pipeline: DataPipeline, symbol: str = "AAPL", days: int = 30) -> Dict[str, Any]:
    """
    Benchmark cache performance by measuring hit/miss ratios.

    Args:
        pipeline: Data pipeline instance
        symbol: Stock symbol to use for testing
        days: Number of days of data

    Returns:
        Dictionary with cache performance results
    """
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    # First run - should be cache misses
    logger.info("First run (cache misses expected)")
    await pipeline.fetch_and_process_data(symbol, start_date, end_date)

    # Second run - should be cache hits
    logger.info("Second run (cache hits expected)")
    await pipeline.fetch_and_process_data(symbol, start_date, end_date)

    # Third run with different date range - should be cache miss
    logger.info("Third run with different date range (cache miss expected)")
    start_date_new = datetime.now() - timedelta(days=days + 10)
    await pipeline.fetch_and_process_data(symbol, start_date_new, end_date)

    return {"cache_test_completed": True}


async def run_comprehensive_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive data pipeline benchmarking.

    Returns:
        Dictionary with all benchmark results
    """
    logger.info("Starting comprehensive data pipeline benchmarking")

    # Initialize components
    provider = YFinanceProvider()
    cache = CacheManager()  # Always enable cache for performance testing
    pipeline = DataPipeline(provider, cache)

    # Test symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    # Reset metrics
    reset_performance_metrics()

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_start": time.time(),
        "tests": {}
    }

    try:
        # Benchmark data fetching
        logger.info("Benchmarking data fetching performance")
        fetch_results = await benchmark_data_fetching(pipeline, test_symbols[:2], days=30)  # Shorter period for speed
        results["tests"]["data_fetching"] = fetch_results

        # Benchmark indicator calculation
        logger.info("Benchmarking indicator calculation performance")
        indicator_results = await benchmark_indicator_calculation(pipeline, "AAPL", days=60)
        results["tests"]["indicator_calculation"] = indicator_results

        # Benchmark cache performance
        if cache:
            logger.info("Benchmarking cache performance")
            cache_results = await benchmark_cache_performance(pipeline, "AAPL", days=30)
            results["tests"]["cache_performance"] = cache_results
        else:
            results["tests"]["cache_performance"] = {"skipped": "Cache not enabled"}

        # Get performance summary
        performance_summary = get_performance_summary()
        results["performance_summary"] = performance_summary

        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]
        results["status"] = "completed"

        logger.info("Data pipeline benchmarking completed", duration=results["total_duration"])

    except Exception as e:
        logger.error("Error during benchmarking", error=str(e))
        results["status"] = "failed"
        results["error"] = str(e)
        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]

    return results


async def main():
    """Main entry point for the profiling script."""
    configure_logging("INFO")

    logger.info("Starting Data Pipeline Performance Profiling")

    results = await run_comprehensive_benchmark()

    # Log performance report
    log_performance_report()

    # Save results to file
    output_file = f"data_pipeline_profile_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Benchmarking results saved", output_file=output_file)

    # Print summary
    print("\n=== Data Pipeline Performance Profile ===")
    print(f"Total Duration: {results['total_duration']:.2f} seconds")
    print(f"Status: {results['status']}")

    if "performance_summary" in results:
        summary = results["performance_summary"]
        print(f"\nOperations profiled: {len(summary)}")

        for operation, metrics in summary.items():
            print(f"\n{operation}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Avg Duration: {metrics['avg_duration']:.4f}s")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
