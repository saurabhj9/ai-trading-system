#!/usr/bin/env python3
"""
Orchestrator Workflow Performance Profiling Script.

This script benchmarks orchestrator workflow performance including:
- Sequential execution time and idle periods
- Potential for parallel execution of independent agents
- State management overhead between agents
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from src.config.settings import settings
from src.agents.data_structures import AgentConfig
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.risk import RiskManagementAgent
from src.agents.portfolio import PortfolioManagementAgent
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.cache import CacheManager
from src.llm.client import LLMClient
from src.communication.message_bus import MessageBus
from src.communication.state_manager import StateManager
from src.communication.orchestrator import Orchestrator
from src.utils.performance import (
    time_function, time_block, measure_llm_call,
    get_performance_summary, log_performance_report, reset_performance_metrics
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@time_function("orchestrator_benchmark")
async def benchmark_orchestrator_run(
    orchestrator: Orchestrator,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark orchestrator run performance.

    Args:
        orchestrator: Orchestrator instance to benchmark
        symbol: Stock symbol to analyze
        start_date: Start date for data
        end_date: End date for data
        iterations: Number of times to run the benchmark

    Returns:
        Dictionary with benchmark results
    """
    results = []

    for i in range(iterations):
        with time_block(f"orchestrator_run_iter_{i}", {
            "iteration": i,
            "symbol": symbol
        }):
            try:
                start_time = time.time()
                final_state = await orchestrator.run(symbol, start_date, end_date)
                end_time = time.time()

                final_decision = final_state.get("final_decision")
                results.append({
                    "iteration": i,
                    "success": True,
                    "total_time": end_time - start_time,
                    "final_decision": final_decision.signal if final_decision else None,
                    "error": final_state.get("error", "")
                })
            except Exception as e:
                results.append({
                    "iteration": i,
                    "success": False,
                    "error": str(e)
                })

    successful_runs = [r for r in results if r["success"]]
    return {
        "symbol": symbol,
        "iterations": iterations,
        "successful_runs": len(successful_runs),
        "success_rate": len(successful_runs) / iterations,
        "avg_total_time": sum(r["total_time"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
        "results": results
    }


async def benchmark_parallel_vs_sequential(
    agents: Dict[str, Any],
    market_data: Any
) -> Dict[str, Any]:
    """
    Compare parallel vs sequential execution of independent agents.

    Args:
        agents: Dictionary of agent instances
        market_data: Market data to analyze

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # Sequential execution (current implementation)
    with time_block("sequential_execution", {"type": "sequential"}):
        seq_start = time.time()

        with time_block("seq_technical", {"agent": "technical"}):
            tech_decision = await agents["technical"].analyze(market_data)

        with time_block("seq_sentiment", {"agent": "sentiment"}):
            sent_decision = await agents["sentiment"].analyze(market_data)

        seq_end = time.time()
        seq_total = seq_end - seq_start

    # Parallel execution (potential optimization)
    with time_block("parallel_execution", {"type": "parallel"}):
        par_start = time.time()

        # Run technical and sentiment in parallel
        tech_task = asyncio.create_task(agents["technical"].analyze(market_data))
        sent_task = asyncio.create_task(agents["sentiment"].analyze(market_data))

        tech_decision_par, sent_decision_par = await asyncio.gather(tech_task, sent_task)

        par_end = time.time()
        par_total = par_end - par_start

    results["sequential"] = {
        "total_time": seq_total,
        "technical_decision": tech_decision.signal,
        "sentiment_decision": sent_decision.signal
    }

    results["parallel"] = {
        "total_time": par_total,
        "technical_decision": tech_decision_par.signal,
        "sentiment_decision": sent_decision_par.signal,
        "speedup": seq_total / par_total if par_total > 0 else 0
    }

    return results


async def benchmark_state_management_overhead(
    orchestrator: Orchestrator,
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Profile state management overhead in orchestrator workflow.

    Args:
        orchestrator: Orchestrator instance
        symbol: Stock symbol
        start_date: Start date
        end_date: End date

    Returns:
        Dictionary with state management profiling results
    """
    # We'll measure the time spent in state operations by instrumenting the workflow
    # For now, we'll run the orchestrator and analyze the performance metrics

    with time_block("state_management_profile", {"operation": "full_workflow"}):
        final_state = await orchestrator.run(symbol, start_date, end_date)

    # Get performance summary to analyze state operations
    performance_summary = get_performance_summary()

    # Look for state-related operations
    state_operations = {k: v for k, v in performance_summary.items()
                       if any(word in k.lower() for word in ['state', 'dict', 'update'])}

    return {
        "workflow_completed": final_state.get("final_decision") is not None,
        "state_operations": state_operations,
        "total_state_time": sum(op.get("total_duration", 0) for op in state_operations.values())
    }


async def initialize_orchestrator() -> Orchestrator:
    """
    Initialize orchestrator with all dependencies.

    Returns:
        Configured Orchestrator instance
    """
    # Initialize dependencies
    llm_client = LLMClient()
    message_bus = MessageBus()
    state_manager = StateManager()

    # Initialize data pipeline
    provider = YFinanceProvider()
    cache = CacheManager() if settings.data.CACHE_ENABLED else None
    data_pipeline = DataPipeline(provider, cache)

    # Create agent configurations
    agent_configs = {
        "technical": AgentConfig(
            name="technical_agent",
            model_name=settings.llm.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=500
        ),
        "sentiment": AgentConfig(
            name="sentiment_agent",
            model_name=settings.llm.DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=600
        ),
        "risk": AgentConfig(
            name="risk_agent",
            model_name=settings.llm.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=400
        ),
        "portfolio": AgentConfig(
            name="portfolio_agent",
            model_name=settings.llm.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=500
        )
    }

    # Initialize agents
    technical_agent = TechnicalAnalysisAgent(
        config=agent_configs["technical"],
        llm_client=llm_client,
        message_bus=message_bus,
        state_manager=state_manager
    )
    sentiment_agent = SentimentAnalysisAgent(
        config=agent_configs["sentiment"],
        llm_client=llm_client,
        message_bus=message_bus,
        state_manager=state_manager
    )
    risk_agent = RiskManagementAgent(
        config=agent_configs["risk"],
        llm_client=llm_client,
        message_bus=message_bus,
        state_manager=state_manager
    )
    portfolio_agent = PortfolioManagementAgent(
        config=agent_configs["portfolio"],
        llm_client=llm_client,
        message_bus=message_bus,
        state_manager=state_manager
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(
        data_pipeline=data_pipeline,
        technical_agent=technical_agent,
        sentiment_agent=sentiment_agent,
        risk_agent=risk_agent,
        portfolio_agent=portfolio_agent,
        state_manager=state_manager
    )

    return orchestrator


async def run_comprehensive_orchestrator_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive orchestrator workflow benchmarking.

    Returns:
        Dictionary with all benchmark results
    """
    logger.info("Starting comprehensive orchestrator workflow benchmarking")

    # Reset metrics
    reset_performance_metrics()

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_start": time.time(),
        "tests": {}
    }

    try:
        # Initialize orchestrator
        orchestrator = await initialize_orchestrator()

        # Test parameters
        symbols = ["AAPL", "GOOGL", "MSFT"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Benchmark orchestrator runs with different symbols
        logger.info("Benchmarking orchestrator runs with different symbols")
        orchestrator_results = {}
        for symbol in symbols:
            logger.info(f"Benchmarking orchestrator with symbol: {symbol}")
            symbol_results = await benchmark_orchestrator_run(
                orchestrator, symbol, start_date, end_date, iterations=2
            )
            orchestrator_results[symbol] = symbol_results
        results["tests"]["orchestrator_runs"] = orchestrator_results

        # Get market data for parallel vs sequential analysis
        provider = YFinanceProvider()
        cache = CacheManager() if settings.data.CACHE_ENABLED else None
        pipeline = DataPipeline(provider, cache)
        market_data = await pipeline.fetch_and_process_data("AAPL", start_date, end_date)

        if market_data:
            # Initialize agents for parallel comparison
            llm_client = LLMClient()
            message_bus = MessageBus()
            state_manager = StateManager()

            agent_configs = {
                "technical": AgentConfig(name="technical_agent", model_name=settings.llm.DEFAULT_MODEL, temperature=0.1, max_tokens=500),
                "sentiment": AgentConfig(name="sentiment_agent", model_name=settings.llm.DEFAULT_MODEL, temperature=0.2, max_tokens=600)
            }

            agents = {
                "technical": TechnicalAnalysisAgent(
                    config=agent_configs["technical"],
                    llm_client=llm_client,
                    message_bus=message_bus,
                    state_manager=state_manager
                ),
                "sentiment": SentimentAnalysisAgent(
                    config=agent_configs["sentiment"],
                    llm_client=llm_client,
                    message_bus=message_bus,
                    state_manager=state_manager
                )
            }

            # Benchmark parallel vs sequential
            logger.info("Analyzing parallel vs sequential execution potential")
            parallel_results = await benchmark_parallel_vs_sequential(agents, market_data)
            results["tests"]["parallel_vs_sequential"] = parallel_results

            # Benchmark state management overhead
            logger.info("Profiling state management overhead")
            state_results = await benchmark_state_management_overhead(
                orchestrator, "AAPL", start_date, end_date
            )
            results["tests"]["state_management"] = state_results

        # Get performance summary
        performance_summary = get_performance_summary()
        results["performance_summary"] = performance_summary

        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]
        results["status"] = "completed"

        logger.info("Orchestrator workflow benchmarking completed", duration=results["total_duration"])

    except Exception as e:
        logger.error("Error during orchestrator benchmarking", error=str(e))
        results["status"] = "failed"
        results["error"] = str(e)
        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]

    return results


async def main():
    """Main entry point for the orchestrator profiling script."""
    configure_logging("INFO")

    logger.info("Starting Orchestrator Workflow Performance Profiling")

    results = await run_comprehensive_orchestrator_benchmark()

    # Log performance report
    log_performance_report()

    # Save results to file
    output_file = f"orchestrator_profile_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Benchmarking results saved", output_file=output_file)

    # Print summary
    print("\n=== Orchestrator Workflow Performance Profile ===")
    print(f"Total Duration: {results['total_duration']:.2f} seconds")
    print(f"Status: {results['status']}")

    if "tests" in results and "orchestrator_runs" in results["tests"]:
        runs = results["tests"]["orchestrator_runs"]
        print("\nOrchestrator Run Performance:")
        for symbol, symbol_data in runs.items():
            print(f"  {symbol}: {symbol_data['success_rate']:.2%} success, {symbol_data['avg_total_time']:.2f}s avg time")

    if "tests" in results and "parallel_vs_sequential" in results["tests"]:
        par_seq = results["tests"]["parallel_vs_sequential"]
        print("\nParallel vs Sequential Analysis:")
        print(f"  Sequential: {par_seq['sequential']['total_time']:.4f}s")
        print(f"  Parallel: {par_seq['parallel']['total_time']:.4f}s")
        print(f"  Speedup: {par_seq['parallel'].get('speedup', 0):.2f}x")

    if "tests" in results and "state_management" in results["tests"]:
        state = results["tests"]["state_management"]
        print(f"\nState Management Overhead: {state.get('total_state_time', 0):.6f}s")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
