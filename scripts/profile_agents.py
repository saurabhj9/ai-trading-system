#!/usr/bin/env python3
"""
Agent Processing Performance Profiling Script.

This script benchmarks agent processing performance including:
- Individual agent execution times
- LLM call overhead within agents
- Response parsing performance
- Comparison across different agent types
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from src.config.settings import settings
from src.agents.data_structures import AgentConfig, MarketData, AgentDecision
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
from src.utils.performance import (
    time_function, time_block, measure_llm_call,
    get_performance_summary, log_performance_report, reset_performance_metrics
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@time_function("agent_benchmark")
async def benchmark_single_agent(
    agent_name: str,
    agent_instance: Any,
    market_data: MarketData,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark a single agent's performance.

    Args:
        agent_name: Name of the agent
        agent_instance: Agent instance to benchmark
        market_data: Market data to analyze
        iterations: Number of times to run the benchmark

    Returns:
        Dictionary with benchmark results
    """
    results = []

    for i in range(iterations):
        with time_block(f"{agent_name}_analysis_iter_{i}", {
            "agent": agent_name,
            "iteration": i,
            "symbol": market_data.symbol
        }):
            try:
                decision = await agent_instance.analyze(market_data)
                results.append({
                    "iteration": i,
                    "success": True,
                    "signal": decision.signal,
                    "confidence": decision.confidence,
                    "reasoning_length": len(decision.reasoning)
                })
            except Exception as e:
                results.append({
                    "iteration": i,
                    "success": False,
                    "error": str(e)
                })

    successful_runs = [r for r in results if r["success"]]
    return {
        "agent_name": agent_name,
        "iterations": iterations,
        "successful_runs": len(successful_runs),
        "success_rate": len(successful_runs) / iterations,
        "avg_confidence": sum(r["confidence"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
        "avg_reasoning_length": sum(r["reasoning_length"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
        "results": results
    }


async def benchmark_all_agents(
    agents: Dict[str, Any],
    market_data: MarketData,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark all agents with the same market data.

    Args:
        agents: Dictionary of agent instances
        market_data: Market data to analyze
        iterations: Number of times to run each benchmark

    Returns:
        Dictionary with benchmark results for all agents
    """
    results = {}

    for agent_name, agent_instance in agents.items():
        logger.info(f"Benchmarking agent: {agent_name}")
        agent_results = await benchmark_single_agent(agent_name, agent_instance, market_data, iterations)
        results[agent_name] = agent_results

    return results


async def benchmark_agent_with_different_symbols(
    agent_name: str,
    agent_instance: Any,
    symbols: List[str],
    iterations: int = 2
) -> Dict[str, Any]:
    """
    Benchmark an agent with different stock symbols.

    Args:
        agent_name: Name of the agent
        agent_instance: Agent instance to benchmark
        symbols: List of stock symbols to test
        iterations: Number of times to run each benchmark

    Returns:
        Dictionary with benchmark results
    """
    # Initialize data pipeline
    provider = YFinanceProvider()
    cache = CacheManager() if settings.data.CACHE_ENABLED else None
    pipeline = DataPipeline(provider, cache)

    results = {}

    for symbol in symbols:
        logger.info(f"Benchmarking {agent_name} with symbol: {symbol}")

        # Get market data for this symbol
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        market_data = await pipeline.fetch_and_process_data(symbol, start_date, end_date)

        if market_data:
            symbol_results = await benchmark_single_agent(
                f"{agent_name}_{symbol}",
                agent_instance,
                market_data,
                iterations
            )
            results[symbol] = symbol_results
        else:
            results[symbol] = {"error": f"Failed to fetch data for {symbol}"}

    return results


async def benchmark_agent_chain(
    agents: Dict[str, Any],
    market_data: MarketData
) -> Dict[str, Any]:
    """
    Benchmark the complete agent processing chain.

    Args:
        agents: Dictionary of agent instances
        market_data: Market data to analyze

    Returns:
        Dictionary with chain benchmark results
    """
    decisions = {}

    # Run agents in sequence (simulating orchestrator)
    with time_block("agent_chain_technical", {"stage": "technical"}):
        technical_decision = await agents["technical"].analyze(market_data)
        decisions["technical"] = technical_decision

    with time_block("agent_chain_sentiment", {"stage": "sentiment"}):
        sentiment_decision = await agents["sentiment"].analyze(market_data)
        decisions["sentiment"] = sentiment_decision

    with time_block("agent_chain_risk", {"stage": "risk"}):
        risk_decision = await agents["risk"].analyze(market_data, proposed_decisions=decisions, portfolio_state={})
        decisions["risk"] = risk_decision

    with time_block("agent_chain_portfolio", {"stage": "portfolio"}):
        portfolio_decision = await agents["portfolio"].analyze(market_data, agent_decisions=decisions, portfolio_state={})
        decisions["portfolio"] = portfolio_decision

    return {
        "chain_completed": True,
        "decisions": {
            "technical": technical_decision.signal,
            "sentiment": sentiment_decision.signal,
            "risk": risk_decision.signal,
            "portfolio": portfolio_decision.signal
        }
    }


async def initialize_agents() -> Dict[str, Any]:
    """
    Initialize all agent instances for benchmarking.

    Returns:
        Dictionary of agent instances
    """
    # Initialize dependencies
    llm_client = LLMClient()
    message_bus = MessageBus()
    state_manager = StateManager()

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
        ),
        "risk": RiskManagementAgent(
            config=agent_configs["risk"],
            llm_client=llm_client,
            message_bus=message_bus,
            state_manager=state_manager
        ),
        "portfolio": PortfolioManagementAgent(
            config=agent_configs["portfolio"],
            llm_client=llm_client,
            message_bus=message_bus,
            state_manager=state_manager
        )
    }

    return agents


async def run_comprehensive_agent_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive agent processing benchmarking.

    Returns:
        Dictionary with all benchmark results
    """
    logger.info("Starting comprehensive agent processing benchmarking")

    # Reset metrics
    reset_performance_metrics()

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_start": time.time(),
        "tests": {}
    }

    try:
        # Initialize agents
        agents = await initialize_agents()

        # Get test market data
        provider = YFinanceProvider()
        cache = CacheManager() if settings.data.CACHE_ENABLED else None
        pipeline = DataPipeline(provider, cache)

        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        market_data = await pipeline.fetch_and_process_data("AAPL", start_date, end_date)

        if not market_data:
            raise Exception("Failed to fetch test market data")

        # Benchmark individual agents
        logger.info("Benchmarking individual agent performance")
        individual_results = await benchmark_all_agents(agents, market_data, iterations=3)
        results["tests"]["individual_agents"] = individual_results

        # Benchmark agent with different symbols
        logger.info("Benchmarking technical agent with different symbols")
        symbol_results = await benchmark_agent_with_different_symbols(
            "technical",
            agents["technical"],
            ["AAPL", "GOOGL", "MSFT"],
            iterations=2
        )
        results["tests"]["different_symbols"] = symbol_results

        # Benchmark agent chain
        logger.info("Benchmarking complete agent processing chain")
        chain_results = await benchmark_agent_chain(agents, market_data)
        results["tests"]["agent_chain"] = chain_results

        # Get performance summary
        performance_summary = get_performance_summary()
        results["performance_summary"] = performance_summary

        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]
        results["status"] = "completed"

        logger.info("Agent processing benchmarking completed", duration=results["total_duration"])

    except Exception as e:
        logger.error("Error during agent benchmarking", error=str(e))
        results["status"] = "failed"
        results["error"] = str(e)
        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]

    return results


async def main():
    """Main entry point for the agent profiling script."""
    configure_logging("INFO")

    logger.info("Starting Agent Processing Performance Profiling")

    results = await run_comprehensive_agent_benchmark()

    # Log performance report
    log_performance_report()

    # Save results to file
    output_file = f"agent_profile_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Benchmarking results saved", output_file=output_file)

    # Print summary
    print("\n=== Agent Processing Performance Profile ===")
    print(f"Total Duration: {results['total_duration']:.2f} seconds")
    print(f"Status: {results['status']}")

    if "performance_summary" in results:
        summary = results["performance_summary"]
        agent_operations = {k: v for k, v in summary.items() if any(word in k.lower() for word in ['agent', 'technical', 'sentiment', 'risk', 'portfolio'])}

        print(f"\nAgent Operations profiled: {len(agent_operations)}")

        for operation, metrics in agent_operations.items():
            print(f"\n{operation}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Avg Duration: {metrics['avg_duration']:.4f}s")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

    if "tests" in results and "individual_agents" in results["tests"]:
        agents = results["tests"]["individual_agents"]
        print("\nIndividual Agent Performance:")
        for agent_name, agent_data in agents.items():
            print(f"  {agent_name}: {agent_data['success_rate']:.2%} success, {agent_data['avg_confidence']:.2f} avg confidence")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
