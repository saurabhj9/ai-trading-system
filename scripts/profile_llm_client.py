#!/usr/bin/env python3
"""
LLM Client Performance Profiling Script.

This script benchmarks the LLM client performance including:
- Response times for different models
- Network latency
- Token usage correlation with performance
- Error handling and retry behavior
"""
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import json
from datetime import datetime

from src.config.settings import settings
from src.llm.client import LLMClient
from src.utils.performance import (
    measure_llm_call, time_block, get_performance_summary,
    log_performance_report, reset_performance_metrics
)
from src.utils.logging import configure_logging, get_logger
from src.agents.data_structures import MarketData

logger = get_logger(__name__)

# Golden dataset for quality benchmarking
GOLDEN_DATASET = {
    "bullish_technical": {
        "market_data": MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            timestamp=datetime.now(),
            ohlc={"open": 145.0, "high": 152.0, "low": 144.0, "close": 150.0},
            technical_indicators={
                "rsi": 75.0,
                "macd": {"signal": 2.5, "histogram": 0.8},
                "sma_20": 140.0,
                "sma_50": 135.0
            }
        ),
        "news_headlines": [
            "Apple reports record quarterly earnings",
            "Analysts upgrade AAPL stock rating",
            "Strong demand for iPhone drives Apple shares higher"
        ],
        "portfolio_state": {"equity": 100000.0, "cash": 50000.0},
        "expected_signals": {
            "technical": "BUY",
            "sentiment": "BULLISH",
            "risk": "APPROVE",
            "portfolio": "BUY"
        }
    },
    "bearish_technical": {
        "market_data": MarketData(
            symbol="TSLA",
            price=200.0,
            volume=500000,
            timestamp=datetime.now(),
            ohlc={"open": 210.0, "high": 212.0, "low": 195.0, "close": 200.0},
            technical_indicators={
                "rsi": 25.0,
                "macd": {"signal": -3.2, "histogram": -1.5},
                "sma_20": 220.0,
                "sma_50": 225.0
            }
        ),
        "news_headlines": [
            "Tesla faces production delays",
            "Competition intensifies in EV market",
            "Tesla stock drops on regulatory concerns"
        ],
        "portfolio_state": {"equity": 100000.0, "cash": 50000.0},
        "expected_signals": {
            "technical": "SELL",
            "sentiment": "BEARISH",
            "risk": "REJECT",
            "portfolio": "SELL"
        }
    },
    "neutral_scenario": {
        "market_data": MarketData(
            symbol="GOOGL",
            price=2800.0,
            volume=800000,
            timestamp=datetime.now(),
            ohlc={"open": 2790.0, "high": 2820.0, "low": 2780.0, "close": 2800.0},
            technical_indicators={
                "rsi": 55.0,
                "macd": {"signal": 0.5, "histogram": 0.1},
                "sma_20": 2790.0,
                "sma_50": 2780.0
            }
        ),
        "news_headlines": [
            "Google announces new AI features",
            "Alphabet quarterly results meet expectations",
            "Mixed analyst reactions to Google earnings"
        ],
        "portfolio_state": {"equity": 100000.0, "cash": 50000.0},
        "expected_signals": {
            "technical": "HOLD",
            "sentiment": "NEUTRAL",
            "risk": "APPROVE",
            "portfolio": "HOLD"
        }
    }
}


@measure_llm_call("anthropic/claude-3-haiku")
async def benchmark_llm_response_time(
    client: LLMClient,
    prompt: str,
    model: str = "anthropic/claude-3-haiku",
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark LLM response times for a given prompt.

    Args:
        client: LLM client instance
        prompt: Prompt to send to LLM
        model: Model to use
        iterations: Number of times to run the benchmark

    Returns:
        Dictionary with benchmark results
    """
    results = []

    for i in range(iterations):
        with time_block(f"llm_call_{model}_iter_{i}", {
            "iteration": i,
            "model": model,
            "prompt_length": len(prompt)
        }):
            try:
                response = await client.generate(model, prompt, "You are a helpful assistant.")
                usage = client.last_usage
                results.append({
                    "iteration": i,
                    "success": True,
                    "response_length": len(response),
                    "prompt_length": len(prompt),
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else None,
                        "completion_tokens": usage.completion_tokens if usage else None,
                        "total_tokens": usage.total_tokens if usage else None
                    } if usage else None
                })
            except Exception as e:
                results.append({
                    "iteration": i,
                    "success": False,
                    "error": str(e),
                    "prompt_length": len(prompt)
                })

    successful_runs = [r for r in results if r["success"]]
    usages = [r["usage"] for r in successful_runs if r["usage"]]
    return {
        "iterations": iterations,
        "successful_runs": len(successful_runs),
        "success_rate": len(successful_runs) / iterations,
        "avg_response_length": sum(r["response_length"] for r in successful_runs) / len(successful_runs) if successful_runs else 0,
        "avg_prompt_tokens": sum(u["prompt_tokens"] for u in usages) / len(usages) if usages else 0,
        "avg_completion_tokens": sum(u["completion_tokens"] for u in usages) / len(usages) if usages else 0,
        "avg_total_tokens": sum(u["total_tokens"] for u in usages) / len(usages) if usages else 0,
        "results": results
    }


async def benchmark_different_prompt_sizes(client: LLMClient, model: str = "anthropic/claude-3-haiku") -> Dict[str, Any]:
    """
    Benchmark LLM performance with different prompt sizes.

    Args:
        client: LLM client instance
        model: Model to use

    Returns:
        Dictionary with benchmark results
    """
    prompt_sizes = [100, 500, 1000, 2000]

    results = {}

    for size in prompt_sizes:
        # Create a prompt of approximately the desired size
        prompt = f"Please analyze this market data: {'A' * (size - 50)}"

        logger.info(f"Benchmarking prompt size: {size} characters")
        result = await benchmark_llm_response_time(client, prompt, model, iterations=2)
        results[f"size_{size}"] = result

    return results


async def benchmark_different_models(client: LLMClient) -> Dict[str, Any]:
    """
    Benchmark different LLM models if available.

    Args:
        client: LLM client instance

    Returns:
        Dictionary with benchmark results
    """
    models_to_test = [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        # Add more models as needed
    ]

    results = {}
    test_prompt = "Analyze the current market trend for AAPL stock. Provide a brief summary."

    for model in models_to_test:
        try:
            logger.info(f"Benchmarking model: {model}")
            result = await benchmark_llm_response_time(client, test_prompt, model, iterations=3)
            results[model] = result
        except Exception as e:
            logger.error(f"Failed to benchmark model {model}", error=str(e))
            results[model] = {"error": str(e)}

    return results


async def benchmark_error_handling(client: LLMClient) -> Dict[str, Any]:
    """
    Benchmark error handling and retry behavior.

    Args:
        client: LLM client instance

    Returns:
        Dictionary with error handling benchmark results
    """
    # Test with invalid model to trigger errors
    invalid_model = "invalid-model-name"

    start_time = time.time()
    try:
        await client.generate(invalid_model, "Test prompt", "Test system prompt")
        error_result = {"success": True, "unexpected": "Expected error but got success"}
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "duration": time.time() - start_time
        }

    return {"invalid_model_test": error_result}


async def benchmark_decision_quality(client: LLMClient) -> Dict[str, Any]:
    """
    Benchmark the quality of LLM decision-making using agent-like prompts.

    Args:
        client: LLM client instance

    Returns:
        Dictionary with quality benchmark results for each model and scenario
    """
    models_to_test = [
        "anthropic/claude-3-haiku",  # Baseline
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "x-ai/grok-4-fast",
        "deepseek/deepseek-v3.1-terminus",
        "openai/gpt-5-mini",
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash",
    ]

    results = {}

    for model in models_to_test:
        logger.info(f"Benchmarking decision quality for model: {model}")
        model_results = {}

        for scenario_name, scenario_data in GOLDEN_DATASET.items():
            logger.info(f"Testing scenario: {scenario_name}")
            scenario_results = {}

            # Technical Analysis Prompt
            technical_system = (
                "You are a specialized AI assistant for financial technical analysis. "
                "Your goal is to analyze the provided market data and technical indicators. "
                "Determine a trading signal (BUY, SELL, or HOLD) and a confidence score (0.0 to 1.0). "
                "You must provide your reasoning in a brief, data-driven explanation. "
                "Your final output must be a single JSON object with three keys: "
                "'signal', 'confidence', and 'reasoning'."
            )
            market_data = scenario_data["market_data"]
            technical_prompt = (
                f"Analyze the following market data for {market_data.symbol}:\n"
                f"- Current Price: {market_data.price}\n"
                f"- Trading Volume: {market_data.volume}\n"
                f"- OHLC: {market_data.ohlc}\n"
                f"- Technical Indicators: {market_data.technical_indicators}\n\n"
                "Based on this data, provide your trading signal, confidence, and reasoning "
                "as a single JSON object."
            )

            try:
                technical_response = await client.generate(model, technical_prompt, technical_system)
                scenario_results["technical"] = {
                    "response": technical_response,
                    "expected_signal": scenario_data["expected_signals"]["technical"]
                }
            except Exception as e:
                scenario_results["technical"] = {"error": str(e)}

            # Sentiment Analysis Prompt
            sentiment_system = (
                "You are a specialized AI assistant for financial sentiment analysis. "
                "Your goal is to analyze news headlines related to a stock and determine "
                "the market sentiment (BULLISH, BEARISH, or NEUTRAL). Provide a confidence "
                "score (0.0 to 1.0) and a brief, data-driven reasoning. Your final output "
                "must be a single JSON object with three keys: 'signal', 'confidence', "
                "and 'reasoning'."
            )
            headlines = scenario_data["news_headlines"]
            sentiment_prompt = (
                f"Analyze the sentiment from the following news headlines for {market_data.symbol}:\n"
                + "\n".join(f"- {headline}" for headline in headlines)
                + "\n\nBased on these headlines, provide your sentiment (BULLISH, BEARISH, or NEUTRAL), "
                "confidence, and reasoning as a single JSON object."
            )

            try:
                sentiment_response = await client.generate(model, sentiment_prompt, sentiment_system)
                scenario_results["sentiment"] = {
                    "response": sentiment_response,
                    "expected_signal": scenario_data["expected_signals"]["sentiment"]
                }
            except Exception as e:
                scenario_results["sentiment"] = {"error": str(e)}

            # Risk Management Prompt (simplified)
            risk_system = (
                "You are a specialized AI assistant for financial risk management. "
                "Your goal is to assess the risk of a proposed trade. Analyze the "
                "provided market data and portfolio state. "
                "Determine a risk assessment signal (APPROVE or REJECT), a confidence score (0.0 to 1.0), "
                "and provide data-driven reasoning. Your final output must be a single JSON object with three keys: "
                "'signal', 'confidence', and 'reasoning'."
            )
            portfolio_state = scenario_data["portfolio_state"]
            risk_prompt = (
                f"Assess the risk for a trade in {market_data.symbol} given the following:\n"
                f"- Market Data: Price=${market_data.price}, Volume={market_data.volume}\n"
                f"- Current Portfolio: {portfolio_state}\n\n"
                "Based on this data, provide your risk assessment (APPROVE or REJECT), "
                "confidence, and reasoning as a single JSON object."
            )

            try:
                risk_response = await client.generate(model, risk_prompt, risk_system)
                scenario_results["risk"] = {
                    "response": risk_response,
                    "expected_signal": scenario_data["expected_signals"]["risk"]
                }
            except Exception as e:
                scenario_results["risk"] = {"error": str(e)}

            # Portfolio Management Prompt (simplified)
            portfolio_system = (
                "You are a specialized AI assistant for portfolio management. Your goal is to "
                "synthesize inputs from technical and sentiment analysis. "
                "Consider the current portfolio state and make a final trading decision "
                "(BUY, SELL, or HOLD). Provide a confidence score (0.0 to 1.0) and "
                "data-driven reasoning. Your final output must be a single JSON object with "
                "three keys: 'signal', 'confidence', and 'reasoning'."
            )
            # Mock agent decisions for portfolio
            mock_decisions = {
                "technical": {"signal": scenario_results.get("technical", {}).get("response", "HOLD")},
                "sentiment": {"signal": scenario_results.get("sentiment", {}).get("response", "NEUTRAL")}
            }
            portfolio_prompt = (
                f"Synthesize the following analyses for {market_data.symbol} to make a final trade decision:\n"
                f"- Market Data: Current Price=${market_data.price}\n"
                f"- Agent Decisions: {json.dumps(mock_decisions)}\n"
                f"- Current Portfolio: {portfolio_state}\n\n"
                "Based on all available data, provide your final decision (BUY, SELL, or HOLD), "
                "confidence, and reasoning as a single JSON object."
            )

            try:
                portfolio_response = await client.generate(model, portfolio_prompt, portfolio_system)
                scenario_results["portfolio"] = {
                    "response": portfolio_response,
                    "expected_signal": scenario_data["expected_signals"]["portfolio"]
                }
            except Exception as e:
                scenario_results["portfolio"] = {"error": str(e)}

            model_results[scenario_name] = scenario_results

        results[model] = model_results

    return results


async def run_comprehensive_llm_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive LLM client benchmarking.

    Returns:
        Dictionary with all benchmark results
    """
    logger.info("Starting comprehensive LLM client benchmarking")

    # Initialize LLM client
    try:
        client = LLMClient()
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Failed to initialize LLM client: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

    # Reset metrics
    reset_performance_metrics()

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_start": time.time(),
        "tests": {}
    }

    try:
        # Benchmark response times
        logger.info("Benchmarking LLM response times")
        response_results = await benchmark_llm_response_time(
            client,
            "What is the current market sentiment for technology stocks?",
            iterations=5
        )
        results["tests"]["response_times"] = response_results

        # Benchmark different prompt sizes
        logger.info("Benchmarking different prompt sizes")
        prompt_size_results = await benchmark_different_prompt_sizes(client)
        results["tests"]["prompt_sizes"] = prompt_size_results

        # Benchmark different models
        logger.info("Benchmarking different models")
        model_results = await benchmark_different_models(client)
        results["tests"]["models"] = model_results

        # Benchmark error handling
        logger.info("Benchmarking error handling")
        error_results = await benchmark_error_handling(client)
        results["tests"]["error_handling"] = error_results

        # Benchmark decision quality
        logger.info("Benchmarking decision quality")
        quality_results = await benchmark_decision_quality(client)
        results["tests"]["decision_quality"] = quality_results

        # Get performance summary
        performance_summary = get_performance_summary()
        results["performance_summary"] = performance_summary

        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]
        results["status"] = "completed"

        logger.info("LLM client benchmarking completed", duration=results["total_duration"])

    except Exception as e:
        logger.error("Error during LLM benchmarking", error=str(e))
        results["status"] = "failed"
        results["error"] = str(e)
        results["benchmark_end"] = time.time()
        results["total_duration"] = results["benchmark_end"] - results["benchmark_start"]

    return results


async def main():
    """Main entry point for the LLM profiling script."""
    configure_logging("INFO")

    logger.info("Starting LLM Client Performance Profiling")

    results = await run_comprehensive_llm_benchmark()

    # Log performance report
    log_performance_report()

    # Save results to file
    output_file = f"llm_client_profile_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Benchmarking results saved", output_file=output_file)

    # Print summary
    print("\n=== LLM Client Performance Profile ===")
    print(f"Total Duration: {results['total_duration']:.2f} seconds")
    print(f"Status: {results['status']}")

    if "performance_summary" in results:
        summary = results["performance_summary"]
        llm_operations = {k: v for k, v in summary.items() if 'llm' in k.lower()}

        print(f"\nLLM Operations profiled: {len(llm_operations)}")

        for operation, metrics in llm_operations.items():
            print(f"\n{operation}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Avg Duration: {metrics['avg_duration']:.4f}s")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

    if "tests" in results and "response_times" in results["tests"]:
        rt = results["tests"]["response_times"]
        print(f"\nResponse Time Test Results:")
        print(f"  Iterations: {rt['iterations']}")
        print(f"  Success Rate: {rt['success_rate']:.2%}")
        print(f"  Avg Response Length: {rt['avg_response_length']:.0f} characters")
        if rt.get('avg_total_tokens'):
            print(f"  Avg Prompt Tokens: {rt['avg_prompt_tokens']:.0f}")
            print(f"  Avg Completion Tokens: {rt['avg_completion_tokens']:.0f}")
            print(f"  Avg Total Tokens: {rt['avg_total_tokens']:.0f}")

    if "tests" in results and "decision_quality" in results["tests"]:
        dq = results["tests"]["decision_quality"]
        print(f"\nDecision Quality Test Results:")
        print(f"  Models tested: {len(dq)}")
        print(f"  Scenarios tested: {len(GOLDEN_DATASET)}")
        print("  Quality results saved for manual review in the JSON output")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
