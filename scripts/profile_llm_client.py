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

from src.config.settings import settings
from src.llm.client import LLMClient
from src.utils.performance import (
    measure_llm_call, time_block, get_performance_summary,
    log_performance_report, reset_performance_metrics
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


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
        # Add other models as needed
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

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
