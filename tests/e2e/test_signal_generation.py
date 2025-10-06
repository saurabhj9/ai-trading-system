#!/usr/bin/env python3
"""
Test script to validate the AI Trading System signal generation with real data.

This script tests:
1. API endpoint functionality
2. Signal generation success
3. Response format correctness
4. Local Signal Generation Framework integration
5. Performance metrics collection
"""

import asyncio
import json
import requests
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import os


class SignalGenerationTester:
    """Test class for validating signal generation system."""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_symbol = "AAPL"
        self.test_days = 30
        self.server_process = None
        self.test_results = {
            "server_status": "unknown",
            "api_response": None,
            "validation_results": {},
            "performance_metrics": None,
            "issues_found": [],
            "recommendations": []
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("=" * 80)
        print("AI Trading System - Signal Generation Validation Test")
        print("=" * 80)
        print(f"Test started at: {datetime.now().isoformat()}")
        print(f"Testing symbol: {self.test_symbol}")
        print(f"Analysis period: {self.test_days} days")
        print()

        try:
            # Step 1: Check if server is running, start if needed
            await self._check_server_status()

            # Step 2: Make test API call
            await self._test_api_call()

            # Step 3: Validate response format
            await self._validate_response_format()

            # Step 4: Check Local Signal Generation Framework usage
            await self._check_local_signal_generation()

            # Step 5: Validate performance metrics
            await self._validate_performance_metrics()

            # Step 6: Generate final report
            self._generate_final_report()

        except Exception as e:
            print(f"ERROR during testing: {e}")
            self.test_results["issues_found"].append(f"Test execution error: {str(e)}")

        finally:
            # Clean up server if we started it
            if self.server_process:
                print("\nShutting down test server...")
                self.server_process.terminate()
                self.server_process.wait()

        return self.test_results

    async def _check_server_status(self):
        """Check if server is running, start if needed."""
        print("1. Checking server status...")

        try:
            # Try to connect to the server
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("   [OK] Server is already running")
                self.test_results["server_status"] = "running"
                return
        except requests.exceptions.RequestException:
            print("   [ERROR] Server is not running, starting it...")

        # Start the server
        try:
            print("   Starting server with 'uv run main.py'...")
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uv", "run", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start
            max_wait = 30  # seconds
            wait_interval = 2
            for i in range(0, max_wait, wait_interval):
                try:
                    response = requests.get(f"{self.base_url}/", timeout=2)
                    if response.status_code == 200:
                        print(f"   [OK] Server started successfully (took {i + wait_interval} seconds)")
                        self.test_results["server_status"] = "started"
                        return
                except requests.exceptions.RequestException:
                    pass

                print(f"   Waiting for server to start... ({i + wait_interval}/{max_wait}s)")
                await asyncio.sleep(wait_interval)

            raise Exception("Server failed to start within timeout period")

        except Exception as e:
            print(f"   [ERROR] Failed to start server: {e}")
            self.test_results["issues_found"].append(f"Server startup failed: {str(e)}")
            raise

    async def _test_api_call(self):
        """Make test API call to generate signal."""
        print("\n2. Testing API call for signal generation...")

        try:
            url = f"{self.base_url}/api/v1/signals/{self.test_symbol}"
            params = {"days": self.test_days}

            print(f"   Making request to: {url}")
            print(f"   Parameters: {params}")

            start_time = time.time()
            response = requests.get(url, params=params, timeout=60)
            end_time = time.time()

            print(f"   Response status: {response.status_code}")
            print(f"   Response time: {end_time - start_time:.2f} seconds")

            if response.status_code == 200:
                print("   [OK] API call successful")
                self.test_results["api_response"] = response.json()
                self.test_results["response_time"] = end_time - start_time
            else:
                print(f"   [ERROR] API call failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results["issues_found"].append(
                    f"API call failed: HTTP {response.status_code} - {response.text}"
                )
                raise Exception(f"API call failed: HTTP {response.status_code}")

        except Exception as e:
            print(f"   ✗ Error making API call: {e}")
            self.test_results["issues_found"].append(f"API call error: {str(e)}")
            raise

    async def _validate_response_format(self):
        """Validate that response contains all expected fields."""
        print("\n3. Validating response format...")

        if not self.test_results["api_response"]:
            print("   ✗ No API response to validate")
            return

        response = self.test_results["api_response"]
        validation_results = {}

        # Check top-level fields
        required_fields = ["symbol", "analysis_period", "final_decision", "agent_decisions"]
        for field in required_fields:
            if field in response:
                print(f"   ✓ Found required field: {field}")
                validation_results[f"has_{field}"] = True
            else:
                print(f"   ✗ Missing required field: {field}")
                validation_results[f"has_{field}"] = False
                self.test_results["issues_found"].append(f"Missing required field: {field}")

        # Check final_decision structure
        if "final_decision" in response:
            final_decision = response["final_decision"]
            decision_fields = ["signal", "confidence", "reasoning", "timestamp"]
            for field in decision_fields:
                if field in final_decision:
                    print(f"   ✓ Found final_decision field: {field}")
                    validation_results[f"has_final_decision_{field}"] = True
                else:
                    print(f"   ✗ Missing final_decision field: {field}")
                    validation_results[f"has_final_decision_{field}"] = False
                    self.test_results["issues_found"].append(f"Missing final_decision field: {field}")

            # Validate signal value
            if "signal" in final_decision:
                signal = final_decision["signal"]
                if signal in ["BUY", "SELL", "HOLD"]:
                    print(f"   ✓ Valid signal value: {signal}")
                    validation_results["valid_signal_value"] = True
                else:
                    print(f"   ✗ Invalid signal value: {signal}")
                    validation_results["valid_signal_value"] = False
                    self.test_results["issues_found"].append(f"Invalid signal value: {signal}")

            # Validate confidence range
            if "confidence" in final_decision:
                confidence = final_decision["confidence"]
                if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                    print(f"   ✓ Valid confidence value: {confidence}")
                    validation_results["valid_confidence_value"] = True
                else:
                    print(f"   ✗ Invalid confidence value: {confidence}")
                    validation_results["valid_confidence_value"] = False
                    self.test_results["issues_found"].append(f"Invalid confidence value: {confidence}")

        # Check agent_decisions structure
        if "agent_decisions" in response:
            agent_decisions = response["agent_decisions"]
            expected_agents = ["technical", "sentiment", "risk", "portfolio"]

            for agent in expected_agents:
                if agent in agent_decisions:
                    print(f"   ✓ Found agent decision: {agent}")
                    validation_results[f"has_agent_{agent}"] = True

                    # Check agent decision structure
                    agent_decision = agent_decisions[agent]
                    agent_fields = ["signal", "confidence", "reasoning", "timestamp"]
                    for field in agent_fields:
                        if field in agent_decision:
                            validation_results[f"has_agent_{agent}_{field}"] = True
                        else:
                            print(f"   ✗ Missing {agent} field: {field}")
                            validation_results[f"has_agent_{agent}_{field}"] = False
                            self.test_results["issues_found"].append(f"Missing {agent} field: {field}")
                else:
                    print(f"   ✗ Missing agent decision: {agent}")
                    validation_results[f"has_agent_{agent}"] = False
                    self.test_results["issues_found"].append(f"Missing agent decision: {agent}")

        self.test_results["validation_results"] = validation_results

    async def _check_local_signal_generation(self):
        """Check if Local Signal Generation Framework is being used."""
        print("\n4. Checking Local Signal Generation Framework usage...")

        if not self.test_results["api_response"]:
            print("   [ERROR] No API response to check")
            return

        response = self.test_results["api_response"]
        local_generation_detected = False

        # Check technical agent for local signal generation indicators
        if "agent_decisions" in response and "technical" in response["agent_decisions"]:
            technical_decision = response["agent_decisions"]["technical"]

            # Check for signal source in supporting data
            if "supporting_data" in technical_decision:
                supporting_data = technical_decision["supporting_data"]

                if "signal_source" in supporting_data:
                    signal_source = supporting_data["signal_source"]
                    print(f"   ✓ Signal source detected: {signal_source}")

                    if signal_source == "LOCAL":
                        print("   ✓ Local Signal Generation Framework is being used!")
                        local_generation_detected = True
                    elif signal_source == "LLM":
                        print("   ⚠ LLM-based signal generation (Local framework not used)")
                        self.test_results["recommendations"].append(
                            "Consider enabling local signal generation for better performance"
                        )
                    elif signal_source == "ESCALATED":
                        print("   ⚠ Hybrid mode: Local signal escalated to LLM")
                        local_generation_detected = True
                        self.test_results["recommendations"].append(
                            "Hybrid mode detected - monitor escalation reasons"
                        )

                # Check for local generation metadata
                if "local_generation_metadata" in supporting_data:
                    print("   ✓ Local generation metadata found")
                    local_generation_detected = True

                    metadata = supporting_data["local_generation_metadata"]

                    # Check for performance metrics in metadata
                    if "performance_metrics" in metadata:
                        print("   ✓ Performance metrics found in local generation")
                        self.test_results["performance_metrics"] = metadata["performance_metrics"]

                    # Check for market regime detection
                    if "market_regime" in metadata:
                        regime = metadata["market_regime"]
                        print(f"   ✓ Market regime detected: {regime.get('regime', 'Unknown')}")
                        print(f"     Confidence: {regime.get('confidence', 0):.2f}")

                    # Check for escalation info
                    if "escalation" in metadata:
                        escalation = metadata["escalation"]
                        if escalation.get("required", False):
                            print(f"   ⚠ Escalation required: {escalation.get('reasoning', 'No reason')}")
                            self.test_results["recommendations"].append(
                                f"Escalation triggered: {escalation.get('reasoning', 'No reason')}"
                            )

                # Check for signal strength and market regime
                if "signal_strength" in supporting_data:
                    print(f"   ✓ Signal strength: {supporting_data['signal_strength']}")

                if "market_regime" in supporting_data:
                    print(f"   ✓ Market regime: {supporting_data['market_regime']}")

                # Check for indicators
                if "indicators" in supporting_data:
                    indicators = supporting_data["indicators"]
                    print(f"   ✓ Technical indicators calculated: {list(indicators.keys())}")

        if not local_generation_detected:
            print("   ⚠ Local Signal Generation Framework not detected")
            self.test_results["recommendations"].append(
                "Local Signal Generation Framework may not be enabled. "
                "Check SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED setting."
            )

        self.test_results["local_generation_detected"] = local_generation_detected

    async def _validate_performance_metrics(self):
        """Validate that performance metrics are being collected."""
        print("\n5. Validating performance metrics...")

        metrics_found = False

        # Check metrics in API response
        if self.test_results["api_response"]:
            response = self.test_results["api_response"]

            # Check for performance metrics in agent decisions
            if "agent_decisions" in response:
                for agent_name, agent_decision in response["agent_decisions"].items():
                    if "supporting_data" in agent_decision:
                        supporting_data = agent_decision["supporting_data"]

                        # Check for local generation metrics
                        if "local_generation_metadata" in supporting_data:
                            metadata = supporting_data["local_generation_metadata"]
                            if "performance_metrics" in metadata:
                                print(f"   ✓ Performance metrics found in {agent_name} agent")
                                self._analyze_performance_metrics(metadata["performance_metrics"])
                                metrics_found = True

        # If no metrics found in response, try to get them from monitoring endpoint
        if not metrics_found:
            try:
                print("   Checking monitoring endpoint for metrics...")
                response = requests.get(f"{self.base_url}/api/v1/monitoring/metrics", timeout=10)

                if response.status_code == 200:
                    metrics = response.json()
                    print("   ✓ Performance metrics found in monitoring endpoint")
                    self._analyze_performance_metrics(metrics)
                    metrics_found = True
                else:
                    print(f"   ⚠ Monitoring endpoint returned status {response.status_code}")

            except Exception as e:
                print(f"   ⚠ Could not fetch metrics from monitoring endpoint: {e}")

        if not metrics_found:
            print("   ⚠ No performance metrics found")
            self.test_results["issues_found"].append("Performance metrics not being collected")
            self.test_results["recommendations"].append(
                "Enable performance metrics collection in settings"
            )

    def _analyze_performance_metrics(self, metrics: Dict[str, Any]):
        """Analyze and report on performance metrics."""
        if not metrics:
            return

        print("   Performance Metrics Analysis:")

        # Check for signal generation metrics
        if "total_signals_generated" in metrics:
            total_signals = metrics["total_signals_generated"]
            print(f"     - Total signals generated: {total_signals}")

        if "avg_generation_time" in metrics:
            avg_time = metrics["avg_generation_time"]
            print(f"     - Average generation time: {avg_time:.3f}s")

            # Check against target
            target_time = 0.1  # 100ms target from config
            if avg_time <= target_time:
                print(f"     ✓ Meets performance target (< {target_time}s)")
            else:
                print(f"     ⚠ Exceeds performance target (> {target_time}s)")

        # Check for local vs LLM metrics
        if "local_signals" in metrics and "llm_signals" in metrics:
            local_signals = metrics["local_signals"]
            llm_signals = metrics["llm_signals"]
            total = local_signals + llm_signals

            if total > 0:
                local_percentage = (local_signals / total) * 100
                print(f"     - Local signals: {local_signals} ({local_percentage:.1f}%)")
                print(f"     - LLM signals: {llm_signals} ({100 - local_percentage:.1f}%)")

                if local_percentage > 50:
                    print("     ✓ Local signal generation is primary")
                else:
                    print("     ⚠ LLM signal generation is primary")

        # Check for escalation metrics
        if "escalations" in metrics:
            escalations = metrics["escalations"]
            print(f"     - Escalations: {escalations}")

            if "total_signals_generated" in metrics and metrics["total_signals_generated"] > 0:
                escalation_rate = (escalations / metrics["total_signals_generated"]) * 100
                print(f"     - Escalation rate: {escalation_rate:.1f}%")

                if escalation_rate > 20:
                    print("     ⚠ High escalation rate detected")
                    self.test_results["recommendations"].append(
                        "High escalation rate - review escalation thresholds"
                    )

        # Check for error rates
        if "local_errors" in metrics and "local_signals" in metrics:
            if metrics["local_signals"] > 0:
                error_rate = (metrics["local_errors"] / metrics["local_signals"]) * 100
                print(f"     - Local signal error rate: {error_rate:.1f}%")

                if error_rate > 5:
                    print("     ⚠ High local signal error rate")
                    self.test_results["issues_found"].append(
                        f"High local signal error rate: {error_rate:.1f}%"
                    )

        self.test_results["performance_metrics"] = metrics

    def _generate_final_report(self):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("FINAL TEST REPORT")
        print("=" * 80)

        # Server status
        print(f"\nServer Status: {self.test_results['server_status']}")

        # API Response
        if self.test_results["api_response"]:
            print(f"\nAPI Response: ✓ Successful")
            if "response_time" in self.test_results:
                print(f"Response Time: {self.test_results['response_time']:.2f}s")
        else:
            print(f"\nAPI Response: ✗ Failed")

        # Signal Generation
        print(f"\nLocal Signal Generation: {'✓ Detected' if self.test_results.get('local_generation_detected') else '✗ Not Detected'}")

        # Validation Results
        validation_results = self.test_results.get("validation_results", {})
        if validation_results:
            total_checks = sum(1 for v in validation_results.values() if isinstance(v, bool))
            passed_checks = sum(1 for v in validation_results.values() if v is True)
            print(f"\nResponse Validation: {passed_checks}/{total_checks} checks passed")

        # Issues Found
        issues = self.test_results.get("issues_found", [])
        if issues:
            print(f"\nIssues Found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\nIssues Found: None ✓")

        # Recommendations
        recommendations = self.test_results.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\nRecommendations: None ✓")

        # Overall Status
        print(f"\nOverall Status: {'✓ PASS' if not issues else '⚠ ISSUES DETECTED'}")

        # Save detailed report to file
        report_file = f"signal_generation_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\nDetailed report saved to: {report_file}")
        print("=" * 80)


async def main():
    """Main test execution function."""
    tester = SignalGenerationTester()
    results = await tester.run_all_tests()

    # Exit with appropriate code
    if results.get("issues_found"):
        print("\nTest completed with issues. Check the report for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
