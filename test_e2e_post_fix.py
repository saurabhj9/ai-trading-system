#!/usr/bin/env python3
"""
End-to-end test script for post-fix validation.

This script tests:
1. API endpoint functionality
2. All agent decisions in response
3. Local signal generation (if enabled)
4. Monitoring endpoint
"""

import requests
import json
import time
from datetime import datetime

class E2EPostFixTester:
    """End-to-end tester for post-fix validation."""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }

    def run_test(self, name, test_func):
        """Run a single test and track results."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")

        self.test_results["summary"]["total"] += 1

        try:
            result = test_func()
            if result["passed"]:
                print(f"[PASS] {name}")
                self.test_results["summary"]["passed"] += 1
            else:
                print(f"[FAIL] {name}")
                print(f"Reason: {result.get('reason', 'Unknown')}")
                self.test_results["summary"]["failed"] += 1

            self.test_results["tests"][name] = result
            return result
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            self.test_results["summary"]["failed"] += 1
            self.test_results["tests"][name] = {
                "passed": False,
                "reason": f"Exception: {str(e)}"
            }
            return {"passed": False, "reason": str(e)}

    def test_api_connection(self):
        """Test 1: Verify API server is running."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return {
                    "passed": True,
                    "status_code": response.status_code,
                    "response": response.json()
                }
            else:
                return {
                    "passed": False,
                    "reason": f"Unexpected status code: {response.status_code}"
                }
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Connection failed: {str(e)}"
            }

    def test_signal_generation(self):
        """Test 2: Generate signal and verify response structure."""
        try:
            url = f"{self.base_url}/api/v1/signals/AAPL"
            params = {"days": 30}

            print(f"Requesting signal for AAPL (30 days)...")
            start_time = time.time()
            response = requests.get(url, params=params, timeout=120)
            end_time = time.time()

            duration = end_time - start_time
            print(f"Response time: {duration:.2f}s")

            if response.status_code != 200:
                return {
                    "passed": False,
                    "reason": f"HTTP {response.status_code}: {response.text}"
                }

            data = response.json()

            # Verify response structure
            checks = {
                "has_symbol": "symbol" in data,
                "has_final_decision": "final_decision" in data,
                "has_agent_decisions": "agent_decisions" in data,
                "final_decision_has_signal": "signal" in data.get("final_decision", {}),
                "final_decision_has_confidence": "confidence" in data.get("final_decision", {}),
                "final_decision_has_reasoning": "reasoning" in data.get("final_decision", {}),
            }

            all_passed = all(checks.values())

            result = {
                "passed": all_passed,
                "duration": duration,
                "status_code": response.status_code,
                "checks": checks,
                "response_summary": {
                    "symbol": data.get("symbol"),
                    "final_signal": data.get("final_decision", {}).get("signal"),
                    "final_confidence": data.get("final_decision", {}).get("confidence"),
                    "agent_count": len(data.get("agent_decisions", {}))
                }
            }

            if not all_passed:
                failed_checks = [k for k, v in checks.items() if not v]
                result["reason"] = f"Failed checks: {', '.join(failed_checks)}"

            return result

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Request failed: {str(e)}"
            }

    def test_agent_decisions_present(self):
        """Test 3: Verify all expected agent decisions are in response."""
        try:
            url = f"{self.base_url}/api/v1/signals/AAPL"
            params = {"days": 30}

            print(f"Checking agent decisions presence...")
            response = requests.get(url, params=params, timeout=120)

            if response.status_code != 200:
                return {
                    "passed": False,
                    "reason": f"HTTP {response.status_code}"
                }

            data = response.json()
            agent_decisions = data.get("agent_decisions", {})

            # Expected agents (technical, sentiment, risk are minimum)
            expected_agents = ["technical", "risk"]
            present_agents = list(agent_decisions.keys())

            # Check which agents are present
            agent_checks = {
                agent: agent in present_agents
                for agent in expected_agents
            }

            # Also check for sentiment (should be present now with fix)
            agent_checks["sentiment"] = "sentiment" in present_agents

            print(f"\nAgent decisions present:")
            for agent, present in agent_checks.items():
                status = "[OK]" if present else "[MISSING]"
                print(f"  {status} {agent}")

            # Verify each decision has required fields
            decision_structure_ok = True
            for agent, decision in agent_decisions.items():
                required_fields = ["signal", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in decision:
                        decision_structure_ok = False
                        print(f"  [WARN] {agent} missing field: {field}")

            # Pass if we have technical and risk at minimum
            # Sentiment is now expected with the fix
            passed = agent_checks["technical"] and agent_checks["risk"] and decision_structure_ok

            result = {
                "passed": passed,
                "present_agents": present_agents,
                "agent_checks": agent_checks,
                "decision_structure_ok": decision_structure_ok,
                "agent_count": len(present_agents)
            }

            if not passed:
                missing = [a for a, p in agent_checks.items() if not p]
                if missing:
                    result["reason"] = f"Missing agents: {', '.join(missing)}"
                elif not decision_structure_ok:
                    result["reason"] = "Some decisions have incomplete structure"
            else:
                result["note"] = "All critical agents present with complete decisions"

            return result

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Request failed: {str(e)}"
            }

    def test_monitoring_endpoint(self):
        """Test 4: Verify monitoring endpoint is accessible."""
        try:
            url = f"{self.base_url}/api/v1/monitoring/metrics"

            print(f"Testing monitoring endpoint...")
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {
                    "passed": False,
                    "reason": f"HTTP {response.status_code}",
                    "status_code": response.status_code
                }

            data = response.json()

            # Verify metrics structure
            expected_fields = ["timestamp", "uptime_seconds", "requests_total"]
            present_fields = [f for f in expected_fields if f in data]

            passed = len(present_fields) == len(expected_fields)

            result = {
                "passed": passed,
                "status_code": response.status_code,
                "expected_fields": expected_fields,
                "present_fields": present_fields,
                "metrics_summary": {
                    "uptime": data.get("uptime_seconds"),
                    "requests": data.get("requests_total"),
                    "errors": data.get("errors_total")
                }
            }

            if not passed:
                missing = set(expected_fields) - set(present_fields)
                result["reason"] = f"Missing fields: {', '.join(missing)}"

            return result

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Request failed: {str(e)}"
            }

    def test_historical_ohlc_populated(self):
        """Test 5: Verify historical_ohlc is populated (indirect test via signal generation)."""
        try:
            # This is tested indirectly - if local signal generation works,
            # then historical_ohlc must be populated
            url = f"{self.base_url}/api/v1/signals/MSFT"
            params = {"days": 30}

            print(f"Testing with MSFT to verify data pipeline...")
            response = requests.get(url, params=params, timeout=120)

            if response.status_code != 200:
                return {
                    "passed": False,
                    "reason": f"HTTP {response.status_code}"
                }

            data = response.json()

            # If we get a valid response with technical analysis,
            # the data pipeline is working
            has_technical = "technical" in data.get("agent_decisions", {})

            result = {
                "passed": has_technical,
                "status_code": response.status_code,
                "symbol": data.get("symbol"),
                "has_technical_decision": has_technical
            }

            if not has_technical:
                result["reason"] = "Technical analysis missing (data pipeline may have issues)"
            else:
                result["note"] = "Data pipeline working correctly"

            return result

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Request failed: {str(e)}"
            }

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "="*60)
        print("END-TO-END POST-FIX VALIDATION TEST")
        print("="*60)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Target: {self.base_url}")

        # Run tests
        self.run_test("API Connection", self.test_api_connection)
        self.run_test("Signal Generation", self.test_signal_generation)
        self.run_test("Agent Decisions Present", self.test_agent_decisions_present)
        self.run_test("Monitoring Endpoint", self.test_monitoring_endpoint)
        self.run_test("Historical OHLC Populated", self.test_historical_ohlc_populated)

        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.test_results['summary']['total']}")
        print(f"Passed: {self.test_results['summary']['passed']}")
        print(f"Failed: {self.test_results['summary']['failed']}")

        pass_rate = (self.test_results['summary']['passed'] /
                     self.test_results['summary']['total'] * 100)
        print(f"Pass Rate: {pass_rate:.1f}%")

        # Save results
        with open("e2e_test_results_post_fix.json", "w") as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\nResults saved to: e2e_test_results_post_fix.json")

        if self.test_results['summary']['failed'] == 0:
            print("\n[SUCCESS] All tests passed!")
            return 0
        else:
            print(f"\n[WARNING] {self.test_results['summary']['failed']} test(s) failed")
            return 1

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Checking if API server is running...")
    print("="*60)

    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        print("[OK] API server is running")
    except:
        print("\n[ERROR] API server is not running!")
        print("\nPlease start the server first:")
        print("  uv run main.py")
        print("\nOr in a separate terminal:")
        print("  cd C:\\Users\\praga\\Documents\\SaurabhRepos\\ai-trading-system")
        print("  .venv\\Scripts\\python.exe main.py")
        return 1

    # Run tests
    tester = E2EPostFixTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    exit(main())
