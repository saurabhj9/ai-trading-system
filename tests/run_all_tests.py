#!/usr/bin/env python3
"""
Comprehensive Test Runner for AI Trading System

This script provides a unified interface to run different categories of tests
with appropriate configuration and reporting.

Usage:
    python tests/run_all_tests.py --category unit
    python tests/run_all_tests.py --category all
    python tests/run_all_tests.py --coverage
    python tests/run_all_tests.py --performance
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json


class TestRunner:
    """Comprehensive test runner for the AI Trading System."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run a command and return results."""
        if cwd is None:
            cwd = self.project_root

        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd}")
        print("-" * 60)

        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        end_time = time.time()

        return {
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": end_time - start_time,
            "success": result.returncode == 0
        }

    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        cmd = ["uv", "run", "pytest", "tests/unit", "-m", "unit"]

        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80"
            ])

        return self.run_command(cmd)

    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        cmd = ["uv", "run", "pytest", "tests/integration", "-m", "integration"]

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd)

    def run_e2e_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run end-to-end tests."""
        cmd = ["uv", "run", "pytest", "tests/e2e", "-m", "e2e"]

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd)

    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        cmd = ["uv", "run", "pytest", "tests/performance", "-m", "performance", "-s"]

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd)

    def run_validation_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run validation tests."""
        cmd = ["uv", "run", "pytest", "tests/validation", "-m", "validation"]

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd)

    def run_comparison_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run comparison tests."""
        cmd = ["uv", "run", "pytest", "tests/comparison", "-m", "comparison"]

        if verbose:
            cmd.append("-v")

        return self.run_command(cmd)

    def run_all_tests(self, verbose: bool = False, coverage: bool = False) -> Dict[str, Any]:
        """Run all test categories in sequence."""
        results = {}

        # Run unit tests first
        print("Running Unit Tests...")
        results["unit"] = self.run_unit_tests(verbose=verbose, coverage=coverage)

        # Run integration tests
        print("\nRunning Integration Tests...")
        results["integration"] = self.run_integration_tests(verbose=verbose)

        # Run validation tests
        print("\nRunning Validation Tests...")
        results["validation"] = self.run_validation_tests(verbose=verbose)

        # Run comparison tests
        print("\nRunning Comparison Tests...")
        results["comparison"] = self.run_comparison_tests(verbose=verbose)

        # Run performance tests (separate as they take longer)
        print("\nRunning Performance Tests...")
        results["performance"] = self.run_performance_tests(verbose=verbose)

        # Run e2e tests last
        print("\nRunning End-to-End Tests...")
        results["e2e"] = self.run_e2e_tests(verbose=verbose)

        return results

    def run_quick_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run a quick subset of tests for development."""
        results = {}

        # Run only unit and integration tests for quick feedback
        print("Running Quick Tests (Unit + Integration)...")
        results["unit"] = self.run_unit_tests(verbose=verbose)
        results["integration"] = self.run_integration_tests(verbose=verbose)

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a test execution report."""
        total_duration = sum(result.get("duration", 0) for result in results.values())
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        report_lines = [
            "=" * 80,
            "AI TRADING SYSTEM - TEST EXECUTION REPORT",
            "=" * 80,
            f"Total Duration: {total_duration:.2f} seconds",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        for category, result in results.items():
            if result:
                # Parse pytest output to get test counts
                stdout = result.get("stdout", "")
                lines = stdout.split('\n')

                category_passed = 0
                category_failed = 0
                category_total = 0

                for line in lines:
                    # Look for the summary line with test counts
                    if " passed" in line and (" failed" in line or " deselected" in line or " warnings" in line):
                        # Parse line like "=============== 20 passed, 41 deselected, 3 warnings in 0.23s ================"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "passed" in part and i > 0:
                                # Extract the number before "passed"
                                passed_str = parts[i-1].replace('=', '').replace(',', '')
                                try:
                                    category_passed = int(passed_str)
                                except ValueError:
                                    pass
                            elif "failed" in part and i > 0:
                                # Extract the number before "failed"
                                failed_str = parts[i-1].replace('=', '').replace(',', '')
                                try:
                                    category_failed = int(failed_str)
                                except ValueError:
                                    pass

                category_total = category_passed + category_failed
                total_tests += category_total
                passed_tests += category_passed
                failed_tests += category_failed

                status = "PASSED" if result["success"] else "FAILED"
                report_lines.extend([
                    f"{category.upper()} TESTS: {status}",
                    f"  Duration: {result.get('duration', 0):.2f}s",
                    f"  Tests: {category_passed} passed, {category_failed} failed",
                    ""
                ])

        # Summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        overall_status = "PASSED" if failed_tests == 0 else "FAILED"

        report_lines.extend([
            "SUMMARY",
            "-" * 40,
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Success Rate: {success_rate:.1f}%",
            f"Overall Status: {overall_status}",
            "=" * 80
        ])

        return "\n".join(report_lines)

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        output_path = self.project_root / "test_results" / filename
        output_path.parent.mkdir(exist_ok=True)

        # Prepare results for JSON serialization
        json_results = {}
        for category, result in results.items():
            if result:
                json_results[category] = {
                    "command": result["command"],
                    "returncode": result["returncode"],
                    "duration": result["duration"],
                    "success": result["success"],
                    "stdout": result["stdout"][:1000] + "..." if len(result["stdout"]) > 1000 else result["stdout"],
                    "stderr": result["stderr"][:1000] + "..." if len(result["stderr"]) > 1000 else result["stderr"]
                }

        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration": sum(result.get("duration", 0) for result in results.values()),
                "results": json_results
            }, f, indent=2)

        return str(output_path)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="AI Trading System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --category unit                    # Run unit tests only
  %(prog)s --category all                     # Run all tests
  %(prog)s --category all --coverage          # Run all tests with coverage
  %(prog)s --category quick                   # Run quick tests (unit + integration)
  %(prog)s --category performance             # Run performance tests only
  %(prog)s --category unit --verbose          # Run unit tests with verbose output
        """
    )

    parser.add_argument(
        "--category",
        choices=["unit", "integration", "e2e", "performance", "validation", "comparison", "all", "quick"],
        default="quick",
        help="Test category to run (default: quick)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report (only for unit tests)"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to JSON file"
    )

    parser.add_argument(
        "--output-file",
        help="Custom output file for test results"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner()
    runner.start_time = time.time()

    print("AI Trading System Test Runner")
    print("=" * 50)
    print(f"Category: {args.category}")
    print(f"Verbose: {args.verbose}")
    print(f"Coverage: {args.coverage}")
    print()

    # Run tests based on category
    try:
        if args.category == "unit":
            results = {"unit": runner.run_unit_tests(verbose=args.verbose, coverage=args.coverage)}
        elif args.category == "integration":
            results = {"integration": runner.run_integration_tests(verbose=args.verbose)}
        elif args.category == "e2e":
            results = {"e2e": runner.run_e2e_tests(verbose=args.verbose)}
        elif args.category == "performance":
            results = {"performance": runner.run_performance_tests(verbose=args.verbose)}
        elif args.category == "validation":
            results = {"validation": runner.run_validation_tests(verbose=args.verbose)}
        elif args.category == "comparison":
            results = {"comparison": runner.run_comparison_tests(verbose=args.verbose)}
        elif args.category == "all":
            results = runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)
        elif args.category == "quick":
            results = runner.run_quick_tests(verbose=args.verbose)
        else:
            print(f"Unknown category: {args.category}")
            sys.exit(1)

        # Generate and display report
        report = runner.generate_report(results)
        print(report)

        # Save results if requested
        if args.save_results:
            output_file = runner.save_results(results, args.output_file)
            print(f"\nTest results saved to: {output_file}")

        # Exit with appropriate code
        failed_tests = any(not result.get("success", False) for result in results.values() if result)
        sys.exit(1 if failed_tests else 0)

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
