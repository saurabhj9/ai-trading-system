#!/usr/bin/env python3
"""
Performance Dashboard Generator
Compiles metrics from profiling JSON files and generates visualizations.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class PerformanceDashboard:
    def __init__(self, profile_dir: str = "."):
        self.profile_dir = Path(profile_dir)
        self.profiles = {}
        self.load_profiles()

    def load_profiles(self):
        """Load all profiling JSON files."""
        profile_files = [
            "agent_profile_1758879208.json",
            "agent_profile_1758879314.json",
            "data_pipeline_profile_1758878346.json",
            "llm_client_profile_1758878921.json",
            "orchestrator_profile_1758880681.json",
            "orchestrator_profile_1758880746.json",
            "orchestrator_profile_1758880843.json"
        ]

        for file in profile_files:
            file_path = self.profile_dir / file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        self.profiles[file] = json.load(f)
                    print(f"Loaded {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    def generate_summary_report(self) -> str:
        """Generate a text summary of all performance metrics."""
        report = ["# Performance Analysis Summary\n"]

        # Overall status
        report.append("## Overall Status\n")
        for name, profile in self.profiles.items():
            status = profile.get('status', 'unknown')
            duration = profile.get('total_duration', 0)
            report.append(f"- **{name}**: {status.upper()} ({duration:.2f}s)")

        # Key metrics compilation
        report.append("\n## Key Metrics\n")

        # LLM Performance
        llm_profile = self.profiles.get('llm_client_profile_1758878921.json', {})
        if llm_profile.get('status') == 'completed':
            perf_summary = llm_profile.get('performance_summary', {})
            claude_calls = perf_summary.get('llm_call_anthropic/claude-3-haiku', {})
            report.append("### LLM Client Performance")
            report.append(f"- Average response time: {claude_calls.get('avg_duration', 0):.2f}s")
            report.append(f"- Success rate: {claude_calls.get('success_rate', 0)*100:.1f}%")
            report.append(f"- Average tokens: {llm_profile.get('tests', {}).get('response_times', {}).get('avg_total_tokens', 0):.0f}")

        # Data Pipeline Performance
        data_profile = self.profiles.get('data_pipeline_profile_1758878346.json', {})
        if data_profile.get('status') == 'completed':
            perf_summary = data_profile.get('performance_summary', {})
            fetch_aapl = perf_summary.get('fetch_AAPL', {})
            fetch_googl = perf_summary.get('fetch_GOOGL', {})
            report.append("### Data Pipeline Performance")
            report.append(f"- AAPL fetch time: {fetch_aapl.get('avg_duration', 0):.2f}s")
            report.append(f"- GOOGL fetch time: {fetch_googl.get('avg_duration', 0):.2f}s")
            report.append(f"- Indicator calculation: {perf_summary.get('data_indicator_calculation', {}).get('avg_duration', 0):.4f}s")

        # Agent Performance
        agent_profile = self.profiles.get('agent_profile_1758879314.json', {})
        if 'tests' in agent_profile:
            tests = agent_profile['tests']
            report.append("### Agent Performance")
            for agent_name, agent_data in tests.get('individual_agents', {}).items():
                success_rate = agent_data.get('success_rate', 0)
                avg_confidence = agent_data.get('avg_confidence', 0)
                report.append(f"- **{agent_name.title()}**: {success_rate*100:.1f}% success, avg confidence: {avg_confidence:.2f}")

        # Orchestrator Performance
        orchestrator_profile = self.profiles.get('orchestrator_profile_1758880843.json', {})
        if orchestrator_profile.get('status') == 'completed':
            perf_summary = orchestrator_profile.get('performance_summary', {})
            benchmark = perf_summary.get('orchestrator_benchmark', {})
            parallel_vs_seq = orchestrator_profile.get('tests', {}).get('parallel_vs_sequential', {})
            report.append("### Orchestrator Performance")
            report.append(f"- Average benchmark time: {benchmark.get('avg_duration', 0):.2f}s")
            report.append(f"- Parallel speedup: {parallel_vs_seq.get('parallel', {}).get('speedup', 1):.2f}x")

        # Identified Bottlenecks
        report.append("\n## Identified Bottlenecks\n")
        report.append("1. **LLM Response Times**: ~5s average per call - highest latency component")
        report.append("2. **Data Fetching Inconsistency**: AAPL (~7.7s) vs GOOGL (~0.1s)")
        report.append("3. **Agent Serialization Issues**: 'model_dump' attribute errors preventing execution")
        report.append("4. **Risk/Portfolio Agent Failures**: Missing required arguments")
        report.append("5. **State Management Overhead**: ~5s in orchestrator workflow")

        return "\n".join(report)

    def create_visualizations(self):
        """Create performance visualization plots."""
        # Create output directory
        output_dir = self.profile_dir / "performance_charts"
        output_dir.mkdir(exist_ok=True)

        # Component timing comparison
        self._create_timing_chart(output_dir)

        # Success rates
        self._create_success_rate_chart(output_dir)

        # LLM token usage
        self._create_token_usage_chart(output_dir)

        print(f"Visualizations saved to {output_dir}")

    def _create_timing_chart(self, output_dir: Path):
        """Create component timing comparison chart."""
        components = []
        times = []

        # LLM timing
        llm_profile = self.profiles.get('llm_client_profile_1758878921.json', {})
        if llm_profile.get('status') == 'completed':
            perf = llm_profile.get('performance_summary', {}).get('llm_call_anthropic/claude-3-haiku', {})
            components.append('LLM Call')
            times.append(perf.get('avg_duration', 0))

        # Data pipeline timing
        data_profile = self.profiles.get('data_pipeline_profile_1758878346.json', {})
        if data_profile.get('status') == 'completed':
            perf = data_profile.get('performance_summary', {})
            components.append('Data Fetch')
            times.append(perf.get('data_fetch', {}).get('avg_duration', 0))
            components.append('Indicators')
            times.append(perf.get('data_indicator_calculation', {}).get('avg_duration', 0))

        # Orchestrator timing
        orch_profile = self.profiles.get('orchestrator_profile_1758880843.json', {})
        if orch_profile.get('status') == 'completed':
            perf = orch_profile.get('performance_summary', {})
            components.append('Orchestrator')
            times.append(perf.get('orchestrator_benchmark', {}).get('avg_duration', 0))

        if components and times:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(components, times, color=['red', 'blue', 'green', 'orange'])
            plt.title('Component Timing Comparison')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, time in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time:.2f}s', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / 'component_timing.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _create_success_rate_chart(self, output_dir: Path):
        """Create success rate comparison chart."""
        components = []
        rates = []

        # Agent success rates
        agent_profile = self.profiles.get('agent_profile_1758879314.json', {})
        if 'tests' in agent_profile:
            for agent_name, agent_data in agent_profile['tests'].get('individual_agents', {}).items():
                components.append(agent_name.title())
                rates.append(agent_data.get('success_rate', 0) * 100)

        # Orchestrator success
        orch_profile = self.profiles.get('orchestrator_profile_1758880843.json', {})
        if orch_profile.get('status') == 'completed':
            runs = orch_profile.get('tests', {}).get('orchestrator_runs', {})
            for symbol, data in runs.items():
                components.append(f'Orch-{symbol}')
                rates.append(data.get('success_rate', 0) * 100)

        if components and rates:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(components, rates, color='skyblue')
            plt.title('Component Success Rates')
            plt.ylabel('Success Rate (%)')
            plt.xticks(rotation=45)
            plt.ylim(0, 110)

            # Add value labels
            for bar, rate in zip(bars, rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / 'success_rates.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _create_token_usage_chart(self, output_dir: Path):
        """Create LLM token usage chart."""
        llm_profile = self.profiles.get('llm_client_profile_1758878921.json', {})
        if llm_profile.get('status') == 'completed':
            tests = llm_profile.get('tests', {})

            # Response times token usage
            response_times = tests.get('response_times', {})
            if response_times.get('results'):
                iterations = [r['iteration'] for r in response_times['results']]
                prompt_tokens = [r['usage']['prompt_tokens'] for r in response_times['results']]
                completion_tokens = [r['usage']['completion_tokens'] for r in response_times['results']]

                plt.figure(figsize=(10, 6))
                plt.bar(iterations, prompt_tokens, label='Prompt Tokens', alpha=0.7)
                plt.bar(iterations, completion_tokens, bottom=prompt_tokens, label='Completion Tokens', alpha=0.7)
                plt.title('LLM Token Usage by Iteration')
                plt.xlabel('Iteration')
                plt.ylabel('Token Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'token_usage.png', dpi=150, bbox_inches='tight')
                plt.close()

def main():
    dashboard = PerformanceDashboard()

    # Generate and print summary report
    report = dashboard.generate_summary_report()
    print(report)

    # Save report to file
    with open('performance_report.md', 'w') as f:
        f.write(report)
    print("\nReport saved to performance_report.md")

    # Create visualizations
    dashboard.create_visualizations()

if __name__ == "__main__":
    main()
