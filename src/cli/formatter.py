"""
Output formatting for different display modes.
"""
import json
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class OutputFormatter:
    """Format analysis results for display."""

    @staticmethod
    def _color_code_signal(signal: str) -> str:
        """Apply color coding to signals based on type."""
        if signal == "BUY":
            return "[green]BUY[/green]"
        elif signal == "SELL":
            return "[red]SELL[/red]"
        elif signal == "HOLD":
            return "[yellow]HOLD[/yellow]"
        elif signal == "BULLISH":
            return "[green]BULLISH[/green]"
        elif signal == "BEARISH":
            return "[red]BEARISH[/red]"
        elif signal == "NEUTRAL":
            return "[yellow]NEUTRAL[/yellow]"
        elif signal == "APPROVE":
            return "[green]APPROVE[/green]"
        elif signal == "REJECT":
            return "[red]REJECT[/red]"
        elif signal == "ERROR":
            return "[red]ERROR[/red]"
        else:
            # Fallback for any unknown signal type
            return f"[dim]{signal}[/dim]"

    @staticmethod
    def format_table(
        results: List[Dict[str, Any]],
        detailed: bool = False,
        skip_single_detail: bool = False,
    ) -> None:
        """
        Format results as a rich table with colors.

        Args:
            results: List of analysis results
            detailed: Show detailed information
        """
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return

        # Handle single result
        if len(results) == 1 and not skip_single_detail:
            OutputFormatter._format_single_detailed(results[0])
            return

        # Multiple results - compact table
        table = Table(title="Analysis Results", show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Signal", no_wrap=True)
        table.add_column("Confidence", justify="right")
        table.add_column("Summary", style="dim")

        for result in results:
            if "error" in result:
                table.add_row(
                    result["symbol"],
                    "[red]ERROR[/red]",
                    "-",
                    result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"],
                )
            else:
                signal = result["signal"]
                confidence = result["confidence"]

                # Color code signal using helper method
                signal_text = OutputFormatter._color_code_signal(signal)

                # Truncate reasoning
                reasoning = result["reasoning"]
                summary = reasoning[:50] + "..." if len(reasoning) > 50 else reasoning

                table.add_row(
                    result["symbol"],
                    signal_text,
                    f"{confidence:.1%}",
                    summary,
                )

        console.print(table)

    @staticmethod
    def format_detailed_table(results: List[Dict[str, Any]]) -> None:
        """
        Format results as a detailed comparison table with agent decisions.

        Args:
            results: List of analysis results
        """
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return

        # Create detailed comparison table
        table = Table(title="Agent Decision Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Technical", no_wrap=True)
        table.add_column("Sentiment", no_wrap=True)
        table.add_column("Risk", no_wrap=True)
        table.add_column("Portfolio", no_wrap=True)
        table.add_column("Signal", no_wrap=True)
        table.add_column("Confidence", justify="right")
        table.add_column("Summary", overflow="fold", style="dim", max_width=80)

        for result in results:
            if "error" in result:
                error_summary = result.get("error", "Error generating analysis")
                table.add_row(
                    result["symbol"],
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "[red]ERROR[/red]",
                    "-",
                    Text(error_summary, style="dim"),
                )
                continue

            # Extract agent decisions with defaults
            agent_decisions = result.get("agent_decisions", {})

            # Get signals for each agent
            technical_signal = agent_decisions.get("technical", {}).get("signal", "N/A")
            sentiment_signal = agent_decisions.get("sentiment", {}).get("signal", "N/A")
            risk_signal = agent_decisions.get("risk", {}).get("signal", "N/A")
            portfolio_signal = agent_decisions.get("portfolio", {}).get("signal", "N/A")

            # Color code each agent's signal
            technical_colored = OutputFormatter._color_code_signal(technical_signal) if technical_signal != "N/A" else "[dim]N/A[/dim]"
            sentiment_colored = OutputFormatter._color_code_signal(sentiment_signal) if sentiment_signal != "N/A" else "[dim]N/A[/dim]"
            risk_colored = OutputFormatter._color_code_signal(risk_signal) if risk_signal != "N/A" else "[dim]N/A[/dim]"
            portfolio_colored = OutputFormatter._color_code_signal(portfolio_signal) if portfolio_signal != "N/A" else "[dim]N/A[/dim]"

            # Color code final signal
            final_signal = result["signal"]
            final_signal_colored = OutputFormatter._color_code_signal(final_signal)

            reasoning = result.get("reasoning", "")
            summary_cell = Text(reasoning or "No summary available", style="dim")

            table.add_row(
                result["symbol"],
                technical_colored,
                sentiment_colored,
                risk_colored,
                portfolio_colored,
                final_signal_colored,
                f"{result['confidence']:.1%}",
                summary_cell,
            )

        console.print(table)

    @staticmethod
    def _format_single_detailed(result: Dict[str, Any]) -> None:
        """Format single result with detailed information."""
        if "error" in result:
            console.print(
                Panel(
                    f"[red]Error analyzing {result['symbol']}:[/red]\n{result['error']}",
                    title="[X] Analysis Error",
                    border_style="red",
                )
            )
            return

        symbol = result["symbol"]
        signal = result["signal"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        # Create signal display with color
        signal_display = OutputFormatter._color_code_signal(signal)
        if signal == "BUY":
            border_style = "green"
        elif signal == "SELL":
            border_style = "red"
        else:
            border_style = "yellow"

        # Main panel content
        content = f"""[bold]Signal:[/bold] {signal_display}    [bold]Confidence:[/bold] {confidence:.1%}

[bold]Analysis Period:[/bold] {result['analysis_period']['days']} days
({result['analysis_period']['start']} to {result['analysis_period']['end']})

[bold]Reasoning:[/bold]
{reasoning}
"""

        # Agent decisions table
        if result.get("agent_decisions"):
            agent_table = Table(show_header=True, box=None, padding=(0, 1))
            agent_table.add_column("Agent", style="cyan")
            agent_table.add_column("Signal")
            agent_table.add_column("Confidence", justify="right")

            for agent_name, decision in result["agent_decisions"].items():
                agent_signal = decision["signal"]

                # Color-code using helper method
                signal_text = OutputFormatter._color_code_signal(agent_signal)

                agent_table.add_row(
                    agent_name.capitalize(),
                    signal_text,
                    f"{decision['confidence']:.1%}",
                )

            content += "\n[bold]Agent Decisions:[/bold]\n"

            # Render table to string (workaround for nested rendering)
            from io import StringIO
            buffer = StringIO()
            temp_console = Console(file=buffer, force_terminal=True)
            temp_console.print(agent_table)
            content += buffer.getvalue()

        console.print(
            Panel(
                content,
                title=f"[*] {symbol} Analysis",
                subtitle=f"Updated: {result['timestamp'][:19]}",
                border_style=border_style,
            )
        )

    @staticmethod
    def format_json(results: List[Dict[str, Any]]) -> str:
        """
        Format results as JSON.

        Args:
            results: List of analysis results

        Returns:
            JSON string
        """
        if len(results) == 1:
            return json.dumps(results[0], indent=2)
        return json.dumps(results, indent=2)

    @staticmethod
    def format_csv(results: List[Dict[str, Any]]) -> str:
        """
        Format results as CSV.

        Args:
            results: List of analysis results

        Returns:
            CSV string
        """
        if not results:
            return ""

        # CSV header
        csv_lines = ["Symbol,Signal,Confidence,Reasoning,Timestamp"]

        for result in results:
            if "error" in result:
                csv_lines.append(
                    f"{result['symbol']},ERROR,0.0,\"{result['error']}\",{result['timestamp']}"
                )
            else:
                # Escape quotes in reasoning
                reasoning = result["reasoning"].replace('"', '""')
                csv_lines.append(
                    f"{result['symbol']},{result['signal']},{result['confidence']:.3f},\"{reasoning}\",{result['timestamp']}"
                )

        return "\n".join(csv_lines)

    @staticmethod
    def print_progress(message: str, emoji: str = "[*]") -> None:
        """Print a progress message."""
        console.print(f"{emoji} {message}", style="dim")

    @staticmethod
    def print_success(message: str) -> None:
        """Print a success message."""
        console.print(f"[OK] {message}", style="green")

    @staticmethod
    def print_error(message: str) -> None:
        """Print an error message."""
        console.print(f"[ERROR] {message}", style="red bold")

    @staticmethod
    def print_warning(message: str) -> None:
        """Print a warning message."""
        console.print(f"[WARN] {message}", style="yellow")
