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
    def format_table(results: List[Dict[str, Any]], detailed: bool = False) -> None:
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
        if len(results) == 1:
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

                # Color code signal
                if signal == "BUY":
                    signal_text = "[green]BUY[/green]"
                elif signal == "SELL":
                    signal_text = "[red]SELL[/red]"
                else:
                    signal_text = "[yellow]HOLD[/yellow]"

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
        if signal == "BUY":
            signal_display = "[green]BUY[/green]"
            border_style = "green"
        elif signal == "SELL":
            signal_display = "[red]SELL[/red]"
            border_style = "red"
        else:
            signal_display = "[yellow]HOLD[/yellow]"
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

                # Color-code based on signal type and agent
                if agent_signal == "BUY":
                    signal_text = "[green]BUY[/green]"
                elif agent_signal == "SELL":
                    signal_text = "[red]SELL[/red]"
                elif agent_signal == "HOLD":
                    signal_text = "[yellow]HOLD[/yellow]"
                elif agent_signal == "BULLISH":
                    signal_text = "[green]BULLISH[/green]"
                elif agent_signal == "BEARISH":
                    signal_text = "[red]BEARISH[/red]"
                elif agent_signal == "NEUTRAL":
                    signal_text = "[yellow]NEUTRAL[/yellow]"
                elif agent_signal == "APPROVE":
                    signal_text = "[green]APPROVE[/green]"
                elif agent_signal == "REJECT":
                    signal_text = "[red]REJECT[/red]"
                else:
                    # Fallback for any unknown signal type
                    signal_text = f"[dim]{agent_signal}[/dim]"

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
