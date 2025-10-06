#!/usr/bin/env python3
"""
AI Trading System - Command Line Interface

A professional CLI for analyzing stocks without needing to start a web server.

Usage:
    uv run cli.py AAPL
    uv run cli.py AAPL GOOGL MSFT --days 60
    uv run cli.py AAPL --watch --interval 300
    uv run cli.py --watchlist stocks.txt --format table

Examples:
    # Basic analysis
    uv run cli.py AAPL

    # Multiple symbols
    uv run cli.py AAPL GOOGL MSFT

    # Custom time period
    uv run cli.py AAPL --days 90

    # JSON output
    uv run cli.py AAPL --format json

    # Save to file
    uv run cli.py AAPL --output analysis.json

    # Watch mode (continuous monitoring)
    uv run cli.py AAPL --watch --interval 300

    # Analyze from watchlist file
    uv run cli.py --watchlist my_stocks.txt
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.cli.analyzer import StockAnalyzer
from src.cli.formatter import OutputFormatter
from src.utils.cli_logging import configure_cli_logging

# Load environment variables
load_dotenv()

console = Console()
formatter = OutputFormatter()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trading System - Stock Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                              # Analyze Apple stock
  %(prog)s AAPL GOOGL MSFT                   # Analyze multiple stocks
  %(prog)s AAPL --days 90                    # Custom time period
  %(prog)s AAPL --format json                # JSON output
  %(prog)s AAPL --output report.json         # Save to file
  %(prog)s AAPL --watch --interval 300       # Watch mode (5 min updates)
  %(prog)s --watchlist stocks.txt            # Analyze from file

For more information, visit: https://github.com/your-username/ai-trading-system
        """,
    )

    parser.add_argument(
        "symbols",
        nargs="*",
        help="Stock symbols to analyze (e.g., AAPL GOOGL MSFT)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to analyze (default: 30)",
    )

    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save output to file",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Enable continuous monitoring mode",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Update interval for watch mode in seconds (default: 300)",
    )

    parser.add_argument(
        "--watchlist",
        "-w",
        type=str,
        help="File containing list of symbols (one per line)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output with detailed information",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Save detailed logs to file",
    )

    return parser.parse_args()


def load_watchlist(filepath: str) -> List[str]:
    """
    Load symbols from a watchlist file.

    Args:
        filepath: Path to watchlist file

    Returns:
        List of symbols
    """
    try:
        with open(filepath, "r") as f:
            symbols = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Support format: SYMBOL or SYMBOL:days
                symbol = line.split(":")[0].strip().upper()
                if symbol:
                    symbols.append(symbol)
            return symbols
    except FileNotFoundError:
        formatter.print_error(f"Watchlist file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        formatter.print_error(f"Error reading watchlist file: {e}")
        sys.exit(1)


async def analyze_single(analyzer: StockAnalyzer, symbol: str, days: int, quiet: bool = False):
    """Analyze a single symbol."""
    if not quiet:
        formatter.print_progress(f"Analyzing {symbol}...")

    result = await analyzer.analyze_symbol(symbol, days)

    if not quiet:
        formatter.print_success(f"Analysis complete for {symbol}")

    return result


async def analyze_multiple(
    analyzer: StockAnalyzer, symbols: List[str], days: int, quiet: bool = False
):
    """Analyze multiple symbols with progress tracking."""
    if not quiet:
        formatter.print_progress(f"Analyzing {len(symbols)} symbols...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(symbols)} symbols...", total=len(symbols))

        results = []
        for symbol in symbols:
            try:
                result = await analyzer.analyze_symbol(symbol, days)
                results.append(result)
                progress.advance(task)
            except Exception as e:
                results.append(
                    {"symbol": symbol, "error": str(e), "timestamp": ""}
                )
                progress.advance(task)

    if not quiet:
        formatter.print_success(f"Completed analysis of {len(symbols)} symbols")

    return results


async def watch_mode(analyzer: StockAnalyzer, symbol: str, interval: int, days: int):
    """Continuous monitoring mode."""
    console.print(f"\n[bold cyan]Monitoring {symbol}[/bold cyan] (updates every {interval}s)")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        iteration = 0
        while True:
            if iteration > 0:
                console.print(f"\n[dim]--- Update #{iteration} ---[/dim]\n")

            result = await analyzer.analyze_symbol(symbol, days)
            formatter.format_table([result])

            iteration += 1
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")


async def main():
    """Main CLI entry point."""
    args = parse_arguments()

    # Configure logging based on verbosity
    configure_cli_logging(
        verbose=args.verbose,
        log_file=args.log_file
    )

    # Disable color if requested
    if args.no_color:
        console.no_color = True

    # Validate arguments
    if not args.symbols and not args.watchlist:
        formatter.print_error("Error: No symbols provided. Use --help for usage information.")
        sys.exit(1)

    # Load symbols
    if args.watchlist:
        symbols = load_watchlist(args.watchlist)
        if not args.quiet:
            formatter.print_progress(f"Loaded {len(symbols)} symbols from {args.watchlist}")
    else:
        symbols = [s.upper() for s in args.symbols]

    # Initialize analyzer
    analyzer = StockAnalyzer()

    # Watch mode (only for single symbol)
    if args.watch:
        if len(symbols) > 1:
            formatter.print_error("Watch mode only supports a single symbol")
            sys.exit(1)
        await watch_mode(analyzer, symbols[0], args.interval, args.days)
        return

    # Analyze symbols
    try:
        if len(symbols) == 1:
            results = [await analyze_single(analyzer, symbols[0], args.days, args.quiet)]
        else:
            results = await analyze_multiple(analyzer, symbols, args.days, args.quiet)

        # Format and display output
        if args.format == "table":
            formatter.format_table(results, detailed=args.verbose)
        elif args.format == "json":
            output = formatter.format_json(results)
            console.print(output)
        elif args.format == "csv":
            output = formatter.format_csv(results)
            console.print(output)

        # Save to file if requested
        if args.output:
            if args.format == "json":
                content = formatter.format_json(results)
            elif args.format == "csv":
                content = formatter.format_csv(results)
            else:
                content = formatter.format_json(results)  # Default to JSON for table output

            with open(args.output, "w") as f:
                f.write(content)

            formatter.print_success(f"Results saved to {args.output}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        formatter.print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
