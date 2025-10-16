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

# Load environment variables FIRST before any imports that use settings
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.cli.analyzer import StockAnalyzer
from src.cli.formatter import OutputFormatter
from src.utils.cli_logging import configure_cli_logging



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
  
Verbosity Levels:
  %(prog)s AAPL --verbose=0                  # Silent (errors only)
  %(prog)s AAPL --verbose=1                  # Normal (warnings + errors)
  %(prog)s AAPL --verbose=2                  # Detailed (info + warnings + errors)
  %(prog)s AAPL --verbose=3                  # Debug (full verbose output)
  
Summary Mode:
  %(prog)s AAPL --summary-only               # Final table only (clean output)

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
        nargs="?",
        const=1,
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Verbosity level: 0=errors-only, 1=normal (default), 2=detailed, 3=debug (more verbose = more detail)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Silent mode (equivalent to --verbose=0)",
    )

    parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Show only the final results table and errors",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="[DEPRECATED] Use --verbose=2 instead",
    )

    parser.add_argument(
        "--log",
        type=str,
        metavar="FILE",
        help="Save debug logs to file (use with --verbose=3)",
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

    # Handle deprecation warnings and argument conflicts
    verbosity_level = args.verbose
    
    # Handle --quiet flag overriding verbosity
    if args.quiet:
        verbosity_level = 0
    # Handle --summary-only flag 
    summary_only_mode = args.summary_only
    if summary_only_mode:
        verbosity_level = 1  # Use normal verbosity level, control display separately
    
    # Handle --detailed deprecation
    if args.detailed and not summary_only_mode:
        formatter.print_warning("Warning: --detailed is deprecated. Use --verbose=2 instead.")
        verbosity_level = 2
    
    # Configure logging based on verbosity
    # Handle summary-only mode with special logging
    if summary_only_mode:
        configure_cli_logging(
            verbose=4,  # Use level 4 for summary-only
            log_file=getattr(args, 'log', None)
        )
    else:
        configure_cli_logging(
            verbose=verbosity_level,
            log_file=getattr(args, 'log', None)
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
        if verbosity_level > 0:
            formatter.print_progress(f"Loaded {len(symbols)} symbols from {args.watchlist}")
    else:
        symbols = [s.upper() for s in args.symbols]

    # Initialize analyzer
    analyzer = StockAnalyzer()

    # Show system status on higher verbosity levels (after analyzer initialization)
    if verbosity_level >= 2 and not summary_only_mode:
        formatter.print_header("AI Trading System CLI")
        import os
        redis_disabled = os.getenv('ENABLE_REDIS', '').lower() in ('false', '0', 'no')
        if redis_disabled:
            cache_status = "Cache: Memory (Redis disabled)"
        else:
            cache_status = "Cache: Auto-detecting Redis"
        if not summary_only_mode and verbosity_level < 4:
            formatter.print_info(f"Verbosity: Level {verbosity_level} | {cache_status}")

    # Watch mode (only for single symbol)
    if args.watch:
        if len(symbols) > 1:
            formatter.print_error("Watch mode only supports a single symbol")
            sys.exit(1)
        await watch_mode(analyzer, symbols[0], args.interval, args.days)
        return

    # Analyze symbols
    try:
        # Set quiet mode for silent (0) or summary-only (4) modes
        is_quiet = verbosity_level == 0 or summary_only_mode
        
        if len(symbols) == 1:
            results = [await analyze_single(analyzer, symbols[0], args.days, is_quiet)]
        else:
            results = await analyze_multiple(analyzer, symbols, args.days, is_quiet)

        # Format and display output
        if verbosity_level >= 2 and not summary_only_mode:  # Detailed output
            if len(results) > 1:
                formatter.format_detailed_table(results)
            else:
                formatter.format_table(results, detailed=verbosity_level >= 2)
        elif args.format == "table":
            formatter.format_table(results, detailed=verbosity_level >= 2)
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
            elif verbosity_level >= 2 and not summary_only_mode:
                content = formatter.format_json(results)  # Save detailed as JSON
            else:
                content = formatter.format_json(results)  # Default to JSON for table output

            with open(args.output, "w") as f:
                f.write(content)

            if verbosity_level > 0:
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
