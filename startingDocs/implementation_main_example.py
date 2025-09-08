#!/usr/bin/env python3
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = typer.Typer()
console = Console()

async def initialize_system():
    """Initialize all system components"""
    # Import your classes here
    from communication.message_bus import MessageBus
    from communication.state_manager import StateManager
    from data.pipeline import DataPipelineManager
    from agents.technical import TechnicalAnalysisAgent
    from agents.sentiment import SentimentAnalysisAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.portfolio_manager import PortfolioManagerAgent

    # Initialize core components
    message_bus = MessageBus()
    state_manager = StateManager()  # Redis-backed
    data_pipeline = DataPipelineManager()
    orchestrator = AgentOrchestrator(message_bus, state_manager)

    # Create and register agents
    agents = {
        'technical': TechnicalAnalysisAgent(
            config=AgentConfig(name="technical"),
            llm_client=get_llm_client(),
            message_bus=message_bus,
            state_manager=state_manager
        ),
        'sentiment': SentimentAnalysisAgent(
            config=AgentConfig(name="sentiment"),
            llm_client=get_llm_client(),
            message_bus=message_bus,
            state_manager=state_manager
        ),
        'risk': RiskManagerAgent(
            config=AgentConfig(name="risk"),
            llm_client=get_llm_client(),
            message_bus=message_bus,
            state_manager=state_manager
        ),
        'portfolio': PortfolioManagerAgent(
            config=AgentConfig(name="portfolio"),
            llm_client=get_llm_client(),
            message_bus=message_bus,
            state_manager=state_manager
        )
    }

    # Register all agents
    for agent in agents.values():
        orchestrator.register_agent(agent)

    return {
        'message_bus': message_bus,
        'state_manager': state_manager,
        'data_pipeline': data_pipeline,
        'orchestrator': orchestrator,
        'agents': agents
    }

def get_llm_client():
    """Get configured LLM client"""
    import openai
    return openai.AsyncClient(api_key=os.getenv('OPENAI_API_KEY'))

@app.command()
def run(
    symbols: str = typer.Option("AAPL,GOOGL,MSFT", help="Comma-separated symbols"),
    mode: str = typer.Option("analysis", help="Mode: analysis, backtest, live"),
    interval: int = typer.Option(300, help="Analysis interval in seconds")
):
    """Run the AI Trading System"""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    console.print(f"[green]Starting AI Trading System[/green]")
    console.print(f"Symbols: {', '.join(symbol_list)}")
    console.print(f"Mode: {mode}")

    asyncio.run(run_analysis_loop(symbol_list, mode, interval))

async def run_analysis_loop(symbols: List[str], mode: str, interval: int):
    """Main analysis loop"""
    system = await initialize_system()

    # Start message bus
    await system['message_bus'].start()

    try:
        console.print("[blue]System initialized successfully[/blue]")

        while True:
            console.print(f"\n[yellow]Running analysis cycle...[/yellow]")

            # Analyze all symbols
            results = await system['orchestrator'].analyze_multiple_symbols(symbols)

            # Display results
            display_analysis_results(results)

            # In live mode, you would execute trades here
            if mode == "live":
                console.print("[red]LIVE MODE: Would execute trades here[/red]")

            # Wait for next cycle
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        await system['message_bus'].stop()

def display_analysis_results(results: dict):
    """Display analysis results in a nice table"""
    table = Table(title="AI Trading Analysis Results")

    table.add_column("Symbol", style="cyan")
    table.add_column("Technical", style="green")
    table.add_column("Sentiment", style="blue")
    table.add_column("Risk", style="yellow")
    table.add_column("Final Decision", style="bold red")

    for symbol, decisions in results.items():
        if len(decisions) >= 4:  # All agents responded
            technical = f"{decisions[0].signal} ({decisions[0].confidence:.2f})"
            sentiment = f"{decisions[1].signal} ({decisions[1].confidence:.2f})"
            risk = f"{decisions[2].signal} ({decisions[2].confidence:.2f})"
            final = f"{decisions[3].signal} ({decisions[3].confidence:.2f})"

            table.add_row(symbol, technical, sentiment, risk, final)
        else:
            table.add_row(symbol, "Error", "Error", "Error", "Error")

    console.print(table)

@app.command()
def backtest(
    symbols: str = typer.Option("AAPL,GOOGL", help="Symbols to backtest"),
    start: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)"),
    capital: float = typer.Option(100000, help="Initial capital")
):
    """Run backtesting"""
    console.print(f"[blue]Running backtest from {start} to {end}[/blue]")
    console.print(f"Symbols: {symbols}")
    console.print(f"Initial capital: ${capital:,.2f}")

    # Implement backtesting logic
    asyncio.run(run_backtest_async(symbols.split(","), start, end, capital))

async def run_backtest_async(symbols, start_date, end_date, initial_capital):
    """Async backtest implementation"""
    # Initialize backtesting engine
    from backtesting.engine import BacktestingEngine
    from strategies.multi_agent_strategy import MultiAgentStrategy

    system = await initialize_system()
    strategy = MultiAgentStrategy(system['orchestrator'])

    backtest_engine = BacktestingEngine(initial_capital)

    results = await backtest_engine.run_backtest(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_provider=system['data_pipeline']
    )

    # Display results
    console.print(f"\n[green]Backtest Results:[/green]")
    console.print(f"Total Return: {results.total_return:.2%}")
    console.print(f"Annual Return: {results.annual_return:.2%}")
    console.print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    console.print(f"Max Drawdown: {results.max_drawdown:.2%}")
    console.print(f"Win Rate: {results.win_rate:.2%}")
    console.print(f"Total Trades: {results.total_trades}")

if __name__ == "__main__":
    app()
