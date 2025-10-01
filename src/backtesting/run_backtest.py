"""
Script to run backtesting using full LLM-driven agents.
"""
import asyncio
import os
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta

from src.agents.data_structures import AgentConfig, MarketData
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.backtesting.engine import BacktestingEngine
from src.communication.message_bus import MessageBus
from src.communication.state_manager import StateManager
from src.data.cache import CacheManager
from src.data.pipeline import DataPipeline
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.llm.client import LLMClient


def get_agents():
    """Factory function to create LLM-driven agents with dependencies."""
    # Core components
    llm_client = LLMClient()
    message_bus = MessageBus()
    state_manager = StateManager()

    # Initialize data providers
    yfinance_provider = YFinanceProvider(rate_limit=10, period=60.0)
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_api_key:
        alpha_vantage_provider = AlphaVantageProvider(api_key=alpha_vantage_api_key)
    else:
        alpha_vantage_provider = None  # Sentiment agent will use mock

    # Initialize data pipeline
    data_pipeline = DataPipeline(provider=yfinance_provider, cache=CacheManager())

    # Initialize agents
    agent_dependencies = {
        "llm_client": llm_client,
        "message_bus": message_bus,
        "state_manager": state_manager,
    }
    technical_agent = TechnicalAnalysisAgent(
        config=AgentConfig(name="technical"), **agent_dependencies
    )
    sentiment_agent = SentimentAnalysisAgent(
        config=AgentConfig(name="sentiment"),
        news_provider=alpha_vantage_provider,
        **agent_dependencies,
    )
    risk_agent = RiskManagementAgent(
        config=AgentConfig(name="risk"), **agent_dependencies
    )
    portfolio_agent = PortfolioManagementAgent(
        config=AgentConfig(name="portfolio"), **agent_dependencies
    )

    return technical_agent, sentiment_agent, risk_agent, portfolio_agent


async def generate_decisions(data: pd.DataFrame, symbol: str, tech_agent, sent_agent, risk_agent, port_agent, sample_every: int = 5) -> list:
    """Generate decisions for each data point using LLM-driven agents with batch processing for independent agents."""
    decisions = []
    portfolio_state = {"cash": 100000, "positions": {}}

    # Sample data points to reduce LLM calls (every sample_every trading days)
    sampled_data = data.iloc[::sample_every]

    # Prepare batch requests for technical and sentiment agents (independent)
    batch_requests = []
    request_metadata = []  # To track which request corresponds to which agent and time step

    for idx, row in sampled_data.iterrows():
        market_data = MarketData(
            symbol=symbol,
            price=row['Close'],
            volume=row['Volume'],
            timestamp=idx.to_pydatetime(),
            ohlc={
                "Open": row['Open'], "High": row['High'],
                "Low": row['Low'], "Close": row['Close']
            },
            technical_indicators={
                "RSI": row.get('RSI_14', 50),
                "MACD": row.get('MACD', 0)
            }
        )

        # Prepare prompts for technical and sentiment agents
        tech_prompt = await tech_agent.get_user_prompt(market_data)
        sent_prompt = await sent_agent.get_user_prompt(market_data)

        tech_system = tech_agent.get_system_prompt()
        sent_system = sent_agent.get_system_prompt()

        batch_requests.append((tech_agent.config.model_name, tech_prompt, tech_system))
        request_metadata.append(("technical", idx))

        batch_requests.append((sent_agent.config.model_name, sent_prompt, sent_system))
        request_metadata.append(("sentiment", idx))

    # Execute batch requests for technical and sentiment agents
    llm_client = tech_agent.llm_client  # All agents share the same LLM client
    batch_responses = await llm_client.generate_batch(batch_requests)

    # Organize responses by time step
    responses_by_timestamp = {}
    for (agent_type, timestamp), response in zip(request_metadata, batch_responses):
        if timestamp not in responses_by_timestamp:
            responses_by_timestamp[timestamp] = {}
        responses_by_timestamp[timestamp][agent_type] = response

    # Now process each time step sequentially for risk and portfolio agents (which depend on previous decisions)
    for idx, row in data.iterrows():
        timestamp = idx
        market_data = MarketData(
            symbol=symbol,
            price=row['Close'],
            volume=row['Volume'],
            timestamp=idx.to_pydatetime(),
            ohlc={
                "Open": row['Open'], "High": row['High'],
                "Low": row['Low'], "Close": row['Close']
            },
            technical_indicators={
                "RSI": row.get('RSI_14', 50),
                "MACD": row.get('MACD', 0)
            }
        )

        # Get pre-computed technical and sentiment decisions
        tech_dec = tech_agent.create_decision(market_data, responses_by_timestamp[timestamp]["technical"])
        sent_dec = sent_agent.create_decision(market_data, responses_by_timestamp[timestamp]["sentiment"])

        decisions_dict = {"technical": tech_dec, "sentiment": sent_dec}

        # Risk and portfolio agents still need to be called sequentially due to dependencies
        risk_dec = await risk_agent.analyze(market_data, proposed_decisions=decisions_dict, portfolio_state=portfolio_state)
        decisions_dict["risk"] = risk_dec
        port_dec = await port_agent.analyze(market_data, agent_decisions=decisions_dict, portfolio_state=portfolio_state)
        decisions.append(port_dec)  # Use portfolio decision for trading

    return decisions


async def main():
    symbols = ["AAPL", "GOOGL", "MSFT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    provider = YFinanceProvider()

    # Get LLM-driven agents once
    tech_agent, sent_agent, risk_agent, port_agent = get_agents()

    for symbol in symbols:
        print(f"\nRunning backtest for {symbol}...")
        data = await provider.fetch_data(symbol, start_date, end_date)
        if data is None:
            print(f"Failed to fetch data for {symbol}")
            continue

        # Calculate indicators
        data.ta.rsi(append=True)
        data.ta.macd(append=True)

        decisions = await generate_decisions(data, symbol, tech_agent, sent_agent, risk_agent, port_agent)

        engine = BacktestingEngine()
        results = engine.run_backtest(data, decisions)

        print(f"Backtest Results for {symbol}:")
        print(f"Initial Cash: ${results['initial_cash']}")
        print(f"Final Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Number of Trades: {results['trades']}")
        print(f"Win Rate: {results['win_rate_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}" if results['sharpe_ratio'] else "Sharpe Ratio: N/A")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
