"""
Script to run an initial backtest using deterministic agents.
"""
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta

from src.agents.data_structures import MarketData
from src.backtesting.deterministic_agents import (
    DeterministicPortfolioAgent,
    DeterministicRiskAgent,
    DeterministicSentimentAgent,
    DeterministicTechnicalAgent,
)
from src.backtesting.engine import BacktestingEngine
from src.data.pipeline import DataPipeline
from src.data.providers.yfinance_provider import YFinanceProvider


async def generate_decisions(data: pd.DataFrame, symbol: str) -> list:
    """Generate decisions for each data point using deterministic agents."""
    tech_agent = DeterministicTechnicalAgent()
    sent_agent = DeterministicSentimentAgent()
    risk_agent = DeterministicRiskAgent()
    port_agent = DeterministicPortfolioAgent()

    decisions = []
    portfolio_state = {"cash": 100000, "positions": {}}

    for idx, row in data.iterrows():
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

        tech_dec = tech_agent.analyze(market_data)
        sent_dec = sent_agent.analyze(market_data)
        decisions_dict = {"technical": tech_dec, "sentiment": sent_dec}
        risk_dec = risk_agent.analyze(market_data, decisions_dict, portfolio_state)
        decisions_dict["risk"] = risk_dec
        port_dec = port_agent.analyze(market_data, decisions_dict, portfolio_state)
        decisions.append(port_dec)  # Use portfolio decision for trading

    return decisions


async def main():
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    provider = YFinanceProvider()
    pipeline = DataPipeline(provider)

    data = await provider.fetch_data(symbol, start_date, end_date)
    if data is None:
        print("Failed to fetch data")
        return

    # Calculate indicators

    data.ta.rsi(append=True)
    data.ta.macd(append=True)

    decisions = await generate_decisions(data, symbol)

    engine = BacktestingEngine()
    results = engine.run_backtest(data, decisions)

    print("Backtest Results:")
    print(f"Initial Cash: ${results['initial_cash']}")
    print(f"Final Value: ${results['final_value']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Number of Trades: {results['trades']}")


if __name__ == "__main__":
    asyncio.run(main())
