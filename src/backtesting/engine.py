"""
Backtesting engine using backtrader to evaluate trading strategies.
"""
from datetime import datetime
from typing import List, Dict, Any

import backtrader as bt
import pandas as pd

from src.agents.data_structures import AgentDecision


class BacktestStrategy(bt.Strategy):
    """
    Backtrader strategy that uses agent decisions to generate signals.
    """

    def __init__(self, decisions: List[AgentDecision]):
        self.decisions = {d.timestamp: d for d in decisions}
        self.order = None

    def next(self):
        current_time = self.datas[0].datetime.datetime()
        decision = self.decisions.get(current_time)
        if decision:
            if decision.signal == "BUY" and not self.position:
                self.order = self.buy()
            elif decision.signal == "SELL" and self.position:
                self.order = self.sell()


class BacktestingEngine:
    """
    Engine for running backtests using backtrader.
    """

    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    def run_backtest(self, data: pd.DataFrame, decisions: List[AgentDecision], initial_cash: float = 100000) -> Dict[str, Any]:
        """
        Runs a backtest with the given data and decisions.

        Args:
            data: OHLCV data as DataFrame.
            decisions: List of agent decisions with timestamps.
            initial_cash: Starting cash.

        Returns:
            Backtest results.
        """
        # Prepare data for backtrader
        data_feed = bt.feeds.PandasData(dataname=data, datetime='Date' if 'Date' in data.columns else None)

        self.cerebro.adddata(data_feed)
        self.cerebro.addstrategy(BacktestStrategy, decisions=decisions)
        self.cerebro.broker.setcash(initial_cash)

        # Run backtest
        results = self.cerebro.run()

        # Get results
        final_value = self.cerebro.broker.getvalue()
        returns = (final_value - initial_cash) / initial_cash * 100

        trade_analysis = results[0].analyzers.trade_analyzer.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0) if trade_analysis else 0
        won_trades = trade_analysis.get('won', {}).get('total', 0) if trade_analysis else 0
        lost_trades = trade_analysis.get('lost', {}).get('total', 0) if trade_analysis else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        sharpe_analysis = results[0].analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get('sharperatio', None)

        drawdown_analysis = results[0].analyzers.drawdown.get_analysis()
        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) if drawdown_analysis else 0

        return {
            "initial_cash": initial_cash,
            "final_value": final_value,
            "total_return_pct": returns,
            "trades": total_trades,
            "win_rate_pct": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
        }
