# TODO

This document lists the immediate, actionable tasks for the current development phase.

## Phase 3: Risk, Portfolio Management & Backtesting

### Milestone 1: Advanced Agent Development

-   [ ] Create `src/agents/risk_manager.py` with a `RiskManagementAgent` class.
-   [ ] Implement basic risk calculations (e.g., stop-loss levels) in the `RiskManagementAgent`.
-   [ ] Create `src/agents/portfolio_manager.py` with a `PortfolioManagementAgent` class.
-   [ ] Implement logic in the `PortfolioManagementAgent` to synthesize decisions from other agents (for now, just the `TechnicalAnalysisAgent`).
-   [ ] Update the `Orchestrator` to include the new agents in the workflow.

### Milestone 2: Backtesting Engine Setup

-   [ ] Choose a backtesting library (`backtrader` or `vectorbt`) and add it to the project dependencies.
-   [ ] Create `src/backtesting/engine.py`.
-   [ ] Implement a basic backtesting `Strategy` that uses the `Orchestrator` to generate signals.
-   [ ] Implement a script in `scripts/run_backtest.py` to configure and run the backtesting engine with the strategy.

### Milestone 3: Initial Backtest & Reporting

-   [ ] Run an initial backtest using the implemented strategy.
-   [ ] Create `src/backtesting/reports.py` to generate basic performance metrics (e.g., Total Return, Sharpe Ratio).
-   [ ] Ensure the backtest script outputs a simple performance report.
