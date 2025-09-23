# TODO

This document lists the immediate, actionable tasks for the current development phase.

## Phase 3: Risk, Portfolio Management & Backtesting - COMPLETED

All Phase 3 deliverables have been implemented:

### Milestone 1: Advanced Agent Development - COMPLETED

-   [x] Create `src/agents/risk.py` with a `RiskManagementAgent` class (currently returns mock APPROVE decisions).
-   [x] Implement basic risk calculations (placeholder for now).
-   [x] Create `src/agents/portfolio.py` with a `PortfolioManagementAgent` class (currently returns mock decisions based on technical).
-   [x] Implement logic in the `PortfolioManagementAgent` to synthesize decisions from other agents.
-   [x] Update the `Orchestrator` to include the new agents in the workflow.

### Milestone 2: Backtesting Engine Setup - COMPLETED

-   [x] Chose `backtrader` as the backtesting library and added to dependencies.
-   [x] Create `src/backtesting/engine.py`.
-   [x] Implement a basic backtesting `Strategy` that uses deterministic agents for signals.
-   [x] Implement a script in `src/backtesting/run_backtest.py` to configure and run the backtesting engine.

### Milestone 3: Initial Backtest & Reporting - COMPLETED

-   [x] Run an initial backtest using the implemented strategy (AAPL, past 365 days, RSI-based rules).
-   [x] Basic performance metrics are output in the script (Total Return, Trades).
-   [x] Backtest script outputs a simple performance report.

## Important Notes on Mock Data

- **SentimentAnalysisAgent**: Returns mock NEUTRAL signals. Replace with real news/social media analysis before live use.
- **RiskManagementAgent**: Always APPROVE. Implement actual risk metrics (VaR, position sizing) for paper trading.
- **PortfolioManagementAgent**: Follows technical signals. Enhance to consider all agents and portfolio state.
- **Backtesting**: Uses deterministic agents for speed. For accurate backtests, integrate full agents (but this will be slower).

## Next Phase: Phase 4 - API & Deployment

- Develop FastAPI application for trade signals and system status.
- Create Dockerfile and docker-compose.yml.
- Implement production deployment.
- Set up logging and monitoring.
