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

## Phase 4: API & Deployment - COMPLETED

All Phase 4 deliverables have been implemented:

### Milestone 1: FastAPI Application Development - COMPLETED

-   [x] Create `src/api/app.py` with FastAPI application setup.
-   [x] Implement `src/api/routes/signals.py` for trade signal generation endpoint.
-   [x] Implement `src/api/routes/status.py` for system status and health checks.
-   [x] Implement `src/api/routes/monitoring.py` for metrics and monitoring.
-   [x] Set up proper dependency injection for agents with LLM client, message bus, and state manager.

### Milestone 2: Containerization & Orchestration - COMPLETED

-   [x] Create `Dockerfile` for the main application container.
-   [x] Create `docker-compose.yml` for multi-service orchestration (app, Redis, PostgreSQL).
-   [x] Configure environment variables and service dependencies.

### Milestone 3: Production Deployment - COMPLETED

-   [x] Implement `scripts/deploy.sh` for automated production deployment.
-   [x] Set up structured logging with structlog and JSON output.
-   [x] Implement basic monitoring with request metrics and health checks.

## Next Phase: Phase 5 - Refinement & Optimization

This phase focuses on replacing mock implementations with production-ready logic, externalizing configurations, and enhancing overall system performance and reliability.

### Milestone 1: Implement Core Agent Logic

-   [ ] **LLM Response Parsing:** Implement robust parsing logic in all agents (`Technical`, `Sentiment`, `Risk`, `Portfolio`) to replace mock `AgentDecision` returns with analysis derived from the LLM.
-   [ ] **Sentiment Analysis Agent:** Integrate a live news data provider (e.g., NewsAPI, Alpha Vantage News) to replace the current `mock_news` list.
-   [ ] **Risk Management Agent:** Implement actual risk calculations (e.g., position sizing, Value at Risk) instead of the mock "APPROVE" signal.

### Milestone 2: Externalize Configurations

-   [ ] **Centralize Settings:** Move hardcoded values from the codebase into a structured configuration system (e.g., `.env` files with a `config.py` loader).
    -   [ ] LLM provider URL and model names (`src/llm/client.py`, `src/agents/data_structures.py`).
    -   [ ] Default portfolio starting cash (`src/communication/orchestrator.py`).
-   [ ] **Secure API:** Update the FastAPI CORS policy in `src/api/app.py` to use a specific list of allowed origins instead of `["*"]`.

### Milestone 3: Optimization and Tuning

-   [ ] **Performance Analysis:** Profile the data pipeline and agent processing to identify and address bottlenecks.
-   [ ] **Backtesting:** Conduct comprehensive backtesting cycles using the full, LLM-driven agent logic to evaluate strategy performance.
-   [ ] **Parameter Tuning:** Tune agent parameters and strategy rules based on backtesting results.

### Milestone 4: Enhance Monitoring

-   [ ] **Add Alerts:** Implement an alerting mechanism for critical system events (e.g., repeated API failures, large drawdown).
-   [ ] **Expand Metrics:** Add more granular metrics for agent performance, LLM response times, and data pipeline throughput.
