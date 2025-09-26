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

### Milestone 1: Implement Core Agent Logic - COMPLETED

-   [x] **LLM Response Parsing:** Implement robust parsing logic in all agents (`Technical`, `Sentiment`, `Risk`, `Portfolio`) to replace mock `AgentDecision` returns with analysis derived from the LLM.
-   [x] **Sentiment Analysis Agent:** Integrate a live news data provider (e.g., NewsAPI, Alpha Vantage News) to replace the current `mock_news` list.
-   [x] **Risk Management Agent:** Implement actual risk calculations (e.g., position sizing, Value at Risk) instead of the mock "APPROVE" signal.

### Milestone 2: Externalize Configurations - COMPLETED

-   [x] **Centralize Settings:** Moved hardcoded values from the codebase into a structured configuration system (`.env` files with a `config.py` loader).
    -   [x] LLM provider URL and model names (`src/llm/client.py`, `src/agents/data_structures.py`).
    -   [x] Default portfolio starting cash (`src/communication/orchestrator.py`).
-   [x] **Secure API:** Updated the FastAPI CORS policy in `src/api/app.py` to use a specific list of allowed origins from the new configuration system.

### Milestone 3: Optimization and Tuning

#### Performance Analysis: Profile the data pipeline and agent processing to identify and address bottlenecks

-   [x] **Set Up Profiling Tools and Metrics Collection**
    -   [x] Create timing decorators and context managers for performance measurement
    -   [x] Enhance existing monitoring system with detailed performance metrics
    -   [x] Create profiling scripts for benchmarking critical components

-   [x] **Profile Data Pipeline Performance**
    -   [x] Measure data fetching performance from providers (Alpha Vantage, Yahoo Finance)
    -   [x] Profile technical indicator calculations (RSI, MACD)
    -   [x] Analyze cache performance and hit/miss ratios

-   [x] **Profile LLM Client Performance**
    -   [x] Measure LLM response times and network latency
    -   [x] Track token usage and correlate with performance
    -   [x] Analyze retry behavior and error handling impact

-   [x] **Profile Agent Processing Performance**
    -   [x] Measure individual agent execution times (Technical, Sentiment, Risk, Portfolio)
    -   [x] Profile LLM call overhead and response parsing
    -   [x] Identify slowest agents and bottlenecks

-   [x] **Profile Orchestrator Workflow Performance**
  -   [x] Measure sequential execution time and identify idle periods
  -   [x] Analyze potential for parallel execution of independent agents
  -   [x] Profile state management overhead between agents

-   [x] **Analyze Results and Identify Bottlenecks**
     -   [x] Compile performance metrics from all components
     -   [x] Create performance dashboards and visualizations
     -   [x] Prioritize bottlenecks by impact and effort

-   [x] **Create Performance Optimization Recommendations**
     -   [x] Develop specific optimization strategies for each bottleneck
     -   [x] Estimate potential performance improvements
     -   [x] Create implementation roadmap with priorities

-   [x] **Implement Identified Optimizations** (Prioritized by Impact/Effort)
      -   [x] **HIGH PRIORITY: LLM Response Time Optimization** (~5s avg, 40% of total time)
          -   [x] Implement LLM response caching for repeated queries
          -   [x] Add connection pooling to reduce network latency
          -   [ ] Implement request batching for multiple agent calls
          -   [ ] Explore faster LLM models or providers
      -   [x] **MEDIUM PRIORITY: Data Pipeline Optimization**
          -   [x] Investigate and fix AAPL data fetching bottleneck (~7.7s vs 0.1s for GOOGL)
          -   [x] Implement parallel data fetching for multiple symbols
          -   [ ] Optimize technical indicator calculations
          -   [ ] Improve cache hit ratios
      -   [x] **MEDIUM PRIORITY: Agent Processing Parallelization**
          -   [x] Implement parallel execution for independent agents (Technical/Sentiment)
          -   [ ] Optimize agent prompts to reduce LLM token usage
          -   [ ] Streamline Risk/Portfolio agent parameter passing
     -   [ ] **LOW PRIORITY: State Management Optimization**
         -   [ ] Reduce state serialization overhead (~5s in orchestrator)
         -   [ ] Implement incremental state updates
         -   [ ] Profile and optimize message bus performance

-   [ ] **Backtesting:** Conduct comprehensive backtesting cycles using the full, LLM-driven agent logic to evaluate strategy performance.
     -   [ ] Replace deterministic agents with full LLM-driven agents in backtesting engine
     -   [ ] Run multi-symbol backtests (AAPL, GOOGL, MSFT) over 2-year periods
     -   [ ] Compare performance metrics: Sharpe ratio, max drawdown, win rate
     -   [ ] Analyze agent decision consistency across different market conditions
-   [ ] **Parameter Tuning:** Tune agent parameters and strategy rules based on backtesting results.
     -   [ ] Optimize LLM temperature and prompt engineering for better decision quality
     -   [ ] Adjust risk thresholds and position sizing parameters
     -   [ ] Fine-tune technical indicator parameters (RSI periods, MACD settings)
     -   [ ] Implement A/B testing framework for agent prompt variations

## Critical Issues Resolved

-   [x] **Agent Serialization Errors**: Fixed 'model_dump' attribute errors by ensuring proper serialization compatibility with LangGraph (verified: orchestrator profiling completes successfully)
-   [x] **Risk/Portfolio Agent Failures**: Resolved method signature mismatches by correcting parameter passing in orchestrator calls (verified: all agents execute without errors)

### Milestone 4: Enhance Monitoring

-   [ ] **Add Alerts:** Implement an alerting mechanism for critical system events (e.g., repeated API failures, large drawdown).
    -   [ ] Define alert thresholds and conditions
    -   [ ] Implement alert notification system
    -   [ ] Create alert dashboard and management interface

-   [ ] **Expand Metrics:** Add more granular metrics for agent performance, LLM response times, and data pipeline throughput.
    -   [ ] Add agent-specific performance metrics
    -   [ ] Implement detailed LLM response time tracking
    -   [ ] Enhance data pipeline throughput monitoring
    -   [ ] Create comprehensive performance dashboards
