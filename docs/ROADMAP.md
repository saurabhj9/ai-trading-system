# Project Roadmap

This document outlines the high-level roadmap for the development of the AI Trading System. The project is divided into several phases, each with a specific focus and set of deliverables.

## Phase 1: Core Framework & Agent Development

**Goal:** Build the foundational components of the agentic system.

-   **Deliverables:**
    -   [x] Implement the `BaseAgent` class and core data structures (`AgentConfig`, `MarketData`, `AgentDecision`).
    -   [x] Develop the initial version of the **Technical Analysis Agent**.
    -   [ ] Develop the initial version of the **Sentiment Analysis Agent**.
    -   [x] Set up the **LangGraph** orchestration to manage a simple, sequential workflow between the agents.
    -   [x] Implement the basic **State Manager** for sharing state between agents.
    -   [x] Create unit tests for all new components.

## Phase 2: Data Pipeline & Integration

**Goal:** Build a robust data pipeline to feed the agents with market data.

-   **Deliverables:**
    -   [x] Implement the data provider interface for `yfinance` and `Alpha Vantage`.
    -   [x] Create the data processing pipeline for calculating technical indicators.
    -   [x] Implement the multi-level data caching system (in-memory and Redis).
    -   [x] Integrate the data pipeline with the agent framework.
    -   [x] Write integration tests for the data pipeline.

## Phase 3: Risk, Portfolio Management & Backtesting

**Goal:** Add advanced decision-making and performance evaluation capabilities.

-   **Deliverables:**
    -   [ ] Develop the **Risk Management Agent**.
    -   [ ] Develop the **Portfolio Management Agent** to synthesize agent inputs and make trade decisions.
    -   [ ] Set up the **Backtesting Engine** using `backtrader` or `vectorbt`.
    -   [ ] Create a simplified, deterministic version of the agent workflow for fast and repeatable backtests.
    -   [ ] Generate initial backtest reports and performance metrics.

## Phase 4: API & Deployment

**Goal:** Expose the system's functionality via an API and prepare for deployment.

-   **Deliverables:**
    -   [ ] Develop a **FastAPI** application to expose trade signals and system status.
    -   [ ] Create `Dockerfile` and `docker-compose.yml` for all services.
    -   [ ] Implement a production deployment script.
    -   [ ] Set up structured logging and basic monitoring.

## Phase 5: Refinement & Optimization

**Goal:** Improve the performance, intelligence, and reliability of the system.

-   **Deliverables:**
    -   [ ] Optimize the data pipeline and agent processing for speed.
    -   [ ] Enhance the intelligence of the agents with more sophisticated analysis techniques.
    -   [ ] Conduct extensive backtesting and parameter tuning.
    -   [ ] Improve monitoring and alerting.
