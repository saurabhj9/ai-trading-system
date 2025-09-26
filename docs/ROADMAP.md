# Project Roadmap

This document outlines the high-level roadmap for the development of the AI Trading System. The project is divided into several phases, each with a specific focus and set of deliverables.

## Phase 1: Core Framework & Agent Development

**Goal:** Build the foundational components of the agentic system.

-   **Deliverables:**
    -   [x] Implement the `BaseAgent` class and core data structures (`AgentConfig`, `MarketData`, `AgentDecision`).
    -   [x] Develop the initial version of the **Technical Analysis Agent**.
    -   [x] Develop the initial version of the **Sentiment Analysis Agent** (currently uses mock data for sentiment analysis).
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
    -   [x] Develop the **Risk Management Agent**.
    -   [x] Develop the **Portfolio Management Agent** to synthesize agent inputs and make trade decisions.
    -   [x] Set up the **Backtesting Engine** using `backtrader` or `vectorbt`.
    -   [x] Create a simplified, deterministic version of the agent workflow for fast and repeatable backtests.
    -   [x] Generate initial backtest reports and performance metrics.

    **Note on Mock Data:** The Sentiment, Risk, and Portfolio Management Agents currently return mock decisions for development purposes. These must be replaced with real analysis logic before paper trading or live deployment. The backtesting uses deterministic, rule-based versions of the agents for speed, but the live system will use the full LLM-based agents.

## Phase 4: API & Deployment

**Goal:** Expose the system's functionality via an API and prepare for deployment.

-   **Deliverables:**
    -   [x] Develop a **FastAPI** application to expose trade signals and system status.
    -   [x] Create `Dockerfile` and `docker-compose.yml` for all services.
    -   [x] Implement a production deployment script.
    -   [x] Set up structured logging and basic monitoring.

## Phase 5: Refinement & Optimization

**Goal:** Improve the performance, intelligence, and reliability of the system by replacing mock implementations with production-ready logic and externalizing configurations.

-   **Deliverables:**
    -   [x] **Agent Logic Implementation:**
            -   [x] Implement robust LLM response parsing for all agents (`Technical`, `Sentiment`, `Risk`, `Portfolio`).
            -   [x] Replace mock `AgentDecision` returns with logic derived from LLM analysis.
            -   [x] Integrate a real-time news API for the `SentimentAnalysisAgent`, replacing the current mock news data.
    -   [x] **Configuration Management:**
        -   [x] Externalized hardcoded configurations (e.g., LLM provider URL, model names) into a dedicated config module using `.env` files.
        -   [x] Made the default portfolio starting cash configurable.
        -   [x] Secured the FastAPI CORS policy by allowing configuration of trusted domains.
    -   [ ] **Optimization & Tuning:**
        -   [ ] Optimize the data pipeline and agent processing for speed.
        -   [ ] Conduct extensive backtesting with full agent logic and perform parameter tuning.
    -   [ ] **Monitoring & Reliability:**
        -   [ ] Enhance system monitoring with more detailed metrics and alerting.
