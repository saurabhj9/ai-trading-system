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

    **Note on Mock Data:** The Sentiment Agent currently uses mock news data for development purposes. The Risk and Portfolio agents are fully implemented with LLM-based analysis. For production deployment, integrate a live news API for sentiment analysis. The backtesting uses deterministic, rule-based versions of the agents for speed, but the live system uses the full LLM-based agents.

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
    -   [x] **Optimization & Tuning:**
        -   [x] **Performance Analysis Setup:** Created comprehensive profiling tools and metrics collection system
        -   [x] **Data Pipeline Profiling:** Completed initial performance analysis of data fetching, caching, and indicator calculations
        -   [x] **LLM Client Profiling:** Completed performance analysis of LLM response times, token usage, and error handling
        -   [x] **Agent Processing Profiling:** Completed performance analysis of individual agent execution times and LLM call overhead
        -   [x] **LLM Response Time Optimization:** Implemented caching, connection pooling, parallel agent execution, and request batching (62% reduction in LLM response time, 42% reduction in orchestrator execution time)
        -   [x] **LLM Model Evaluation:** Completed comprehensive benchmarking of 8 models from 4 providers; selected Claude-3-haiku for optimal speed-quality balance
        -   [x] **Data Pipeline Optimization:** Added retry logic for data fetching and parallel multi-symbol fetching
        -   [x] **Agent Processing Parallelization:** Implemented parallel execution of independent agents (Technical/Sentiment)
        -   [ ] **Prompt Engineering Optimization:** Refine agent prompts to address decision quality issues (e.g., oversold RSI misinterpretation in bearish scenarios)
        -   [ ] Conduct extensive backtesting with full agent logic and perform parameter tuning.
    -   [ ] **Monitoring & Reliability:**
        -   [ ] Enhance system monitoring with more detailed metrics and alerting.
