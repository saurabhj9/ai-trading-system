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

    **Note on Mock Data:** The Sentiment Agent currently uses mock news data for development purposes. The Risk and Portfolio agents are fully implemented with LLM-based analysis. For production deployment, integrate a live news API for sentiment analysis. Earlier backtests used deterministic, rule-based agents for speed; we are now migrating to a local-first, event-driven LLM escalation approach in backtests to control cost while maintaining fidelity.

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
        -   [x] **Prompt Engineering Optimization:** Refine agent prompts to address decision quality issues (e.g., oversold RSI misinterpretation in bearish scenarios) and implement historical indicator data for trend analysis and divergence detection
        -   [x] Conduct extensive backtesting with full agent logic and perform parameter tuning.
    -   [x] **Monitoring & Reliability:**
        -   [x] Enhance system monitoring with more detailed metrics and alerting.

## Phase 6: Enhanced Technical Analysis & Local-First Strategy

**Goal:** Transform the system from LLM-dependent to a sophisticated hybrid that leverages local processing for routine analysis and strategic LLM escalation for complex scenarios. Optimize performance, reliability, and cost as we scale symbols and data granularity.

### Strategic Shift: Local-First Analysis

The system is pivoting from LLM-first decision making to a local-first approach where:
- **80-90% of decisions** are made locally using rule-based systems
- **LLM escalation** occurs only for complex scenarios requiring nuanced reasoning
- **Cost reduction** of 80-90% while maintaining or improving decision quality
- **Faster response times** through local signal generation
- **Better scalability** for multiple symbols and timeframes

### Implementation Strategy

#### Phase 6.1: Foundation (Weeks 1-4)
-   [x] **Expand Technical Indicators:** - COMPLETED 2025-10-01
    -   [x] Add momentum indicators (Stochastic, Williams %R, CCI)
    -   [x] Add mean reversion indicators (Bollinger Bands, Keltner Channels)
    -   [x] Add volatility indicators (ATR, Historical Volatility)
    -   [x] Add trend indicators (EMAs, ADX, Aroon, Parabolic SAR)
    -   [x] Add volume indicators (OBV, VWAP, MFI)
    -   [x] Add statistical indicators (Hurst Exponent, Z-Score, Correlation)

    **Summary**: Successfully implemented 20+ new technical indicators across 6 categories, expanding the system's analytical capabilities from basic indicators to a comprehensive technical analysis toolkit. All indicators are now organized by category with proper metadata for reliability and usage patterns.
-   [ ] **Local Signal Generation Framework:**
    -   [ ] Create `SignalGenerator` class for local decision making
    -   [ ] Implement market regime detection and context-aware weighting
    -   [ ] Build consensus-based signal combination logic
    -   [ ] Add confidence calculation based on signal agreement
-   [ ] **Data Provider Optimization:**
    -   [ ] Optimize yfinance usage with better caching
    -   [ ] Add IEX Cloud as secondary provider (50K free calls/month)
    -   [ ] Implement provider selection logic with automatic fallbacks

#### Phase 6.2: Event-Driven System (Weeks 5-8)
-   [ ] **Event-Driven Triggers:**
    -   [ ] Implement comprehensive trigger detection system
    -   [ ] Add technical, volatility, trend, conflict, and regime triggers
    -   [ ] Implement decision TTL and cooldown mechanisms
    -   [ ] Add configurable trigger thresholds and sensitivity
-   [ ] **State-Hash Decision Cache:**
    -   [ ] Implement feature vector quantization for compact representation
    -   [ ] Create state hashing system for decision caching
    -   [ ] Add cache persistence and retrieval logic with Redis
    -   [ ] Implement cache hit/miss tracking and effectiveness metrics

#### Phase 6.3: Strategic LLM Integration (Weeks 9-10)
-   [ ] **LLM Escalation Logic:**
    -   [ ] Implement escalation triggers for complex scenarios
    -   [ ] Design specialized prompts for conflict resolution
    -   [ ] Add LLM decision caching for similar conflict patterns
    -   [ ] Create escalation confidence scoring
-   [ ] **Budget and Governance:**
    -   [ ] Implement per-session LLM call budgets and tracking
    -   [ ] Add rate limiting and degradation logging
    -   [ ] Create cost tracking and optimization metrics
    -   [ ] Implement explicit degradation modes when budgets exhausted

#### Phase 6.4: Optimization & Validation (Weeks 11-14)
-   [ ] **Performance Optimization:**
    -   [ ] Optimize technical indicator calculations with vectorization
    -   [ ] Implement incremental state updates to reduce overhead
    -   [ ] Profile and optimize message bus performance
    -   [ ] Add comprehensive monitoring metrics
-   [ ] **Comprehensive Backtesting & Validation:**
    -   [ ] Run multi-symbol backtests (AAPL, GOOGL, MSFT) over 2-year periods
    -   [ ] Compare performance: local-only vs. local+LLM hybrid approaches
    -   [ ] Analyze decision consistency across market regimes
    -   [ ] Optimize parameters based on backtest results
    -   [ ] Validate cost reduction and performance improvements

### Expected Outcomes

#### Performance Improvements
- **80-90% reduction in LLM calls** through local-first approach
- **Significant cost reduction** while maintaining decision quality
- **Faster response times** with local signal generation
- **Better scalability** for multiple symbols and timeframes

#### Decision Quality
- **Improved consistency** through rule-based local analysis
- **Enhanced accuracy** with strategic LLM escalation for complex scenarios
- **Reduced overfitting** through diversified signal sources
- **Better risk management** through consensus-based decisions

#### Technical Benefits
- **Reduced latency** for routine trading decisions
- **Improved reliability** with local fallback capabilities
- **Better cost control** through budget governance
- **Enhanced monitoring** with comprehensive metrics and alerting

## Future Phases (Beyond Phase 6)

### Phase 7: Advanced Features & Scaling
- Multi-timeframe analysis
- Portfolio optimization algorithms
- Advanced risk management techniques
- Machine learning model integration
- Real-time streaming data processing

### Phase 8: Production Hardening
- High availability deployment
- Disaster recovery procedures
- Advanced security implementations
- Compliance and audit features
- Performance optimization at scale

### Phase 9: Intelligence Enhancement
- Reinforcement learning for strategy optimization
- Market microstructure analysis
- Alternative data integration
- Advanced sentiment analysis
- Predictive analytics

This roadmap provides a clear path forward for transforming the AI Trading System into a cost-effective, high-performance, and intelligent trading platform that combines the best of rule-based systems with strategic LLM intelligence.
