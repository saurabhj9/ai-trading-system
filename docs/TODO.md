# TODO

This document lists the immediate, actionable tasks for the current development phase.

## Phase 6: Enhanced Technical Analysis & Local-First Strategy

This phase focuses on transforming the system from LLM-dependent to a sophisticated hybrid that leverages local processing for routine analysis and strategic LLM escalation for complex scenarios. The goal is to reduce costs by 80-90% while maintaining or improving decision quality.

### Priority 1: Foundation (High Impact, Low Complexity)

#### Milestone 1: Expand Technical Indicators - COMPLETED
**Timeline: 1-2 weeks**
**Completion Date: 2025-10-01**

-   [x] Extend `src/data/pipeline.py` to calculate additional indicators
    -   [x] **Momentum Indicators**: Stochastic Oscillator, Williams %R, CCI
    -   [x] **Mean Reversion Indicators**: Bollinger Bands, Keltner Channels, Donchian Channels
    -   [x] **Volatility Indicators**: ATR, Historical Volatility, Chaikin Volatility
    -   [x] **Trend Indicators**: EMAs (multiple periods), ADX, Aroon, Parabolic SAR
    -   [x] **Volume Indicators**: OBV, VWAP, MFI, Accumulation/Distribution Line
    -   [x] **Statistical Indicators**: Hurst Exponent, Z-Score, Correlation analysis
-   [x] Organize indicators by category in data structures
-   [x] Add indicator metadata (category, reliability, typical usage)

**Summary**: Successfully implemented 20+ new technical indicators across 6 categories, expanding the system's analytical capabilities from basic indicators to a comprehensive technical analysis toolkit. All indicators are now organized by category with proper metadata for reliability and usage patterns.

#### Milestone 2: Local Signal Generation Framework
**Timeline: 2-3 weeks**

-   [ ] Create `SignalGenerator` class for local decision making
    -   [ ] Implement indicator categorization and scoring system
    -   [ ] Build market regime detection (trending vs. ranging vs. volatile)
    -   [ ] Create context-aware weighting system per regime
    -   [ ] Implement consensus-based signal combination logic
    -   [ ] Add confidence calculation based on signal agreement
-   [ ] Design hierarchical decision tree for signal processing
-   [ ] Create signal validation and conflict detection

#### Milestone 3: Data Provider Optimization
**Timeline: 1 week**

-   [ ] Optimize current yfinance usage with better caching and batch requests
-   [ ] Add IEX Cloud as secondary provider (50K free calls/month)
-   [ ] Implement provider selection logic with automatic fallbacks
-   [ ] Add data quality scoring and validation
-   [ ] Create data provider manager for intelligent source selection

### Priority 2: Event-Driven System (High Impact, Medium Complexity)

#### Milestone 4: Event-Driven Triggers
**Timeline: 2-3 weeks**

-   [ ] Implement comprehensive trigger detection system
    -   [ ] **Technical Triggers**: RSI/MACD crossovers, Bollinger Band breaches, Stochastic extremes
    -   [ ] **Volatility Triggers**: ATR spikes, volatility regime changes, VIX-like calculations
    -   [ ] **Trend Triggers**: ADX threshold crosses, moving average crossovers, Aroon signals
    -   [ ] **Conflict Triggers**: Divergent signals across indicator categories
    -   [ ] **Regime Triggers**: Market structure breaks, trend direction changes
-   [ ] Add configurable trigger thresholds and sensitivity settings
-   [ ] Implement decision TTL and cooldown mechanisms
-   [ ] Create trigger validation and false-positive filtering

#### Milestone 5: State-Hash Decision Cache
**Timeline: 1-2 weeks**

-   [ ] Implement feature vector quantization for compact representation
-   [ ] Create state hashing system for decision caching
-   [ ] Add cache persistence and retrieval logic with Redis
-   [ ] Implement cache hit/miss tracking and effectiveness metrics
-   [ ] Add cache invalidation logic for significant market changes

### Priority 3: Strategic LLM Integration (Medium Impact, Medium Complexity)

#### Milestone 6: LLM Escalation Logic
**Timeline: 2 weeks**

-   [ ] Implement escalation triggers for complex scenarios
    -   [ ] Conflicting signals across indicator categories
    -   [ ] Regime transition uncertainty
    -   [ ] Unusual market conditions and black swan events
    -   [ ] Multi-timeframe divergences
    -   [ ] Fundamental-technical divergences
-   [ ] Design specialized prompts for conflict resolution
-   [ ] Add LLM decision caching for similar conflict patterns
-   [ ] Create escalation confidence scoring

#### Milestone 7: Budget and Governance
**Timeline: 1 week**

-   [ ] Implement per-session LLM call budgets and tracking
-   [ ] Add rate limiting and degradation logging
-   [ ] Create cost tracking and optimization metrics
-   [ ] Implement explicit degradation modes when budgets exhausted
-   [ ] Add LLM usage analytics and reporting

### Priority 4: Optimization & Testing (Medium Impact, High Complexity)

#### Milestone 8: Performance Optimization
**Timeline: 2-3 weeks**

-   [ ] Optimize technical indicator calculations with vectorization
-   [ ] Implement incremental state updates to reduce overhead
-   [ ] Profile and optimize message bus performance
-   [ ] Add comprehensive monitoring metrics
    -   [ ] Cache hit rates, LLM call frequency, token usage
    -   [ ] Decision time, event counts, cost tracking
    -   [ ] Signal accuracy and confidence distributions

#### Milestone 9: Comprehensive Backtesting & Validation
**Timeline: 3-4 weeks**

-   [ ] Run multi-symbol backtests (AAPL, GOOGL, MSFT) over 2-year periods
-   [ ] Compare performance: local-only vs. local+LLM hybrid approaches
-   [ ] Analyze decision consistency across market regimes
-   [ ] Optimize parameters based on backtest results
-   [ ] Validate cost reduction and performance improvements
-   [ ] Create performance reports and recommendations

---

<details>
<summary>Completed Work (Phases 3-5)</summary>

## Phase 3: Risk, Portfolio Management & Backtesting - COMPLETED

All Phase 3 deliverables have been implemented.

### Milestone 1: Advanced Agent Development - COMPLETED

-   [x] Create `src/agents/risk.py` with a `RiskManagementAgent` class.
-   [x] Implement basic risk calculations.
-   [x] Create `src/agents/portfolio.py` with a `PortfolioManagementAgent` class.
-   [x] Implement logic in the `PortfolioManagementAgent` to synthesize decisions from other agents.
-   [x] Update the `Orchestrator` to include the new agents in the workflow.

### Milestone 2: Backtesting Engine Setup - COMPLETED

-   [x] Chose `backtrader` as the backtesting library and added to dependencies.
-   [x] Create `src/backtesting/engine.py`.
-   [x] Implement a basic backtesting `Strategy` that uses deterministic agents for signals.
-   [x] Implement a script in `src/backtesting/run_backtest.py` to configure and run the backtesting engine.

### Milestone 3: Initial Backtest & Reporting - COMPLETED

-   [x] Run an initial backtest using the implemented strategy.
-   [x] Basic performance metrics are output in the script.
-   [x] Backtest script outputs a simple performance report.

## Phase 4: API & Deployment - COMPLETED

All Phase 4 deliverables have been implemented.

### Milestone 1: FastAPI Application Development - COMPLETED

-   [x] Create `src/api/app.py` with FastAPI application setup.
-   [x] Implement `src/api/routes/signals.py`, `src/api/routes/status.py`, and `src/api/routes/monitoring.py`.
-   [x] Set up proper dependency injection for agents.

### Milestone 2: Containerization & Orchestration - COMPLETED

-   [x] Create `Dockerfile` and `docker-compose.yml`.
-   [x] Configure environment variables and service dependencies.

### Milestone 3: Production Deployment - COMPLETED

-   [x] Implement `scripts/deploy.sh` for automated deployment.
-   [x] Set up structured logging and basic monitoring.

## Phase 5: Refinement & Optimization - COMPLETED

This phase focused on replacing mock implementations with production-ready logic, externalizing configurations, and enhancing overall system performance and reliability.

### Milestone 1: Implement Core Agent Logic - COMPLETED

-   [x] **LLM Response Parsing:** Implemented robust parsing logic in all agents.
-   [x] **Sentiment Analysis Agent:** Integrated a live news data provider.
-   [x] **Risk Management Agent:** Implemented actual risk calculations.

### Milestone 2: Externalize Configurations - COMPLETED

-   [x] **Centralize Settings:** Moved hardcoded values into a structured configuration system.
-   [x] **Secure API:** Updated the FastAPI CORS policy to use configured allowed origins.

### Milestone 3: Optimization and Tuning - COMPLETED

-   [x] **Performance Analysis & Profiling:** Completed comprehensive profiling of the data pipeline, LLM client, and agent processing.
-   [x] **Implement Identified Optimizations:**
    -   [x] **LLM Response Time Optimization:** Implemented caching, connection pooling, and batching.
    -   [x] **LLM Model Evaluation:** Benchmarked models and selected `Claude-3-haiku`.
    -   [x] **Prompt Engineering & Historical Data:** Refined prompts and implemented historical data in `TechnicalAgent`.
    -   [x] **Data Pipeline Optimization:** Fixed data fetching bottlenecks and implemented parallel fetching.
    -   [x] **Agent Processing Parallelization:** Implemented parallel execution for independent agents.
-   [x] **Critical Issues Resolved:** Fixed agent serialization and parameter passing errors.

</details>
