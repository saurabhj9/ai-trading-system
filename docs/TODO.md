# TODO

This document lists the immediate, actionable tasks for the current development phase.

## Phase 2: Data Pipeline & Integration

### Milestone 1: Data Provider Interface

-   [ ] Create `src/data/providers/base_provider.py` with an abstract `BaseDataProvider` class.
-   [ ] Implement a `yfinance_provider.py` that inherits from the base provider and fetches data using the `yfinance` library.
-   [ ] Implement an `alpha_vantage_provider.py` that inherits from the base provider.
-   [ ] Add basic error handling and rate limiting to the providers.

### Milestone 2: Data Processing Pipeline

-   [ ] Create `src/data/pipeline.py`.
-   [ ] Implement a `DataPipeline` class that takes a data provider as input.
-   [ ] Add a method to the pipeline to fetch data and calculate a basic set of technical indicators (e.g., RSI, MACD) using `pandas-ta`.
-   [ ] The pipeline should return a `MarketData` object.

### Milestone 3: Caching Layer

-   [ ] Create `src/data/cache.py`.
-   [ ] Implement a `CacheManager` with methods for saving and retrieving data.
-   [ ] For now, the cache can be a simple in-memory dictionary.
-   [ ] Integrate the cache into the `DataPipeline` to avoid redundant data fetching.

### Milestone 4: Integration

-   [ ] Update the `Orchestrator` to use the `DataPipeline` to fetch data before running the agents.
-   [ ] Create a new integration test in `tests/integration/test_data_pipeline.py` to verify the end-to-end data flow.
