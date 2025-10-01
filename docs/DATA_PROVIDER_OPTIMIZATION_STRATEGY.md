# Data Provider Optimization Strategy

This document outlines the comprehensive strategy for optimizing data providers to support the enhanced technical analysis and local-first signal generation approach.

## Current State Analysis

### Existing Providers
1. **yfinance** - Primary provider for OHLCV data
   - Free, unlimited usage
   - Good historical data coverage
   - Some reliability issues
   - Rate limiting can be aggressive

2. **Alpha Vantage** - Secondary provider for news sentiment
   - Limited free tier (25 calls/day, 5 calls/minute)
   - Good for news sentiment data
   - Expensive paid plans
   - Too restrictive for primary data needs

### Current Limitations
- Single point of failure with yfinance
- No backup provider for OHLCV data
- Limited real-time data capabilities
- No data quality scoring or validation
- Basic caching without persistence

## Optimization Strategy

### Multi-Provider Architecture

#### Primary Provider: yfinance (Enhanced)
**Role**: Bulk historical data, backup real-time data
**Strengths**: Free, good historical coverage
**Optimizations**:
- Implement intelligent batching
- Add retry logic with exponential backoff
- Improve error handling and validation
- Optimize request patterns

#### Secondary Provider: IEX Cloud
**Role**: Real-time data, high-quality validation, backup historical data
**Strengths**:
- Generous free tier (50K calls/month)
- High-quality professional data
- Real-time capabilities
- Good reliability
**Free Tier Allocation**:
- Real-time quotes: ~1,000 calls/day
- Historical data: ~1,500 calls/day
- Total: ~50K calls/month

#### Tertiary Provider: Polygon.io
**Role**: Additional backup, specialized data
**Strengths**:
- Free tier: 5 calls/minute, 1,000 calls/month
- High-quality data
- Good for specific use cases
**Usage**: Emergency backup only

### Provider Selection Logic

#### Intelligent Provider Manager
```python
class DataProviderManager:
    """
    Manages multiple data providers with intelligent selection logic.
    """

    def __init__(self):
        self.providers = {
            "yfinance": YFinanceProvider(),
            "iex_cloud": IEXCloudProvider(),
            "polygon": PolygonProvider()
        }
        self.health_monitor = ProviderHealthMonitor()
        self.cost_tracker = CostTracker()

    def select_provider(self,
                       data_type: DataType,
                       urgency: Urgency,
                       cost_sensitivity: CostSensitivity) -> str:
        """
        Selects the best provider based on multiple factors.
        """

    def get_data_with_fallback(self,
                              symbol: str,
                              data_request: DataRequest) -> Optional[MarketData]:
        """
        Attempts to get data from providers in order of preference.
        """
```

#### Provider Selection Matrix
| Data Type | Urgency | Cost Sensitivity | Primary | Secondary | Tertiary |
|-----------|---------|------------------|---------|-----------|----------|
| Historical Bulk | Low | High | yfinance | IEX Cloud | Polygon |
| Real-time Quote | High | Medium | IEX Cloud | yfinance | Polygon |
| Intraday Data | Medium | High | IEX Cloud | yfinance | Polygon |
| Validation Data | Medium | Low | IEX Cloud | Polygon | yfinance |
| Emergency Backup | High | Low | IEX Cloud | Polygon | yfinance |

### Data Quality Management

#### Quality Scoring System
```python
class DataQualityScorer:
    """
    Scores data quality based on multiple factors.
    """

    def score_data_quality(self,
                          data: MarketData,
                          provider: str,
                          validation_results: Dict) -> QualityScore:
        """
        Calculates quality score (0.0 to 1.0) for data.
        """

        factors = {
            "completeness": self.check_completeness(data),
            "freshness": self.check_freshness(data),
            "consistency": self.check_consistency(data, provider),
            "accuracy": self.check_accuracy(data, validation_results),
            "provider_reliability": self.get_provider_reliability(provider)
        }

        return self.calculate_weighted_score(factors)
```

#### Validation Rules
1. **Completeness Check**: All required fields present
2. **Range Validation**: Prices within reasonable bounds
3. **Consistency Check**: OHLC relationships valid
4. **Freshness Check**: Data timestamps current
5. **Cross-Provider Validation**: Compare against other providers

#### Data Cleaning Pipeline
```python
class DataCleaningPipeline:
    """
    Cleans and standardizes data from multiple providers.
    """

    def clean_data(self, raw_data: Dict, provider: str) -> MarketData:
        """
        Standardizes data format and applies cleaning rules.
        """

        # Standardize column names
        standardized = self.standardize_columns(raw_data, provider)

        # Handle missing values
        filled = self.handle_missing_values(standardized)

        # Remove outliers
        cleaned = self.remove_outliers(filled)

        # Validate relationships
        validated = self.validate_relationships(cleaned)

        return self.convert_to_market_data(validated)
```

### Caching Strategy

#### Multi-Level Caching Architecture
```
┌─────────────────┐
│   Application   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   L1 Cache      │ ← Memory (Redis)
│   (Hot Data)    │   - Recent data
│   TTL: 5 min    │   - High frequency access
└─────────┬───────┘
          │
┌─────────▼───────┐
│   L2 Cache      │ ← Disk (SQLite/File)
│   (Warm Data)   │   - Historical data
│   TTL: 1 day    │   - Medium frequency access
└─────────┬───────┘
          │
┌─────────▼───────┐
│   Providers     │ ← External APIs
│   (Cold Data)   │   - Real-time fetching
│   TTL: Varies   │   - Low frequency access
└─────────────────┘
```

#### Cache Implementation
```python
class MultiLevelCache:
    """
    Implements multi-level caching with intelligent eviction.
    """

    def __init__(self):
        self.l1_cache = RedisCache()  # Hot data
        self.l2_cache = DiskCache()   # Warm data
        self.cache_stats = CacheStatistics()

    def get(self, key: str) -> Optional[MarketData]:
        """
        Gets data from cache, checking L1 then L2.
        """

        # Try L1 cache first
        data = self.l1_cache.get(key)
        if data:
            self.cache_stats.record_hit("L1")
            return data

        # Try L2 cache
        data = self.l2_cache.get(key)
        if data:
            self.cache_stats.record_hit("L2")
            # Promote to L1 if frequently accessed
            if self.should_promote_to_l1(key):
                self.l1_cache.set(key, data, ttl=300)
            return data

        self.cache_stats.record_miss()
        return None

    def set(self, key: str, data: MarketData, ttl: int):
        """
        Sets data in appropriate cache level.
        """

        # Always store in L2
        self.l2_cache.set(key, data, ttl)

        # Store in L1 for hot data
        if self.is_hot_data(key):
            self.l1_cache.set(key, data, min(ttl, 300))
```

#### Cache Key Strategy
```python
def generate_cache_key(symbol: str,
                      data_type: DataType,
                      date_range: DateRange,
                      provider: str) -> str:
    """
    Generates consistent cache keys.
    """

    components = [
        symbol.upper(),
        data_type.value,
        date_range.start.isoformat(),
        date_range.end.isoformat(),
        provider
    ]

    return hashlib.md5("_".join(components).encode()).hexdigest()
```

### Cost Optimization

#### Cost Tracking System
```python
class CostTracker:
    """
    Tracks API costs and optimizes provider usage.
    """

    def __init__(self):
        self.usage_by_provider = defaultdict(int)
        self.cost_by_provider = defaultdict(float)
        self.monthly_budgets = {
            "iex_cloud": 0,  # Free tier
            "polygon": 0,     # Free tier
            "yfinance": 0     # Free
        }

    def track_api_call(self, provider: str, cost: float):
        """
        Tracks API usage and costs.
        """

        self.usage_by_provider[provider] += 1
        self.cost_by_provider[provider] += cost

        # Check budget limits
        if self.cost_by_provider[provider] > self.monthly_budgets[provider]:
            self.handle_budget_exceeded(provider)

    def optimize_provider_selection(self,
                                   data_request: DataRequest) -> str:
        """
        Selects provider based on cost optimization.
        """

        # Prefer free providers when possible
        if data_request.can_use_free_data:
            return "yfinance"

        # Use paid providers strategically
        if data_request.requires_high_quality:
            return "iex_cloud"

        # Default to most cost-effective
        return self.get_most_cost_effective_provider(data_request)
```

#### Smart Batching
```python
class BatchOptimizer:
    """
    Optimizes API calls through intelligent batching.
    """

    def optimize_requests(self,
                         requests: List[DataRequest]) -> List[BatchRequest]:
        """
        Combines individual requests into optimal batches.
        """

        # Group by provider and data type
        grouped = self.group_by_provider_and_type(requests)

        # Optimize batch sizes
        optimized_batches = []
        for provider, provider_requests in grouped.items():
            batches = self.create_optimal_batches(provider, provider_requests)
            optimized_batches.extend(batches)

        return optimized_batches

    def create_optimal_batches(self,
                              provider: str,
                              requests: List[DataRequest]) -> List[BatchRequest]:
        """
        Creates batches optimized for provider limits.
        """

        provider_limits = self.get_provider_limits(provider)
        batches = []
        current_batch = BatchRequest(provider)

        for request in requests:
            if current_batch.can_add(request, provider_limits):
                current_batch.add(request)
            else:
                batches.append(current_batch)
                current_batch = BatchRequest(provider)
                current_batch.add(request)

        if current_batch.has_requests():
            batches.append(current_batch)

        return batches
```

### Performance Optimization

#### Parallel Data Fetching
```python
class ParallelDataFetcher:
    """
    Fetches data from multiple providers in parallel.
    """

    async def fetch_multiple_symbols(self,
                                   symbols: List[str],
                                   data_request: DataRequest) -> Dict[str, MarketData]:
        """
        Fetches data for multiple symbols concurrently.
        """

        # Create tasks for each symbol
        tasks = []
        for symbol in symbols:
            task = self.fetch_single_symbol(symbol, data_request)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        data_by_symbol = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                continue
            data_by_symbol[symbol] = result

        return data_by_symbol
```

#### Connection Pooling
```python
class ConnectionPoolManager:
    """
    Manages connection pools for optimal performance.
    """

    def __init__(self):
        self.pools = {
            "yfinance": aiohttp.TCPConnector(limit=10, limit_per_host=5),
            "iex_cloud": aiohttp.TCPConnector(limit=5, limit_per_host=2),
            "polygon": aiohttp.TCPConnector(limit=2, limit_per_host=1)
        }

    def get_session(self, provider: str) -> aiohttp.ClientSession:
        """
        Gets a session with appropriate connection pool.
        """

        return aiohttp.ClientSession(
            connector=self.pools[provider],
            timeout=aiohttp.ClientTimeout(total=30)
        )
```

### Reliability & Fault Tolerance

#### Health Monitoring
```python
class ProviderHealthMonitor:
    """
    Monitors provider health and performance.
    """

    def __init__(self):
        self.health_scores = defaultdict(float)
        self.response_times = defaultdict(list)
        self.error_rates = defaultdict(float)

    def record_api_call(self,
                       provider: str,
                       response_time: float,
                       success: bool):
        """
        Records API call performance metrics.
        """

        self.response_times[provider].append(response_time)

        # Keep only recent measurements
        if len(self.response_times[provider]) > 100:
            self.response_times[provider] = self.response_times[provider][-100:]

        # Update error rate
        recent_calls = 50
        if not hasattr(self, 'recent_results'):
            self.recent_results = defaultdict(list)

        self.recent_results[provider].append(success)
        if len(self.recent_results[provider]) > recent_calls:
            self.recent_results[provider] = self.recent_results[provider][-recent_calls:]

        success_rate = sum(self.recent_results[provider]) / len(self.recent_results[provider])
        self.error_rates[provider] = 1.0 - success_rate

        # Update health score
        self.update_health_score(provider)

    def is_healthy(self, provider: str) -> bool:
        """
        Determines if a provider is healthy enough to use.
        """

        return self.health_scores[provider] > 0.7
```

#### Automatic Failover
```python
class FailoverManager:
    """
    Manages automatic failover between providers.
    """

    def __init__(self, provider_manager: DataProviderManager):
        self.provider_manager = provider_manager
        self.health_monitor = ProviderHealthMonitor()
        self.failover_history = []

    async def get_data_with_failover(self,
                                   symbol: str,
                                   data_request: DataRequest) -> Optional[MarketData]:
        """
        Attempts to get data with automatic failover.
        """

        providers = self.get_provider_order(data_request)

        for provider in providers:
            if not self.health_monitor.is_healthy(provider):
                continue

            try:
                data = await self.provider_manager.get_data_from_provider(
                    provider, symbol, data_request
                )

                if data and self.validate_data(data):
                    self.health_monitor.record_api_call(provider, 0.1, True)
                    return data

            except Exception as e:
                self.health_monitor.record_api_call(provider, 0.1, False)
                logger.warning(f"Provider {provider} failed: {e}")
                continue

        # All providers failed
        self.handle_complete_failure(symbol, data_request)
        return None
```

### Implementation Timeline

#### Phase 1: Foundation (Week 1)
1. Implement IEX Cloud provider
2. Create provider manager interface
3. Add basic health monitoring
4. Implement multi-level caching

#### Phase 2: Optimization (Week 2)
1. Add intelligent provider selection
2. Implement data quality scoring
3. Add cost tracking system
4. Optimize caching strategies

#### Phase 3: Performance (Week 3)
1. Implement parallel data fetching
2. Add connection pooling
3. Optimize batching strategies
4. Add performance monitoring

#### Phase 4: Reliability (Week 4)
1. Enhance health monitoring
2. Implement automatic failover
3. Add comprehensive error handling
4. Create provider reliability scoring

### Expected Benefits

#### Performance Improvements
- **50-70% reduction** in data fetch latency through caching
- **10x improvement** in multi-symbol data fetching through parallelization
- **90% reduction** in API calls through intelligent caching

#### Cost Optimization
- **Zero cost** for most historical data through yfinance optimization
- **Strategic use** of IEX Cloud free tier for high-value data
- **80% reduction** in paid API calls through caching and optimization

#### Reliability Improvements
- **99.9% uptime** through multi-provider architecture
- **Automatic failover** ensures data availability
- **Quality scoring** ensures data reliability

This optimization strategy provides a robust, cost-effective, and high-performance data infrastructure that supports the enhanced technical analysis and local-first signal generation approach.
