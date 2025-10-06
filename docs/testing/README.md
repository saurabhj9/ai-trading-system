# AI Trading System Testing Guide

Welcome to the comprehensive testing guide for the AI Trading System. This document provides an overview of the testing structure, conventions, and how to get started with running and writing tests.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Structure Overview](#test-structure-overview)
- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Data and Fixtures](#test-data-and-fixtures)
- [Mocking and Test Isolation](#mocking-and-test-isolation)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

## Testing Philosophy

Our testing approach is based on the following principles:

1. **Comprehensive Coverage**: We test at multiple levels - unit, integration, end-to-end, performance, and validation
2. **Test-Driven Development**: Write tests before or alongside production code
3. **Fast Feedback**: Unit tests should be fast and provide immediate feedback
4. **Realistic Scenarios**: Integration and e2e tests should use realistic data and scenarios
5. **Maintainable Tests**: Tests should be clear, concise, and easy to maintain
6. **Continuous Testing**: Tests run automatically in CI/CD pipelines

## Test Structure Overview

```
tests/
├── unit/                    # Fast, isolated tests for individual components
│   ├── agents/             # Agent-specific unit tests
│   ├── data/               # Data-related unit tests
│   │   └── indicators/     # Technical indicator tests
│   ├── signal_generation/  # Signal generation component tests
│   ├── communication/      # Communication layer tests
│   ├── llm/               # LLM client tests
│   └── api/               # API endpoint tests
├── integration/           # Tests for component interactions
├── e2e/                  # End-to-end workflow tests
├── performance/          # Performance and profiling tests
├── validation/           # Validation and quality tests
├── comparison/           # Comparison tests between approaches
└── fixtures/             # Shared test data and utilities
```

## Quick Start

### Prerequisites

- Python 3.8+
- uv package manager installed
- Test dependencies installed (`uv sync --dev`)

### Running Tests

The easiest way to run tests is using our test runner script:

```bash
# Run quick tests (unit + integration) - recommended for development
uv run scripts/testing/run_all_tests.py --category quick

# Run all tests
uv run scripts/testing/run_all_tests.py --category all

# Run specific test categories
uv run scripts/testing/run_all_tests.py --category unit
uv run scripts/testing/run_all_tests.py --category integration
uv run scripts/testing/run_all_tests.py --category performance

# Run with coverage
uv run scripts/testing/run_all_tests.py --category unit --coverage

# Run with verbose output
uv run scripts/testing/run_all_tests.py --category unit --verbose
```

### Using pytest directly

You can also use pytest directly:

```bash
# Run all unit tests
uv run pytest tests/unit -m unit

# Run specific test file
uv run pytest tests/unit/agents/test_technical_agent.py -v

# Run with coverage
uv run pytest tests/unit --cov=src --cov-report=html

# Run tests matching a pattern
uv run pytest tests/unit -k "test_signal_generation" -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation
**Characteristics**: Fast, focused, no external dependencies
**Examples**: Testing a single agent's decision logic, indicator calculations

```python
@pytest.mark.unit
def test_technical_agent_signal_generation(sample_market_data):
    agent = TechnicalAnalysisAgent(config, mock_llm, mock_bus, mock_state)
    decision = agent.analyze(sample_market_data)
    assert decision.signal in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= decision.confidence <= 1.0
```

### Integration Tests (`tests/integration/`)

**Purpose**: Test interactions between components
**Characteristics**: Real dependencies, focused on interfaces
**Examples**: Testing agent communication, data pipeline integration

```python
@pytest.mark.integration
async def test_data_pipeline_integration():
    pipeline = DataPipeline(provider, cache)
    data = await pipeline.fetch_and_process_data("AAPL", start_date, end_date)
    assert data is not None
    assert "RSI" in data.technical_indicators
```

### End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete workflows from start to finish
**Characteristics**: Real environment, full system integration
**Examples**: Complete trading workflow, API endpoint testing

```python
@pytest.mark.e2e
async def test_complete_trading_workflow():
    # Test the entire workflow from data fetch to final decision
    orchestrator = Orchestrator(...)
    result = await orchestrator.run("AAPL", start_date, end_date)
    assert result["final_decision"] is not None
```

### Performance Tests (`tests/performance/`)

**Purpose**: Validate performance requirements
**Characteristics**: Measure timing, throughput, resource usage
**Examples**: Signal generation latency, agent performance

```python
@pytest.mark.performance
@pytest.mark.slow
async def test_signal_generation_latency():
    start_time = time.time()
    signal = await generate_signal(market_data)
    latency = time.time() - start_time
    assert latency < 0.1  # 100ms target
```

### Validation Tests (`tests/validation/`)

**Purpose**: Validate system quality and correctness
**Characteristics**: Quality checks, validation criteria
**Examples**: Signal validation, rollback procedures

### Comparison Tests (`tests/comparison/`)

**Purpose**: Compare different approaches or implementations
**Characteristics**: Side-by-side comparisons, A/B testing
**Examples**: Local vs LLM signal generation comparison

## Running Tests

### Test Categories

```bash
# Unit tests only (fastest)
pytest tests/unit -m unit

# Integration tests
pytest tests/integration -m integration

# End-to-end tests (slowest)
pytest tests/e2e -m e2e

# Performance tests
pytest tests/performance -m performance

# Validation tests
pytest tests/validation -m validation

# Comparison tests
pytest tests/comparison -m comparison
```

### Test Selection

```bash
# Run tests by keyword
pytest -k "signal_generation" -v

# Run tests by file
pytest tests/unit/agents/test_technical_agent.py

# Run tests by marker
pytest -m "unit and not slow"

# Run failed tests only
pytest --lf

# Run tests with specific verbosity
pytest -v  # Verbose output
pytest -vv # Very verbose output
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/unit --cov=src --cov-report=html

# Generate terminal coverage report
pytest tests/unit --cov=src --cov-report=term-missing

# Fail if coverage is below threshold
pytest tests/unit --cov=src --cov-fail-under=80
```

## Writing Tests

### Test Structure

Follow the AAA pattern (Arrange, Act, Assert):

```python
def test_indicator_calculation():
    # Arrange
    market_data = create_test_market_data()
    indicator = RSIIndicator(period=14)

    # Act
    result = indicator.calculate(market_data)

    # Assert
    assert 0 <= result <= 100
    assert isinstance(result, float)
```

### Test Naming

Use descriptive names that explain what the test does:

```python
# Good
def test_rsi_indicator_returns_value_between_0_and_100()
def test_technical_agent_generates_buy_signal_when_rsi_oversold()

# Avoid
def test_rsi()
def test_agent()
```

### Fixtures

Use fixtures for common test setup:

```python
@pytest.fixture
def oversold_market_data():
    return MarketData(
        symbol="TEST",
        technical_indicators={"RSI": 25.0}  # Oversold condition
    )

def test_technical_agent_buy_signal_oversold(oversold_market_data):
    agent = TechnicalAnalysisAgent(...)
    decision = agent.analyze(oversold_market_data)
    assert decision.signal == "BUY"
```

### Async Tests

For async functions, use the `async` keyword and appropriate test markers:

```python
@pytest.mark.asyncio
async def test_async_agent_analysis():
    agent = TechnicalAnalysisAgent(...)
    decision = await agent.analyze(market_data)
    assert decision is not None
```

## Test Data and Fixtures

### Using Fixtures

The `tests/conftest.py` file provides common fixtures:

- `sample_market_data`: Standard market data for testing
- `agent_config`: Default agent configuration
- `mock_dependencies`: Mocked LLM, message bus, and state manager
- `test_symbols`: Common test symbols (AAPL, GOOGL, MSFT, etc.)

### Creating Custom Fixtures

```python
@pytest.fixture
def bullish_market_data():
    return MarketData(
        symbol="BULL",
        price=150.0,
        technical_indicators={
            "RSI": 70.0,  # Overbought
            "MACD": 2.0,  # Bullish
        }
    )
```

### Test Data Files

Store test data in `tests/fixtures/`:

```
tests/fixtures/
├── market_data/
│   ├── sample_ohlcv.csv
│   └── historical_data.json
├── mock_responses/
│   └── llm_responses.json
└── test_configs/
    └── agent_configs.json
```

## Mocking and Test Isolation

### Mocking External Dependencies

Use the `unittest.mock` library for mocking:

```python
from unittest.mock import Mock, AsyncMock, patch

def test_agent_with_mock_llm():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = '{"signal": "BUY", "confidence": 0.8}'

    agent = TechnicalAnalysisAgent(config, mock_llm, ...)
    decision = await agent.analyze(market_data)

    assert decision.signal == "BUY"
    mock_llm.generate.assert_called_once()
```

### Patching

Use patch to temporarily replace components:

```python
@patch('src.data.providers.yfinance_provider.YFinanceProvider.fetch_data')
def test_data_pipeline_with_mock_provider(mock_fetch):
    mock_fetch.return_value = sample_dataframe

    pipeline = DataPipeline(provider)
    data = pipeline.fetch_and_process_data("AAPL", start, end)

    assert data is not None
    mock_fetch.assert_called_once()
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Push to main branch
- Scheduled runs

### Test Categories in CI

- **PR Checks**: Unit + Integration tests (fast feedback)
- **Main Branch**: All tests including performance
- **Nightly**: Full test suite with coverage

### Coverage Requirements

- Unit tests: Minimum 80% coverage
- Integration tests: Minimum 70% coverage
- Overall: Minimum 75% coverage

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure you're running from project root
python -m pytest tests/

# Or add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Async Test Issues

```python
# Make sure to use the async marker
@pytest.mark.asyncio
async def test_async_function():
    await some_async_function()
```

#### Fixture Not Found

```bash
# Check that conftest.py is in the right place
ls tests/conftest.py

# Verify fixture name spelling
pytest --collect-only | grep fixture_name
```

#### Performance Test Timeouts

```bash
# Increase timeout for slow tests
pytest tests/performance --timeout=300
```

### Debugging Tests

```bash
# Run with pdb debugger
pytest --pdb tests/unit/test_file.py::test_function

# Show local variables on failure
pytest -l tests/unit/test_file.py

# Stop on first failure
pytest -x tests/unit/

# Run with maximum verbosity
pytest -vv tests/unit/test_file.py
```

### Getting Help

1. Check the test logs for specific error messages
2. Look at existing similar tests for patterns
3. Review the testing documentation in this folder
4. Ask in the team's testing channel

## Best Practices

1. **Keep tests focused**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should explain what they're testing
3. **Mock external dependencies**: Tests should be reliable and fast
4. **Use fixtures**: Avoid code duplication in test setup
5. **Test edge cases**: Don't just test the happy path
6. **Maintain test independence**: Tests shouldn't depend on each other
7. **Keep tests updated**: Update tests when code changes
8. **Document complex tests**: Add comments for complex test logic

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Mocking Guide](https://docs.python.org/3/library/unittest.mock.html)
