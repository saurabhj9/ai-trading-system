# AI Trading System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source project to develop a multi-agent system for financial analysis and trade signal generation. This system uses a graph-based agentic workflow to analyze assets from multiple perspectives, leveraging a team of specialized AI agents to perform a comprehensive analysis.

**Note:** This project includes a functional signal generation framework with both local and LLM-based analysis capabilities. It is for educational and research purposes only and is not intended for live trading.

## Table of Contents

- [Features](#features)
- [Local Signal Generation Framework](#local-signal-generation-framework)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the API Server](#running-the-api-server)
  - [Generating Trading Signals](#generating-trading-signals)
  - [Signal Generation Workflow](#signal-generation-workflow)
  - [Local Signal Generation](#local-signal-generation)
  - [Alternative API Server Start Method](#alternative-api-server-start-method)
  - [Migration Framework](#migration-framework)
  - [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Testing Documentation](#testing-documentation)

## Features

- **Multi-Agent Analysis:** Utilizes a team of specialized AI agents (e.g., Technical Analyst, Sentiment Analyst, Risk Manager) for comprehensive market analysis.
- **Local Signal Generation Framework:** A complete local signal generation system that produces trading signals without relying on LLM calls, featuring market regime detection, indicator scoring, consensus building, and conflict resolution.
- **Hybrid Signal Generation:** Seamlessly integrates local signal generation with LLM-based analysis, providing the best of both approaches with intelligent escalation logic.
- **Graph-Based Workflow:** Employs LangGraph to orchestrate agent interactions, ensuring a stateful and logical flow of information.
- **Migration Framework:** Provides a structured approach for transitioning from LLM-only to hybrid signal generation with rollback capabilities.
- **Extensible Provider Network:** Supports multiple data providers (e.g., yfinance, Alpha Vantage) and LLM providers (e.g., OpenAI, Anthropic, Groq, Google).
- **REST API:** Provides a FastAPI-based backend for interaction with the system, allowing for easy integration with other services.
- **Backtesting Framework:** Includes tools to backtest trading strategies and evaluate performance.

## Local Signal Generation Framework

The Local Signal Generation Framework enables the system to generate trading signals without relying on LLM calls. This provides significant improvements in:

- **Performance**: Signal generation latency reduced from seconds to milliseconds
- **Cost**: Up to 70% reduction in operational costs by minimizing LLM API calls
- **Reliability**: Consistent signal generation without external API dependencies

### Key Components

1. **Market Regime Detector**: Identifies current market conditions (trending, ranging, volatile)
2. **Indicator Scorer**: Evaluates multiple technical indicators and assigns scores
3. **Consensus Signal Combiner**: Aggregates indicator signals using weighted consensus
4. **Hierarchical Decision Tree**: Makes final signal decisions based on market context
5. **Signal Validator**: Validates signals against historical performance and market conditions
6. **Conflict Detector**: Identifies conflicts between different signal sources
7. **LLM Escalation Logic**: Intelligently decides when to escalate to LLM-based analysis

### Signal Types

The framework generates three types of signals:
- **BUY**: Indicates a potential upward price movement
- **SELL**: Indicates a potential downward price movement
- **HOLD**: Indicates no clear directional signal

Each signal includes:
- Signal strength (WEAK, MODERATE, STRONG, VERY_STRONG)
- Confidence score (0.0 to 1.0)
- Market regime at time of generation
- Reasoning explaining the signal decision
- Contributing indicators and their weights

## Architecture

For a detailed explanation of the system design, agent workflow, and technology stack, please see the [**Architecture Document**](docs/ARCHITECTURE.md).

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for environment and package management)
- An active internet connection

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/ai-trading-system.git
    cd ai-trading-system
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```sh
    uv sync
    ```

4.  **Set up environment variables:**
    -   Copy the example environment file:
        ```sh
        cp .env.example .env
        ```
    -   Open the `.env` file and add your API keys. At a minimum, you need to set:
        -   `OPENROUTER_API_KEY`: Your API key for OpenRouter.
        -   `DATA_ALPHA_VANTAGE_API_KEY`: Your API key for Alpha Vantage.

## Usage

### Running the API Server

The `main.py` script starts the FastAPI web server that provides API endpoints for generating trading signals:

```sh
uv run main.py
```

**What to expect when running main.py:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The server will start and wait for API requests. No trading analysis is performed until you make specific API calls.

### Generating Trading Signals

Once the server is running, you can generate trading signals by making API requests:

1. **Interactive API Documentation**: Visit `http://127.0.0.1:8000/docs` in your browser
2. **Direct API Calls**: Use curl or any HTTP client to request signals

Example curl command to generate a signal for Apple (AAPL):
```sh
curl "http://127.0.0.1:8000/api/v1/signals/AAPL?days=30"
```

**Example API Response:**
```json
{
  "symbol": "AAPL",
  "analysis_period": {
    "start_date": "2023-01-01T00:00:00",
    "end_date": "2023-01-31T00:00:00",
    "days": 30
  },
  "final_decision": {
    "signal": "BUY",
    "confidence": 0.75,
    "reasoning": "Technical indicators show bullish momentum with strong support at current levels...",
    "timestamp": "2023-01-31T12:00:00"
  },
  "agent_decisions": {
    "technical": {
      "signal": "BUY",
      "confidence": 0.80,
      "reasoning": "Price above key moving averages with positive MACD crossover...",
      "timestamp": "2023-01-31T12:00:00"
    },
    "sentiment": {
      "signal": "HOLD",
      "confidence": 0.60,
      "reasoning": "Mixed news sentiment with neutral analyst ratings...",
      "timestamp": "2023-01-31T12:00:00"
    },
    "risk": {
      "signal": "BUY",
      "confidence": 0.70,
      "reasoning": "Acceptable risk-reward ratio with proper position sizing...",
      "timestamp": "2023-01-31T12:00:00"
    }
  }
}
```

### Signal Generation Workflow

When you request a trading signal, the system executes this workflow:

1. **Data Collection**: Fetches market data for the requested symbol and time period
2. **Multi-Agent Analysis**: Runs analysis through specialized agents:
   - Technical Analysis Agent: Analyzes price patterns and technical indicators
   - Sentiment Analysis Agent: Analyzes news and market sentiment
   - Risk Management Agent: Evaluates risk factors
   - Portfolio Management Agent: Makes final trading decision
3. **Signal Generation**: Returns a structured response with:
   - Final signal (BUY/SELL/HOLD)
   - Confidence score (0.0-1.0)
   - Detailed reasoning for the decision
   - Individual agent decisions for transparency

### Local Signal Generation

The system includes a Local Signal Generation Framework that can produce trading signals without relying on LLM calls. You can run an example to see it in action:

```sh
uv run python examples/signal_generation_example.py
```

This example demonstrates:
- Market regime detection
- Indicator scoring and consensus building
- Signal validation and conflict detection
- Performance metrics tracking

### Alternative API Server Start Method

You can also start the API server directly with uvicorn:

```sh
uv run uvicorn src.api.app:app --reload
```

This is equivalent to running `main.py` and provides the same functionality.

### Migration Framework

The project includes a migration framework to help transition from LLM-only to hybrid signal generation:

```python
from src.migration.signal_generation_migration import initialize_migration

# Initialize migration manager
migration = initialize_migration()

# Apply current migration phase configuration
config_updates = migration.apply_phase_configuration()

# Check migration status
status = migration.get_migration_status()
print(f"Current phase: {status['phase_name']} ({status['migration_progress']:.1f}%)")
```

The migration framework supports:
- Gradual rollout of local signal generation
- Performance comparison between approaches
- Automated rollback if issues arise
- Configuration management for migration phases

### Running Tests

The AI Trading System features a comprehensive, organized testing framework with multiple test categories. The easiest way to run tests is using our universal test runner:

```sh
# Run quick tests (unit + integration) - recommended for development
uv run tests/run_all_tests.py --category quick

# Run all tests
uv run tests/run_all_tests.py --category all

# Run specific test categories
uv run tests/run_all_tests.py --category unit
uv run tests/run_all_tests.py --category integration
uv run tests/run_all_tests.py --category performance

# Run with coverage
uv run tests/run_all_tests.py --category unit --coverage
```

#### Test Categories

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests for component interactions
- **End-to-End Tests**: Complete workflow tests
- **Performance Tests**: Performance and profiling tests
- **Validation Tests**: Quality and validation tests
- **Comparison Tests**: Comparison between different approaches

For more detailed information about testing, including best practices and writing tests, see the [Testing Documentation](docs/testing/).

#### Using pytest directly

You can also use pytest directly for more control:

```sh
# Run all unit tests
uv run pytest tests/unit -m unit

# Run specific test file
uv run pytest tests/unit/agents/test_technical_agent.py -v

# Run with coverage
uv run pytest tests/unit --cov=src --cov-report=html
```

## Project Structure

Here is a high-level overview of the project's directory structure:

```
├───.github/         # GitHub Actions and CI/CD workflows
├───data/            # Data storage for logs, cache, and backtest results
├───docs/            # Project documentation
│   └───testing/     # Comprehensive testing documentation
├───examples/        # Example usage scripts
├───performance_charts/ # Performance visualization charts
├───scripts/         # Utility and deployment scripts
├───src/             # Main source code
│   ├───agents/      # Core analysis agents
│   ├───analysis/    # Market analysis components
│   ├───api/         # FastAPI application
│   ├───backtesting/ # Backtesting engine and components
│   ├───communication/ # Message bus and orchestration
│   ├───config/      # Configuration and settings
│   ├───data/        # Data pipelines and providers
│   ├───llm/         # LLM client integrations
│   ├───migration/   # Migration utilities for transitioning to hybrid signal generation
│   ├───signal_generation/ # Local Signal Generation Framework
│   │   ├───components/ # Signal generation components
│   │   ├───core.py   # Core data structures and interfaces
│   │   └───signal_generator.py # Main signal generator class
│   ├───strategies/  # Trading strategies
│   └───utils/       # Utility functions and helpers
├───tests/           # Comprehensive test suite
│   ├───unit/        # Fast, isolated tests for individual components
│   │   ├───agents/  # Agent-specific unit tests
│   │   ├───data/    # Data-related unit tests
│   │   │   └───indicators/ # Technical indicator tests
│   │   ├───signal_generation/ # Signal generation component tests
│   │   ├───communication/ # Communication layer tests
│   │   ├───llm/     # LLM client tests
│   │   └───api/     # API endpoint tests
│   ├───integration/ # Tests for component interactions
│   ├───e2e/         # End-to-end workflow tests
│   ├───performance/ # Performance and profiling tests
│   ├───validation/  # Validation and quality tests
│   ├───comparison/  # Comparison tests between approaches
│   ├───fixtures/    # Shared test data and utilities
│   ├───conftest.py  # Shared fixtures and test configuration
│   ├───run_all_tests.py # Universal test runner
│   └───__init__.py  # Test package initialization
├───main.py          # Main entry point for the application
└───pyproject.toml   # Project metadata and dependencies
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please read our [**Contributing Guide**](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Testing Documentation

For comprehensive information about the testing framework, including:

- Detailed testing philosophy and best practices
- Step-by-step guides for writing different types of tests
- Advanced testing techniques and patterns
- Troubleshooting common testing issues
- Migration guide for the new testing structure

Please refer to the [Testing Documentation](docs/testing/) which provides a complete guide to the testing framework, including quick start instructions, test categories, and examples.

## Disclaimer

This software is for educational purposes only. Do not risk money that you are not prepared to lose.
