# AI Trading System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source project to develop a multi-agent system for financial analysis and trade signal generation. This system uses a graph-based agentic workflow to analyze assets from multiple perspectives, leveraging a team of specialized AI agents to perform a comprehensive analysis.

**Note:** This project is in the early stages of development and is for educational and research purposes only. It is not intended for live trading.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Analysis](#running-the-analysis)
  - [Running the API](#running-the-api)
  - [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Features

- **Multi-Agent Analysis:** Utilizes a team of specialized AI agents (e.g., Technical Analyst, Sentiment Analyst, Risk Manager) for comprehensive market analysis.
- **Graph-Based Workflow:** Employs LangGraph to orchestrate agent interactions, ensuring a stateful and logical flow of information.
- **Extensible Provider Network:** Supports multiple data providers (e.g., yfinance, Alpha Vantage) and LLM providers (e.g., OpenAI, Anthropic, Groq, Google).
- **REST API:** Provides a FastAPI-based backend for interaction with the system, allowing for easy integration with other services.
- **Backtesting Framework:** Includes tools to backtest trading strategies and evaluate performance.

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
    uv pip install -r requirements.txt
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

### Running the Analysis

To run the core signal generation process, execute the `main.py` script:

```sh
python main.py
```

### Running the API

The system includes a FastAPI server for API-based interaction.

```sh
uvicorn src.api.app:app --reload
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

### Running Tests

To ensure everything is working as expected, run the test suite using pytest:

```sh
pytest
```

## Project Structure

Here is a high-level overview of the project's directory structure:

```
├───.github/         # GitHub Actions and CI/CD workflows
├───data/            # Data storage for logs, cache, and backtest results
├───docs/            # Project documentation
├───src/             # Main source code
│   ├───agents/      # Core analysis agents
│   ├───api/         # FastAPI application
│   ├───backtesting/ # Backtesting engine and components
│   ├───config/      # Configuration and settings
│   ├───data/        # Data pipelines and providers
│   ├───llm/         # LLM client integrations
│   ├───signal_generation/ # Logic for generating trade signals
│   └───utils/       # Utility functions and helpers
├───tests/           # Unit and integration tests
├───main.py          # Main entry point for the application
└───pyproject.toml   # Project metadata and dependencies
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please read our [**Contributing Guide**](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Disclaimer

This software is for educational purposes only. Do not risk money that you are not prepared to lose.
