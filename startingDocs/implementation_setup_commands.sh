#!/bin/bash
# Complete setup script for AI Trading System

echo "Setting up AI Trading System..."

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Create project
mkdir ai-trading-system
cd ai-trading-system
uv init --app

# Create directory structure
mkdir -p src/{agents,communication,data/providers,strategies,backtesting,utils,api/routes,config}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p scripts docs data/{cache,backtest_results,logs} notebooks

# Create main files
touch src/main.py
touch src/agents/{base,technical,sentiment,risk_manager,portfolio_manager}.py
touch src/communication/{message_bus,state_manager,orchestrator}.py
touch src/data/{pipeline,cache}.py
touch src/data/providers/{alpha_vantage,twelve_data,yfinance_provider}.py

echo "Project structure created!"
echo "Next: Configure pyproject.toml and install dependencies"
echo "Run: uv add pandas numpy aiohttp python-dotenv pydantic openai"
