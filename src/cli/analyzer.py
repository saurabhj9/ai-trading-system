"""
Core analysis logic for CLI.
"""
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.agents.data_structures import AgentConfig
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.communication.message_bus import MessageBus
from src.communication.orchestrator import Orchestrator
from src.communication.state_manager import StateManager
from src.data.cache import CacheManager
from src.data.pipeline import DataPipeline
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StockAnalyzer:
    """Handles stock analysis orchestration for CLI."""

    def __init__(self):
        """Initialize the stock analyzer."""
        self.orchestrator = None

    def _get_orchestrator(self) -> Orchestrator:
        """Create or return cached orchestrator instance."""
        if self.orchestrator is not None:
            return self.orchestrator

        # Core components
        llm_client = LLMClient()
        message_bus = MessageBus()
        state_manager = StateManager()

        # Initialize data providers
        yfinance_provider = YFinanceProvider(rate_limit=10, period=60.0)
        alpha_vantage_api_key = os.getenv("DATA_ALPHA_VANTAGE_API_KEY")

        if not alpha_vantage_api_key:
            raise ValueError(
                "DATA_ALPHA_VANTAGE_API_KEY is required for sentiment analysis.\n"
                "Please set it in your .env file to fetch real news data.\n"
                "Get your free API key at: https://www.alphavantage.co/support/#api-key"
            )

        alpha_vantage_provider = AlphaVantageProvider(api_key=alpha_vantage_api_key)

        # Initialize data pipeline
        data_pipeline = DataPipeline(provider=yfinance_provider, cache=CacheManager())

        # Initialize agents
        agent_dependencies = {
            "llm_client": llm_client,
            "message_bus": message_bus,
            "state_manager": state_manager,
        }

        technical_agent = TechnicalAnalysisAgent(
            config=AgentConfig(name="technical"), **agent_dependencies
        )

        sentiment_agent = SentimentAnalysisAgent(
            config=AgentConfig(name="sentiment"),
            news_provider=alpha_vantage_provider,
            **agent_dependencies,
        )

        risk_agent = RiskManagementAgent(
            config=AgentConfig(name="risk"), **agent_dependencies
        )

        portfolio_agent = PortfolioManagementAgent(
            config=AgentConfig(name="portfolio"), **agent_dependencies
        )

        self.orchestrator = Orchestrator(
            data_pipeline=data_pipeline,
            technical_agent=technical_agent,
            sentiment_agent=sentiment_agent,
            risk_agent=risk_agent,
            portfolio_agent=portfolio_agent,
            state_manager=state_manager,
        )

        return self.orchestrator

    async def analyze_symbol(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze a single stock symbol.

        Args:
            symbol: Stock symbol to analyze
            days: Number of days of historical data

        Returns:
            Analysis results dictionary
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        orchestrator = self._get_orchestrator()
        result = await orchestrator.run(symbol, start_date, end_date)

        # Format the result
        if result.get("error"):
            return {
                "symbol": symbol,
                "error": result["error"],
                "timestamp": datetime.now().isoformat(),
            }

        formatted_result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal": result["final_decision"].signal,
            "confidence": result["final_decision"].confidence,
            "reasoning": result["final_decision"].reasoning,
            "analysis_period": {
                "start": start_date.date().isoformat(),
                "end": end_date.date().isoformat(),
                "days": days,
            },
            "agent_decisions": {},
        }

        # Add individual agent decisions
        for agent_name, decision in result.get("decisions", {}).items():
            # Log warning if confidence is 0 (potential bug)
            if decision.confidence == 0.0:
                logger.warning(
                    f"Agent '{agent_name}' returned 0% confidence. "
                    f"Signal: {decision.signal}, Reasoning: {decision.reasoning[:100]}"
                )

            formatted_result["agent_decisions"][agent_name] = {
                "signal": decision.signal,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            }

        return formatted_result

    async def analyze_batch(
        self, symbols: List[str], days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple symbols in parallel.

        Args:
            symbols: List of stock symbols
            days: Number of days of historical data

        Returns:
            List of analysis results
        """
        tasks = [self.analyze_symbol(symbol, days) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        formatted_results = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                formatted_results.append(
                    {
                        "symbol": symbol,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                formatted_results.append(result)

        # Post-process results to show summary of symbol validation errors
        valid_results = []
        invalid_results = []
        symbol_errors = {}
        
        for result in formatted_results:
            if result.get("error"):
                invalid_results.append(result)
                symbol_errors[result["symbol"]] = result["error"]
            else:
                valid_results.append(result)
        
        # Print summary of symbol validation errors if any
        if invalid_results:
            invalid_count = len(invalid_results)
            valid_count = len(valid_results)
            
            print(f"\n[WARN] {invalid_count} symbol{'s' if invalid_count > 1 else ''} had validation errors:")
            for result in invalid_results:
                symbol = result["symbol"]
                error_msg = result["error"]
                
                # Check if it's a symbol validation error with suggestions
                if "Did you mean" in error_msg:
                    # Extract suggestion
                    parts = error_msg.split("Did you mean")
                    if len(parts) > 1:
                        suggestion = parts[1].strip().rstrip("?")
                        print(f"  • {symbol}: Invalid symbol. Did you mean{suggestion}?")
                    else:
                        print(f"  • {symbol}: {error_msg}")
                else:
                    print(f"  • {symbol}: {error_msg}")
            
            if valid_count > 0:
                print(f"\n[INFO] Proceeding with {valid_count} valid symbol{'s' if valid_count > 1 else ''}...")
            else:
                print(f"\n[ERROR] No valid symbols to analyze.")
        
        return formatted_results

    async def watch_symbol(
        self, symbol: str, interval: int, days: int, callback=None
    ):
        """
        Continuously monitor a symbol.

        Args:
            symbol: Stock symbol to monitor
            interval: Update interval in seconds
            days: Number of days of historical data
            callback: Optional callback function for updates
        """
        while True:
            try:
                result = await self.analyze_symbol(symbol, days)
                if callback:
                    callback(result)
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in watch mode: {e}")
                if callback:
                    callback({"symbol": symbol, "error": str(e)})
                await asyncio.sleep(interval)
