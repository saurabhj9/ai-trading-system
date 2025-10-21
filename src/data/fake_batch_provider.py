"""
Fake batch provider for safe testing without API calls.

This provider simulates market data without requiring real API keys,
making it perfect for testing, development, and demonstrations.
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class FakeBatchProvider:
    """Fake data provider that simulates market prices without API calls."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Simulated base prices for realistic data
        self.base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 250.0,
            'AMZN': 3500.0,
            'META': 350.0,
            'NVDA': 450.0,
            'NFLX': 400.0,
            'DIS': 90.0,
            'AMD': 120.0
        }

        # Simulated price movements
        self.price_movements = {}

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple symbols."""
        quotes = {}

        for symbol in symbols:
            # Generate realistic price based on base price with small random movement
            base_price = self.base_prices.get(symbol, 100.0)

            # Simulate price movement
            if symbol not in self.price_movements:
                self.price_movements[symbol] = base_price

            # Small random walk
            movement = random.uniform(-0.02, 0.02)  # +/- 2% max movement
            self.price_movements[symbol] *= (1 + movement)

            # Ensure price stays reasonable (within 50% of base price)
            min_price = base_price * 0.5
            max_price = base_price * 1.5
            self.price_movements[symbol] = max(min_price, min(max_price, self.price_movements[symbol]))

            quotes[symbol] = round(self.price_movements[symbol], 2)

        return quotes

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[Dict]:
        """Fetch historical data (simulated for testing)."""
        # For testing, we don't actually fetch historical data
        # This is handled by the market data generator in tests
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a single symbol."""
        quotes = asyncio.run(self.get_multiple_quotes([symbol]))
        return quotes.get(symbol)
