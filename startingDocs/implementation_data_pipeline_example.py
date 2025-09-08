import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf

@dataclass
class DataProvider:
    name: str
    priority: int
    rate_limit: int  # calls per minute

class AlphaVantageProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = asyncio.Semaphore(5)  # 5 calls per minute

    async def get_current_price(self, symbol: str) -> float:
        async with self.rate_limiter:
            async with aiohttp.ClientSession() as session:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    price = data['Global Quote']['05. price']
                    return float(price)

class YFinanceProvider:
    def __init__(self):
        self.name = "yfinance"

    async def get_current_price(self, symbol: str) -> float:
        # Use asyncio to run sync yfinance in thread
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
        info = await loop.run_in_executor(None, lambda: ticker.info)
        return float(info.get('currentPrice', info.get('regularMarketPrice', 0)))

    async def get_historical_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
        hist = await loop.run_in_executor(None, lambda: ticker.history(period=period))
        return hist

class DataPipelineManager:
    def __init__(self, providers: List, cache_manager=None):
        self.providers = sorted(providers, key=lambda x: getattr(x, 'priority', 999))
        self.cache = cache_manager

    async def get_market_data(self, symbol: str):
        # Try cache first
        if self.cache:
            cached_data = await self.cache.get(f"market_data_{symbol}")
            if cached_data and not self._is_stale(cached_data):
                return cached_data

        # Try providers in order
        for provider in self.providers:
            try:
                price = await provider.get_current_price(symbol)

                # Get historical data for technical indicators
                if hasattr(provider, 'get_historical_data'):
                    historical = await provider.get_historical_data(symbol)
                    tech_indicators = self._calculate_indicators(historical)
                else:
                    tech_indicators = {}

                market_data = {
                    'symbol': symbol,
                    'price': price,
                    'volume': 0,  # Would get from historical data
                    'timestamp': datetime.now(),
                    'ohlc': {},  # Would extract from historical
                    'technical_indicators': tech_indicators
                }

                # Cache the result
                if self.cache:
                    await self.cache.set(f"market_data_{symbol}", market_data, ttl=60)

                return market_data

            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue

        raise Exception(f"All providers failed for {symbol}")

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if df.empty:
            return {}

        indicators = {}

        # Simple moving averages
        if len(df) >= 20:
            indicators['sma_20'] = df['Close'].rolling(20).mean().iloc[-1]
        if len(df) >= 50:
            indicators['sma_50'] = df['Close'].rolling(50).mean().iloc[-1]

        # Simple RSI calculation
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))

        return indicators

    def _is_stale(self, cached_data: Dict, max_age_minutes: int = 5) -> bool:
        """Check if cached data is too old"""
        timestamp = cached_data.get('timestamp')
        if not timestamp:
            return True

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return (datetime.now() - timestamp) > timedelta(minutes=max_age_minutes)

# Usage example
async def main():
    # Setup providers
    providers = [
        YFinanceProvider(),  # Primary (free)
        # AlphaVantageProvider("your-api-key"),  # Secondary
    ]

    pipeline = DataPipelineManager(providers)

    # Get market data
    try:
        data = await pipeline.get_market_data("AAPL")
        print(f"AAPL Price: ${data['price']}")
        print(f"Technical Indicators: {data['technical_indicators']}")
    except Exception as e:
        print(f"Failed to get data: {e}")

if __name__ == "__main__":
    asyncio.run(main())
