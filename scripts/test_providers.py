#!/usr/bin/env python3
"""
Individual provider testing script for validating data provider capabilities.

This script tests each data provider individually to ensure they can provide
the data types needed for the event-driven trigger system.

Usage:
    python scripts/test_providers.py --provider finnhub
    python scripts/test_providers.py --provider all
    python scripts/test_providers.py --provider finnhub --type current_price

Available providers: finnhub, alpha_vantage, yfinance, marketaux, all
Available data types: current_price, historical_data, news_sentiment, company_profile, all
"""
import asyncio
import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import providers
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.finnhub_provider import FinnhubProvider
from src.data.providers.marketaux_provider import MarketauxProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check which API keys are available."""
    keys = {
        'yfinance': 'Always available',
        'finnhub': os.getenv('DATA_FINNHUB_API_KEY', 'NOT_SET'),
        'alpha_vantage': os.getenv('DATA_ALPHA_VANTAGE_API_KEY', 'NOT SET'),
        'marketaux': os.getenv('DATA_MARKETAUX_API_KEY', 'NOT SET')
    }

    logger.info("API Key Status:")
    for provider, status in keys.items():
        logger.info(f"  {provider}: {status}")

    return keys

class ProviderTestResult:
    """Container for test results."""
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.successful_tests = 0
        self.failed_tests = []
        self.error_messages = []
        self.data_samples = {}
        self.response_times = []

    def add_success(self, test_name: str, data: Any = None, response_time: float = 0):
        """Add a successful test result."""
        self.successful_tests += 1
        if data is not None:
            self.data_samples[test_name] = data
        if response_time > 0:
            self.response_times.append(response_time)

    def add_failure(self, test_name: str, error: str):
        """Add a failed test result."""
        self.failed_tests.append(test_name)
        self.error_messages.append(error)

    def get_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total_tests = self.successful_tests + len(self.failed_tests)
        if total_tests == 0:
            return 0.0
        return (self.successful_tests / total_tests) * 100

    def get_average_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def print_summary(self):
        """Print comprehensive test summary."""
        logger.info(f"\n=== {self.provider_name} Test Summary ===")
        logger.info(f"✅ Successful Tests: {self.successful_tests}")
        logger.info(f"❌ Failed Tests: {len(self.failed_tests)}")
        logger.info(f"📊 Success Rate: {self.get_success_rate():.1f}%")

        if self.failed_tests:
            logger.error("Failed Tests:")
            for i, error in enumerate(self.error_messages, 1):
                logger.error(f"  {i}. {error}")

        if self.response_times:
            avg_time = self.get_average_response_time()
            logger.info(f"⚡ Response Time: {avg_time:.3f} seconds average")

        logger.info(f"📊 Data Samples: {len(self.data_samples)} types")
        for data_type, sample in self.data_samples.items():
            if isinstance(sample, (list, dict)):
                logger.info(f"  - {data_type}: {len(sample)} items")
            else:
                logger.info(f"  - {data_type}: {type(sample).__name__}")

        ready = "✅" if self.get_success_rate() >= 80 else "❌"
        logger.info(f"🔧 Provider Ready: {ready}")

class ProviderTester:
    """Comprehensive provider testing class."""

    def __init__(self):
        self.providers = {}
        self.results = {}
        self.api_keys = check_api_keys()

    def get_provider(self, provider_name: str):
        """Get or create a provider instance."""
        if provider_name in self.providers:
            return self.providers[provider_name]

        provider = None
        try:
            if provider_name == "yfinance":
                provider = YFinanceProvider()
            elif provider_name == "finnhub":
                key = self.api_keys['finnhub']
                if key != 'NOT_SET':
                    provider = FinnhubProvider(key)
            elif provider_name == "alpha_vantage":
                key = self.api_keys['alpha_vantage']
                if key != 'NOT_SET':
                    provider = AlphaVantageProvider(key)
            elif provider_name == "marketaux":
                key = self.api_keys['marketaux']
                if key != 'NOT_SET':
                    provider = MarketauxProvider(key)
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")

        if provider:
            self.providers[provider_name] = provider
        return provider

    async def test_provider(self, provider_name: str, test_type: str = "all") -> ProviderTestResult:
        """Run tests for a specific provider."""
        logger.info(f"\n=== Testing {provider_name} Provider ===")

        provider = self.get_provider(provider_name)
        if not provider:
            result = ProviderTestResult(provider_name)
            result.add_failure("provider_creation", f"Could not create {provider_name} provider")
            return result

        result = ProviderTestResult(provider_name)

        # Test connectivity
        await self._test_connectivity(provider, result)

        # Test specific data types
        if test_type in ["all", "current_price"]:
            await self._test_current_price(provider, result)

        if test_type in ["all", "historical_data"]:
            await self._test_historical_data(provider, result)

        if test_type in ["all", "news_sentiment"]:
            await self._test_news_sentiment(provider, result)

        if test_type in ["all", "company_profile"]:
            await self._test_company_profile(provider, result)

        # Test error handling
        await self._test_error_handling(provider, result)

        result.print_summary()
        return result

    async def _test_connectivity(self, provider, result):
        """Test basic connectivity."""
        try:
            start_time = datetime.now()

            # Test with a simple method call
            if hasattr(provider, 'get_current_price'):
                price = await provider.get_current_price("AAPL")
                response_time = (datetime.now() - start_time).total_seconds()

                if price is not None:
                    result.add_success("connectivity", price, response_time)
                else:
                    result.add_failure("connectivity", "Returned None")
            else:
                result.add_failure("connectivity", "No get_current_price method")

        except Exception as e:
            result.add_failure("connectivity", str(e))

    async def _test_current_price(self, provider, result):
        """Test current price fetching."""
        try:
            start_time = datetime.now()
            price = await provider.get_current_price("AAPL")
            response_time = (datetime.now() - start_time).total_seconds()

            if price and price > 0:
                result.add_success("current_price", price, response_time)
                logger.info(f"  💰 AAPL Price: ${price:.2f}")
            else:
                result.add_failure("current_price", f"Invalid price: {price}")

        except Exception as e:
            result.add_failure("current_price", str(e))

    async def _test_historical_data(self, provider, result):
        """Test historical data fetching."""
        try:
            start_time = datetime.now()
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)

            data = await provider.fetch_data("AAPL", start_date, end_date)
            response_time = (datetime.now() - start_time).total_seconds()

            if data is not None and len(data) > 0:
                result.add_success("historical_data", data, response_time)
                logger.info(f"  📈 Historical data: {len(data)} days")
            else:
                result.add_failure("historical_data", "No data returned")

        except Exception as e:
            result.add_failure("historical_data", str(e))

    async def _test_news_sentiment(self, provider, result):
        """Test news sentiment fetching."""
        try:
            if not hasattr(provider, 'fetch_news_sentiment'):
                result.add_failure("news_sentiment", "Method not available")
                return

            start_time = datetime.now()
            news = await provider.fetch_news_sentiment("AAPL", limit=5)
            response_time = (datetime.now() - start_time).total_seconds()

            if news and len(news) > 0:
                result.add_success("news_sentiment", news, response_time)
                logger.info(f"  📰 News articles: {len(news)}")

                # Show sample article
                if news and len(news) > 0:
                    article = news[0]
                    logger.info(f"    Sample: {article.get('title', 'No title')[:50]}...")
            else:
                result.add_failure("news_sentiment", "No news returned")

        except Exception as e:
            result.add_failure("news_sentiment", str(e))

    async def _test_company_profile(self, provider, result):
        """Test company profile fetching."""
        try:
            if not hasattr(provider, 'fetch_company_profile'):
                result.add_failure("company_profile", "Method not available")
                return

            start_time = datetime.now()
            profile = await provider.fetch_company_profile("AAPL")
            response_time = (datetime.now() - start_time).total_seconds()

            if profile:
                result.add_success("company_profile", profile, response_time)
                name = profile.get('name', profile.get('companyName', 'Unknown'))
                logger.info(f"  🏢 Company: {name}")
            else:
                result.add_failure("company_profile", "No profile returned")

        except Exception as e:
            result.add_failure("company_profile", str(e))

    async def _test_error_handling(self, provider, result):
        """Test error handling with invalid symbol."""
        try:
            price = await provider.get_current_price("INVALID-SYMBOL-123")

            # Should return None or raise an exception
            if price is None:
                result.add_success("error_handling", "Correctly returned None for invalid symbol")
            else:
                result.add_failure("error_handling", f"Should return None, got {price}")

        except Exception as e:
            # Exception is also acceptable
            result.add_success("error_handling", f"Correctly raised exception: {type(e).__name__}")

    async def test_all_providers(self, test_type: str = "all"):
        """Test all available providers."""
        providers = ['yfinance', 'finnhub', 'alpha_vantage', 'marketaux']
        results = {}

        for provider_name in providers:
            if provider_name == 'yfinance' or self.api_keys[provider_name] != 'NOT_SET':
                result = await self.test_provider(provider_name, test_type)
                results[provider_name] = result
            else:
                logger.warning(f"Skipping {provider_name} - no API key")

        return results

async def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test individual data providers')
    parser.add_argument('--provider',
                       choices=['all', 'yfinance', 'finnhub', 'alpha_vantage', 'marketaux'],
                       default='all',
                       help='Provider to test')
    parser.add_argument('--type',
                       choices=['all', 'current_price', 'historical_data', 'news_sentiment', 'company_profile'],
                       default='all',
                       help='Type of data to test')

    args = parser.parse_args()

    logger.info("🔍 Data Provider Testing Suite")
    logger.info("=" * 50)

    # Check API keys
    api_keys = check_api_keys()

    # Create tester
    tester = ProviderTester()

    # Run tests
    if args.provider == 'all':
        results = await tester.test_all_providers(args.type)
    else:
        result = await tester.test_provider(args.provider, args.type)
        results = {args.provider: result}

    # Print overall summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 OVERALL SUMMARY")
    logger.info(f"{'='*50}")

    total_providers = len(results)
    successful_providers = sum(1 for r in results.values() if r.get_success_rate() >= 80)

    logger.info(f"Providers Tested: {total_providers}")
    logger.info(f"Ready for Production: {successful_providers}")

    for provider_name, result in results.items():
        status = "✅" if result.get_success_rate() >= 80 else "❌"
        logger.info(f"{status} {provider_name}: {result.get_success_rate():.1f}%")

    if successful_providers == total_providers:
        logger.info("\n🎉 All providers are ready for production!")
    elif successful_providers > 0:
        logger.info(f"\n⚠️ {successful_providers}/{total_providers} providers are ready")
    else:
        logger.info("\n❌ No providers are ready for production")

if __name__ == "__main__":
    asyncio.run(main())
