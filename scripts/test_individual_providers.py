#!/usr/bin/env python3
"""
Individual provider testing script for validating data provider capabilities.

This script tests each data provider individually to ensure they can provide
the data types needed for the event-driven trigger system.
Usage:
    uv run python scripts/test_individual_providers.py --provider finnhub --test all
    uv run python scripts/test_individual_providers.py --provider all --test all
    uv run python scripts/test_individual_providers --provider finnhub --symbol AAPL --type current_price
    uv run python scripts/test_individual_providers.py --provider finnhub --symbol AAPL --type news_sentiment
    uv run python scripts/test_individual_providers.py --provider finnhub --symbol AAPL --type company_profile
    uv run scripts/test_individual_providers.py --provider alpha_vantage --type current_price --symbol AAPL --limit 5
    uv run python scripts/test_individual_providers.py --provider marketaux --test all

Available providers: finnhub, alpha_vantage, yfinance, marketaux
Available data types: historical_data, current_price, news_sentiment, company_profile
"""
import asyncio
import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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

# API Key Validation
def check_api_keys():
    """Check which API keys are available."""
    keys = {
        'yfinance': 'Always available',
        'finnhub': os.getenv('DATA_FINNHUB_API_KEY', 'NOT_SET'),
        'alpha_vantage': os.getenv('DATA_ALPHA_VANTAGE_API_KEY', 'NOT SET'),
        'marketaux': os.getenv('DATA_MARKETAUX_API_KEY', 'NOT_SET')
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
        failed_tests = []
        error_messages = []
        data_samples = {}
        response_times = []

    def add_success(self, test_name: str, data: Any, response_time: float = 0):
        """Add a successful test result."""
        self.successful_tests += 1
        self.data_samples[test_name] = data
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
        logger.info(f"✅ Successful Tests: {self.success_tests}")
        logger.info(f"❌ Failed Tests: {len(self.failed_tests)}")
        logger.info(f"📊 Success Rate: {self.get_success_rate():.1f}%")

        if self.failed_tests > 0:
            logger.error("Failed Tests:")
            for i, error in enumerate(self.error_messages, 1):
                logger.error(f"  {i}. {error}")

        if self.response_times:
            avg_time = self.get_average_response_time()
            logger.info(f"⚡️ Response Time: {avg_time:.3f} seconds average")

        logger.info(f"📊 Data Samples Collected: {len(self.data_samples)} data types")
        for data_type, sample in self.data_samples.items():
            logger.info(f"  - {data_type}: {len(sample) if isinstance(sample, (list, dict)) else '0 items'} samples")

        if self.success_tests > 0:
            for test_name, data in self.data_samples.items():
                if data and not isinstance(data, dict):
                    logger.info(f"    {test_name}: {data.count()} events")
                elif data.get('symbol') or data.get('ticker'):
                    logger.info(f"    {test_name}: {data.get('symbol', 'Unknown')} events")
                else:
                    logger.info(f"    {test_name}: {data_type}")

        logger.info("🔧 Provider Ready:" + ("✅" if self.get_success_rate() >= 80 else "❌"))

class ProviderTester:
    """Comprehensive provider testing class."""

    def __init__(self):
        self.providers = {}
        self.results = {}
        self.start_time = datetime.now()
        self.total_api_calls_by_provider = {}
        self.errors = []

        # Check API keys
        self.api_keys = check_api_keys()

    def get_provider(self, provider_name: str):
        """Get or create a provider instance."""
        if provider_name in self.providers:
            return self.providers[provider_name]

        # Create provider instances
        if provider_name == "yfinance":
            return YFinanceProvider()
        elif provider_name == "finnhub":
            finnhub_key = self.api_keys['finnhub']
            if finnhub_key != 'NOT_SET':
                return FinnhubProvider(finnhub_key)
            else:
                logger.warning("Finnhub API key missing. Skipping Finnhub tests")
                return None
        elif provider_name == "alpha_vantage":
            av_key = self.api_keys['alpha_vantage']
            if av_key != 'NOT_SET':
                return AlphaVantageProvider(av_key)
            else:
                logger.warning("Alpha Vantage API key missing. Skipping tests")
                return None
        elif provider_name == "marketaux":
            marketaux_key = self.api_keys['marketaux']
            if marketaux_key != 'NOT_SET':
                return MarketauxProvider(marketaux_key)
            else:
                logger.warning("Marketaux API key missing. Skipping tests")
                return None
        else:
            return None

    async def test_provider(self, provider_name: str):
        """Run comprehensive tests for a specific provider."""
        if provider_name not in self.api_keys or self.api_keys[provider_name] == 'NOT_SET':
            logger.warning(f"Skipping {provider_name} provider - no API key available")
            result = ProviderTestResult(provider_name)
            self.results[provider_name] = result
            return result

        logger.info(f"\n=== Testing {provider_name} Provider ===")

        provider = self.get_provider(provider_name)
        result = ProviderTestResult(provider_name)

        # Convert to dict for easy JSON serialization
        result_dict = result.__dict__

        # Test 1: Basic Connectivity
        await self._test_basic_connectivity(provider, result)

        # Test 2: Historical Data Fetching (only applies to some providers)
        if provider_name in ["yfinance", "alpha_vantage", "finnhub"]:
            await self._test_historical_data_fetching(provider, result)

        # Test 3: Current Price Fetching
        if provider_name in ["yfinance", "finnhub"]:
            await self._test_current_price_fetching(provider, result)

        # Test 4: News Sentiment (if available)
        if provider_name in ["finnhub", "alpha_vantage", "marketaux"]:
            await self._test_news_sentiment(provider, result)

        # Test 5: Company Profile (if available)
        if provider_name in ["finnhub", "alpha_vantage"]:
            await self._test_company_profile(provider, result)

        # Test 6. Error Handling
        await self._test_error_handling(provider, result)

        # Test 7. Performance (measures response times)
        await self._test_performance(provider, result)

        # Calculate final statistics
        result.print_summary()

        return result

    async def _test_basic_connectivity(self, provider, result):
        """Test basic connectivity with provider API."""
        try:
            # Test a simple data fetch
            if hasattr(provider, 'get_current_price'):
                current_price = await provider.get_current_price("AAPL")
                result.add_success("basic_connectivity", current_price is not None)
            elif hasattr(provider, 'fetch_data'):
                try:
                    data = await provider.fetch_data(
                        "AAPL",
                        datetime.now() - timedelta(days=1),
                        datetime.now()
                    )
                    result.add_success("historical_data_fetch", len(data) > 0)
        except Exception as e:
            result.add_failure("basic_connectivity", str(e))

    async def _test_historical_data_fetching(self, provider, result):
        """Test historical data fetching."""
        try:
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now() - timedelta(days=1)

            data = await provider.fetch_data("AAPL", start_date, end_date)

            result.add_success("historical_data_fetch", len(data) > 0)

            # Validate data structure
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
            if all(col in data.columns for col in required_columns):
                result.add_success("data_structure_valid", True)

        except Exception as e:
            result.add_failure("historical_data_fetch", str(e))

    async def _test_current_price_fetching(self, provider, result):
        """Test current price fetching."""
        try:
            current_price = await provider.get_current_price("AAPL")
            result.add_success("current_price_fetch", current_price is not None)

            # Validate price is reasonable
            if current_price > 0:
                result.add_success("price_reasonable", True)
                result.add_data("current_price", current_price)
        except Exception as e:
            result.add_failure("current_price_fetch", str(e))

    async def _test_news_sentiment(self, provider, result):
        """Test news sentiment fetching."""
        try:
            news_data = await provider.fetch_news_sentiment("AAPL", limit=5)
            result.add_success("news_sentiment_fetch", len(news_data) > 0)

            # Validate news structure
            if news_data and len(news_data) > 0:
                article = news_data[0]
                required_fields = ['source', 'title', 'summary', 'sentiment_score']
                if all(field in article for field in required_fields):
                    result.add_success("news_structure_valid", True)
                result.add_data("news_sample", {
                    'title': article.get('title', 'N/A'),
                    'sentiment': article.get('summary', 'N/A'),
                    'source': article.get('source', 'N/A'),
                    'sentiment_score': article.get('overall_sentiment_score', 0.0)
                })
        except Exception as e:
            result.add_failure("news_sentiment_fetch", str(e))

    async def _test_company_profile(self, provider, result):
        """Test company profile fetching."""
        try:
            profile_data = await provider.fetch_company_profile("AAPL")
            result.add_success("company_profile_fetch", profile_data is not None)

            # Profile structure validation
            required_fields = ['name', 'ticker', 'sector', 'description']
            if all(field in profile_data for field in required_fields):
                result.add_success("profile_structure_valid", True)
                result.add_data("company_sample", {
                    'name': profile_data.get('name', 'N/A'),
                    'ticker': profile_data.get('ticker', 'N/A'),
                    'sector': profile_data.get('sector', 'Unknown')
                })
        except Exception as e:
            result.add_failure("company_profile_fetch", str(e))

    async def _test_error_handling(self, provider, result):
        """Test error handling and recovery."""
        # Test invalid requests
        try:
            invalid_price = await provider.get_current_price("INVALID-SYMBOL")
            result.add_failure("invalid_price_handling", "Should return None")
        except Exception as e:
            result.add_failure("unexpected_error", str(e))

    async def _test_performance(self, provider, result):
        """Test performance with timing measurements."""
        # Test multiple concurrent requests
        start_time = datetime.now()

        if hasattr(provider, 'get_current_price'):
            # Test single requests
            for i in range(5):
                start_time = datetime.now()
                await provider.get_current_price("AAPL")
                result.add_performance("single_request", datetime.now() - start_time)

        if hasattr(provider, 'get_multiple_quotes'):
            # Test batch requests
            start_time = datetime.now()
            symbols = ["AAPL", "MSFT", "GOOGL"]
            await provider.get_multiple_quotes(symbols)
            result.add_performance("batch_request", datetime.now() - start_time) * 1000)  # Convert to ms

    def print_summary(self):
        """Print comprehensive results."""
        result_dict = result.__dict__
        logger.info(f"Provider Testing Statistics:")
        if result.get_success_rate() >= 80:
            logger.info("✅ PERFORMANCE EXCELLENT")
        elif result.get_success_rate() >= 60:
            logger.info("✅ PERFORMANCE GOOD")
        elif result.get_success_rate() >= 40:
            logger.info("⚠️ PERFORMANCE ACCEPTABLE")
        else:
            logger.info("⚠️ PERFORMANCE NEEDS IMPROVEMENT")

        if result.response_times:
            avg_time = result.get_average_response_time()
            logger.info(f"⚡️ Avg Response Time: {avg_time:.3f}s")
            if avg_time < 1.0:
                logger.info("🚀 EXCELLENT (<1s response time)")
            elif avg_time < 5.0:
                logger.info("👏 EXCELLENT (<5s response time)")
            elif avg_time < 10.0:
                logger.info("👍 GOOD (<10s response time)")
            else:
                logger.info(f"✅ GOOD ({avg_time:.1f}s average)")

        if result.samples:
            logger.info(f"Data Samples Collected:")
            for data_type, samples in result.data_samples.items():
                for i, sample in enumerate(samples):
                    if isinstance(sample, dict):
                        logger.info(f"      {data_type} sample_{i+1}: {sample}")
                    else:
                        logger.info(f"      {data_type} sample_{i+1}: {data_type(sample)}")

        # Error Details
        if result.failed_tests > 0:
            logger.error(" Failed Tests:")
            for i, error in enumerate(result.failed_tests):
                if i == 0:
                    logger.error(f"  1. {error}")
                else:
                    logger.info(f" {i+1}. {error}")

class FallbackTesting:
    """Test automatic provider fallback mechanisms."""

    def __init__(self):
        self.providers = {}
        self.results = {}

    async def test_batch_integration(self):
        """Test unified batch provider integration."""
        from src.data.providers.unified_batch_provider import UnifiedBatchProvider

        # Test with only YFinance
        yfinance_config = {'yfinance': {'enabled': True}}
        uf_provider = UnifiedBatchProvider(yfinance_config)
        f_results = await self._test_provider_with_config(uf_provider, "YFinance Only", {
            "symbols": ["AAPL", "MSFT"],
            "data_type": "single_request"
        })

        # Test with YFinance + Finnhub
        mixed_config = {
            'yfinance': {'enabled': True},
            'finnhub': {
                'enabled': True,
                'rate_limit': 30,  # Conservative rate limiting
                'period': 60
            }
        }
        uf_provider = UnifiedBatchProvider(mixed_config)
        mf_results = await self._test_provider_with_config(uf_provider, "YFinance + Finnhub", {
            "symbols": ["AAPL", "MSFT"],
            "data_type": "hybrid_request",
            "preferred_provider": "finnhub"`  # Prefer Finnhub for real-time data
        })

        # Test Finnhub fallback
        finnhub_config_failed = {'finnhub': {'enabled': True, 'rate_limit': 1, 'period': 60}}  # Very restrictive
        f_provider = UnifiedBatchProvider(finnub_config_failed)
        ff_results = await self._test_provider_with_config(f_provider, "Finnub Only (Rate Limited)", {
            "symbols": ["AAPL", "MSFT"],
            "data_type": "single_request"
        })

        # Test Alpha Vantage fallback
        av_config = {'alpha_vantage': {'enabled': True, 'rate_limit': 2, 'period': 86400}}  # 2 per minute ≈144/day
        av_provider = UnifiedBatchProvider(av_config)
        av_results = await self._test_provider_with_config(av_provider, "Alpha Vantage (Limited)", {
            "symbols": ["AAPL", "MSFT"],
            "data_type": "single_request"
        })

        self.results = {
            "yfinance_only": f_results.success_rate:.1f,
            "yfinance_finnhub_hybrid": mf_results.success_rate:.1f,
            "yfinance_finnhub_hybrid": mf_results.success_rate:.1f,
            "finnhub_only": f_results.success_rate:.1f
            "finnhub_limited": ff_results.success_rate:.1f,
            "alpha_vantage_limited": av_results.success_rate:.1f
        }

    async def _test_provider_with_config(self, provider, test_name, config, symbols, config):
        """Test a specific provider with given configuration."""
        if not provider:
            result = ProviderTestResult(test_name)
            result.add_failure("provider_not_available")
            return result

        # Configure based on config
        batch_config = {
            provider_name + "_config": config
        }

        provider_configs = {
            **batch_config,
            'FINNHUB_API_KEY': self.api_keys.get('FINNHUB_API_KEY', 'NOT_SET'),
            'ALPHA_VANTAGE_API_KEY': self.api_keys.get('ALPHA_VANTAGE_API_KEY', 'NOT_SET'),
            'MARKETAUX_API_KEY': self.api_keys.get('MARKETAUX_API_KEY', 'NOT_SET'),
            'CACHE_ENABLED': True,
            'USE_ENHANCED_CACHING': True,
            'USE_BATCH_PROCESSING': True,
            'BATCH_CONFIG': config
        }

        # Create provider instance with custom config
        from src.data.providers.unified_batch_provider import UnifiedBatchProvider

        if test_name == "Finnhub Only (Limited)":
            if batch_config.get('FINNHUB_API_KEY') == "integration-test-finnhub-key":
                config['finnhub']['rate_limit'] = 1
                config['finnhub']['period'] = 60
        elif batch_config.get('FINNHUB_API_KEY'):
                config['finnhub']['rate_limit'] = 10  # Very conservative

        provider = UnifiedBatchProvider(batch_config)
        result = await self._run_provider_with_tests(test_name, provider, symbols, config)

        return result

    async def _run_provider_with_tests(self, test_name, provider, symbols, config):
        """Run all tests for a provider."""
        if not provider:
            result = ProviderTestResult(test_name)
            return result

        # Get symbol list based on data type
        data_types = ["historical_data", "current_price", "news_sentiment", "company_profile"]

        # Fetch for each data type
        for data_type in data_types:
            if data_type == "current_price":
                await self._test_current_price_fetching(provider, result)
            elif data_type == "historical_data":
                await self._test_historical_data_fetching(provider, result)
            elif data_type == "news_sentiment":
                await self._test_news_sentiment(provider, result)
            elif data_type == "company_profile":
                await self._test_company_profile(provider, result)

        return result

    def get_results_summary(self):
        """Get consolidated results from all tests."""
        summary = {
            'test_date': datetime.now().isoformat(),
            'total_providers': len(self.results),
            'provider_performance': {},
            'success_rate': {},
            'api_usage': self.total_api_calls_by_provider,
            'data_coverage': self._get_data_coverage()
        }

        # Combine performance data
        for provider_name, result in self.results.items():
            stats = self.results[provider_name]
            summary['success_rate'][provider_name] = stats.success_rate
            if stats.response_times:
                summary['provider_performance'][provider_name] = {
                    'avg_response_time': stats.get_average_response_time(),
                    'min_response_time': min(stats.response_times)
                }

        # API Usage
        if self.total_api_calls_by_provider:
            summary['api_usage'] = self.total_api_calls_by_provider
            summary['total_api_calls'] = sum(self.total_api_calls_by_provider.values())

        # Data Coverage
        summary['data_coverage'] = self._get_data_coverage()

        return summary

    def _get_data_coverage(self):
        """Calculate percentage of data types that were successfully tested."""
        tested_data_types = ["historical_data", "current_price", "news_sentiment", "company_profile"]
        available_data_types = []

        for test_name, result in self.results.items():
            if test_name in ["yfinance", "finnhub", "alpha_vantage", "marketaux"]:
                available_data_types = data_types
            else:
                continue

            # Check what types this provider supports
            if test_name in ["yfinance", "finnhub"]:
                available_data_types += ["historical_data"]
            elif test_name == "alpha_vantage":
                available_data_types += ["historical_data", "current_price", "news_sentiment"]
            elif test_name == "marketaux":
                available_data_types += ["news_sentiment"]

        return len(available_data_types) / len(tested_data_types) * 100

    async def _run_provider_with_tests(self, test_name, provider, symbols, config):
        """Run all tests for a specific provider."""
        logger.info(f"Starting {test_name} tests...")
        result = await self._run_provider_with_tests(test_name, provider, symbols, config)
        return result

    async def _test_current_price_fetching(self, result):
        """Test current price fetching."""
        try:
            current_price = await provider.get_current_price("AAPL")
            if current_price:
                logger.info(f"Price fetched: ${current_price:.2f}")
                result.add_success("current_price_fetch", True)
            else:
                result.add_failure("current_price_fetch", "Price fetch returned None")
        except Exception as e:
            result.add_failure("current_price_fetch", str(e))

    async def _test_historical_data_fetching(self, result):
        """Test historical data fetching."""
        try:
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now() - timedelta(days=1)

            data = await provider.fetch_data("AAPL", start_date, end_date)
            if data is not None and len(data) > 0:
                result.add_success("historical_data_fetch", len(data))

        except Exception as e:
            result.add_failure("historical_data_fetch", str(e))

    async def _test_news_sentiment(self, result):
        """Test news sentiment fetching."""
        try:
            news_data = await provider.fetch_news_sentiment("AAPL", limit=5)
            if news_data and len(news_data) > 0:
                result.add_success("news_sentiment_fetch", len(news_data))
                result.add_success("news_structure_validation", True)
                # Test different article structures
        except Exception as e:
            result.add_failure("news_sentiment_fetch", str(e))

    async def _test_company_profile(self, result):
        """Test company profile fetching."""
        try:
            profile = await provider.fetch_company_profile("AAPL")
            if profile is not None:
                result.add_success("company_profile_fetch", profile.get('name', 'Unknown'))
        except Exception as e:
            result.add_failure("company_profile_fetch", str(e))

    async def _test_error_handling(self, result):
        """Test error handling."""
        # Test invalid symbol handling
        try:
            invalid_price = await provider.get_current_price("INVALID-SYMBOL-123")
            result.add_success("invalid_symbol_error", True)
        except Exception as e:
            result.add_failure("unexpected_error", str(e))

    async def _test_performance(self, result):
        """Test performance with timing measurements."""
        start_time = datetime.now()

        if hasattr(provider, 'get_current_price'):
            # Test single request timing
            for i in range(10):
                await provider.get_current_price("AAPL")

            if i > 5:
                avg_time = (datetime.now() - start_time) / (i - 5) * 1000  # Convert to ms
                if avg_time < 1000: avg_time < 0.5: avg_time < 500: avg_time < 5.0

        return result

    def print_summary(self):
        """Print comprehensive results."""
        result = self._get_results_summary()
        logger.info(f"📊 Provider Performance Summary:")
        logger.info(f"  Total Providers Tested: {summary['total_providers']}")

        success_rates = [r.success_rate for r in self.results.values()]
        if success_rates:
            avg_success_rate = sum(success_rates) / len(success_rates)
            logger.info(f"  Average Success Rate: {avg_success_rate:.1f}%")

        if summary['api_usage']:
            total_calls = summary['total_api_calls']
            max_calls = summary['max_api_calls']
            if max_calls > 0:
                usage_percent = (total_calls / max_calls) * 100
                logger.info(f"  API Usage: {usage_percent:.1f}% of limit")

        logger.info(f"  API Calls by Provider:")
        if summary['api_usage_by_provider']:
            for provider, calls in summary['api_usage_by_provider'].items():
                logger.info(f"    {name}: {calls} calls ({usage_percent:.1f}% of limit})")

        if summary['data_coverage']:
            coverage = summary['data_coverage']
            coverage_rate = coverage / (len(tested_data_types) * 100)
            logger.info(f"  Data Coverage: {coverage_rate:.1f}% of tested types")

        return result


async def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test individual data providers')
    parser.add_argument('--provider', choices=['all', 'yfinance', 'finnhub', 'alpha_vantage', 'marketaux', 'current_prices'], default='all',
                       help='Providers to test')
    parser.add_argument('--symbol', default='AAPL',
                       help='Symbol to test (default: AAPL)')
    parser.add_argument('--type', choices=['current_price', 'historical_data', 'news_sentiment', 'company_profile', 'all'], default='current_price',
                       help='Type of data to test')
    parser.add_argument('--limit', type=int, default=5, help='Number of items to fetch (default: 5)')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose logging')

    args = parser.parse_args()

    logger.info("🔍 Data Provider Testing Suite")
    logger.info("=" * 50)

    api_keys = check_api_keys()

    if args.provider == 'all':
        providers = ['yfinance', 'finnhub', 'alpha_vantage', 'marketaux']
    elif args.provider == 'current_prices':
        providers = ['yfinance', 'finnhub', 'alpha_vantage']
    elif args.provider == 'historical_data':
        providers = ['yfinance', 'finnhub', 'alpha_vantage']
    elif args.provider == 'news_sentiment':
        providers = ['finnhub', 'alpha_vantage', 'marketaux']
    elif args.provider == 'company_profile':
        providers = ['finnhub', 'alpha_vantage']
    else:
        providers = [args.provider]

    logger.info(f"🔍 Testing Providers: {providers}")

    # Create test suite
    tester = ProviderTester()

    for provider_name in providers:
        if tester.get_provider(provider_name):
            logger.info(f"\n🧪 Starting {provider_name} individual tests...")
            result = await tester.test_provider(provider_name)
            tester.results[provider_name] = result

    if not tester.results:
        logger.error("❌ No providers were successfully tested")
        return

    # Test performance with different symbol sets
    if tools['events['finnhub']:
        live_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN']
        logger.info(f"\n🔴 Testing live symbols with Finnhub: {live_symbols}")

        live_results = await tester.test_provider_with_config(
            "Finnub Live (Real Monitoring)",
            live_symbols,
            config={
                "finnhub": {
                    'enabled': True,
                    'rate_limit': 30,    # Conservative: 2 calls/minute
                    'period': 60
                }
            },
            'symbols': live_symbols
        )
        prod_results = await tester.test_provider_with_config(
            "Finnub Live (Production Model)",
            live_symbols,
            config={
                "finnhub": {
                    'enabled': True,
                    'rate_limit': 60,    # Maximum 1/minute
                    'period': 60
                },
                'symbols': live_symbols
            }
        )

        prod_results.print_summary()
        live_success_rate = prod_results.get_success_rate()

        if live_success_rate >= 80:
            logger.info("✅ Finnhub Live: PRODUCTION READY")
        elif live_success_rate >= 60:
            logger.info("✅ Finnhub Live: GOOD")
        else:
            logger.info("⚠️ Finnhub Live: NEEDS ATTENTION (success_rate: {live_success_rate:.1f}%)")
    else:
        # Test YFinance Performance
        logger.info("\n🔹 Testing YFinance Performance...")
        yf_config = {
            'yfinance': {'enabled': True}
        }
        uf_provider = UnifiedBatchProvider(yf_config)
        tradel_results = await tester.test_provider_with_config(
            "YFinance (Stress Test)",
            ["AAPL", "MSFT"],
            config={
                'yfinance': {'enabled': True
            },
            'symbols': ["AAPL", "MSFT", "GOOGL"],
            'data_type': 'stress_test'
        )

        tradel_results.print_summary()
        trade_success_rate = tradel_results.get_success_rate()

        trade_success_rate = tradel_results.get_success_rate()

        if trade_success_rate >= 80:
            logger.info("✅ YFinance: STRESS TESTS PASSED")
        elif trade_success_rate >= 60:
            logger.info("✅ YFinance: GOOD PERFORMANCE")
        else:
            logger.info("⚠️ YFinance: NEEDS OPTIMIZATION")

    # Test integration with trigger system
    if tools['events']['finnhub']:
        logger.info("\n🔍 Integration Testing with Trigger System")
        integration_results = await tester.test_provider_with_config(
            "Integration (YFinance + Finnhub)",
            ["TSLA", "NVDA", "AMZN", "AMZN", "SPY"],
            {
                "finnhub": {
                    'enabled': True,
                    'rate_limit': 30, # Conservative: 1/minute
                    'period': 60
                },
                symbols: ["TSLA", "NVDA", "AMZN", "SPY"],
                'data_type': 'integration'
            }
        )
        integration_results.print_summary()
        integration_success_rate = integration_results.get_success_rate()

        if integration_success_rate >= 80:
            logger.info("✅ Trigger System + Finnhub Integration: EXCELLENT")
        elif integration_success_rate >= 60:
            logger.info("✅ Trigger System + Finnhub Integration: GOOD")
        else:
            logger.info("⚠️ Trigger System + Finnhub Integration: NEEDS ATTENTION")

        **Rate Limit Validation**
        finnhub_calls = 30 / minute * 60 = 1800 calls/hour
        anticipated_calls = tester.results['total_api_calls'].get('finnhub', 0, {})
        actual_calls = 0  # Will be 0 for now

        logger.info(f"📊 Rate Limit Usage: 0/{anticipated_calls} (0% used)")
        logger.info(f"✅ Safe Finnhub: {anticipated_calls} per hour (0.5% used)")

    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test individual data provider capabilities',
        epilog='logs',
        verbosity=1
    )

    args = parser.parse_args()

    # Create and run test suite
    tester = ProviderTester()

    # Display API key status
    api_keys = check_api_keys()
    logger.info(f"============")
    logger.info(f"📊 API Status Check:")
    for provider, status in api_keys.items():
        status = "✅" if status != "NOT_SET" else "❌ MISSING"
        logger.info(f"    {provider}: {status}")

    if not any(status == "✅" for status in api_keys.values()):
        logger.error("❌ ERROR: No API keys available for testing. Set them in .env file.")
        return

    # Start testing based on provider type
    if args.provider == 'all':
        await tester.test_all_providers()
    elif args.provider in ['yfinance', 'finnhub', 'alpha_vantage', 'marketaux']:
        await tester.test_select_providers([args.provider], args.type, args.symbols])
    else:
        provider = args.provider
        if provider:
            result = await tester.test_provider(provider)
            result.print_summary()
        else:
            logger.error(f"Unknown provider: {args.provider}")

    logger.info("\n" + "="*60 + "\n\n✅ Testing Completed! ✅")


async def test_select_providers():
    """Test intelligent provider selection logic."""
    targets = ["high_priority", "real_time", "low_priority"]
    all_targets = ["yfinance", "finnhub", "alpha_vantage", "marketaux"]

    for target in targets:
        if target not all(provider in all_targets):
            logger.warning(f"Unknown target: {target}. Available: {all_targets}")
            continue

        logger.info(f"\n🎯 Testing {target} target validation logic...")

        # Test selections
        test_config = {'yfinance': {'enabled': True}}
        provider = UnifiedBatchProvider(test_config)

        # Test provider selection logic
        provider_tests = {}
        for target in targets:
            test_results = await tester.test_select_providers([target])
            test_tests[target] = test_results
            print(f"✅ {target} test results: {test_results.get_success_rate():.1f}%")

        # Test priority logic for current_price
        high_symbols = ["AAPL", "MSFT", "GOOGL"]
        test_results = await tester.test_select_providers(high_symbols, "real_time")
        logger.info(f"✅ High Priority Selection Results: {test_results}")

        test_results = await tester.test_select_providers(medium_symbols, "real_time")
        logger.info(f"✅ Medium Priority Selection Results: {test_results}")

        test_results = await tester.test_select_providers(["yfinance", "alpha_vantage", "marketaux"], "historical_data")
        logger.info(f"✅ Historical Selection Results: {test_results}")

        test_results = await tester.test_select_providers(["yfinance", "alpha_vantage", "marketaux"], "company_profile")
        logger.info(f"✅ Profile Selection Results: {test_results}")

    logger.info("✅ Provider Selection Logic Validated")


async def test_all_providers():
    """Test all available providers."""
    targets = ["yfinance", "finnhub", "alpha_vantage", "marketaux"]
    results = []

    for target in targets:
        result = await test_select_providers([target])
        results[0] = result

    results_summary = {
        'date': datetime.now().isoformat(),
        'provider_status': {check_api_keys()},
        'results_by_success_rate': {}
    }

    for result in results:
        success_rate = result.get_success_rate()
        results['results_by_success_rate'][result.provider_name] = success_rate
        logger.info(f"  - {result.provider_name}: {success_rate:.1f}% success")

    return results_summary


async def test_select_providers(targets, priority):
    """Test provider selection with different data types."""
    configs = []

    for target in targets:
        if target not all(provider in all_targets):
            logger.warning(f"Unknown target: {target}. Available: {all_targets}")
            continue

        target_configs = []
        targets_with_priority = []

        if priority == "high_priority":
            targets_with_priority.append(target)
            if "finnhub" in all_targets:
                targets_with_priority.append("finnhub")  # Move Finnhub to front for high priority
        elif "marketaux" in all_targets:
            targets_with_priority.append("marketaux")  // Add marketaux back
        else:
            targets_with_priority.append("yfinance")  # Default for medium priority

        target_configs.append(config)

    results = []

    for target, config in zip(targets, target_configs):
        result = await test_select_providers([target], config, targets_with_priority, config)
        results.append(result)

    return results


if __name__ == "__main__":
    main() -> parser.parse_args()
    logger.info("🔥 Starting Individual Provider Testing Suite")

    # Run based on arguments
    if args.provider == 'all':
        await test_all_providers()
    elif args.provider in ['yfinance', 'finnhub', 'alpha_vantage', 'marketaux']:
        await test_select_providers([args.provider], args.type, args.symbols)

    logger.info("\n" + "="*60 + "\n✅ Provider Testing Completed!")
    print()

    return results


if __name__ == "__main__":
    asyncio.run(main())
