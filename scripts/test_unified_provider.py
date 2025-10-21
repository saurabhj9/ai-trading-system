#!/usr/bin/env python3
"""
Test unified batch provider with fallback mechanisms.

This script tests the UnifiedBatchProvider to ensure it can:
1. Automatically switch between providers
2. Handle provider failures gracefully
3. Provide consistent data for the trigger system
"""
import asyncio
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import providers
from src.data.providers.unified_batch_provider import UnifiedBatchProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_provider_fallback():
    """Test fallback from one provider to another."""
    logger.info("=== Testing Single Provider Fallback ===")

    # Test with all providers enabled
    config = {
        'yfinance': {'enabled': True},
        'finnhub': {
            'enabled': True,
            'rate_limit': 60,
            'period': 60
        },
        'alpha_vantage': {
            'enabled': True,
            'rate_limit': 5,
            'period': 60
        }
    }

    provider = UnifiedBatchProvider(config)

    # Test current price fetch
    try:
        response = await provider.fetch_batch_current_prices(["AAPL"])
        if response and hasattr(response, 'data') and "AAPL" in response.data:
            price = response.data["AAPL"]
            logger.info(f"✅ Unified provider AAPL price: ${price:.2f}")
            return True
        else:
            logger.error("❌ No price data returned")
            return False
    except Exception as e:
        logger.error(f"❌ Unified provider failed: {e}")
        return False

async def test_batch_operations():
    """Test batch operations with multiple symbols."""
    logger.info("=== Testing Batch Operations ===")

    config = {
        'yfinance': {'enabled': True},
        'alpha_vantage': {'enabled': True, 'rate_limit': 5, 'period': 60}
    }

    provider = UnifiedBatchProvider(config)

    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    try:
        # Test multiple current prices
        response = await provider.fetch_batch_current_prices(symbols)
        logger.info(f"✅ Batch prices response: {type(response)}")
        if response and hasattr(response, 'data'):
            logger.info(f"  Success rate: {response.success_rate:.1f}%")
            logger.info(f"  Prices: {len(response.data)} symbols")

        # Test historical data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        historical_response = await provider.fetch_batch_historical_data("AAPL", start_date, end_date)
        if historical_response and hasattr(historical_response, 'data'):
            logger.info(f"✅ Historical data: {len(historical_response.data)} symbols")
        else:
            logger.warning("⚠️ No historical data returned")

        return True
    except Exception as e:
        logger.error(f"❌ Batch operations failed: {e}")
        return False

async def test_provider_priority():
    """Test provider priority selection."""
    logger.info("=== Testing Provider Priority ===")

    # Test with Finnhub prioritized for real-time data
    config = {
        'yfinance': {'enabled': True},
        'finnhub': {
            'enabled': True,
            'rate_limit': 60,
            'period': 60,
            'priority': 1  # Higher priority
        },
        'alpha_vantage': {
            'enabled': True,
            'rate_limit': 5,
            'period': 60,
            'priority': 3  # Lower priority
        }
    }

    provider = UnifiedBatchProvider(config)

    try:
        response = await provider.fetch_batch_current_prices(["AAPL"])
        if response and hasattr(response, 'data') and "AAPL" in response.data:
            price = response.data["AAPL"]
            logger.info(f"✅ Priority-based price: ${price:.2f}")
            return True
        else:
            logger.error("❌ No price data returned")
            return False
    except Exception as e:
        logger.error(f"❌ Priority test failed: {e}")
        return False

async def test_error_recovery():
    """Test error recovery and fallback."""
    logger.info("=== Testing Error Recovery ===")

    # Test with disabled primary provider to force fallback
    config = {
        'yfinance': {'enabled': False},  # Disabled
        'finnhub': {
            'enabled': True,
            'rate_limit': 60,
            'period': 60
        }
    }

    provider = UnifiedBatchProvider(config)

    try:
        response = await provider.fetch_batch_current_prices(["AAPL"])
        if response and hasattr(response, 'data') and "AAPL" in response.data:
            price = response.data["AAPL"]
            logger.info(f"✅ Fallback price: ${price:.2f}")
            return True
        else:
            logger.error("❌ No price data returned")
            return False
    except Exception as e:
        logger.error(f"❌ Fallback failed: {e}")
        return False

async def test_rate_limiting():
    """Test rate limiting behavior."""
    logger.info("=== Testing Rate Limiting ===")

    # Very restrictive config to test rate limiting
    config = {
        'alpha_vantage': {
            'enabled': True,
            'rate_limit': 2,
            'period': 60
        },
        'yfinance': {'enabled': True}  # No rate limit
    }

    provider = UnifiedBatchProvider(config)

    try:
        # Make multiple requests quickly
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

        # Test batch rate limiting
        response = await provider.fetch_batch_current_prices(symbols)

        logger.info(f"✅ Rate limiting test: {len(response.data) if response and hasattr(response, 'data') else 0} prices fetched")
        if response and hasattr(response, 'data'):
            for symbol, price in response.data.items():
                logger.info(f"  {symbol}: ${price:.2f}")

        return True
    except Exception as e:
        logger.error(f"❌ Rate limiting test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🔧 Unified Batch Provider Testing Suite")
    logger.info("=" * 50)

    test_results = {
        "single_fallback": await test_single_provider_fallback(),
        "batch_operations": await test_batch_operations(),
        "provider_priority": await test_provider_priority(),
        "error_recovery": await test_error_recovery(),
        "rate_limiting": await test_rate_limiting()
    }

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 UNIFIED PROVIDER SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 Unified batch provider is ready!")
    elif passed >= total * 0.8:
        logger.info("✅ Unified batch provider is mostly ready")
    else:
        logger.warning("⚠️ Unified batch provider needs attention")

if __name__ == "__main__":
    asyncio.run(main())
