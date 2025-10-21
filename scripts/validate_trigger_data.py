#!/usr/bin/env python3
"""
Validate data quality for trigger detection requirements.

This script tests whether the data providers can provide the necessary
data quality and structure needed for the event-driven trigger system.
"""
import asyncio
import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TriggerDataValidator:
    """Validates data quality for trigger detection."""

    def __init__(self):
        self.providers = {
            'yfinance': YFinanceProvider(),
            'finnhub': FinnhubProvider(os.getenv('DATA_FINNHUB_API_KEY')),
            'alpha_vantage': AlphaVantageProvider(os.getenv('DATA_ALPHA_VANTAGE_API_KEY'))
        }
        self.validation_results = {}

    def validate_tick_data_structure(self, data: pd.DataFrame) -> bool:
        """Validate tick data structure for trigger detection."""
        if data is None or len(data) == 0:
            return False

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False

        # Check for reasonable data values
        if (data['Close'] <= 0).any():
            return False

        if (data['Volume'] < 0).any():
            return False

        # Check data continuity (no huge gaps in dates)
        if len(data) > 1:
            date_diffs = data.index.to_series().diff().dt.days.dropna()
            if (date_diffs > 7).any():  # Gaps larger than a week
                logger.warning("Found large gaps in historical data")

        return True

    def validate_price_data_quality(self, price: float) -> bool:
        """Validate individual price data quality."""
        if price is None:
            return False

        if price <= 0:
            return False

        # Reasonable price range for stocks (prevent erroneous data)
        if price > 1000000:  # > $1M per share is unreasonable
            return False

        return True

    def validate_technical_indicators_feasibility(self, data: pd.DataFrame) -> dict:
        """Check if data is sufficient for technical indicators."""
        validation = {
            'sufficient_for_rsi': len(data) >= 14,
            'sufficient_for_macd': len(data) >= 26,
            'sufficient_for_bollinger': len(data) >= 20,
            'sufficient_for_ema_20': len(data) >= 20,
            'sufficient_for_ema_50': len(data) >= 50,
            'sufficient_for_sma_200': len(data) >= 200,
            'has_volume_data': 'Volume' in data.columns and data['Volume'].notna().all(),
            'price_volatility': data['Close'].std() / data['Close'].mean() > 0.01  # At least 1% volatility
        }

        return validation

    async def validate_provider_for_triggers(self, provider_name: str) -> dict:
        """Validate a single provider for trigger detection requirements."""
        logger.info(f"\n=== Validating {provider_name} for Trigger Detection ===")

        provider = self.providers.get(provider_name)
        if not provider:
            return {'status': 'error', 'message': 'Provider not available'}

        results = {
            'provider': provider_name,
            'current_price_quality': False,
            'historical_data_quality': False,
            'technical_indicators_feasible': False,
            'data_freshness': False,
            'error_handling': False,
            'performance_acceptable': False,
            'overall_score': 0
        }

        try:
            # Test 1: Current Price Quality
            logger.info("Testing current price quality...")
            start_time = datetime.now()
            price = await provider.get_current_price("AAPL")
            response_time = (datetime.now() - start_time).total_seconds()

            results['current_price_quality'] = self.validate_price_data_quality(price)
            results['performance_acceptable'] = response_time < 5.0  # < 5 seconds

            if results['current_price_quality']:
                logger.info(f"  ✅ Current price: ${price:.2f} ({response_time:.3f}s)")
            else:
                logger.warning("  ❌ Current price quality failed")

            # Test 2: Historical Data Quality
            logger.info("Testing historical data quality...")
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=60)  # 2 months

            start_time = datetime.now()
            historical_data = await provider.fetch_data("AAPL", start_date, end_date)
            response_time = (datetime.now() - start_time).total_seconds()

            if historical_data is not None:
                results['historical_data_quality'] = self.validate_tick_data_structure(historical_data)
                results['data_freshness'] = (datetime.now() - historical_data.index[-1]).days <= 2

                # Test 3: Technical Indicators Feasibility
                tech_validation = self.validate_technical_indicators_feasibility(historical_data)
                results['technical_indicators_feasible'] = all(tech_validation.values())

                logger.info(f"  ✅ Historical data: {len(historical_data)} days")
                logger.info(f"  Data freshness: {results['data_freshness']}")
                logger.info(f"  Technical indicators: {tech_validation}")

                # Log specific technical indicator feasibility
                for indicator, feasible in tech_validation.items():
                    status = "✅" if feasible else "❌"
                    logger.info(f"    {status} {indicator}")
            else:
                logger.warning("  ❌ No historical data returned")

            # Test 4: Error Handling
            logger.info("Testing error handling...")
            try:
                invalid_price = await provider.get_current_price("INVALID-SYMBOL-123")
                results['error_handling'] = invalid_price is None  # Should return None for invalid symbols
                logger.info(f"  {'✅' if results['error_handling'] else '❌'} Error handling")
            except Exception:
                results['error_handling'] = True  # Exception is also acceptable
                logger.info("  ✅ Error handling (exception thrown)")

            # Calculate overall score
            score_components = [
                results['current_price_quality'],
                results['historical_data_quality'],
                results['technical_indicators_feasible'],
                results['data_freshness'],
                results['error_handling'],
                results['performance_acceptable']
            ]
            results['overall_score'] = sum(score_components) / len(score_components) * 100

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            results['status'] = 'error'
            results['error'] = str(e)

        return results

    async def validate_all_providers(self) -> dict:
        """Validate all providers for trigger detection."""
        logger.info("🔍 Trigger System Data Validation Suite")
        logger.info("=" * 60)

        validation_results = {}

        for provider_name in self.providers.keys():
            if provider_name == 'yfinance' or os.getenv(f'DATA_{provider_name.upper()}_API_KEY'):
                result = await self.validate_provider_for_triggers(provider_name)
                validation_results[provider_name] = result
            else:
                logger.warning(f"Skipping {provider_name} - no API key")

        return validation_results

    def print_validation_summary(self, results: dict):
        """Print comprehensive validation summary."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 TRIGGER SYSTEM DATA VALIDATION SUMMARY")
        logger.info(f"{'='*60}")

        ready_providers = []
        marginal_providers = []
        failed_providers = []

        for provider_name, result in results.items():
            if 'overall_score' in result:
                score = result['overall_score']
                status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"

                logger.info(f"{status} {provider_name}: {score:.1f}%")

                if score >= 80:
                    ready_providers.append(provider_name)
                elif score >= 60:
                    marginal_providers.append(provider_name)
                else:
                    failed_providers.append(provider_name)

                # Show specific capabilities
                capabilities = []
                if result.get('current_price_quality'):
                    capabilities.append("price")
                if result.get('historical_data_quality'):
                    capabilities.append("historical")
                if result.get('technical_indicators_feasible'):
                    capabilities.append("indicators")
                if result.get('data_freshness'):
                    capabilities.append("fresh")

                if capabilities:
                    logger.info(f"    Capabilities: {', '.join(capabilities)}")
            else:
                logger.error(f"❌ {provider_name}: validation error")
                failed_providers.append(provider_name)

        # Overall assessment
        logger.info(f"\n📈 ASSESSMENT")
        logger.info(f"Ready for Production: {len(ready_providers)} providers")
        logger.info(f"Marginal (needs attention): {len(marginal_providers)} providers")
        logger.info(f"Failed: {len(failed_providers)} providers")

        if ready_providers:
            logger.info(f"✅ Recommended: {', '.join(ready_providers)}")

        if marginal_providers:
            logger.info(f"⚠️ Partial support: {', '.join(marginal_providers)}")

        if failed_providers:
            logger.info(f"❌ Not suitable: {', '.join(failed_providers)}")

        # Final recommendation
        if len(ready_providers) >= 1:
            logger.info("\n🎉 TRIGGER SYSTEM DATA VALIDATION: PASSED")
            logger.info("✅ At least one provider is suitable for trigger detection")
        elif len(marginal_providers) >= 1:
            logger.info("\n⚠️ TRIGGER SYSTEM DATA VALIDATION: MARGINAL")
            logger.info("⚠️ Providers have limitations but may work with reduced functionality")
        else:
            logger.info("\n❌ TRIGGER SYSTEM DATA VALIDATION: FAILED")
            logger.info("❌ No suitable providers found for trigger detection")

async def main():
    """Main validation function."""
    validator = TriggerDataValidator()
    results = await validator.validate_all_providers()
    validator.print_validation_summary(results)

if __name__ == "__main__":
    asyncio.run(main())
