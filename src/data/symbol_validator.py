"""
Symbol validation and suggestion utilities for better error handling.
"""
from typing import Dict, Optional, List, Tuple
from difflib import get_close_matches
import asyncio
import yfinance as yf
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SymbolValidationError:
    """Represents a symbol validation error with details."""

    def __init__(self, symbol: str, error_type: str, message: str, suggestion: Optional[str] = None):
        self.symbol = symbol
        self.error_type = error_type  # 'invalid_symbol', 'delisted', 'no_data', 'rate_limit'
        self.message = message
        self.suggestion = suggestion

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "error": self.message,
            "error_type": self.error_type,
            "suggestion": self.suggestion
        }


class SymbolValidator:
    """
    Handles symbol validation and provides user-friendly error messages and suggestions.
    """

    # Common symbol corrections (typo -> correct symbol)
    COMMON_CORRECTIONS: Dict[str, str] = {
        "GOOGL": "GOOG",      # Google (Class C vs Class A)
        "FB": "META",        # Facebook/Meta
        "BRK.B": "BRK-B",    # Berkshire Hathaway
        "BRK.A": "BRK-A",    # Berkshire Hathaway
        "AMZN1": "AMZN",     # Amazon typo
        "AAPL1": "AAPL",     # Apple typo
        "MSFT1": "MSFT",     # Microsoft typo
    }

    # List of major valid symbols for fuzzy matching
    MAJOR_SYMBOLS: List[str] = [
        "AAPL", "GOOG", "MSFT", "NVDA", "TSLA", "AMZN", "META", "BRK-B",
        "JNJ", "JPM", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "CVX",
        "LLY", "TMO", "ABBV", "DHR", "ABT", "CRM", "NVO", "ACN", "ADBE",
        "PFE", "CMCSA", "NFLX", "ORCL", "COST", "KO", "PEP", "MDT", "T",
        "WMT", "LIN", "NVS", "UPS", "DIS", "RTX", "LOW", "CAT", "DE",
        "HON", "TJX", "PLD", "AMT", "EQIX", "TXN", "BABA", "NEE", "BA",
        "SAP", "GS", "MS", "GE", "CVS", "AMD", "INTC", "IBM", "NOW",
        "MU", "ATVI", "QCOM", "MDLZ"
    ]

    def __init__(self):
        self.cached_valid_symbols = set(self.MAJOR_SYMBOLS)

    async def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[SymbolValidationError]]:
        """
        Validate a symbol and return error details if invalid.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Tuple of (is_valid, error_object)
        """
        # Check for common corrections first
        if symbol in self.COMMON_CORRECTIONS:
            correct_symbol = self.COMMON_CORRECTIONS[symbol]
            return False, SymbolValidationError(
                symbol=symbol,
                error_type="invalid_symbol",
                message=f"Invalid symbol '{symbol}'. Did you mean '{correct_symbol}'?",
                suggestion=correct_symbol
            )

        # Quick validation with yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got meaningful data
            if not info:
                # Try fuzzy matching
                suggestion = self.suggest_correction(symbol)
                if suggestion:
                    return False, SymbolValidationError(
                        symbol=symbol,
                        error_type="invalid_symbol",
                        message=f"Invalid symbol '{symbol}'. Did you mean '{suggestion}'?",
                        suggestion=suggestion
                    )
                else:
                    return False, SymbolValidationError(
                        symbol=symbol,
                        error_type="invalid_symbol",
                        message=f"Invalid symbol '{symbol}' - symbol not found or not supported"
                    )

            # Check if it looks like a valid stock
            if 'regularMarketPrice' not in info and 'currentPrice' not in info:
                # Check for other indicators of invalid symbol
                if all(key not in info for key in ['symbol', 'longName', 'exchange']):
                    suggestion = self.suggest_correction(symbol)
                    if suggestion:
                        return False, SymbolValidationError(
                            symbol=symbol,
                            error_type="invalid_symbol",
                            message=f"Invalid symbol '{symbol}'. Did you mean '{suggestion}'?",
                            suggestion=suggestion
                        )
                    else:
                        return False, SymbolValidationError(
                            symbol=symbol,
                            error_type="invalid_symbol",
                            message=f"Invalid symbol '{symbol}' - no market data available"
                        )

            # Symbol looks valid
            return True, None

        except Exception as e:
            error_str = str(e).lower()

            # Parse yfinance specific errors
            if "yftzmissingerror" in error_str:
                if "possibly delisted" in error_str:
                    return False, SymbolValidationError(
                        symbol=symbol,
                        error_type="delisted",
                        message=f"Invalid symbol '{symbol}' - appears to be delisted or inactive"
                    )
                elif "no timezone found" in error_str:
                    suggestion = self.suggest_correction(symbol)
                    return False, SymbolValidationError(
                        symbol=symbol,
                        error_type="invalid_symbol",
                        message=f"Invalid symbol '{symbol}' - no timezone or exchange data" +
                                (f". Did you mean '{suggestion}'?" if suggestion else ""),
                        suggestion=suggestion
                    )
                else:
                    return False, SymbolValidationError(
                        symbol=symbol,
                        error_type="invalid_symbol",
                        message=f"Invalid symbol '{symbol}' - possibly delisted or not found"
                    )

            # Other errors
            logger.warning(f"Error validating symbol {symbol}: {e}")
            suggestion = self.suggest_correction(symbol)
            return False, SymbolValidationError(
                symbol=symbol,
                error_type="invalid_symbol",
                message=f"Invalid symbol '{symbol}' - validation failed" +
                        (f". Did you mean '{suggestion}'?" if suggestion else ""),
                suggestion=suggestion
            )

    def suggest_correction(self, symbol: str) -> Optional[str]:
        """
        Suggest a correction for a potentially misspelled symbol.

        Args:
            symbol: The symbol to find corrections for

        Returns:
            Suggested symbol or None if no good match
        """
        matches = get_close_matches(symbol.upper(),
                                    list(self.cached_valid_symbols),
                                    n=1,
                                    cutoff=0.6)
        return matches[0] if matches else None

    def validate_batch(self, symbols: List[str]) -> Dict[str, Tuple[bool, Optional[SymbolValidationError]]]:
        """
        Validate multiple symbols in batch.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dict mapping symbol to (is_valid, error_object)
        """
        results = {}
        for symbol in symbols:
            is_valid, error = asyncio.run(self.validate_symbol(symbol))
            results[symbol] = (is_valid, error)
        return results
