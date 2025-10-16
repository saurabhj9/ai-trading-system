"""
Market-aware TTL (Time-To-Live) calculation for caching.

This module provides intelligent TTL calculation based on market hours,
data characteristics, and trading schedules for different exchanges.
"""
import datetime
from datetime import time, timedelta
from typing import Optional

import pytz

from .cache_config import CacheDataType, TTLConfig, MarketHours


class MarketAwareTTL:
    """
    Calculates TTL based on market awareness and data characteristics.
    
    This class determines how long different types of data should be cached
    based on market hours, trading schedules, and data freshness requirements.
    """
    
    def __init__(self, ttl_config: Optional[TTLConfig] = None):
        """
        Initialize market-aware TTL calculator.
        
        Args:
            ttl_config: TTL configuration settings
        """
        self.ttl_config = ttl_config or TTLConfig()
        
        # Timezone configurations
        self.ny_tz = pytz.timezone('America/New_York')
        self.utc_tz = pytz.UTC
        
        # Market schedules
        self.nyse_open = MarketHours.NYSE_OPEN
        self.nyse_close = MarketHours.NYSE_CLOSE
        self.nasdaq_open = MarketHours.NASDAQ_OPEN
        self.nasdaq_close = MarketHours.NASDAQ_CLOSE
        
        # Holiday calendar (simplified - in production would use proper holiday API)
        self.holidays = self._get_holiday_dates()
    
    def _get_holiday_dates(self) -> set:
        """
        Get holiday dates for market closures.
        
        Returns:
            Set of holiday dates (simplified for demo)
        """
        # Simplified holiday list - in production would use a proper holiday API
        current_year = datetime.datetime.now().year
        holidays = set()
        
        # Add major US holidays (simplified dates)
        holidays.add(datetime.date(current_year, 1, 1))  # New Year's Day
        holidays.add(datetime.date(current_year, 7, 4))   # Independence Day
        holidays.add(datetime.date(current_year, 12, 25)) # Christmas Day
        
        return holidays
    
    def _is_market_hours(self, timestamp: Optional[datetime.datetime] = None) -> bool:
        """
        Check if current time is during market hours.
        
        Args:
            timestamp: Timestamp to check (defaults to now)
            
        Returns:
            True if during market hours, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(self.utc_tz)
        
        # Convert to Eastern Time
        eastern_time = timestamp.astimezone(self.ny_tz)
        
        # Check if it's a weekend
        if eastern_time.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # Check if it's a holiday
        if eastern_time.date() in self.holidays:
            return False
        
        # Check if within market hours
        current_time = eastern_time.time()
        return self.nyse_open <= current_time <= self.nyse_close
    
    def _get_next_market_open(self, timestamp: Optional[datetime.datetime] = None) -> datetime.datetime:
        """
        Get the next market open time.
        
        Args:
            timestamp: Reference timestamp (defaults to now)
            
        Returns:
            Next market open datetime
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(self.utc_tz)
        
        eastern_time = timestamp.astimezone(self.ny_tz)
        
        # Start from tomorrow
        next_day = eastern_time + timedelta(days=1)
        
        # Find the next weekday that's not a holiday
        while next_day.weekday() >= 5 or next_day.date() in self.holidays:
            next_day += timedelta(days=1)
        
        # Set the open time
        next_open = datetime.datetime.combine(
            next_day.date(),
            self.nyse_open,
            tzinfo=self.ny_tz
        )
        
        return next_open
    
    def calculate_ttl(
        self,
        data_type: CacheDataType,
        timestamp: Optional[datetime.datetime] = None
    ) -> timedelta:
        """
        Calculate TTL for a specific data type based on market awareness.
        
        Args:
            data_type: Type of data being cached
            timestamp: Reference timestamp for calculation
            
        Returns:
            TTL duration
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(self.utc_tz)
        
        is_market_hours = self._is_market_hours(timestamp)
        
        # Calculate TTL based on data type and market status
        if data_type == CacheDataType.CURRENT_PRICE:
            return self._calculate_price_ttl(is_market_hours, timestamp)
        
        elif data_type == CacheDataType.HISTORICAL_OHLCV:
            return self._calculate_historical_ttl(timestamp)
        
        elif data_type == CacheDataType.NEWS_SENTIMENT:
            return self.ttl_config.news_sentiment
        
        elif data_type == CacheDataType.TECHNICAL_INDICATORS:
            return self._calculate_indicators_ttl(timestamp)
        
        elif data_type == CacheDataType.COMPANY_PROFILE:
            return self.ttl_config.company_profile
        
        elif data_type == CacheDataType.FUNDAMENTAL_DATA:
            return self.ttl_config.fundamental_data
        
        elif data_type == CacheDataType.MARKET_REGIME:
            return self.ttl_config.market_regime
        
        # Default TTL for unknown data types
        return timedelta(hours=1)
    
    def _calculate_price_ttl(
        self,
        is_market_hours: bool,
        timestamp: datetime.datetime
    ) -> timedelta:
        """
        Calculate TTL for current price data.
        
        Args:
            is_market_hours: Whether market is currently open
            timestamp: Reference timestamp
            
        Returns:
            TTL for price data
        """
        if is_market_hours:
            # Short TTL during market hours for freshness
            return self.ttl_config.current_price_market_hours
        else:
            # Longer TTL after hours
            return self.ttl_config.current_price_after_hours
    
    def _calculate_historical_ttl(self, timestamp: datetime.datetime) -> timedelta:
        """
        Calculate TTL for historical OHLCV data.
        
        Args:
            timestamp: Reference timestamp
            
        Returns:
            TTL for historical data
        """
        # Historical data doesn't change during the day
        # Cache until next market day
        next_market_open = self._get_next_market_open(timestamp)
        
        # Add some buffer time to ensure we don't have stale data
        ttl = next_market_open - timestamp + timedelta(minutes=30)
        
        # Cap at the configured maximum
        return min(ttl, self.ttl_config.historical_ohlcv)
    
    def _calculate_indicators_ttl(self, timestamp: datetime.datetime) -> timedelta:
        """
        Calculate TTL for technical indicators.
        
        Args:
            timestamp: Reference timestamp
            
        Returns:
            TTL for technical indicators
        """
        # Technical indicators depend on underlying data
        # Use similar logic to historical data
        return self._calculate_historical_ttl(timestamp)
    
    def get_cache_expiry(
        self,
        data_type: CacheDataType,
        timestamp: Optional[datetime.datetime] = None
    ) -> datetime.datetime:
        """
        Get the absolute expiry time for cached data.
        
        Args:
            data_type: Type of data being cached
            timestamp: Reference timestamp
            
        Returns:
            Absolute expiry datetime
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(self.utc_tz)
        
        ttl = self.calculate_ttl(data_type, timestamp)
        return timestamp + ttl
    
    def should_refresh(
        self,
        data_type: CacheDataType,
        cached_timestamp: datetime.datetime,
        current_timestamp: Optional[datetime.datetime] = None
    ) -> bool:
        """
        Determine if cached data should be refreshed.
        
        Args:
            data_type: Type of cached data
            cached_timestamp: When the data was cached
            current_timestamp: Current timestamp for comparison
            
        Returns:
            True if data should be refreshed
        """
        if current_timestamp is None:
            current_timestamp = datetime.datetime.now(self.utc_tz)
        
        ttl = self.calculate_ttl(data_type, cached_timestamp)
        expiry_time = cached_timestamp + ttl
        
        return current_timestamp >= expiry_time
    
    def get_cache_info(
        self,
        data_type: CacheDataType,
        symbol: str,
        cached_timestamp: datetime.datetime
    ) -> dict:
        """
        Get cache information for logging and monitoring.
        
        Args:
            data_type: Type of cached data
            symbol: Symbol for the data
            cached_timestamp: When the data was cached
            
        Returns:
            Dictionary with cache information
        """
        current_time = datetime.datetime.now(self.utc_tz)
        ttl = self.calculate_ttl(data_type, cached_timestamp)
        expiry_time = cached_timestamp + ttl
        
        is_stale = current_time >= expiry_time
        time_remaining = max(timedelta(0), expiry_time - current_time)
        
        return {
            "symbol": symbol,
            "data_type": data_type.value,
            "cached_at": cached_timestamp.isoformat(),
            "expires_at": expiry_time.isoformat(),
            "ttl_seconds": int(ttl.total_seconds()),
            "time_remaining_seconds": int(time_remaining.total_seconds()),
            "is_stale": is_stale,
            "is_market_hours": self._is_market_hours(current_time)
        }
