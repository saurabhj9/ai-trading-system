"""
Cache performance monitoring and metrics collection.

This module provides comprehensive monitoring capabilities for the caching system,
including performance metrics, hit rates, and detailed analytics.
"""
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .cache_config import CacheDataType


class CacheMetrics:
    """Metrics for cache performance tracking."""

    def __init__(self, retention_hours: int = 24):
        """
        Initialize cache metrics.

        Args:
            retention_hours: How long to retain metrics data
        """
        self.retention_period = timedelta(hours=retention_hours)

        # Performance counters
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0

        # Detailed tracking
        self.hits_by_type = defaultdict(int)
        self.misses_by_type = defaultdict(int)
        self.sets_by_type = defaultdict(int)
        self.hits_by_source = defaultdict(int)  # redis, memory

        # Response time tracking
        self.response_times = deque(maxlen=1000)  # Keep last 1000 measurements
        self.response_times_by_type = defaultdict(lambda: deque(maxlen=1000))

        # Time series data
        self.time_series = deque(maxlen=retention_hours * 60)  # One entry per minute

        # Error tracking
        self.errors_log = deque(maxlen=100)  # Keep last 100 errors

        self._last_cleanup = datetime.utcnow()

    def record_hit(self, key: str, data_type: CacheDataType, source: str) -> None:
        """
        Record a cache hit.

        Args:
            key: Cache key
            data_type: Type of data
            source: Cache source (redis, memory)
        """
        self.hits += 1
        self.hits_by_type[data_type] += 1
        self.hits_by_source[source] += 1

    def record_miss(self, key: str, data_type: CacheDataType) -> None:
        """
        Record a cache miss.

        Args:
            key: Cache key
            data_type: Type of data
        """
        self.misses += 1
        self.misses_by_type[data_type] += 1

    def record_set(self, key: str, data_type: CacheDataType, ttl_seconds: int) -> None:
        """
        Record a cache set operation.

        Args:
            key: Cache key
            data_type: Type of data
            ttl_seconds: TTL in seconds
        """
        self.sets += 1
        self.sets_by_type[data_type] += 1

    def record_error(self, operation: str, error: str) -> None:
        """
        Record a cache error.

        Args:
            operation: Operation being performed
            error: Error message
        """
        self.errors += 1
        error_entry = {
            "timestamp": datetime.utcnow(),
            "operation": operation,
            "error": error
        }
        self.errors_log.append(error_entry)

    def record_response_time(self, duration_ms: float, data_type: CacheDataType) -> None:
        """
        Record cache operation response time.

        Args:
            duration_ms: Duration in milliseconds
            data_type: Type of data
        """
        self.response_times.append(duration_ms)
        self.response_times_by_type[data_type].append(duration_ms)

    def _cleanup_old_data(self) -> None:
        """Clean up old data beyond retention period."""
        now = datetime.utcnow()
        if now - self._last_cleanup < timedelta(hours=1):
            return

        cutoff_time = now - self.retention_period

        # Clean up time series data
        while self.time_series and self.time_series[0]["timestamp"] < cutoff_time:
            self.time_series.popleft()

        # Clean up error logs
        while self.errors_log and self.errors_log[0]["timestamp"] < cutoff_time:
            self.errors_log.popleft()

        self._last_cleanup = now

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with all cache metrics
        """
        self._cleanup_old_data()

        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        # Calculate response time statistics
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)

            # Calculate percentiles
            sorted_times = sorted(self.response_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50 = p95 = p99 = 0

        stats = {
            "performance": {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "errors": self.errors,
                "total_requests": total_requests,
                "hit_rate_percent": round(hit_rate, 2)
            },
            "response_times": {
                "average_ms": round(avg_response_time, 3),
                "min_ms": round(min_response_time, 3),
                "max_ms": round(max_response_time, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "sample_size": len(self.response_times)
            },
            "by_data_type": {
                data_type.value: {
                    "hits": self.hits_by_type[data_type],
                    "misses": self.misses_by_type[data_type],
                    "sets": self.sets_by_type[data_type],
                    "hit_rate_percent": round(
                        (self.hits_by_type[data_type] /
                         (self.hits_by_type[data_type] + self.misses_by_type[data_type]) * 100)
                        if (self.hits_by_type[data_type] + self.misses_by_type[data_type]) > 0 else 0, 2
                    )
                }
                for data_type in CacheDataType
            },
            "by_source": {
                source: count
                for source, count in self.hits_by_source.items()
            },
            "recent_errors": [
                {
                    "timestamp": error["timestamp"].isoformat(),
                    "operation": error["operation"],
                    "error": error["error"]
                }
                for error in list(self.errors_log)[-10:]  # Last 10 errors
            ]
        }

        return stats

    def get_time_series_data(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get time series data for the specified period.

        Args:
            minutes: Number of minutes of data to return

        Returns:
            List of time series data points
        """
        self._cleanup_old_data()

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        return [
            point for point in self.time_series
            if point["timestamp"] >= cutoff_time
        ]

    def get_top_misses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get cache misses grouped by key pattern.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of cache miss statistics
        """
        # This would require additional tracking of misses by key
        # For now, return empty list
        return []

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0

        self.hits_by_type.clear()
        self.misses_by_type.clear()
        self.sets_by_type.clear()
        self.hits_by_source.clear()

        self.response_times.clear()
        self.response_times_by_type.clear()

        self.time_series.clear()
        self.errors_log.clear()


class CacheMonitor:
    """
    High-level cache monitoring interface.

    Provides monitoring capabilities for the caching system with
    real-time metrics collection and performance analysis.
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize cache monitor.

        Args:
            retention_hours: How long to retain metrics data
        """
        self.metrics = CacheMetrics(retention_hours)
        self._start_time = datetime.utcnow()

    def record_hit(self, key: str, data_type: CacheDataType, source: str) -> None:
        """Record a cache hit with timing."""
        start_time = time.time()
        self.metrics.record_hit(key, data_type, source)
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_response_time(duration_ms, data_type)

    def record_miss(self, key: str, data_type: CacheDataType) -> None:
        """Record a cache miss with timing."""
        start_time = time.time()
        self.metrics.record_miss(key, data_type)
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_response_time(duration_ms, data_type)

    def record_set(self, key: str, data_type: CacheDataType, ttl_seconds: int) -> None:
        """Record a cache set operation with timing."""
        start_time = time.time()
        self.metrics.record_set(key, data_type, ttl_seconds)
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_response_time(duration_ms, data_type)

    def record_error(self, operation: str, error: str) -> None:
        """Record a cache error."""
        self.metrics.record_error(operation, error)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metrics.get_stats()
        stats["monitor"] = {
            "uptime_seconds": int((datetime.utcnow() - self._start_time).total_seconds()),
            "start_time": self._start_time.isoformat()
        }
        return stats

    def reset_metrics(self) -> None:
        """Reset all monitoring metrics."""
        self.metrics.reset_metrics()
        self._start_time = datetime.utcnow()

    def get_performance_report(self) -> str:
        """
        Generate a human-readable performance report.

        Returns:
            Formatted performance report string
        """
        stats = self.get_stats()

        report = []
        report.append("=== Cache Performance Report ===")
        report.append(f"Uptime: {stats['monitor']['uptime_seconds']:,} seconds")
        report.append("")

        # Performance summary
        perf = stats["performance"]
        report.append("Performance Summary:")
        report.append(f"  Total Requests: {perf['total_requests']:,}")
        report.append(f"  Cache Hits: {perf['hits']:,}")
        report.append(f"  Cache Misses: {perf['misses']:,}")
        report.append(f"  Hit Rate: {perf['hit_rate_percent']:.1f}%")
        report.append(f"  Cache Sets: {perf['sets']:,}")
        report.append(f"  Errors: {perf['errors']:,}")
        report.append("")

        # Response times
        resp = stats["response_times"]
        report.append("Response Times (ms):")
        report.append(f"  Average: {resp['average_ms']:.3f}")
        report.append(f"  Min: {resp['min_ms']:.3f}")
        report.append(f"  Max: {resp['max_ms']:.3f}")
        report.append(f"  50th percentile: {resp['p50_ms']:.3f}")
        report.append(f"  95th percentile: {resp['p95_ms']:.3f}")
        report.append(f"  99th percentile: {resp['p99_ms']:.3f}")
        report.append("")

        # By data type
        report.append("Performance by Data Type:")
        for data_type, stats_by_type in stats["by_data_type"].items():
            if stats_by_type["hits"] + stats_by_type["misses"] > 0:
                report.append(f"  {data_type}:")
                report.append(f"    Hit Rate: {stats_by_type['hit_rate_percent']:.1f}%")
                report.append(f"    Hits: {stats_by_type['hits']:,}")
                report.append(f"    Misses: {stats_by_type['misses']:,}")

        return "\n".join(report)
