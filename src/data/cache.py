"""
In-memory cache manager with Time-To-Live (TTL) support.
"""
import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    A simple in-memory cache with Time-To-Live (TTL) support.

    This manager stores data in a dictionary and automatically handles the
    expiration of cached items.
    """

    def __init__(self):
        """Initializes the cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl_seconds: int):
        """
        Adds an item to the cache with a specified TTL.

        Args:
            key: The key to store the item under.
            value: The item to store.
            ttl_seconds: The time-to-live for the item, in seconds.
        """
        if ttl_seconds <= 0:
            return  # Do not cache if TTL is non-positive

        expires_at = time.time() + ttl_seconds
        self._cache[key] = {"value": value, "expires_at": expires_at}
        logger.debug(f"Cache SET for key: {key} with TTL: {ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and has not expired.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached item, or None if it is not found or has expired.
        """
        item = self._cache.get(key)
        if item is None:
            logger.debug(f"Cache MISS for key: {key}")
            return None

        if time.time() > item["expires_at"]:
            # Item has expired, so remove it from the cache
            del self._cache[key]
            print(f"Cache EXPIRED for key: {key}")
            return None

        logger.debug(f"Cache HIT for key: {key}")
        return item["value"]

    def clear(self):
        """Clears all items from the cache."""
        self._cache.clear()
        print("Cache CLEARED.")
