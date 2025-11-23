"""
Caching layer for MCP tools to improve performance.

Implements in-memory cache with TTL (time-to-live) for regime detection results.
Supports caching of regime detection, statistics, and transition probabilities.
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
import json


class CacheEntry:
    """A cache entry with timestamp tracking."""

    def __init__(self, value: Any, ttl_seconds: int = 14400):
        """
        Initialize a cache entry.

        Args:
            value: The cached value
            ttl_seconds: Time-to-live in seconds (default: 4 hours = 14400 seconds)
        """
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def get_age_minutes(self) -> float:
        """Get age of cache entry in minutes."""
        return (time.time() - self.created_at) / 60


class MCP_Cache:
    """
    Simple in-memory cache for MCP tool results.

    Features:
    - TTL-based expiration (default 4 hours)
    - Automatic cleanup of expired entries
    - Thread-safe operations using dict
    - Cache statistics tracking
    """

    def __init__(self, default_ttl_seconds: int = 14400):
        """
        Initialize the cache.

        Args:
            default_ttl_seconds: Default time-to-live for cache entries (default: 4 hours)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.default_ttl_seconds = default_ttl_seconds

    def _make_key(
        self, ticker: str, n_states: int, start_date: Optional[str], end_date: Optional[str], tool_name: str = "default"
    ) -> str:
        """Create a cache key from parameters."""
        key_parts = [tool_name, ticker.upper(), str(n_states)]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        return "|".join(key_parts)

    def get(
        self,
        ticker: str,
        n_states: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tool_name: str = "default",
    ) -> Optional[Any]:
        """
        Get a value from cache if it exists and is not expired.

        Args:
            ticker: Stock symbol
            n_states: Number of states
            start_date: Start date (optional)
            end_date: End date (optional)
            tool_name: Name of the tool calling cache (for key uniqueness)

        Returns:
            Cached value if found and valid, None otherwise
        """
        key = self._make_key(ticker, n_states, start_date, end_date, tool_name)

        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        if entry.is_expired():
            del self._cache[key]
            self._stats["evictions"] += 1
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        return entry.value

    def set(
        self,
        ticker: str,
        n_states: int,
        value: Any,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        tool_name: str = "default",
    ) -> None:
        """
        Set a value in the cache.

        Args:
            ticker: Stock symbol
            n_states: Number of states
            value: Value to cache
            start_date: Start date (optional)
            end_date: End date (optional)
            ttl_seconds: Time-to-live in seconds (uses default if not specified)
            tool_name: Name of the tool calling cache (for key uniqueness)
        """
        key = self._make_key(ticker, n_states, start_date, end_date, tool_name)
        ttl = ttl_seconds or self.default_ttl_seconds
        self._cache[key] = CacheEntry(value, ttl)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, etc.)
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        # Calculate memory usage estimate
        cache_size_estimate = len(self._cache)

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests,
            "cached_entries": cache_size_estimate,
        }

    def get_cache_info(self) -> str:
        """Get human-readable cache information."""
        stats = self.get_stats()
        return (
            f"Cache Info: {stats['cached_entries']} entries, "
            f"{stats['hits']} hits, {stats['misses']} misses, "
            f"hit rate {stats['hit_rate']*100:.1f}%, "
            f"{stats['evictions']} evictions"
        )


# Global cache instance
_global_cache = MCP_Cache(default_ttl_seconds=14400)  # 4 hours


def get_cache() -> MCP_Cache:
    """Get the global cache instance."""
    return _global_cache
