"""
Caching system for trained regime detection models.

This module provides intelligent caching to avoid re-training HMM models
unnecessarily, significantly improving backtest performance.
"""

import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import hashlib


class RegimeModelCache:
    """
    Cache for trained regime detection models.

    Stores trained HMM models to avoid repeated training on same data.
    Uses data fingerprinting to detect when retraining is needed.
    """

    def __init__(self, max_cache_size: int = 100):
        """
        Initialize model cache.

        Args:
            max_cache_size: Maximum number of models to cache
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_cache_key(
        self,
        ticker: str,
        n_states: int,
        data_hash: str,
        config_hash: str,
    ) -> str:
        """
        Generate unique cache key for model.

        Args:
            ticker: Asset ticker
            n_states: Number of HMM states
            data_hash: Hash of training data
            config_hash: Hash of configuration

        Returns:
            Cache key string
        """
        key_str = f"{ticker}_{n_states}_{data_hash}_{config_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _hash_data(self, data: Any) -> str:
        """
        Generate hash of DataFrame for cache key.

        Args:
            data: pandas DataFrame

        Returns:
            Hash string
        """
        try:
            # Hash based on data shape and first/last values
            shape_str = str(data.shape)
            first_val = str(data.iloc[0].values if len(data) > 0 else "")
            last_val = str(data.iloc[-1].values if len(data) > 0 else "")
            hash_input = f"{shape_str}_{first_val}_{last_val}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
        except Exception:
            return "default"

    def _hash_config(self, config: Dict) -> str:
        """
        Generate hash of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string
        """
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def get(
        self,
        ticker: str,
        n_states: int,
        data: Any,
        config: Dict,
    ) -> Optional[Any]:
        """
        Retrieve cached model if available.

        Args:
            ticker: Asset ticker
            n_states: Number of states
            data: Training data
            config: Configuration dict

        Returns:
            Cached model or None
        """
        data_hash = self._hash_data(data)
        config_hash = self._hash_config(config)
        cache_key = self._generate_cache_key(ticker, n_states, data_hash, config_hash)

        if cache_key in self.cache:
            self.hit_count += 1
            cached_entry = self.cache[cache_key]
            cached_entry["last_accessed"] = datetime.now()
            cached_entry["access_count"] += 1
            return cached_entry["model"]
        else:
            self.miss_count += 1
            return None

    def put(
        self,
        ticker: str,
        n_states: int,
        data: Any,
        config: Dict,
        model: Any,
    ) -> None:
        """
        Store model in cache.

        Args:
            ticker: Asset ticker
            n_states: Number of states
            data: Training data used
            config: Configuration dict
            model: Trained model to cache
        """
        data_hash = self._hash_data(data)
        config_hash = self._hash_config(config)
        cache_key = self._generate_cache_key(ticker, n_states, data_hash, config_hash)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()

        self.cache[cache_key] = {
            "model": model,
            "ticker": ticker,
            "n_states": n_states,
            "created": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0,
        }

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self.cache:
            return

        # Find entry with oldest last_accessed time
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]["last_accessed"],
        )
        del self.cache[lru_key]

    def clear(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = (
            self.hit_count / total_requests if total_requests > 0 else 0.0
        )

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


class CachedRegimeDetector:
    """
    Regime detector with intelligent caching.

    Wraps regime detection pipeline with caching layer to minimize
    redundant HMM training operations.
    """

    def __init__(
        self,
        cache: Optional[RegimeModelCache] = None,
        retrain_frequency: str = "weekly",
        force_retrain_days: int = 30,
    ):
        """
        Initialize cached regime detector.

        Args:
            cache: Model cache instance (creates new if None)
            retrain_frequency: How often to retrain ('daily', 'weekly', 'monthly', 'never')
            force_retrain_days: Force retrain after this many days regardless
        """
        self.cache = cache or RegimeModelCache()
        self.retrain_frequency = retrain_frequency
        self.force_retrain_days = force_retrain_days
        self.last_train_dates: Dict[str, datetime] = {}
        self.pipelines: Dict[str, Any] = {}

    def should_retrain(self, ticker: str) -> bool:
        """
        Determine if model should be retrained.

        Args:
            ticker: Asset ticker

        Returns:
            True if retraining is needed
        """
        if self.retrain_frequency == "never":
            return ticker not in self.pipelines

        if ticker not in self.last_train_dates:
            return True

        last_train = self.last_train_dates[ticker]
        days_since = (datetime.now() - last_train).days

        # Force retrain if too old
        if days_since >= self.force_retrain_days:
            return True

        # Check frequency
        if self.retrain_frequency == "daily":
            return days_since >= 1
        elif self.retrain_frequency == "weekly":
            return days_since >= 7
        elif self.retrain_frequency == "monthly":
            return days_since >= 30

        return False

    def get_or_train(
        self,
        ticker: str,
        n_states: int,
        data: Any,
        config: Dict,
        train_func: Any,
    ) -> Any:
        """
        Get cached model or train new one.

        Args:
            ticker: Asset ticker
            n_states: Number of HMM states
            data: Training data
            config: Configuration dict
            train_func: Function to train new model if needed

        Returns:
            Trained pipeline/model
        """
        # Check if we should use cached model
        if not self.should_retrain(ticker):
            cached_model = self.cache.get(ticker, n_states, data, config)
            if cached_model is not None:
                return cached_model

        # Train new model
        model = train_func()

        # Cache the new model
        self.cache.put(ticker, n_states, data, config, model)
        self.last_train_dates[ticker] = datetime.now()
        self.pipelines[ticker] = model

        return model

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        return self.cache.get_stats()


def create_cached_detector(
    retrain_frequency: str = "weekly",
    max_cache_size: int = 100,
) -> CachedRegimeDetector:
    """
    Factory function to create cached regime detector.

    Args:
        retrain_frequency: Retraining schedule
        max_cache_size: Maximum models to cache

    Returns:
        CachedRegimeDetector instance
    """
    cache = RegimeModelCache(max_cache_size=max_cache_size)
    return CachedRegimeDetector(
        cache=cache,
        retrain_frequency=retrain_frequency,
    )
