"""
Optimized algorithm base class with performance improvements.

This module provides HiddenRegimeAlgorithmOptimized, an enhanced version
of HiddenRegimeAlgorithm with caching, batch updates, and profiling.
"""

from datetime import datetime
from typing import Dict, Optional

from .algorithm import HiddenRegimeAlgorithm
from .performance import (
    RegimeModelCache,
    CachedRegimeDetector,
    PerformanceProfiler,
    BatchRegimeUpdater,
)


class HiddenRegimeAlgorithmOptimized(HiddenRegimeAlgorithm):
    """
    Optimized version of HiddenRegimeAlgorithm with performance improvements.

    Enhancements over base class:
    - Model caching to avoid redundant training
    - Batch regime updates for multi-asset portfolios
    - Performance profiling and monitoring
    - Optimized data handling

    Example:
        >>> class MyOptimizedStrategy(HiddenRegimeAlgorithmOptimized):
        ...     def Initialize(self):
        ...         self.enable_caching(max_cache_size=200)
        ...         self.enable_profiling()
        ...         self.enable_batch_updates(max_workers=4)
        ...
        ...         # Rest of initialization...
    """

    def __init__(self) -> None:
        """Initialize optimized algorithm."""
        super().__init__()

        # Performance components
        self._model_cache: Optional[RegimeModelCache] = None
        self._cached_detector: Optional[CachedRegimeDetector] = None
        self._profiler: Optional[PerformanceProfiler] = None
        self._batch_updater: Optional[BatchRegimeUpdater] = None

        # Performance flags
        self._caching_enabled = False
        self._profiling_enabled = False
        self._batch_updates_enabled = False

    def enable_caching(
        self,
        max_cache_size: int = 100,
        retrain_frequency: str = "weekly",
    ) -> None:
        """
        Enable model caching.

        Args:
            max_cache_size: Maximum models to cache
            retrain_frequency: How often to retrain models
        """
        self._model_cache = RegimeModelCache(max_cache_size=max_cache_size)
        self._cached_detector = CachedRegimeDetector(
            cache=self._model_cache,
            retrain_frequency=retrain_frequency,
        )
        self._caching_enabled = True

        self.Debug(f"Caching enabled: max_size={max_cache_size}, frequency={retrain_frequency}")

    def enable_profiling(self) -> None:
        """Enable performance profiling."""
        self._profiler = PerformanceProfiler()
        self._profiling_enabled = True

        self.Debug("Performance profiling enabled")

    def enable_batch_updates(
        self,
        max_workers: int = 4,
        use_parallel: bool = True,
    ) -> None:
        """
        Enable batch regime updates for multi-asset strategies.

        Args:
            max_workers: Maximum parallel workers
            use_parallel: Whether to use parallel processing
        """
        self._batch_updater = BatchRegimeUpdater(
            max_workers=max_workers,
            use_parallel=use_parallel,
        )
        self._batch_updates_enabled = True

        self.Debug(f"Batch updates enabled: workers={max_workers}, parallel={use_parallel}")

    def update_regime(self, ticker: Optional[str] = None) -> bool:
        """
        Update regime detection (optimized version).

        This overrides the base class method to add profiling.

        Args:
            ticker: Specific ticker to update (None = update all)

        Returns:
            True if update successful
        """
        if self._profiling_enabled and self._profiler:
            with self._profiler.time_block("regime_update"):
                return super().update_regime(ticker)
        else:
            return super().update_regime(ticker)

    def batch_update_regimes(self, tickers: list) -> Dict[str, bool]:
        """
        Update multiple tickers in batch (optimized).

        Args:
            tickers: List of tickers to update

        Returns:
            Dict mapping tickers to success status
        """
        if not self._batch_updates_enabled or not self._batch_updater:
            # Fall back to sequential updates
            return {ticker: self.update_regime(ticker) for ticker in tickers}

        # Collect data for all tickers
        data_dict = {}
        for ticker in tickers:
            if ticker in self._data_adapters:
                adapter = self._data_adapters[ticker]
                if adapter.is_ready():
                    try:
                        data_dict[ticker] = adapter.to_dataframe()
                    except ValueError:
                        continue

        # Update function for batch processor
        def update_single(ticker, data, pipeline):
            pipeline.data._data = data
            result = pipeline.update()
            return result

        # Batch update all regimes
        results = self._batch_updater.batch_update(
            assets=list(data_dict.keys()),
            data_dict=data_dict,
            pipeline_dict=self._regime_pipelines,
            update_func=update_single,
        )

        # Convert results to success status
        return {ticker: "error" not in result for ticker, result in results.items()}

    def get_profiling_stats(self) -> Optional[str]:
        """
        Get profiling statistics.

        Returns:
            Formatted string with profiling data or None
        """
        if not self._profiling_enabled or not self._profiler:
            return None

        return self._profiler.get_summary()

    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats or None
        """
        if not self._caching_enabled or not self._model_cache:
            return None

        return self._model_cache.get_stats()

    def log_performance_summary(self) -> None:
        """Log performance summary."""
        if self._profiling_enabled and self._profiler:
            summary = self.get_profiling_stats()
            if summary:
                self.Log("=== PERFORMANCE SUMMARY ===")
                for line in summary.split("\n"):
                    self.Log(line)

        if self._caching_enabled and self._model_cache:
            stats = self.get_cache_stats()
            if stats:
                self.Log("=== CACHE STATISTICS ===")
                self.Log(f"Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
                self.Log(f"Hit rate: {stats['hit_rate']:.1%}")
                self.Log(f"Hits: {stats['hit_count']}, Misses: {stats['miss_count']}")

    def _should_retrain(self, ticker: str) -> bool:
        """
        Check if pipeline should be retrained (optimized).

        Overrides base class to use cached detector if enabled.

        Args:
            ticker: Ticker symbol

        Returns:
            True if retraining is needed
        """
        if self._caching_enabled and self._cached_detector:
            return self._cached_detector.should_retrain(ticker)
        else:
            return super()._should_retrain(ticker)


# Example optimized strategy
class OptimizedMultiAssetExample(HiddenRegimeAlgorithmOptimized):
    """
    Example optimized multi-asset strategy.

    Demonstrates use of all performance optimizations:
    - Caching
    - Batch updates
    - Profiling
    """

    def Initialize(self):
        """Initialize the optimized strategy."""
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Enable all optimizations
        self.enable_caching(max_cache_size=200, retrain_frequency="monthly")
        self.enable_profiling()
        self.enable_batch_updates(max_workers=4)

        # Add multiple assets
        self.assets = ["SPY", "QQQ", "TLT", "GLD"]
        self.symbols = {}

        for ticker in self.assets:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols[ticker] = symbol

            # Initialize regime detection for each
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=3,
                lookback_days=252,
                retrain_frequency="monthly",
            )

        self.rebalance_frequency = 5
        self.days_since_rebalance = 0

        self.Debug("OptimizedMultiAssetExample initialized with all optimizations")

    def OnData(self, data):
        """Handle new data."""
        # Update data for all assets
        for ticker, symbol in self.symbols.items():
            if data.ContainsKey(symbol):
                self.on_tradebar(ticker, data[symbol])

        # Check if all ready
        if not all(self.regime_is_ready(t) for t in self.assets):
            return

        # Use batch update for all assets (optimized)
        self.batch_update_regimes(self.assets)

        # Rebalance on schedule
        self.days_since_rebalance += 1
        if self.days_since_rebalance >= self.rebalance_frequency:
            self.rebalance_portfolio()
            self.days_since_rebalance = 0

    def rebalance_portfolio(self):
        """Rebalance based on regimes."""
        # Find assets in Bull regime
        bull_assets = []
        for ticker in self.assets:
            signal = self.get_regime_signal(ticker)
            if signal and signal.regime_name == "Bull" and signal.confidence >= 0.65:
                bull_assets.append(ticker)

        if not bull_assets:
            self.Liquidate()
            return

        # Equal weight among bull assets
        weight = 1.0 / len(bull_assets)
        for ticker in self.assets:
            if ticker in bull_assets:
                self.SetHoldings(self.symbols[ticker], weight)
            else:
                if self.Portfolio[self.symbols[ticker]].Invested:
                    self.Liquidate(self.symbols[ticker])

    def OnEndOfAlgorithm(self):
        """Log performance summary at end."""
        self.log_performance_summary()
