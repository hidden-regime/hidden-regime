"""
Unit tests for QuantConnect performance optimization components.

Tests:
- RegimeModelCache
- CachedRegimeDetector
- PerformanceProfiler
- BatchRegimeUpdater
"""
import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from hidden_regime.quantconnect.performance.caching import (
    RegimeModelCache,
    CachedRegimeDetector
)
from hidden_regime.quantconnect.performance.profiling import PerformanceProfiler
from hidden_regime.quantconnect.performance.batch_updates import BatchRegimeUpdater


class TestRegimeModelCache:
    """Test RegimeModelCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = RegimeModelCache(max_cache_size=100)

        assert cache.max_cache_size == 100
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = RegimeModelCache(max_cache_size=100)

        # Same parameters should generate same key
        key1 = cache._generate_key('SPY', 3, pd.DataFrame({'Close': [100, 101, 102]}))
        key2 = cache._generate_key('SPY', 3, pd.DataFrame({'Close': [100, 101, 102]}))

        assert key1 == key2

    def test_cache_hit(self):
        """Test cache hit functionality."""
        cache = RegimeModelCache(max_cache_size=100)

        data = pd.DataFrame({'Close': [100, 101, 102]})
        model = Mock()

        # Store in cache
        cache.set('SPY', 3, data, None, model)

        # Retrieve from cache
        cached_model = cache.get('SPY', 3, data, None)

        assert cached_model == model
        assert cache.hit_count == 1
        assert cache.miss_count == 0

    def test_cache_miss(self):
        """Test cache miss functionality."""
        cache = RegimeModelCache(max_cache_size=100)

        data = pd.DataFrame({'Close': [100, 101, 102]})

        # Try to get non-existent item
        result = cache.get('SPY', 3, data, None)

        assert result is None
        assert cache.hit_count == 0
        assert cache.miss_count == 1

    def test_cache_eviction_lru(self):
        """Test LRU eviction when cache is full."""
        cache = RegimeModelCache(max_cache_size=3)

        # Add 4 items (should evict oldest)
        for i in range(4):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, Mock())

        # First item should be evicted
        data0 = pd.DataFrame({'Close': [100]})
        result = cache.get('Asset0', 3, data0, None)

        assert result is None  # First item evicted
        assert len(cache.cache) <= 3

    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = RegimeModelCache(max_cache_size=10)

        data = pd.DataFrame({'Close': [100, 101, 102]})

        # 1 set, 2 hits, 1 miss
        cache.set('SPY', 3, data, None, Mock())
        cache.get('SPY', 3, data, None)  # Hit
        cache.get('SPY', 3, data, None)  # Hit
        cache.get('QQQ', 3, data, None)  # Miss

        stats = cache.get_statistics()

        assert stats['hit_count'] == 2
        assert stats['miss_count'] == 1
        assert stats['hit_rate'] == 2 / 3
        assert stats['cache_size'] == 1

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = RegimeModelCache(max_cache_size=10)

        # Add items
        for i in range(5):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, Mock())

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_with_different_n_states(self):
        """Test that different n_states produce different cache keys."""
        cache = RegimeModelCache(max_cache_size=10)

        data = pd.DataFrame({'Close': [100, 101, 102]})

        cache.set('SPY', 3, data, None, Mock(name='model_3'))
        cache.set('SPY', 4, data, None, Mock(name='model_4'))

        # Should be different cache entries
        model_3 = cache.get('SPY', 3, data, None)
        model_4 = cache.get('SPY', 4, data, None)

        assert model_3 != model_4

    def test_cache_with_different_data(self):
        """Test that different data produces cache miss."""
        cache = RegimeModelCache(max_cache_size=10)

        data1 = pd.DataFrame({'Close': [100, 101, 102]})
        data2 = pd.DataFrame({'Close': [100, 101, 103]})  # Different

        cache.set('SPY', 3, data1, None, Mock())

        # Different data should miss
        result = cache.get('SPY', 3, data2, None)

        assert result is None
        assert cache.miss_count == 1


class TestCachedRegimeDetector:
    """Test CachedRegimeDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = CachedRegimeDetector(
            max_cache_size=100,
            retrain_frequency='weekly'
        )

        assert detector.retrain_frequency == 'weekly'
        assert detector.model_cache.max_cache_size == 100

    def test_should_retrain_daily(self):
        """Test daily retrain frequency."""
        detector = CachedRegimeDetector(retrain_frequency='daily')

        detector.last_train_date = pd.Timestamp('2020-01-01')

        # Next day should trigger retrain
        should = detector.should_retrain(pd.Timestamp('2020-01-02'))
        assert should

    def test_should_retrain_weekly(self):
        """Test weekly retrain frequency."""
        detector = CachedRegimeDetector(retrain_frequency='weekly')

        detector.last_train_date = pd.Timestamp('2020-01-01')

        # Same week - no retrain
        should = detector.should_retrain(pd.Timestamp('2020-01-03'))
        assert not should

        # Next week - retrain
        should = detector.should_retrain(pd.Timestamp('2020-01-10'))
        assert should

    def test_should_retrain_monthly(self):
        """Test monthly retrain frequency."""
        detector = CachedRegimeDetector(retrain_frequency='monthly')

        detector.last_train_date = pd.Timestamp('2020-01-01')

        # Same month - no retrain
        should = detector.should_retrain(pd.Timestamp('2020-01-15'))
        assert not should

        # Next month - retrain
        should = detector.should_retrain(pd.Timestamp('2020-02-05'))
        assert should

    def test_detect_regime_with_cache(self, mock_pipeline):
        """Test regime detection with caching."""
        detector = CachedRegimeDetector(
            max_cache_size=10,
            retrain_frequency='weekly'
        )

        data = pd.DataFrame({
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.uniform(1e6, 5e6, 100)
        }, index=pd.date_range('2020-01-01', periods=100))

        # First call - should train and cache
        result1 = detector.detect_regime(
            ticker='SPY',
            n_states=3,
            data=data,
            pipeline=mock_pipeline
        )

        # Second call with same data - should hit cache
        result2 = detector.detect_regime(
            ticker='SPY',
            n_states=3,
            data=data,
            pipeline=mock_pipeline
        )

        # Should get same results
        assert result1 == result2
        assert detector.model_cache.hit_count > 0

    def test_cache_invalidation_on_retrain(self, mock_pipeline):
        """Test that cache is updated when retraining."""
        detector = CachedRegimeDetector(
            max_cache_size=10,
            retrain_frequency='daily'
        )

        data = pd.DataFrame({
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.uniform(1e6, 5e6, 100)
        }, index=pd.date_range('2020-01-01', periods=100))

        # First detection
        detector.detect_regime('SPY', 3, data, mock_pipeline)
        detector.last_train_date = pd.Timestamp('2020-01-01')

        # Second detection next day - should retrain
        detector.current_date = pd.Timestamp('2020-01-02')
        detector.detect_regime('SPY', 3, data, mock_pipeline)

        # Cache should have been updated
        assert detector.model_cache.cache_size > 0


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()

        assert len(profiler.timings) == 0
        assert not profiler.enabled

    def test_enable_disable(self):
        """Test enabling and disabling profiler."""
        profiler = PerformanceProfiler()

        profiler.enable()
        assert profiler.enabled

        profiler.disable()
        assert not profiler.enabled

    def test_time_operation_decorator(self):
        """Test timing operation decorator."""
        profiler = PerformanceProfiler()
        profiler.enable()

        @profiler.time_operation('test_operation')
        def slow_function():
            time.sleep(0.1)
            return 42

        result = slow_function()

        assert result == 42
        assert 'test_operation' in profiler.timings
        assert len(profiler.timings['test_operation']) == 1
        assert profiler.timings['test_operation'][0] >= 0.1

    def test_time_operation_context_manager(self):
        """Test timing operation context manager."""
        profiler = PerformanceProfiler()
        profiler.enable()

        with profiler.time_operation('context_test'):
            time.sleep(0.05)

        assert 'context_test' in profiler.timings
        assert profiler.timings['context_test'][0] >= 0.05

    def test_multiple_timings(self):
        """Test recording multiple timings for same operation."""
        profiler = PerformanceProfiler()
        profiler.enable()

        for _ in range(5):
            with profiler.time_operation('repeated_op'):
                time.sleep(0.01)

        assert len(profiler.timings['repeated_op']) == 5

    def test_get_statistics(self):
        """Test statistics calculation."""
        profiler = PerformanceProfiler()
        profiler.enable()

        # Add some timings manually
        profiler.timings['operation'] = [0.1, 0.2, 0.15, 0.18, 0.12]

        stats = profiler.get_statistics('operation')

        assert 'mean_time' in stats
        assert 'median_time' in stats
        assert 'min_time' in stats
        assert 'max_time' in stats
        assert 'std_dev' in stats
        assert 'call_count' in stats

        assert stats['call_count'] == 5
        assert stats['min_time'] == 0.1
        assert stats['max_time'] == 0.2

    def test_get_all_statistics(self):
        """Test getting statistics for all operations."""
        profiler = PerformanceProfiler()
        profiler.enable()

        profiler.timings['op1'] = [0.1, 0.2]
        profiler.timings['op2'] = [0.3, 0.4]

        all_stats = profiler.get_statistics()

        assert 'op1' in all_stats
        assert 'op2' in all_stats
        assert all_stats['op1']['call_count'] == 2
        assert all_stats['op2']['call_count'] == 2

    def test_profiler_when_disabled(self):
        """Test that profiler doesn't record when disabled."""
        profiler = PerformanceProfiler()
        # Keep disabled

        with profiler.time_operation('disabled_test'):
            time.sleep(0.05)

        assert len(profiler.timings) == 0

    def test_reset_statistics(self):
        """Test resetting profiler statistics."""
        profiler = PerformanceProfiler()
        profiler.enable()

        profiler.timings['op'] = [0.1, 0.2, 0.3]

        profiler.reset()

        assert len(profiler.timings) == 0

    def test_get_summary_report(self):
        """Test generating summary report."""
        profiler = PerformanceProfiler()
        profiler.enable()

        profiler.timings['operation1'] = [0.1, 0.2, 0.15]
        profiler.timings['operation2'] = [0.5, 0.6]

        report = profiler.get_summary_report()

        assert isinstance(report, str)
        assert 'operation1' in report
        assert 'operation2' in report


class TestBatchRegimeUpdater:
    """Test BatchRegimeUpdater class."""

    def test_initialization(self):
        """Test updater initialization."""
        updater = BatchRegimeUpdater(max_workers=4)

        assert updater.max_workers == 4

    def test_batch_update_sequential(self):
        """Test sequential batch updates."""
        updater = BatchRegimeUpdater(max_workers=1, use_parallel=False)

        assets = ['SPY', 'QQQ', 'TLT']

        def update_func(asset, data, pipeline):
            time.sleep(0.01)  # Simulate work
            return {'asset': asset, 'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        results = updater.batch_update(assets, data_dict, pipeline_dict, update_func)

        assert len(results) == 3
        assert all('regime' in r for r in results.values())

    def test_batch_update_parallel(self):
        """Test parallel batch updates."""
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        assets = ['SPY', 'QQQ', 'TLT', 'GLD']

        def update_func(asset, data, pipeline):
            time.sleep(0.05)  # Simulate work
            return {'asset': asset, 'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        start_time = time.time()
        results = updater.batch_update(assets, data_dict, pipeline_dict, update_func)
        elapsed = time.time() - start_time

        assert len(results) == 4
        # Parallel should be faster than 4 * 0.05 = 0.2s
        assert elapsed < 0.15  # Allow some margin

    def test_batch_update_error_handling(self):
        """Test error handling in batch updates."""
        updater = BatchRegimeUpdater(max_workers=2)

        assets = ['SPY', 'QQQ', 'TLT']

        def update_func(asset, data, pipeline):
            if asset == 'QQQ':
                raise ValueError("Simulated error")
            return {'asset': asset, 'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        results = updater.batch_update(
            assets, data_dict, pipeline_dict, update_func,
            continue_on_error=True
        )

        # Should have results for SPY and TLT, error for QQQ
        assert 'SPY' in results
        assert 'TLT' in results
        # QQQ might be missing or have error marker

    def test_performance_speedup(self):
        """Test that parallel execution is faster than sequential."""
        assets = ['Asset' + str(i) for i in range(8)]

        def slow_update(asset, data, pipeline):
            time.sleep(0.05)
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        # Sequential
        updater_seq = BatchRegimeUpdater(max_workers=1, use_parallel=False)
        start = time.time()
        updater_seq.batch_update(assets, data_dict, pipeline_dict, slow_update)
        seq_time = time.time() - start

        # Parallel
        updater_par = BatchRegimeUpdater(max_workers=4, use_parallel=True)
        start = time.time()
        updater_par.batch_update(assets, data_dict, pipeline_dict, slow_update)
        par_time = time.time() - start

        # Parallel should be at least 2x faster
        speedup = seq_time / par_time
        assert speedup >= 2.0

    def test_batch_size_handling(self):
        """Test handling of different batch sizes."""
        updater = BatchRegimeUpdater(max_workers=4)

        def update_func(asset, data, pipeline):
            return {'regime': 'Bull'}

        # Small batch
        small_assets = ['SPY']
        small_results = updater.batch_update(
            small_assets,
            {a: pd.DataFrame() for a in small_assets},
            {a: Mock() for a in small_assets},
            update_func
        )
        assert len(small_results) == 1

        # Large batch
        large_assets = [f'Asset{i}' for i in range(20)]
        large_results = updater.batch_update(
            large_assets,
            {a: pd.DataFrame() for a in large_assets},
            {a: Mock() for a in large_assets},
            update_func
        )
        assert len(large_results) == 20

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        updater = BatchRegimeUpdater(max_workers=4)

        assets = ['SPY', 'QQQ']

        def update_func(asset, data, pipeline):
            time.sleep(0.01)
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        updater.batch_update(assets, data_dict, pipeline_dict, update_func)

        stats = updater.get_statistics()

        assert 'total_batches' in stats
        assert 'total_updates' in stats
        assert stats['total_updates'] == 2


class TestPerformanceIntegration:
    """Integration tests for performance components."""

    def test_caching_with_profiling(self, mock_pipeline, sample_price_data):
        """Test using caching and profiling together."""
        cache = RegimeModelCache(max_cache_size=10)
        profiler = PerformanceProfiler()
        profiler.enable()

        with profiler.time_operation('cache_test'):
            # First call - cache miss
            model = cache.get('SPY', 3, sample_price_data, None)
            if model is None:
                model = Mock()
                cache.set('SPY', 3, sample_price_data, None, model)

            # Second call - cache hit
            cached_model = cache.get('SPY', 3, sample_price_data, None)

        stats = profiler.get_statistics('cache_test')
        cache_stats = cache.get_statistics()

        assert stats['call_count'] == 1
        assert cache_stats['hit_count'] == 1

    def test_full_optimization_stack(self, sample_multi_asset_data):
        """Test complete optimization stack together."""
        cache = RegimeModelCache(max_cache_size=50)
        profiler = PerformanceProfiler()
        profiler.enable()
        updater = BatchRegimeUpdater(max_workers=4)

        assets = list(sample_multi_asset_data.keys())

        def update_with_cache(asset, data, pipeline):
            with profiler.time_operation(f'update_{asset}'):
                # Try cache first
                model = cache.get(asset, 3, data, None)
                if model is None:
                    # Simulate training
                    time.sleep(0.01)
                    model = {'regime': 'Bull', 'confidence': 0.8}
                    cache.set(asset, 3, data, None, model)
                return model

        pipeline_dict = {asset: Mock() for asset in assets}

        # Run batch update with caching and profiling
        results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            update_with_cache
        )

        # Verify all components worked
        assert len(results) == len(assets)
        assert cache.cache_size > 0
        assert len(profiler.timings) > 0
