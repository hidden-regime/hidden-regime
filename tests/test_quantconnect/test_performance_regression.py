"""
Performance regression tests for QuantConnect integration.

These tests ensure that performance optimizations maintain their effectiveness
and that no regressions are introduced in future changes.
"""
import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock

from hidden_regime.quantconnect.data_adapter import QuantConnectDataAdapter
from hidden_regime.quantconnect.performance.caching import (
    RegimeModelCache,
    CachedRegimeDetector
)
from hidden_regime.quantconnect.performance.batch_updates import BatchRegimeUpdater
from hidden_regime.quantconnect.performance.profiling import PerformanceProfiler


# Performance benchmarks (baseline targets)
PERFORMANCE_TARGETS = {
    'cache_hit_rate': 0.70,  # At least 70% cache hit rate after warm-up
    'cache_speedup': 2.0,    # At least 2x faster with cache
    'batch_speedup': 2.0,    # At least 2x faster with batch processing (4 workers)
    'memory_overhead': 0.30,  # Less than 30% memory overhead from caching
}


class TestCachePerformanceRegression:
    """Test cache performance regressions."""

    def test_cache_hit_rate_benchmark(self, performance_test_data, mock_pipeline):
        """Test that cache hit rate meets target."""
        cache = RegimeModelCache(max_cache_size=100)
        detector = CachedRegimeDetector(
            max_cache_size=100,
            retrain_frequency='monthly'
        )

        # Simulate realistic usage pattern
        # - Same data accessed multiple times
        # - Occasional retraining

        for _ in range(10):  # 10 iterations
            detector.detect_regime(
                ticker='SPY',
                n_states=3,
                data=performance_test_data,
                pipeline=mock_pipeline
            )

        stats = detector.model_cache.get_statistics()
        hit_rate = stats.get('hit_rate', 0)

        assert hit_rate >= PERFORMANCE_TARGETS['cache_hit_rate'], \
            f"Cache hit rate {hit_rate:.2%} below target {PERFORMANCE_TARGETS['cache_hit_rate']:.2%}"

    def test_cache_speedup_benchmark(self, performance_test_data):
        """Test that caching provides expected speedup."""
        # Create slow pipeline
        class SlowPipeline:
            def run(self, data):
                time.sleep(0.01)  # Simulate slow operation
                return {'regime': 'Bull', 'confidence': 0.8}

        pipeline = SlowPipeline()

        # Without cache
        start = time.time()
        for _ in range(10):
            pipeline.run(performance_test_data)
        no_cache_time = time.time() - start

        # With cache
        cache = RegimeModelCache(max_cache_size=10)
        start = time.time()
        for _ in range(10):
            model = cache.get('SPY', 3, performance_test_data, None)
            if model is None:
                model = pipeline.run(performance_test_data)
                cache.set('SPY', 3, performance_test_data, None, model)
        cached_time = time.time() - start

        speedup = no_cache_time / cached_time

        assert speedup >= PERFORMANCE_TARGETS['cache_speedup'], \
            f"Cache speedup {speedup:.2f}x below target {PERFORMANCE_TARGETS['cache_speedup']:.2f}x"

    def test_cache_memory_overhead(self, performance_test_data):
        """Test that cache memory overhead is acceptable."""
        import sys

        cache = RegimeModelCache(max_cache_size=100)

        # Measure baseline
        baseline_size = sys.getsizeof(cache.cache)

        # Add 50 items
        for i in range(50):
            data = performance_test_data.iloc[:100 + i]  # Vary data slightly
            cache.set(f'Asset{i}', 3, data, None, {'model': 'mock'})

        # Measure with data
        filled_size = sys.getsizeof(cache.cache)

        # Calculate overhead
        data_size = sys.getsizeof(performance_test_data)
        overhead_ratio = (filled_size - baseline_size) / (50 * data_size)

        assert overhead_ratio <= PERFORMANCE_TARGETS['memory_overhead'], \
            f"Memory overhead {overhead_ratio:.2%} above target {PERFORMANCE_TARGETS['memory_overhead']:.2%}"

    def test_cache_eviction_performance(self):
        """Test that LRU eviction performs well."""
        cache = RegimeModelCache(max_cache_size=100)

        # Fill cache beyond capacity
        start = time.time()
        for i in range(200):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, {'model': i})
        eviction_time = time.time() - start

        # Should complete quickly despite evictions
        assert eviction_time < 1.0, \
            f"Cache eviction took {eviction_time:.2f}s, expected < 1.0s"

        # Should maintain max size
        assert len(cache.cache) <= 100


class TestBatchProcessingPerformanceRegression:
    """Test batch processing performance regressions."""

    def test_batch_speedup_benchmark(self, sample_multi_asset_data):
        """Test that batch processing provides expected speedup."""
        assets = list(sample_multi_asset_data.keys()) * 2  # 8 assets

        def slow_update(asset, data, pipeline):
            time.sleep(0.02)  # Simulate work
            return {'regime': 'Bull'}

        pipeline_dict = {asset: Mock() for asset in assets}

        # Sequential processing
        start = time.time()
        for asset in assets:
            slow_update(asset, sample_multi_asset_data[asset.replace('0', '').replace('1', '')], pipeline_dict[asset])
        sequential_time = time.time() - start

        # Parallel processing with 4 workers
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)
        start = time.time()

        data_dict = {}
        for asset in assets:
            base_asset = asset.replace('0', '').replace('1', '')
            data_dict[asset] = sample_multi_asset_data.get(base_asset, pd.DataFrame())

        updater.batch_update(assets, data_dict, pipeline_dict, slow_update)
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time

        assert speedup >= PERFORMANCE_TARGETS['batch_speedup'], \
            f"Batch speedup {speedup:.2f}x below target {PERFORMANCE_TARGETS['batch_speedup']:.2f}x"

    def test_batch_scaling_efficiency(self):
        """Test that batch processing scales efficiently."""
        def work_function(asset, data, pipeline):
            time.sleep(0.01)
            return {'regime': 'Bull'}

        # Test different asset counts
        results = {}

        for n_assets in [2, 4, 8, 16]:
            assets = [f'Asset{i}' for i in range(n_assets)]
            data_dict = {asset: pd.DataFrame() for asset in assets}
            pipeline_dict = {asset: Mock() for asset in assets}

            updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

            start = time.time()
            updater.batch_update(assets, data_dict, pipeline_dict, work_function)
            elapsed = time.time() - start

            results[n_assets] = elapsed

        # Time should scale sub-linearly (due to parallelization)
        # 16 assets should take less than 8x the time of 2 assets
        assert results[16] < results[2] * 8, \
            f"Batch processing not scaling efficiently: {results}"

    def test_worker_utilization(self):
        """Test that workers are properly utilized."""
        assets = [f'Asset{i}' for i in range(12)]

        execution_times = []

        def track_execution(asset, data, pipeline):
            start = time.time()
            time.sleep(0.02)
            execution_times.append(time.time() - start)
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)
        updater.batch_update(assets, data_dict, pipeline_dict, track_execution)

        # All tasks should complete (12 tasks executed)
        assert len(execution_times) == 12


class TestDataAdapterPerformanceRegression:
    """Test data adapter performance regressions."""

    def test_data_conversion_speed(self):
        """Test that data conversion is fast enough."""
        class MockTradeBar:
            def __init__(self, i):
                self.Time = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
                self.Open = 100 + i * 0.1
                self.High = 101 + i * 0.1
                self.Low = 99 + i * 0.1
                self.Close = 100.5 + i * 0.1
                self.Volume = 1e6

        bars = [MockTradeBar(i) for i in range(1000)]

        adapter = QuantConnectDataAdapter(window_size=1000)

        # Measure conversion time
        start = time.time()
        for bar in bars:
            adapter.add_bar(bar)
        df = adapter.to_dataframe()
        conversion_time = time.time() - start

        # Should convert 1000 bars in under 0.5 seconds
        assert conversion_time < 0.5, \
            f"Data conversion took {conversion_time:.2f}s, expected < 0.5s"

        assert len(df) == 1000

    def test_rolling_window_performance(self):
        """Test rolling window performance."""
        adapter = QuantConnectDataAdapter(window_size=252)

        class MockTradeBar:
            def __init__(self, i):
                self.Time = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
                self.Open = 100.0
                self.High = 101.0
                self.Low = 99.0
                self.Close = 100.5
                self.Volume = 1e6

        # Add many bars (simulating real-time updates)
        start = time.time()
        for i in range(1000):
            adapter.add_bar(MockTradeBar(i))
        elapsed = time.time() - start

        # Should handle 1000 updates quickly
        assert elapsed < 0.2, \
            f"Rolling window updates took {elapsed:.2f}s, expected < 0.2s"


class TestEndToEndPerformanceRegression:
    """Test end-to-end performance regressions."""

    def test_full_pipeline_benchmark(self, performance_test_data, mock_pipeline):
        """Test complete pipeline performance."""
        from hidden_regime.quantconnect.signal_adapter import RegimeSignalAdapter

        profiler = PerformanceProfiler()
        profiler.enable()

        signal_adapter = RegimeSignalAdapter(
            regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
        )

        # Run complete pipeline 100 times
        start = time.time()

        for _ in range(100):
            with profiler.time_operation('full_pipeline'):
                # Data already prepared
                result = mock_pipeline.run(performance_test_data)
                signal = signal_adapter.regime_to_signal(
                    result['regime'],
                    result['confidence']
                )

        total_time = time.time() - start

        # 100 iterations should complete in under 2 seconds with mock pipeline
        assert total_time < 2.0, \
            f"Full pipeline took {total_time:.2f}s for 100 iterations"

        stats = profiler.get_statistics('full_pipeline')
        assert stats['mean_time'] < 0.02  # Average < 20ms per iteration

    def test_multi_asset_portfolio_benchmark(self, sample_multi_asset_data, mock_pipeline):
        """Test multi-asset portfolio performance."""
        from hidden_regime.quantconnect.signal_adapter import (
            RegimeSignalAdapter,
            MultiAssetSignalAdapter
        )

        assets = list(sample_multi_asset_data.keys())

        cache = RegimeModelCache(max_cache_size=50)
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        def update_with_cache(asset, data, pipeline):
            model = cache.get(asset, 3, data, None)
            if model is None:
                model = pipeline.run(data)
                cache.set(asset, 3, data, None, model)
            return model

        pipeline_dict = {asset: mock_pipeline for asset in assets}

        # Benchmark full multi-asset update cycle
        start = time.time()

        for _ in range(10):  # 10 iterations
            # Update all regimes
            results = updater.batch_update(
                assets,
                sample_multi_asset_data,
                pipeline_dict,
                update_with_cache
            )

            # Calculate allocations
            signal_adapter = RegimeSignalAdapter(
                regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
            )

            regime_signals = {}
            for asset, result in results.items():
                regime_signals[asset] = signal_adapter.regime_to_signal(
                    result['regime'],
                    result['confidence']
                )

            multi_adapter = MultiAssetSignalAdapter(
                assets=assets,
                allocation_method='confidence_weighted'
            )
            allocations = multi_adapter.calculate_allocations(regime_signals)

        total_time = time.time() - start

        # 10 iterations of 4-asset portfolio should complete in under 2 seconds
        assert total_time < 2.0, \
            f"Multi-asset portfolio took {total_time:.2f}s for 10 iterations"


class TestMemoryUsageRegression:
    """Test memory usage regressions."""

    def test_cache_memory_leak(self):
        """Test that cache doesn't leak memory."""
        import gc

        cache = RegimeModelCache(max_cache_size=10)

        # Add and evict many items
        for i in range(1000):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, {'model': i})

        # Force garbage collection
        gc.collect()

        # Cache should still be small
        assert len(cache.cache) <= 10

    def test_data_adapter_memory(self):
        """Test that data adapter doesn't grow unbounded."""
        adapter = QuantConnectDataAdapter(window_size=252)

        class MockTradeBar:
            def __init__(self, i):
                self.Time = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
                self.Open = 100.0
                self.High = 101.0
                self.Low = 99.0
                self.Close = 100.5
                self.Volume = 1e6

        # Add many bars
        for i in range(10000):
            adapter.add_bar(MockTradeBar(i))

        # Should only keep window_size bars
        assert len(adapter.bars) == 252


@pytest.mark.benchmark
class TestPerformanceComparison:
    """Comparative performance tests."""

    def test_optimized_vs_unoptimized(self, sample_multi_asset_data, mock_pipeline):
        """Compare optimized vs unoptimized implementation."""
        assets = list(sample_multi_asset_data.keys())

        def slow_update(asset, data, pipeline):
            time.sleep(0.01)
            return pipeline.run(data)

        pipeline_dict = {asset: mock_pipeline for asset in assets}

        # Unoptimized: sequential, no cache
        start = time.time()
        unoptimized_results = {}
        for asset in assets:
            unoptimized_results[asset] = slow_update(
                asset,
                sample_multi_asset_data[asset],
                pipeline_dict[asset]
            )
        unoptimized_time = time.time() - start

        # Optimized: batch + cache
        cache = RegimeModelCache(max_cache_size=10)
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        def fast_update(asset, data, pipeline):
            model = cache.get(asset, 3, data, None)
            if model is None:
                time.sleep(0.01)
                model = pipeline.run(data)
                cache.set(asset, 3, data, None, model)
            return model

        start = time.time()

        # First run - populate cache
        optimized_results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            fast_update
        )

        # Second run - use cache
        optimized_results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            fast_update
        )

        optimized_time = time.time() - start

        # Optimized should be significantly faster
        speedup = unoptimized_time / optimized_time

        assert speedup >= 3.0, \
            f"Optimized implementation only {speedup:.2f}x faster, expected >= 3.0x"

    def test_performance_target_summary(self):
        """Print summary of all performance targets."""
        print("\n" + "="*60)
        print("PERFORMANCE TARGETS SUMMARY")
        print("="*60)

        for metric, target in PERFORMANCE_TARGETS.items():
            if 'rate' in metric:
                print(f"{metric:30s}: >= {target:.0%}")
            elif 'speedup' in metric:
                print(f"{metric:30s}: >= {target:.1f}x")
            else:
                print(f"{metric:30s}: <= {target:.0%}")

        print("="*60)
