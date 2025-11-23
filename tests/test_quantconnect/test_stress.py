"""
Stress tests for QuantConnect integration.

Tests system behavior under:
- High concurrency
- Large data volumes
- Resource constraints
- Edge cases and boundary conditions
"""
import pytest
import time
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

from hidden_regime.quantconnect.performance.caching import RegimeModelCache
from hidden_regime.quantconnect.performance.batch_updates import BatchRegimeUpdater
from hidden_regime.quantconnect.data_adapter import QuantConnectDataAdapter


class TestConcurrencyStress:
    """Stress tests for concurrent operations."""

    def test_concurrent_cache_access(self):
        """Test cache under heavy concurrent access."""
        cache = RegimeModelCache(max_cache_size=100)

        # Pre-populate cache
        for i in range(50):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, {'model': i})

        # Concurrent reads and writes
        def concurrent_operation(thread_id):
            results = []
            for i in range(100):
                asset_id = i % 50
                data = pd.DataFrame({'Close': [100 + asset_id]})

                # Mix of reads and writes
                if i % 3 == 0:
                    # Write
                    cache.set(f'Asset{asset_id}', 3, data, None, {'model': asset_id})
                else:
                    # Read
                    result = cache.get(f'Asset{asset_id}', 3, data, None)
                    results.append(result)

            return results

        # Run with many threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_operation, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # Should complete without errors
        assert len(results) == 10

        # Cache should still be valid
        stats = cache.get_statistics()
        assert stats['cache_size'] <= 100

    def test_concurrent_batch_updates(self, sample_multi_asset_data):
        """Test batch updater under concurrent stress."""
        assets = list(sample_multi_asset_data.keys())

        call_count = {'count': 0}
        lock = threading.Lock()

        def thread_safe_update(asset, data, pipeline):
            with lock:
                call_count['count'] += 1
            time.sleep(0.01)  # Simulate work
            return {'regime': 'Bull', 'asset': asset}

        pipeline_dict = {asset: Mock() for asset in assets}

        # Run multiple batch updates concurrently
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        def run_batch():
            return updater.batch_update(
                assets,
                sample_multi_asset_data,
                pipeline_dict,
                thread_safe_update
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_batch) for _ in range(5)]
            results = [f.result() for f in as_completed(futures)]

        # All batches should complete
        assert len(results) == 5

        # All assets should be processed in each batch
        for result in results:
            assert len(result) == len(assets)

    def test_race_condition_prevention(self):
        """Test that race conditions are prevented in cache."""
        cache = RegimeModelCache(max_cache_size=10)

        data = pd.DataFrame({'Close': [100, 101, 102]})

        results = []

        def racing_operation(thread_id):
            # Try to set the same key concurrently
            cache.set('SPY', 3, data, None, {'model': thread_id})
            time.sleep(0.001)  # Small delay
            result = cache.get('SPY', 3, data, None)
            results.append(result)

        threads = []
        for i in range(10):
            t = threading.Thread(target=racing_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should complete
        assert len(results) == 10

        # Cache should still be consistent
        final_value = cache.get('SPY', 3, data, None)
        assert final_value is not None

    def test_deadlock_prevention(self):
        """Test that system doesn't deadlock under stress."""
        cache = RegimeModelCache(max_cache_size=100)

        def complex_operation(thread_id):
            for i in range(50):
                data = pd.DataFrame({'Close': [100 + i]})

                # Interleave gets and sets
                if i % 2 == 0:
                    cache.set(f'Asset{i}', 3, data, None, {'model': i})
                else:
                    cache.get(f'Asset{i-1}', 3, data, None)

            return thread_id

        # Run with timeout to detect deadlocks
        start = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(complex_operation, i) for i in range(10)]

            # Wait with timeout
            for future in as_completed(futures, timeout=5.0):
                result = future.result()

        elapsed = time.time() - start

        # Should complete quickly without deadlock
        assert elapsed < 5.0


class TestHighVolumeStress:
    """Stress tests for high data volumes."""

    def test_large_dataset_handling(self):
        """Test handling of very large datasets."""
        # Create 10 years of daily data
        dates = pd.date_range(start='2010-01-01', end='2020-12-31', freq='D')

        large_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, len(dates)),
            'High': np.random.uniform(90, 110, len(dates)),
            'Low': np.random.uniform(90, 110, len(dates)),
            'Close': np.random.uniform(90, 110, len(dates)),
            'Volume': np.random.uniform(1e6, 5e6, len(dates))
        }, index=dates)

        adapter = QuantConnectDataAdapter(window_size=len(dates))

        class MockTradeBar:
            def __init__(self, time, row):
                self.Time = time
                self.Open = row['Open']
                self.High = row['High']
                self.Low = row['Low']
                self.Close = row['Close']
                self.Volume = row['Volume']

        # Add all bars
        start = time.time()
        for idx, row in large_data.iterrows():
            adapter.add_bar(MockTradeBar(idx, row))

        df = adapter.to_dataframe()
        elapsed = time.time() - start

        # Should handle large dataset
        assert len(df) == len(dates)

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Large dataset processing took {elapsed:.2f}s"

    def test_many_assets_stress(self):
        """Test handling many assets simultaneously."""
        n_assets = 100
        assets = [f'Asset{i}' for i in range(n_assets)]

        # Generate data for all assets
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

        data_dict = {}
        for asset in assets:
            np.random.seed(hash(asset) % 2**32)  # Reproducible per asset
            data_dict[asset] = pd.DataFrame({
                'Close': np.random.uniform(90, 110, len(dates)),
                'Volume': np.random.uniform(1e6, 5e6, len(dates))
            }, index=dates)

        # Process all assets
        def process_asset(asset, data, pipeline):
            return {'regime': 'Bull', 'confidence': 0.8}

        pipeline_dict = {asset: Mock() for asset in assets}

        updater = BatchRegimeUpdater(max_workers=8, use_parallel=True)

        start = time.time()
        results = updater.batch_update(
            assets,
            data_dict,
            pipeline_dict,
            process_asset
        )
        elapsed = time.time() - start

        # Should process all assets
        assert len(results) == n_assets

        # Should complete in reasonable time
        assert elapsed < 10.0, f"Processing {n_assets} assets took {elapsed:.2f}s"

    def test_rapid_updates_stress(self):
        """Test system under rapid sequential updates."""
        cache = RegimeModelCache(max_cache_size=1000)

        # Simulate rapid trading updates
        n_updates = 10000

        start = time.time()

        for i in range(n_updates):
            asset = f'Asset{i % 100}'  # Cycle through 100 assets
            data = pd.DataFrame({'Close': [100 + (i % 10)]})

            # Rapid cache operations
            model = cache.get(asset, 3, data, None)
            if model is None:
                cache.set(asset, 3, data, None, {'model': i})

        elapsed = time.time() - start

        # Should handle rapid updates
        assert elapsed < 2.0, f"Rapid updates took {elapsed:.2f}s"

        stats = cache.get_statistics()
        assert stats['hit_count'] > 0  # Should have some cache hits


class TestResourceConstraintStress:
    """Stress tests under resource constraints."""

    def test_limited_cache_size_stress(self):
        """Test cache under severe size constraints."""
        # Very small cache
        cache = RegimeModelCache(max_cache_size=5)

        # Access many items
        for i in range(100):
            data = pd.DataFrame({'Close': [100 + i]})
            cache.set(f'Asset{i}', 3, data, None, {'model': i})

        # Cache should still work correctly
        assert len(cache.cache) <= 5

        # Recent items should be cached
        recent_data = pd.DataFrame({'Close': [199]})
        recent = cache.get('Asset99', 3, recent_data, None)
        assert recent is not None

    def test_limited_workers_stress(self):
        """Test batch processing with limited workers."""
        assets = [f'Asset{i}' for i in range(20)]

        def slow_work(asset, data, pipeline):
            time.sleep(0.02)
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        # Only 1 worker - should still work
        updater = BatchRegimeUpdater(max_workers=1, use_parallel=True)

        start = time.time()
        results = updater.batch_update(assets, data_dict, pipeline_dict, slow_work)
        elapsed = time.time() - start

        # Should complete all tasks
        assert len(results) == 20

        # Should take approximately sequential time
        assert elapsed >= 0.4  # 20 * 0.02 = 0.4s

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create many large datasets
        large_datasets = []

        for i in range(10):
            dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
            large_datasets.append(pd.DataFrame({
                'Open': np.random.uniform(90, 110, len(dates)),
                'High': np.random.uniform(90, 110, len(dates)),
                'Low': np.random.uniform(90, 110, len(dates)),
                'Close': np.random.uniform(90, 110, len(dates)),
                'Volume': np.random.uniform(1e6, 5e6, len(dates))
            }, index=dates))

        # Should be able to create datasets
        assert len(large_datasets) == 10

        # Test with cache
        cache = RegimeModelCache(max_cache_size=5)  # Small cache

        for i, data in enumerate(large_datasets):
            cache.set(f'Asset{i}', 3, data, None, {'model': i})

        # Cache should handle gracefully
        assert len(cache.cache) <= 5


class TestEdgeCaseStress:
    """Stress tests for edge cases."""

    def test_zero_data_handling(self):
        """Test handling of zero-length data."""
        adapter = QuantConnectDataAdapter(window_size=252)

        df = adapter.to_dataframe()

        # Should return empty DataFrame without errors
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_worker_edge_case(self):
        """Test batch processing with exactly 1 worker."""
        assets = ['SPY', 'QQQ']

        def work(asset, data, pipeline):
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        updater = BatchRegimeUpdater(max_workers=1, use_parallel=False)

        results = updater.batch_update(assets, data_dict, pipeline_dict, work)

        assert len(results) == 2

    def test_extreme_worker_count(self):
        """Test with very high worker count."""
        assets = ['Asset1', 'Asset2', 'Asset3', 'Asset4']

        def work(asset, data, pipeline):
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        # More workers than tasks
        updater = BatchRegimeUpdater(max_workers=100, use_parallel=True)

        results = updater.batch_update(assets, data_dict, pipeline_dict, work)

        # Should still work correctly
        assert len(results) == 4

    def test_rapid_cache_clear_stress(self):
        """Test rapid cache clearing."""
        cache = RegimeModelCache(max_cache_size=100)

        for iteration in range(10):
            # Fill cache
            for i in range(50):
                data = pd.DataFrame({'Close': [100 + i]})
                cache.set(f'Asset{i}', 3, data, None, {'model': i})

            # Clear cache
            cache.clear()

            # Should be empty
            assert len(cache.cache) == 0


class TestFailureRecoveryStress:
    """Stress tests for failure recovery."""

    def test_partial_failure_recovery(self):
        """Test recovery when some operations fail."""
        assets = [f'Asset{i}' for i in range(20)]

        def failing_work(asset, data, pipeline):
            # Fail for some assets
            if hash(asset) % 5 == 0:
                raise ValueError(f"Simulated failure for {asset}")
            return {'regime': 'Bull'}

        data_dict = {asset: pd.DataFrame() for asset in assets}
        pipeline_dict = {asset: Mock() for asset in assets}

        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        results = updater.batch_update(
            assets,
            data_dict,
            pipeline_dict,
            failing_work,
            continue_on_error=True
        )

        # Should have results for non-failing assets
        assert len(results) > 0

    def test_timeout_handling_stress(self):
        """Test handling of operations that timeout."""
        cache = RegimeModelCache(max_cache_size=10)

        def slow_operation():
            # Simulate very slow operation
            for i in range(10):
                data = pd.DataFrame({'Close': [100 + i]})
                cache.set(f'Asset{i}', 3, data, None, {'model': i})
                time.sleep(0.05)  # Slow

        import threading

        thread = threading.Thread(target=slow_operation)
        thread.start()
        thread.join(timeout=1.0)  # Wait max 1 second

        # Thread may still be running, but system should be responsive
        # Can still access cache from main thread
        data = pd.DataFrame({'Close': [100]})
        result = cache.get('Asset0', 3, data, None)

        # Should not deadlock
        assert True


class TestLongRunningStress:
    """Stress tests for long-running scenarios."""

    def test_sustained_load(self):
        """Test system under sustained load."""
        cache = RegimeModelCache(max_cache_size=50)

        # Simulate 1 hour of trading (1 update per second)
        n_updates = 60  # Reduced for testing speed

        assets = ['SPY', 'QQQ', 'TLT', 'GLD']

        start = time.time()

        for i in range(n_updates):
            asset = assets[i % len(assets)]
            data = pd.DataFrame({'Close': [100 + (i % 100)]})

            # Typical operation cycle
            model = cache.get(asset, 3, data, None)
            if model is None or i % 10 == 0:  # Retrain periodically
                model = {'regime': 'Bull', 'iteration': i}
                cache.set(asset, 3, data, None, model)

        elapsed = time.time() - start

        # Should maintain performance
        avg_time_per_update = elapsed / n_updates
        assert avg_time_per_update < 0.01, \
            f"Average time per update: {avg_time_per_update:.4f}s"

        # Cache should be healthy
        stats = cache.get_statistics()
        assert stats['cache_size'] > 0

    def test_cache_statistics_accuracy(self):
        """Test cache statistics remain accurate over time."""
        cache = RegimeModelCache(max_cache_size=10)

        # Many operations
        for i in range(1000):
            data = pd.DataFrame({'Close': [100 + (i % 50)]})
            asset = f'Asset{i % 10}'

            result = cache.get(asset, 3, data, None)
            if result is None:
                cache.set(asset, 3, data, None, {'model': i})

        stats = cache.get_statistics()

        # Statistics should be consistent
        assert stats['hit_count'] + stats['miss_count'] == 1000
        assert 0 <= stats['hit_rate'] <= 1.0
        assert stats['cache_size'] <= 10


@pytest.mark.stress
class TestExtremeStress:
    """Extreme stress tests (may take longer to run)."""

    @pytest.mark.slow
    def test_marathon_cache_operations(self):
        """Test cache with many operations over extended time."""
        cache = RegimeModelCache(max_cache_size=100)

        # 100,000 operations
        for i in range(100000):
            data = pd.DataFrame({'Close': [100 + (i % 1000)]})
            asset = f'Asset{i % 100}'

            if i % 3 == 0:
                cache.set(asset, 3, data, None, {'model': i})
            else:
                cache.get(asset, 3, data, None)

        # Should still be functional
        stats = cache.get_statistics()
        assert stats['cache_size'] <= 100
        assert stats['hit_count'] > 0

    @pytest.mark.slow
    def test_extreme_concurrency(self):
        """Test with extreme concurrency."""
        cache = RegimeModelCache(max_cache_size=200)

        def concurrent_ops(thread_id):
            for i in range(100):
                data = pd.DataFrame({'Close': [100 + i]})
                asset = f'Asset{thread_id}_{i % 10}'

                if i % 2 == 0:
                    cache.set(asset, 3, data, None, {'model': i})
                else:
                    cache.get(asset, 3, data, None)

        # 50 concurrent threads
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(concurrent_ops, i) for i in range(50)]

            # Wait for all
            for future in as_completed(futures):
                future.result()

        # Should complete without errors
        assert True
