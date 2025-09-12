"""
Performance and benchmark tests for hidden-regime data pipeline.

Tests performance characteristics, memory usage, and benchmarks to detect
regressions and ensure the pipeline scales appropriately with data size.
"""

import gc
import os
import time
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest

from hidden_regime import (
    DataConfig,
    DataLoader,
    DataPreprocessor,
    DataValidator,
    PreprocessingConfig,
    ValidationConfig,
)
from tests.fixtures.sample_data import MockYFinanceTicker, create_sample_stock_data


class TestDataLoadingPerformance:
    """Test performance characteristics of data loading."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.loader = DataLoader()
        self.test_sizes = [30, 100, 250, 500]  # Different data sizes to test

    def test_loading_time_scales_linearly(self):
        """Test that loading time scales reasonably with data size."""
        timing_results = []

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            for size in self.test_sizes:
                # Create mock with specific size
                mock_ticker = MockYFinanceTicker("PERF_TEST")
                mock_ticker_class.return_value = mock_ticker

                # Clear cache to ensure fresh loads
                self.loader.clear_cache()

                # Time the loading operation
                start_time = time.time()

                # Calculate date range for desired size (business days)
                end_date = "2024-12-31"
                # Approximate business days needed (5 days per week)
                calendar_days = int(size * 1.4)  # Buffer for weekends/holidays
                start_date = f"2024-{12 - calendar_days//30:02d}-01"

                data = self.loader.load_stock_data("PERF_TEST", start_date, end_date)

                end_time = time.time()
                load_time = end_time - start_time

                timing_results.append(
                    {
                        "size": len(data),
                        "time": load_time,
                        "time_per_row": (
                            load_time / len(data) if len(data) > 0 else float("inf")
                        ),
                    }
                )

        # Verify results
        assert len(timing_results) == len(self.test_sizes)

        # Check that larger datasets don't take disproportionately longer
        for i in range(1, len(timing_results)):
            prev_result = timing_results[i - 1]
            curr_result = timing_results[i]

            # Time per row should be relatively stable (within 10x)
            if prev_result["time_per_row"] > 0:
                time_ratio = curr_result["time_per_row"] / prev_result["time_per_row"]
                assert (
                    time_ratio < 10.0
                ), f"Performance degradation detected: {time_ratio:.2f}x slower per row"

    def test_caching_performance_improvement(self):
        """Test that caching provides meaningful performance improvements."""
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("CACHE_TEST")
            mock_ticker_class.return_value = mock_ticker

            # First load (should hit API)
            start_time = time.time()
            data1 = self.loader.load_stock_data(
                "CACHE_TEST", "2024-01-01", "2024-12-31"
            )
            first_load_time = time.time() - start_time

            # Second load (should use cache)
            start_time = time.time()
            data2 = self.loader.load_stock_data(
                "CACHE_TEST", "2024-01-01", "2024-12-31"
            )
            cached_load_time = time.time() - start_time

            # Verify data is identical
            pd.testing.assert_frame_equal(data1, data2)

            # Cache should be significantly faster (at least 2x)
            if first_load_time > 0.001:  # Only test if first load was measurable
                speedup_ratio = first_load_time / cached_load_time
                assert (
                    speedup_ratio > 2.0
                ), f"Cache speedup insufficient: {speedup_ratio:.2f}x"

    def test_multi_stock_loading_efficiency(self):
        """Test efficiency of multi-stock vs individual loading."""
        tickers = ["AAPL", "GOOGL", "MSFT"]

        # Create loader with high rate limit to avoid sleep delays in testing
        fast_config = DataConfig(requests_per_minute=10000)  # Very high rate limit
        fast_loader = DataLoader(fast_config)

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:

            def ticker_side_effect(ticker):
                return MockYFinanceTicker(ticker)

            mock_ticker_class.side_effect = ticker_side_effect

            # Clear cache
            fast_loader.clear_cache()

            # Time individual loading
            start_time = time.time()
            individual_results = {}
            for ticker in tickers:
                individual_results[ticker] = fast_loader.load_stock_data(
                    ticker, "2024-01-01", "2024-06-30"
                )
            individual_time = time.time() - start_time

            # Clear cache again
            fast_loader.clear_cache()
            mock_ticker_class.reset_mock()
            mock_ticker_class.side_effect = ticker_side_effect

            # Time batch loading
            start_time = time.time()
            batch_results = fast_loader.load_multiple_stocks(
                tickers, "2024-01-01", "2024-06-30"
            )
            batch_time = time.time() - start_time

            # Verify results are consistent
            assert len(batch_results) == len(individual_results)
            for ticker in tickers:
                assert ticker in batch_results
                # Data should be very similar (allowing for minor timing differences)
                assert len(batch_results[ticker]) == len(individual_results[ticker])

            # Batch loading should not be significantly slower
            # (may be slower due to rate limiting, but not by much)
            time_ratio = batch_time / individual_time
            assert (
                time_ratio < 13.0 # TODO - fix this, number seems high.
            ), f"Batch loading too slow: {time_ratio:.2f}x individual time"


class TestValidationPerformance:
    """Test performance characteristics of data validation."""

    def setup_method(self):
        """Set up validation performance tests."""
        self.validator = DataValidator()
        self.test_sizes = [50, 200, 500, 1000]

    def test_validation_scaling(self):
        """Test that validation time scales appropriately with data size."""
        timing_results = []

        for size in self.test_sizes:
            # Create test data of specified size
            test_data = create_sample_stock_data(n_days=size, add_volume=True)

            # Time validation
            start_time = time.time()
            result = self.validator.validate_data(test_data, f"TEST_{size}")
            validation_time = time.time() - start_time

            timing_results.append(
                {
                    "size": size,
                    "time": validation_time,
                    "time_per_row": validation_time / size,
                }
            )

            # Verify validation completed successfully
            assert isinstance(result.quality_score, float)
            assert 0.0 <= result.quality_score <= 1.0

        # Check scaling characteristics
        for i in range(1, len(timing_results)):
            prev_result = timing_results[i - 1]
            curr_result = timing_results[i]

            # Validation should scale sub-quadratically (time per row shouldn't increase much)
            if prev_result["time_per_row"] > 0:
                time_ratio = curr_result["time_per_row"] / prev_result["time_per_row"]
                assert (
                    time_ratio < 5.0
                ), f"Validation scaling issue: {time_ratio:.2f}x slower per row"

    def test_outlier_detection_performance(self):
        """Test performance of different outlier detection methods."""
        test_data = create_sample_stock_data(n_days=1000, add_outliers=True)
        methods = ["iqr", "zscore"]

        timing_results = {}

        for method in methods:
            config = ValidationConfig(outlier_method=method)
            validator = DataValidator(config)

            start_time = time.time()
            result = validator.validate_data(test_data, f"OUTLIER_{method.upper()}")
            method_time = time.time() - start_time

            timing_results[method] = {
                "time": method_time,
                "quality_score": result.quality_score,
                "outliers_detected": result.metrics.get("n_outliers", 0),
            }

        # Both methods should complete in reasonable time
        for method, results in timing_results.items():
            assert (
                results["time"] < 5.0
            ), f"{method} method took too long: {results['time']:.2f}s"
            assert isinstance(results["quality_score"], float)

    def test_validation_memory_usage(self):
        """Test memory usage characteristics during validation."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_data = create_sample_stock_data(n_days=2000, add_volume=True)

        # Perform validation
        result = self.validator.validate_data(large_data, "MEMORY_TEST")

        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this size dataset)
        assert (
            memory_increase < 100
        ), f"Excessive memory usage: {memory_increase:.1f}MB increase"

        # Cleanup and verify memory is released
        del large_data
        del result
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_retained = final_memory - initial_memory

        # Should not retain excessive memory
        assert (
            memory_retained < 50
        ), f"Memory leak suspected: {memory_retained:.1f}MB retained"


class TestPreprocessingPerformance:
    """Test performance characteristics of data preprocessing."""

    def setup_method(self):
        """Set up preprocessing performance tests."""
        self.preprocessor = DataPreprocessor()
        self.test_sizes = [100, 300, 600, 1000]

    def test_preprocessing_scaling(self):
        """Test preprocessing performance scaling with data size."""
        timing_results = []

        for size in self.test_sizes:
            # Create test data
            test_data = create_sample_stock_data(
                n_days=size, add_volume=True, add_missing=True, missing_pct=0.02
            )

            # Time preprocessing
            start_time = time.time()
            processed_data = self.preprocessor.process_data(test_data, f"PERF_{size}")
            processing_time = time.time() - start_time

            timing_results.append(
                {
                    "size": size,
                    "time": processing_time,
                    "time_per_row": processing_time / size,
                    "output_size": len(processed_data),
                }
            )

            # Verify processing completed
            assert not processed_data.empty
            assert len(processed_data) >= size - 10  # Some rows may be lost

        # Check scaling
        for i in range(1, len(timing_results)):
            prev_result = timing_results[i - 1]
            curr_result = timing_results[i]

            # Processing should scale reasonably
            if prev_result["time_per_row"] > 0:
                time_ratio = curr_result["time_per_row"] / prev_result["time_per_row"]
                assert (
                    time_ratio < 3.0
                ), f"Preprocessing scaling issue: {time_ratio:.2f}x slower per row"

    def test_feature_engineering_performance(self):
        """Test performance impact of different feature engineering options."""
        test_data = create_sample_stock_data(n_days=500, add_volume=True)

        # Test different configuration combinations
        configs = {
            "minimal": PreprocessingConfig(
                calculate_volatility=False, apply_smoothing=False
            ),
            "volatility": PreprocessingConfig(
                calculate_volatility=True, apply_smoothing=False, volatility_window=20
            ),
            "smoothing": PreprocessingConfig(
                calculate_volatility=False, apply_smoothing=True, smoothing_window=5
            ),
            "full": PreprocessingConfig(
                calculate_volatility=True,
                apply_smoothing=True,
                volatility_window=20,
                smoothing_window=5,
            ),
        }

        timing_results = {}

        for config_name, config in configs.items():
            preprocessor = DataPreprocessor(preprocessing_config=config)

            start_time = time.time()
            processed_data = preprocessor.process_data(
                test_data.copy(), f"FEATURE_{config_name}"
            )
            processing_time = time.time() - start_time

            timing_results[config_name] = {
                "time": processing_time,
                "columns": len(processed_data.columns),
                "rows": len(processed_data),
            }

        # All configurations should complete in reasonable time
        for config_name, results in timing_results.items():
            assert (
                results["time"] < 10.0
            ), f"{config_name} config too slow: {results['time']:.2f}s"

        # More features should result in more columns
        assert timing_results["full"]["columns"] > timing_results["minimal"]["columns"]

        # Feature engineering should not be excessively slow
        if timing_results["minimal"]["time"] > 0:
            overhead_ratio = (
                timing_results["full"]["time"] / timing_results["minimal"]["time"]
            )
            assert (
                overhead_ratio < 5.0
            ), f"Feature engineering overhead too high: {overhead_ratio:.2f}x"

    def test_multi_series_processing_performance(self):
        """Test performance of multi-series processing vs individual processing."""
        n_series = 5
        series_size = 200

        # Create multiple series
        data_dict = {}
        for i in range(n_series):
            ticker = f"STOCK_{i}"
            data_dict[ticker] = create_sample_stock_data(
                n_days=series_size, add_volume=True, price_start=100 + i * 10
            )

        # Time individual processing
        start_time = time.time()
        individual_results = {}
        for ticker, data in data_dict.items():
            individual_results[ticker] = self.preprocessor.process_data(data, ticker)
        individual_time = time.time() - start_time

        # Time batch processing
        start_time = time.time()
        batch_results = self.preprocessor.process_multiple_series(data_dict)
        batch_time = time.time() - start_time

        # Verify results
        assert len(batch_results) == len(individual_results)

        # Batch processing should not be significantly slower
        # (may be slower due to alignment, but should be reasonable)
        if individual_time > 0:
            time_ratio = batch_time / individual_time
            assert (
                time_ratio < 3.0
            ), f"Batch processing too slow: {time_ratio:.2f}x individual time"


class TestEndToEndPerformance:
    """Test end-to-end pipeline performance."""

    def setup_method(self):
        """Set up end-to-end performance tests."""
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()

    def test_complete_pipeline_performance(self):
        """Test performance of complete pipeline workflow."""
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("PIPELINE_PERF")
            mock_ticker_class.return_value = mock_ticker

            # Time complete pipeline
            start_time = time.time()

            # Step 1: Load
            raw_data = self.loader.load_stock_data(
                "PIPELINE_PERF", "2024-01-01", "2024-12-31"
            )
            load_time = time.time()

            # Step 2: Validate
            validation = self.validator.validate_data(raw_data, "PIPELINE_PERF")
            validate_time = time.time()

            # Step 3: Process
            processed_data = self.preprocessor.process_data(raw_data, "PIPELINE_PERF")
            process_time = time.time()

            # Step 4: Re-validate
            final_validation = self.validator.validate_data(
                processed_data, "PIPELINE_PERF"
            )
            final_time = time.time()

            # Calculate stage times
            stage_times = {
                "loading": load_time - start_time,
                "validation": validate_time - load_time,
                "processing": process_time - validate_time,
                "final_validation": final_time - process_time,
                "total": final_time - start_time,
            }

            # Verify all stages completed successfully
            assert not raw_data.empty
            assert isinstance(validation.quality_score, float)
            assert not processed_data.empty
            assert isinstance(final_validation.quality_score, float)

            # Check timing constraints
            assert (
                stage_times["total"] < 30.0
            ), f"Pipeline too slow: {stage_times['total']:.2f}s total"

            # No single stage should dominate excessively
            max_stage_time = max(
                stage_times[stage] for stage in ["loading", "validation", "processing"]
            )
            assert (
                max_stage_time < stage_times["total"] * 0.8
            ), "One stage taking too much time"

    def test_pipeline_memory_efficiency(self):
        """Test memory efficiency of complete pipeline."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("MEMORY_PIPELINE")
            mock_ticker_class.return_value = mock_ticker

            # Run pipeline multiple times to detect memory leaks
            for i in range(5):
                raw_data = self.loader.load_stock_data(
                    f"MEMORY_PIPELINE_{i}", "2024-01-01", "2024-12-31"
                )
                validation = self.validator.validate_data(
                    raw_data, f"MEMORY_PIPELINE_{i}"
                )
                processed_data = self.preprocessor.process_data(
                    raw_data, f"MEMORY_PIPELINE_{i}"
                )
                final_validation = self.validator.validate_data(
                    processed_data, f"MEMORY_PIPELINE_{i}"
                )

                # Force cleanup
                del raw_data, validation, processed_data, final_validation

            # Force garbage collection
            gc.collect()

            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Should not accumulate excessive memory
            assert (
                memory_increase < 200
            ), f"Possible memory leak: {memory_increase:.1f}MB increase"


class BenchmarkResults:
    """Store and compare benchmark results for regression testing."""

    @staticmethod
    def run_standard_benchmarks() -> Dict[str, float]:
        """Run standard benchmark suite and return timing results."""
        results = {}

        # Benchmark 1: Data loading (250 days)
        loader = DataLoader()
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("BENCHMARK")
            mock_ticker_class.return_value = mock_ticker

            start_time = time.time()
            data = loader.load_stock_data("BENCHMARK", "2024-01-01", "2024-12-31")
            results["loading_250_days"] = time.time() - start_time

        # Benchmark 2: Validation (500 rows)
        validator = DataValidator()
        test_data = create_sample_stock_data(n_days=500, add_volume=True)

        start_time = time.time()
        validation_result = validator.validate_data(test_data, "BENCHMARK")
        results["validation_500_rows"] = time.time() - start_time

        # Benchmark 3: Processing (300 rows)
        preprocessor = DataPreprocessor()
        process_data = create_sample_stock_data(n_days=300, add_volume=True)

        start_time = time.time()
        processed = preprocessor.process_data(process_data, "BENCHMARK")
        results["processing_300_rows"] = time.time() - start_time

        # Benchmark 4: Complete pipeline
        start_time = time.time()
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("PIPELINE_BENCHMARK")
            mock_ticker_class.return_value = mock_ticker

            raw = loader.load_stock_data(
                "PIPELINE_BENCHMARK", "2024-01-01", "2024-06-30"
            )
            val1 = validator.validate_data(raw, "PIPELINE_BENCHMARK")
            proc = preprocessor.process_data(raw, "PIPELINE_BENCHMARK")
            val2 = validator.validate_data(proc, "PIPELINE_BENCHMARK")

        results["complete_pipeline"] = time.time() - start_time

        return results


def test_benchmark_regression():
    """Test for performance regression by comparing against baseline."""
    # Run benchmarks
    results = BenchmarkResults.run_standard_benchmarks()

    # Define baseline expectations (these should be updated if architecture changes)
    baselines = {
        "loading_250_days": 2.0,  # 2 seconds max for loading
        "validation_500_rows": 1.0,  # 1 second max for validation
        "processing_300_rows": 2.0,  # 2 seconds max for processing
        "complete_pipeline": 5.0,  # 5 seconds max for complete pipeline
    }

    # Check against baselines
    for benchmark, result_time in results.items():
        baseline = baselines.get(benchmark, float("inf"))
        assert (
            result_time < baseline
        ), f"Performance regression in {benchmark}: {result_time:.2f}s > {baseline:.2f}s baseline"


if __name__ == "__main__":
    # Run with benchmark markers
    pytest.main([__file__, "-v", "-m", "not slow"])
