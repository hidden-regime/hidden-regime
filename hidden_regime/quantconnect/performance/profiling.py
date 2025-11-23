"""
Performance profiling utilities for regime detection.

This module provides tools to measure and analyze performance bottlenecks
in regime detection and backtesting.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import statistics


class PerformanceProfiler:
    """
    Performance profiler for regime detection operations.

    Tracks timing of different operations to identify bottlenecks.
    """

    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.enabled = True

    def time_operation(self, operation_name: str) -> Callable:
        """
        Decorator to time an operation.

        Args:
            operation_name: Name of operation being timed

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.timings[operation_name].append(elapsed)
                    self.call_counts[operation_name] += 1

            return wrapper
        return decorator

    def time_block(self, block_name: str):
        """
        Context manager to time a code block.

        Args:
            block_name: Name of code block

        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.time_block("data_loading"):
            ...     data = load_data()
        """
        return TimingContext(self, block_name)

    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get timing statistics.

        Args:
            operation_name: Specific operation (None = all)

        Returns:
            Dictionary with timing stats
        """
        if operation_name:
            timings = self.timings.get(operation_name, [])
            if not timings:
                return {}

            return {
                "operation": operation_name,
                "call_count": self.call_counts[operation_name],
                "total_time": sum(timings),
                "mean_time": statistics.mean(timings),
                "median_time": statistics.median(timings),
                "min_time": min(timings),
                "max_time": max(timings),
                "std_dev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            }
        else:
            # Return stats for all operations
            return {
                op: self.get_stats(op)
                for op in self.timings.keys()
            }

    def get_summary(self) -> str:
        """
        Get human-readable summary of timings.

        Returns:
            Formatted string with timing summary
        """
        if not self.timings:
            return "No profiling data collected"

        lines = ["Performance Profile:", "=" * 60]

        # Sort operations by total time (descending)
        ops_by_time = sorted(
            self.timings.keys(),
            key=lambda op: sum(self.timings[op]),
            reverse=True,
        )

        for op in ops_by_time:
            stats = self.get_stats(op)
            lines.append(f"\n{op}:")
            lines.append(f"  Calls: {stats['call_count']}")
            lines.append(f"  Total: {stats['total_time']:.3f}s")
            lines.append(f"  Mean:  {stats['mean_time']:.3f}s")
            lines.append(f"  Min:   {stats['min_time']:.3f}s")
            lines.append(f"  Max:   {stats['max_time']:.3f}s")

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()

    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False

    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, profiler: PerformanceProfiler, block_name: str):
        """
        Initialize timing context.

        Args:
            profiler: Parent profiler instance
            block_name: Name of code block
        """
        self.profiler = profiler
        self.block_name = block_name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        if self.profiler.enabled:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record."""
        if self.profiler.enabled and self.start_time:
            elapsed = time.time() - self.start_time
            self.profiler.timings[self.block_name].append(elapsed)
            self.profiler.call_counts[self.block_name] += 1


def profile_regime_update(func: Callable) -> Callable:
    """
    Decorator to profile regime update operations.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with profiling
    """
    profiler = PerformanceProfiler()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with profiler.time_block(f"{func.__name__}"):
            result = func(*args, **kwargs)

        # Print stats every 100 calls
        if profiler.call_counts[func.__name__] % 100 == 0:
            print(profiler.get_summary())

        return result

    # Attach profiler to function for access
    wrapper.profiler = profiler
    return wrapper


class RegimeDetectionBenchmark:
    """
    Benchmark suite for regime detection performance.

    Measures performance across different scenarios and configurations.
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: Dict[str, Dict[str, Any]] = {}

    def benchmark_scenario(
        self,
        scenario_name: str,
        setup_func: Callable,
        test_func: Callable,
        iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark a specific scenario.

        Args:
            scenario_name: Name of scenario
            setup_func: Function to set up test
            test_func: Function to benchmark
            iterations: Number of iterations

        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking: {scenario_name}")

        # Setup
        context = setup_func()

        # Warm-up run
        test_func(context)

        # Benchmark runs
        timings = []
        for i in range(iterations):
            start = time.time()
            test_func(context)
            elapsed = time.time() - start
            timings.append(elapsed)

        # Calculate statistics
        results = {
            "scenario": scenario_name,
            "iterations": iterations,
            "mean_time": statistics.mean(timings),
            "median_time": statistics.median(timings),
            "min_time": min(timings),
            "max_time": max(timings),
            "total_time": sum(timings),
            "std_dev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        }

        self.results[scenario_name] = results
        return results

    def print_results(self) -> None:
        """Print benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        for scenario, results in self.results.items():
            print(f"\n{scenario}:")
            print(f"  Iterations: {results['iterations']}")
            print(f"  Mean time:  {results['mean_time']*1000:.2f}ms")
            print(f"  Median:     {results['median_time']*1000:.2f}ms")
            print(f"  Min:        {results['min_time']*1000:.2f}ms")
            print(f"  Max:        {results['max_time']*1000:.2f}ms")
            print(f"  Std dev:    {results['std_dev']*1000:.2f}ms")

        print("=" * 70 + "\n")

    def compare_scenarios(self, baseline: str, comparison: str) -> None:
        """
        Compare two benchmark scenarios.

        Args:
            baseline: Baseline scenario name
            comparison: Comparison scenario name
        """
        if baseline not in self.results or comparison not in self.results:
            print("Both scenarios must be benchmarked first")
            return

        base_time = self.results[baseline]["mean_time"]
        comp_time = self.results[comparison]["mean_time"]

        speedup = base_time / comp_time if comp_time > 0 else 0
        percent_change = ((comp_time - base_time) / base_time) * 100

        print(f"\nComparison: {baseline} vs {comparison}")
        print(f"  Baseline:   {base_time*1000:.2f}ms")
        print(f"  Comparison: {comp_time*1000:.2f}ms")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Change:     {percent_change:+.1f}%")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_global_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler
