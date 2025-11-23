"""
Benchmark script to measure caching performance improvements.

Runs regime detection queries with and without cache to measure latency improvements.
"""

import asyncio
import time
from typing import List, Tuple

from hidden_regime_mcp.tools import detect_regime, get_regime_statistics, get_transition_probabilities
from hidden_regime_mcp.cache import get_cache


async def benchmark_detect_regime(ticker: str = "SPY", n_runs: int = 3) -> Tuple[List[float], float]:
    """
    Benchmark detect_regime function.

    Args:
        ticker: Stock symbol to test
        n_runs: Number of runs for averaging

    Returns:
        Tuple of (list of individual run times, average time, cache stats)
    """
    cache = get_cache()
    cache.clear()  # Start fresh

    times = []

    # First run - cache miss
    start = time.perf_counter()
    result = await detect_regime(ticker, n_states=3)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    first_run = elapsed

    # Subsequent runs - cache hits
    for i in range(n_runs - 1):
        start = time.perf_counter()
        result = await detect_regime(ticker, n_states=3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    cache_stats = cache.get_stats()

    return times, first_run, avg_time, cache_stats


async def benchmark_get_statistics(ticker: str = "SPY", n_runs: int = 3) -> Tuple[List[float], float, float, dict]:
    """Benchmark get_regime_statistics function."""
    cache = get_cache()
    cache.clear()

    times = []

    # First run - cache miss
    start = time.perf_counter()
    result = await get_regime_statistics(ticker, n_states=3)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    first_run = elapsed

    # Subsequent runs - cache hits
    for i in range(n_runs - 1):
        start = time.perf_counter()
        result = await get_regime_statistics(ticker, n_states=3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    cache_stats = cache.get_stats()

    return times, first_run, avg_time, cache_stats


async def benchmark_transitions(ticker: str = "SPY", n_runs: int = 3) -> Tuple[List[float], float, float, dict]:
    """Benchmark get_transition_probabilities function."""
    cache = get_cache()
    cache.clear()

    times = []

    # First run - cache miss
    start = time.perf_counter()
    result = await get_transition_probabilities(ticker, n_states=3)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    first_run = elapsed

    # Subsequent runs - cache hits
    for i in range(n_runs - 1):
        start = time.perf_counter()
        result = await get_transition_probabilities(ticker, n_states=3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    cache_stats = cache.get_stats()

    return times, first_run, avg_time, cache_stats


async def benchmark_multi_ticker(tickers: List[str], n_runs: int = 2) -> None:
    """
    Benchmark with multiple tickers to show cache benefits across portfolio.

    Args:
        tickers: List of stock symbols
        n_runs: Number of runs per ticker
    """
    cache = get_cache()
    cache.clear()

    total_time = 0
    first_run_times = []
    cached_run_times = []

    for ticker in tickers:
        # First run - cache miss
        start = time.perf_counter()
        await detect_regime(ticker, n_states=3)
        elapsed = time.perf_counter() - start
        first_run_times.append(elapsed)
        total_time += elapsed

        # Second run - cache hit
        start = time.perf_counter()
        await detect_regime(ticker, n_states=3)
        elapsed = time.perf_counter() - start
        cached_run_times.append(elapsed)
        total_time += elapsed

    avg_first = sum(first_run_times) / len(first_run_times)
    avg_cached = sum(cached_run_times) / len(cached_run_times)
    speedup = avg_first / avg_cached if avg_cached > 0 else 0

    print("\n" + "=" * 70)
    print("MULTI-TICKER BENCHMARK")
    print("=" * 70)
    print(f"Tickers tested: {', '.join(tickers)}")
    print(f"First run (cache miss) avg: {avg_first:.3f}s")
    print(f"Cached run (cache hit) avg: {avg_cached:.3f}s")
    print(f"Speedup factor: {speedup:.1f}x faster with cache")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Cache stats: {cache.get_cache_info()}")


async def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("HIDDEN REGIME MCP CACHING BENCHMARK")
    print("=" * 70)

    # Benchmark detect_regime
    print("\n1. Benchmarking detect_regime()...")
    print("-" * 70)
    times, first_run, avg_time, stats = await benchmark_detect_regime(n_runs=3)
    speedup = first_run / times[-1] if times[-1] > 0 else 0
    print(f"   First run (cache miss): {first_run:.3f}s")
    print(f"   Avg cached run:         {avg_time:.3f}s")
    print(f"   Individual runs:        {', '.join(f'{t:.3f}s' for t in times)}")
    print(f"   Speedup factor:         {speedup:.1f}x faster with cache")
    print(f"   Cache stats:            {stats}")

    # Benchmark get_regime_statistics
    print("\n2. Benchmarking get_regime_statistics()...")
    print("-" * 70)
    times, first_run, avg_time, stats = await benchmark_get_statistics(n_runs=3)
    speedup = first_run / times[-1] if times[-1] > 0 else 0
    print(f"   First run (cache miss): {first_run:.3f}s")
    print(f"   Avg cached run:         {avg_time:.3f}s")
    print(f"   Individual runs:        {', '.join(f'{t:.3f}s' for t in times)}")
    print(f"   Speedup factor:         {speedup:.1f}x faster with cache")
    print(f"   Cache stats:            {stats}")

    # Benchmark get_transition_probabilities
    print("\n3. Benchmarking get_transition_probabilities()...")
    print("-" * 70)
    times, first_run, avg_time, stats = await benchmark_transitions(n_runs=3)
    speedup = first_run / times[-1] if times[-1] > 0 else 0
    print(f"   First run (cache miss): {first_run:.3f}s")
    print(f"   Avg cached run:         {avg_time:.3f}s")
    print(f"   Individual runs:        {', '.join(f'{t:.3f}s' for t in times)}")
    print(f"   Speedup factor:         {speedup:.1f}x faster with cache")
    print(f"   Cache stats:            {stats}")

    # Multi-ticker benchmark
    await benchmark_multi_ticker(["SPY", "QQQ", "IWM", "EFA", "AGG"], n_runs=2)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
