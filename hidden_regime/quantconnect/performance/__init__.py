"""
Performance optimization utilities for QuantConnect integration.

This module provides tools for optimizing regime detection performance:
- Caching trained models
- Batch regime updates
- Performance profiling
- Memory optimization
"""

from .caching import RegimeModelCache, CachedRegimeDetector
from .profiling import PerformanceProfiler, profile_regime_update
from .batch_updates import BatchRegimeUpdater

__all__ = [
    "RegimeModelCache",
    "CachedRegimeDetector",
    "PerformanceProfiler",
    "profile_regime_update",
    "BatchRegimeUpdater",
]
