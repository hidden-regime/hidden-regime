"""
Technical Indicators Integration Module

Provides seamless integration with the `ta` library for computing traditional
technical analysis indicators. Enables direct comparison between HMM regime
detection and conventional market indicators like MACD, RSI, Bollinger Bands, etc.

Key Features:
- Unified interface for 50+ technical indicators via `ta` library
- HMM-compatible data structures and timing alignment
- Comparative analysis framework for regime vs indicator signals
- Specialized indicator combinations for regime validation
"""

from .calculator import (
    IndicatorCalculator,
    calculate_all_indicators,
    calculate_momentum_indicators,
    calculate_trend_indicators,
    calculate_volatility_indicators,
    calculate_volume_indicators,
)
from .comparison import (
    compare_hmm_vs_indicators,
    generate_indicator_comparison_report,
    validate_regime_with_indicators,
)
from .signals import (
    IndicatorSignalGenerator,
    combine_indicator_signals,
    generate_composite_signal,
)

__all__ = [
    # Core calculator
    "IndicatorCalculator",
    "calculate_all_indicators",
    "calculate_momentum_indicators", 
    "calculate_trend_indicators",
    "calculate_volatility_indicators",
    "calculate_volume_indicators",
    # Comparison framework
    "compare_hmm_vs_indicators",
    "generate_indicator_comparison_report",
    "validate_regime_with_indicators",
    # Signal generation
    "IndicatorSignalGenerator",
    "combine_indicator_signals",
    "generate_composite_signal",
]