"""
Stock Screener Module

High-performance batch processing engine for screening large universes of stocks
using HMM regime detection. Identifies regime changes, anomalous patterns, and
investment opportunities across hundreds of stocks simultaneously.

Key Features:
- Parallel processing for large stock universes (S&P 500, Russell 2000, etc.)
- Regime change detection with confidence scoring
- Multi-criteria filtering (momentum, volatility, technical patterns)
- Performance ranking and signal strength assessment
- Export capabilities for further analysis and reporting
- Real-time monitoring for regime transitions
"""

from .batch import BatchHMMProcessor, ScreeningConfig
from .criteria import (
    RegimeChangeDetector,
    ScreeningCriteria,
    create_momentum_criteria,
    create_regime_change_criteria,
    create_volatility_criteria,
)
from .engine import (
    MarketScreener,
    ScreeningResult,
    create_screening_report,
    screen_stock_universe,
)
from .universes import (
    get_sp500_universe,
    get_russell2000_universe,
    get_custom_universe,
    STOCK_UNIVERSES,
)

__all__ = [
    # Core screening engine
    "MarketScreener",
    "ScreeningResult", 
    "screen_stock_universe",
    "create_screening_report",
    # Batch processing
    "BatchHMMProcessor",
    "ScreeningConfig",
    # Screening criteria
    "ScreeningCriteria",
    "RegimeChangeDetector",
    "create_regime_change_criteria",
    "create_momentum_criteria", 
    "create_volatility_criteria",
    # Stock universes
    "get_sp500_universe",
    "get_russell2000_universe",
    "get_custom_universe",
    "STOCK_UNIVERSES",
]