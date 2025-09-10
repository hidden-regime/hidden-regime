"""
Configuration classes for hidden-regime package.

Provides dataclass-based configuration for data loading,
preprocessing, and validation parameters.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading operations."""

    # Data source settings
    default_source: str = "yfinance"
    use_ohlc_average: bool = True
    include_volume: bool = True

    # Data quality settings
    max_missing_data_pct: float = 0.05  # 5% maximum missing data
    min_observations: int = 30  # Minimum data points required

    # Caching settings
    cache_enabled: bool = True
    cache_expiry_hours: int = 24

    # Rate limiting
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ValidationConfig:
    """Configuration for data validation operations."""

    # Outlier detection
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0  # Standard deviations for zscore
    iqr_multiplier: float = 1.5  # IQR multiplier for outlier detection

    # Price validation
    min_price: float = 0.01  # Minimum valid price
    max_daily_return: float = 0.5  # 50% max daily return (filters splits/errors)

    # Date validation
    min_trading_days_per_month: int = 15  # Minimum trading days expected

    # Missing data handling
    max_consecutive_missing: int = 5  # Max consecutive missing values
    interpolation_method: str = "linear"  # 'linear', 'forward', 'backward'


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing operations."""

    # Return calculation
    return_method: str = "log"  # 'log', 'simple'

    # Smoothing/filtering
    apply_smoothing: bool = False
    smoothing_window: int = 5

    # Feature engineering
    calculate_volatility: bool = True
    volatility_window: int = 20

    # Data alignment
    align_timestamps: bool = True
    fill_method: str = "forward"  # 'forward', 'backward', 'interpolate'
