"""
Test fixtures for hidden-regime package.

Provides sample data and test utilities for unit testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


def create_sample_stock_data(
    n_days: int = 100,
    start_date: Optional[datetime] = None,
    price_start: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005,
    add_volume: bool = True,
    add_missing: bool = False,
    missing_pct: float = 0.02,
    add_outliers: bool = False,
    outlier_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Create sample stock data for testing.

    Args:
        n_days: Number of days of data
        start_date: Starting date (defaults to 90 days ago)
        price_start: Starting price
        volatility: Daily volatility (std of returns)
        trend: Daily trend (mean return)
        add_volume: Whether to include volume data
        add_missing: Whether to add missing values
        missing_pct: Percentage of missing values to add
        add_outliers: Whether to add outlier returns
        outlier_pct: Percentage of outlier returns

    Returns:
        DataFrame with columns: date, price, log_return, volume (optional)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days + 10)

    # Generate business days
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Generate log returns
    np.random.seed(42)  # For reproducible tests
    returns = np.random.normal(trend, volatility, n_days)

    # Add outliers if requested
    if add_outliers:
        n_outliers = max(1, int(n_days * outlier_pct))
        outlier_indices = np.random.choice(n_days, n_outliers, replace=False)
        outlier_magnitudes = np.random.choice([-1, 1], n_outliers) * np.random.uniform(
            0.1, 0.2, n_outliers
        )
        returns[outlier_indices] = outlier_magnitudes

    # Calculate prices from returns
    log_prices = np.log(price_start) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Create DataFrame
    data = pd.DataFrame({"date": dates, "price": prices, "log_return": returns})

    # Add volume if requested
    if add_volume:
        # Generate volume with some correlation to absolute returns
        base_volume = 1000000
        volume_multiplier = 1 + 2 * np.abs(returns)
        volumes = (
            base_volume * volume_multiplier * np.random.lognormal(0, 0.3, n_days)
        ).astype(int)
        data["volume"] = volumes

    # Add missing values if requested
    if add_missing and missing_pct > 0:
        n_missing = max(1, int(n_days * missing_pct))
        missing_indices = np.random.choice(n_days, n_missing, replace=False)

        # Randomly choose columns to make missing
        for idx in missing_indices:
            cols_to_missing = np.random.choice(
                ["price", "log_return"], size=1, replace=False
            )
            for col in cols_to_missing:
                data.loc[idx, col] = np.nan

    return data


def create_invalid_stock_data() -> Dict[str, pd.DataFrame]:
    """
    Create various types of invalid stock data for testing error handling.

    Returns:
        Dictionary with different invalid data scenarios
    """
    scenarios = {}

    # Empty DataFrame
    scenarios["empty"] = pd.DataFrame()

    # Missing required columns
    scenarios["missing_price"] = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),
            "volume": np.random.randint(1000, 10000, 10),
        }
    )

    # Negative prices
    base_data = create_sample_stock_data(n_days=20, add_volume=False)
    scenarios["negative_prices"] = base_data.copy()
    scenarios["negative_prices"].loc[5:7, "price"] = [-10, -5, 0]

    # All missing values
    scenarios["all_missing"] = base_data.copy()
    scenarios["all_missing"]["price"] = np.nan
    scenarios["all_missing"]["log_return"] = np.nan

    # Extreme outliers
    scenarios["extreme_outliers"] = base_data.copy()
    scenarios["extreme_outliers"].loc[5, "log_return"] = 2.0  # 200% daily return
    scenarios["extreme_outliers"].loc[10, "log_return"] = -1.5  # -150% daily return

    # Infinite values
    scenarios["infinite_values"] = base_data.copy()
    scenarios["infinite_values"].loc[3, "log_return"] = np.inf
    scenarios["infinite_values"].loc[7, "log_return"] = -np.inf

    # Duplicate dates
    scenarios["duplicate_dates"] = base_data.copy()
    scenarios["duplicate_dates"].loc[5, "date"] = scenarios["duplicate_dates"].loc[
        4, "date"
    ]

    return scenarios


def create_multi_stock_data(
    tickers: list = ["AAPL", "GOOGL", "MSFT"], n_days: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Create sample data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        n_days: Number of days for each stock

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    np.random.seed(42)
    data = {}

    for i, ticker in enumerate(tickers):
        # Vary parameters slightly for each stock
        price_start = 100 + i * 50
        volatility = 0.015 + i * 0.005
        trend = 0.0002 + i * 0.0001

        data[ticker] = create_sample_stock_data(
            n_days=n_days,
            price_start=price_start,
            volatility=volatility,
            trend=trend,
            add_volume=True,
        )

    return data


class MockYFinanceTicker:
    """Mock yfinance Ticker for testing without API calls."""

    def __init__(self, ticker: str, should_fail: bool = False):
        self.ticker = ticker
        self.should_fail = should_fail

    def history(self, start, end, auto_adjust=True, prepost=False):
        """Mock history method."""
        if self.should_fail:
            raise Exception(f"Mock API failure for {self.ticker}")

        # Calculate number of business days
        date_range = pd.bdate_range(start=start, end=end)
        n_days = len(date_range)

        if n_days == 0:
            return pd.DataFrame()  # Empty DataFrame for invalid date ranges

        # Generate mock OHLC data
        np.random.seed(hash(self.ticker) % 2**32)  # Seed based on ticker

        base_price = {"AAPL": 150, "GOOGL": 2500, "MSFT": 300, "SPY": 400}.get(
            self.ticker, 100
        )
        returns = np.random.normal(0.001, 0.02, n_days)

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * np.exp(ret))

        # Create OHLC from close prices
        highs = [p * np.random.uniform(1.0, 1.02) for p in prices]
        lows = [p * np.random.uniform(0.98, 1.0) for p in prices]
        opens = [p * np.random.uniform(0.99, 1.01) for p in prices]
        volumes = [int(np.random.lognormal(15, 0.3)) for _ in prices]

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            },
            index=date_range,
        )

        # Give the index a name so reset_index() creates 'Date' column (like yfinance)
        df.index.name = "Date"

        return df


def get_test_config_variations() -> Dict[str, Dict[str, Any]]:
    """
    Get various configuration variations for testing.

    Returns:
        Dictionary of configuration scenarios
    """
    from hidden_regime.config import DataConfig, ValidationConfig, PreprocessingConfig

    configs = {}

    # Default configuration
    configs["default"] = {
        "data_config": DataConfig(),
        "validation_config": ValidationConfig(),
        "preprocessing_config": PreprocessingConfig(),
    }

    # Strict validation
    configs["strict"] = {
        "data_config": DataConfig(max_missing_data_pct=0.01, min_observations=100),
        "validation_config": ValidationConfig(
            outlier_threshold=2.0, max_daily_return=0.1, max_consecutive_missing=2
        ),
        "preprocessing_config": PreprocessingConfig(),
    }

    # Lenient validation
    configs["lenient"] = {
        "data_config": DataConfig(max_missing_data_pct=0.15, min_observations=10),
        "validation_config": ValidationConfig(
            outlier_threshold=5.0, max_daily_return=1.0, max_consecutive_missing=10
        ),
        "preprocessing_config": PreprocessingConfig(),
    }

    return configs
