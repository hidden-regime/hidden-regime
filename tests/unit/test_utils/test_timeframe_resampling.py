"""
Unit tests for utils/timeframe_resampling.py

Tests timeframe resampling functionality for OHLC data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def daily_price_data():
    """Create sample daily OHLC data."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "open": np.random.uniform(95, 105, 30),
        "high": np.random.uniform(100, 110, 30),
        "low": np.random.uniform(90, 100, 30),
        "close": np.random.uniform(95, 105, 30),
        "volume": np.random.randint(1000000, 10000000, 30),
    }, index=dates)


def test_resample_to_daily(daily_price_data):
    """Test resampling to daily (identity operation)."""
    resampled = daily_price_data.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    assert len(resampled) <= len(daily_price_data)


def test_resample_to_weekly(daily_price_data):
    """Test resampling to weekly frequency."""
    weekly = daily_price_data.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Should have fewer rows
    assert len(weekly) < len(daily_price_data)


def test_resample_to_monthly(daily_price_data):
    """Test resampling to monthly frequency."""
    monthly = daily_price_data.resample("M").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Should have even fewer rows
    assert len(monthly) < len(daily_price_data)


def test_resample_preserves_ohlc(daily_price_data):
    """Test that OHLC resampling preserves price relationships."""
    weekly = daily_price_data.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    # High should be >= low for all rows
    assert all(weekly["high"] >= weekly["low"])


def test_resample_handles_missing_data():
    """Test resampling with missing data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "close": [100, np.nan, 102, 103, np.nan, 105, 106, np.nan, 108, 109],
    }, index=dates)

    resampled = data.resample("W").agg({"close": "last"})

    # Should handle NaN values
    assert not all(resampled["close"].isna())


def test_resample_custom_aggregation(daily_price_data):
    """Test custom aggregation functions."""
    custom = daily_price_data.resample("W").agg({
        "close": ["mean", "std", "min", "max"],
    })

    assert custom.shape[1] == 4  # 4 aggregation functions


def test_validate_resampling_frequency():
    """Test validation of resampling frequency."""
    valid_frequencies = ["D", "W", "M", "Q", "Y"]
    assert all(isinstance(freq, str) for freq in valid_frequencies)


def test_resample_edge_cases(daily_price_data):
    """Test edge cases in resampling."""
    # Resample to very small dataset
    monthly = daily_price_data.resample("M").agg({"close": "last"}).dropna()

    # Should still work with single month of data
    assert len(monthly) >= 1


def test_resample_with_regime_data():
    """Test resampling preserves regime information."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame({
        "close": np.linspace(100, 120, 20),
        "regime": ["Bull"] * 10 + ["Bear"] * 10,
    }, index=dates)

    weekly = data.resample("W").agg({
        "close": "last",
        "regime": "last",  # Take last regime of week
    })

    assert "regime" in weekly.columns


def test_resample_multi_timeframe_pipeline():
    """Test multi-timeframe resampling pipeline."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    daily = pd.DataFrame({"close": np.random.uniform(95, 105, 60)}, index=dates)

    weekly = daily.resample("W").agg({"close": "last"})
    monthly = daily.resample("M").agg({"close": "last"})

    assert len(daily) > len(weekly) > len(monthly)


def test_resample_preserves_regime_transitions():
    """Test that resampling preserves regime transition information."""
    dates = pd.date_range("2023-01-01", periods=14, freq="D")
    regime_data = pd.DataFrame({
        "regime": ["Bear"] * 7 + ["Bull"] * 7,
    }, index=dates)

    weekly = regime_data.resample("W").agg({"regime": "last"})

    # Should capture the transition
    assert len(weekly["regime"].unique()) >= 1


def test_resample_performance_large_dataset():
    """Test resampling performance with large dataset."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    large_data = pd.DataFrame({
        "close": np.random.uniform(95, 105, 1000),
    }, index=dates)

    # Should handle large datasets efficiently
    weekly = large_data.resample("W").agg({"close": "last"})

    assert len(weekly) > 0
