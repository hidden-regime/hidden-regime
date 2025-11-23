"""
Timeframe resampling utilities for multi-timeframe analysis.

Provides functions to resample daily data to weekly and monthly timeframes
while maintaining data integrity and preventing lookahead bias.

Key Principles:
- Always use last close (no forward-fill across timeframes)
- Daily frequency is source of truth
- Weekly = Friday closes (or last trading day of week)
- Monthly = Last trading day of month
- No lookahead bias - can only see data available at that point in time
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


def resample_to_weekly(
    daily_data: pd.DataFrame, price_column: str = "close"
) -> pd.DataFrame:
    """
    Resample daily data to weekly frequency (Friday closes).

    Args:
        daily_data: DataFrame with daily OHLC data
        price_column: Column to use for weekly aggregation (default: 'close')

    Returns:
        DataFrame resampled to weekly frequency with last close of week

    Notes:
        - Uses week-end aggregation (Friday close or last trading day)
        - Maintains all columns from input
        - Returns only weeks with actual data (no forward-filled gaps)
    """
    if daily_data is None or len(daily_data) == 0:
        return pd.DataFrame()

    # Ensure datetime index
    df = daily_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise ValueError("daily_data must have datetime index or 'date' column")

    # Resample to week-end (last trading day of week)
    # This uses 'W' which defaults to Sunday, so data is Friday's close
    weekly = df.resample("W").last()

    # Remove NaN rows
    weekly = weekly.dropna(how="all")

    return weekly


def resample_to_monthly(
    daily_data: pd.DataFrame, price_column: str = "close"
) -> pd.DataFrame:
    """
    Resample daily data to monthly frequency (month-end closes).

    Args:
        daily_data: DataFrame with daily OHLC data
        price_column: Column to use for monthly aggregation (default: 'close')

    Returns:
        DataFrame resampled to monthly frequency with last close of month

    Notes:
        - Uses month-end aggregation (last trading day of month)
        - Maintains all columns from input
        - Returns only months with actual data
    """
    if daily_data is None or len(daily_data) == 0:
        return pd.DataFrame()

    # Ensure datetime index
    df = daily_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise ValueError("daily_data must have datetime index or 'date' column")

    # Resample to month-end
    monthly = df.resample("M").last()

    # Remove NaN rows
    monthly = monthly.dropna(how="all")

    return monthly


def get_observation_at_timeframe(
    daily_observations: pd.Series, timeframe: str
) -> pd.Series:
    """
    Aggregate daily observations to specified timeframe.

    Args:
        daily_observations: Series of daily log returns or other observations
        timeframe: 'daily', 'weekly', or 'monthly'

    Returns:
        Series of observations at specified timeframe

    Notes:
        - For returns: uses last close (daily observation = daily return)
        - For other metrics: aggregates using last value of period
    """
    if timeframe == "daily":
        return daily_observations.copy()

    # Create temporary dataframe with observations
    df = pd.DataFrame({"observation": daily_observations})

    if timeframe == "weekly":
        return df.resample("W").last()["observation"].dropna()
    elif timeframe == "monthly":
        return df.resample("M").last()["observation"].dropna()
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")


def align_predictions_to_daily(
    timeframe_predictions: pd.Series,
    daily_index: pd.DatetimeIndex,
    method: str = "ffill",
) -> pd.Series:
    """
    Align lower-frequency predictions back to daily index.

    Args:
        timeframe_predictions: Predictions from weekly/monthly model
        daily_index: Target daily datetime index
        method: 'ffill' (forward fill) or 'bfill' (backward fill)

    Returns:
        Series with predictions aligned to daily index

    Notes:
        - Forward fill is recommended (no lookahead bias)
        - Backward fill would create lookahead - use carefully
    """
    if len(timeframe_predictions) == 0:
        return pd.Series(np.nan, index=daily_index)

    # Create series aligned to timeframe index
    # Then reindex to daily with specified method
    aligned = timeframe_predictions.reindex(daily_index, method=method)

    return aligned


def validate_timeframe_data(
    daily_data: pd.DataFrame,
    min_daily_observations: int = 100,
    min_weekly_observations: int = 15,
    min_monthly_observations: int = 5,
) -> Tuple[bool, str]:
    """
    Validate that data has sufficient observations for all timeframes.

    Args:
        daily_data: Daily data to validate
        min_daily_observations: Minimum daily observations required
        min_weekly_observations: Minimum weekly observations required
        min_monthly_observations: Minimum monthly observations required

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if daily_data is None or len(daily_data) == 0:
        return False, "No data provided"

    n_daily = len(daily_data)
    if n_daily < min_daily_observations:
        return (
            False,
            f"Insufficient daily data: {n_daily} < {min_daily_observations}",
        )

    # Estimate weekly and monthly observations
    # Rough: 5 trading days per week, 21 per month
    estimated_weekly = n_daily // 5
    estimated_monthly = n_daily // 21

    if estimated_weekly < min_weekly_observations:
        return (
            False,
            f"Insufficient data for weekly analysis: ~{estimated_weekly} weeks < {min_weekly_observations}",
        )

    if estimated_monthly < min_monthly_observations:
        return (
            False,
            f"Insufficient data for monthly analysis: ~{estimated_monthly} months < {min_monthly_observations}",
        )

    return True, "Data validation passed"


def get_timeframe_info(daily_data: pd.DataFrame) -> dict:
    """
    Get information about data coverage across timeframes.

    Args:
        daily_data: Daily data

    Returns:
        Dict with observation counts and date ranges for each timeframe
    """
    if daily_data is None or len(daily_data) == 0:
        return {
            "daily": {"count": 0, "start": None, "end": None},
            "weekly": {"count": 0, "start": None, "end": None},
            "monthly": {"count": 0, "start": None, "end": None},
        }

    # Ensure datetime index
    df = daily_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")

    daily_info = {
        "count": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
    }

    weekly = df.resample("W").last()
    weekly_info = {
        "count": len(weekly.dropna()),
        "start": weekly.index.min(),
        "end": weekly.index.max(),
    }

    monthly = df.resample("M").last()
    monthly_info = {
        "count": len(monthly.dropna()),
        "start": monthly.index.min(),
        "end": monthly.index.max(),
    }

    return {
        "daily": daily_info,
        "weekly": weekly_info,
        "monthly": monthly_info,
    }
