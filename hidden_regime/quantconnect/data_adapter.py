"""
Data adapters for converting QuantConnect data to hidden-regime format.

This module provides adapters to convert QuantConnect's RollingWindow and TradeBar
objects into pandas DataFrames that hidden-regime expects.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


class QuantConnectDataAdapter:
    """
    Base adapter for converting QuantConnect data to hidden-regime format.

    This adapter handles the conversion between QuantConnect's data structures
    (TradeBar, QuoteBar, etc.) and pandas DataFrames expected by hidden-regime.
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize data adapter.

        Args:
            lookback_days: Number of historical days to maintain
        """
        self.lookback_days = lookback_days
        self._data_buffer: List[Dict[str, Any]] = []

    def add_bar(
        self,
        time: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """
        Add a price bar to the buffer.

        Args:
            time: Timestamp of the bar
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume
        """
        bar_dict = {
            "Date": time,
            "Open": float(open_price),
            "High": float(high),
            "Low": float(low),
            "Close": float(close),
            "Volume": float(volume),
        }
        self._data_buffer.append(bar_dict)

        # Keep only lookback_days of data
        if len(self._data_buffer) > self.lookback_days:
            self._data_buffer = self._data_buffer[-self.lookback_days :]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert buffered data to pandas DataFrame.

        Returns:
            DataFrame with OHLCV data indexed by date

        Raises:
            ValueError: If insufficient data is available
        """
        if len(self._data_buffer) < 2:
            raise ValueError(
                f"Insufficient data: need at least 2 bars, have {len(self._data_buffer)}"
            )

        df = pd.DataFrame(self._data_buffer)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Ensure all columns are numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove any NaN rows
        df.dropna(inplace=True)

        return df

    def is_ready(self, min_bars: int = 30) -> bool:
        """
        Check if adapter has sufficient data for regime detection.

        Args:
            min_bars: Minimum number of bars required

        Returns:
            True if sufficient data is available
        """
        return len(self._data_buffer) >= min_bars

    def clear(self) -> None:
        """Clear the data buffer."""
        self._data_buffer.clear()

    def get_latest_price(self) -> Optional[float]:
        """
        Get the most recent closing price.

        Returns:
            Latest close price or None if no data
        """
        if not self._data_buffer:
            return None
        return self._data_buffer[-1]["Close"]

    def __len__(self) -> int:
        """Return number of bars in buffer."""
        return len(self._data_buffer)


class RollingWindowDataAdapter:
    """
    Adapter specifically for QuantConnect's RollingWindow data structure.

    This adapter converts QC's RollingWindow[TradeBar] directly to DataFrame,
    which is more efficient when using QC's native data structures.

    Note: This class provides a mock interface when QC libraries aren't available.
    In actual QC environment, it will work with real RollingWindow objects.
    """

    def __init__(self, window_size: int = 252):
        """
        Initialize rolling window adapter.

        Args:
            window_size: Size of the rolling window (number of bars)
        """
        self.window_size = window_size
        # In real QC environment, this would be: RollingWindow[TradeBar](window_size)
        self._window_data: List[Dict[str, Any]] = []

    def update(self, bar: Any) -> None:
        """
        Update the rolling window with a new bar.

        Args:
            bar: TradeBar object from QuantConnect (or dict for testing)
        """
        # Handle both QC TradeBar objects and dict for testing
        if hasattr(bar, "Time"):
            # Real QC TradeBar object
            bar_dict = {
                "Date": bar.Time,
                "Open": float(bar.Open),
                "High": float(bar.High),
                "Low": float(bar.Low),
                "Close": float(bar.Close),
                "Volume": float(bar.Volume),
            }
        else:
            # Dict or mock object
            bar_dict = {
                "Date": bar.get("Time", bar.get("Date")),
                "Open": float(bar.get("Open", 0)),
                "High": float(bar.get("High", 0)),
                "Low": float(bar.get("Low", 0)),
                "Close": float(bar.get("Close", 0)),
                "Volume": float(bar.get("Volume", 0)),
            }

        self._window_data.append(bar_dict)

        # Maintain window size
        if len(self._window_data) > self.window_size:
            self._window_data = self._window_data[-self.window_size :]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert rolling window to pandas DataFrame.

        Returns:
            DataFrame with OHLCV data sorted by date

        Raises:
            ValueError: If insufficient data in window
        """
        if len(self._window_data) < 2:
            raise ValueError(
                f"Insufficient data in window: need at least 2 bars, "
                f"have {len(self._window_data)}"
            )

        # Convert to DataFrame
        df = pd.DataFrame(self._window_data)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df

    def is_ready(self) -> bool:
        """
        Check if window has sufficient data.

        Returns:
            True if window has at least 30 bars
        """
        return len(self._window_data) >= 30

    @property
    def count(self) -> int:
        """Return number of bars in window."""
        return len(self._window_data)

    def __len__(self) -> int:
        """Return number of bars in window."""
        return len(self._window_data)


class HistoryDataAdapter:
    """
    Adapter for QuantConnect's History API.

    This adapter converts the DataFrame returned by algorithm.History()
    directly to the format expected by hidden-regime.
    """

    @staticmethod
    def convert_history(history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert QC History DataFrame to hidden-regime format.

        QuantConnect's History returns a multi-index DataFrame with
        (symbol, time) index. This method converts it to single-index
        DataFrame with just time.

        Args:
            history_df: DataFrame from algorithm.History()

        Returns:
            Single-index DataFrame with OHLCV data

        Raises:
            ValueError: If DataFrame format is unexpected
        """
        if history_df is None or history_df.empty:
            raise ValueError("History DataFrame is empty")

        # QC History returns multi-index (symbol, time) - extract data for first symbol
        if isinstance(history_df.index, pd.MultiIndex):
            # Get first symbol level
            symbol = history_df.index.get_level_values(0)[0]
            df = history_df.xs(symbol, level=0)
        else:
            df = history_df.copy()

        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [col.lower() for col in df.columns]

        # Map QC column names to our format
        column_mapping = {}
        for req_col in required_cols:
            for avail_col in df.columns:
                if avail_col.lower() == req_col:
                    column_mapping[avail_col] = req_col.capitalize()
                    break

        if len(column_mapping) != len(required_cols):
            raise ValueError(
                f"Missing required columns. Need {required_cols}, "
                f"have {list(df.columns)}"
            )

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df.sort_index(inplace=True)

        return df[["Open", "High", "Low", "Close", "Volume"]]


def create_dataframe_from_bars(bars: List[Any]) -> pd.DataFrame:
    """
    Create DataFrame from list of TradeBar objects.

    Utility function to convert a list of QuantConnect TradeBar objects
    into a pandas DataFrame for hidden-regime.

    Args:
        bars: List of TradeBar objects (or dicts for testing)

    Returns:
        DataFrame with OHLCV data indexed by date
    """
    adapter = QuantConnectDataAdapter(lookback_days=len(bars))

    for bar in bars:
        if hasattr(bar, "Time"):
            # Real TradeBar
            adapter.add_bar(
                time=bar.Time,
                open_price=bar.Open,
                high=bar.High,
                low=bar.Low,
                close=bar.Close,
                volume=bar.Volume,
            )
        else:
            # Dict or mock
            adapter.add_bar(
                time=bar.get("Time", bar.get("Date")),
                open_price=bar.get("Open", 0),
                high=bar.get("High", 0),
                low=bar.get("Low", 0),
                close=bar.get("Close", 0),
                volume=bar.get("Volume", 0),
            )

    return adapter.to_dataframe()
