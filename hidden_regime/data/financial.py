"""
Financial data loading component for pipeline architecture.

Provides FinancialDataLoader that implements DataComponent interface for loading
stock market data from various sources with robust error handling and validation.
"""

import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.data import FinancialDataConfig
from ..pipeline.interfaces import DataComponent
from ..utils.exceptions import DataLoadError, ValidationError

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")


class FinancialDataLoader(DataComponent):
    """
    Financial data loader component for pipeline architecture.

    Implements DataComponent interface to provide stock market data loading
    with robust error handling, caching, and data quality validation.
    """

    def __init__(self, config: FinancialDataConfig):
        """
        Initialize financial data loader with configuration.

        Args:
            config: FinancialDataConfig with data loading parameters
        """
        self.config = config
        self._cache = {}  # Simple in-memory cache
        self._last_data = None

        # Validate yfinance availability if needed
        if not YFINANCE_AVAILABLE and self.config.source == "yfinance":
            raise DataLoadError(
                "yfinance is not installed but set as data source. "
                "Install with: pip install yfinance"
            )

    def get_all_data(self) -> pd.DataFrame:
        """
        Get complete dataset.

        Returns:
            DataFrame with complete data including timestamps
        """
        if self._last_data is None:
            # Load data if not already loaded
            self._last_data = self._load_data()

        return self._last_data.copy()

    def update(self, current_date: Optional[str] = None) -> pd.DataFrame:
        """
        Update data, optionally fetching new data up to current_date.

        Args:
            current_date: Optional date to update data up to

        Returns:
            Updated DataFrame with any new data
        """
        # For now, we'll reload all data each time
        # In future, this could be optimized to only fetch new data
        self._last_data = self._load_data(end_date_override=current_date)
        return self._last_data.copy()

    def load_data(self, end_date_override: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from configured source.

        Args:
            end_date_override: Optional override for end date

        Returns:
            DataFrame with loaded and processed data
        """
        self._last_data = self._load_data(end_date_override=end_date_override)
        return self._last_data.copy()

    def _load_data(self, end_date_override: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from configured source.

        Args:
            end_date_override: Optional override for end date

        Returns:
            DataFrame with loaded and processed data
        """
        # Determine date range
        start_date = self.config.start_date
        end_date = end_date_override or self.config.end_date

        # Apply num_samples limit if specified
        if (
            self.config.num_samples is not None
            and start_date is None
            and end_date is None
        ):
            # Calculate approximate start date based on frequency and num_samples
            if self.config.frequency == "days":
                days_back = self.config.num_samples
            else:
                days_back = self.config.num_samples  # Fallback

            end_dt = (
                pd.Timestamp.now() if end_date is None else pd.to_datetime(end_date)
            )
            start_dt = end_dt - pd.Timedelta(days=days_back)
            start_date = start_dt.strftime("%Y-%m-%d")
            if end_date is None:
                end_date = end_dt.strftime("%Y-%m-%d")

        # Validate inputs
        self._validate_inputs(self.config.ticker, start_date, end_date)

        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        # Check cache first
        cache_key = f"{self.config.ticker}_{start_dt}_{end_dt}_{self.config.source}"
        if cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            # Simple cache expiry (24 hours)
            if (datetime.now() - cache_time).total_seconds() < 86400:
                return cached_data.copy()

        # Load data based on source
        if self.config.source == "yfinance":
            raw_data = self._load_from_yfinance(self.config.ticker, start_dt, end_dt)
        else:
            raise DataLoadError(f"Unsupported data source: {self.config.source}")

        # Process raw data into standardized format
        processed_data = self._process_raw_data(raw_data)

        # Apply num_samples limit after loading if specified
        if (
            self.config.num_samples is not None
            and len(processed_data) > self.config.num_samples
        ):
            processed_data = processed_data.tail(self.config.num_samples).reset_index(
                drop=True
            )

        # Validate data quality
        self._validate_data_quality(processed_data, self.config.ticker)

        # Cache the result
        self._cache[cache_key] = (processed_data.copy(), datetime.now())

        return processed_data

    def _load_from_yfinance(
        self, ticker: str, start_date: Optional[pd.Timestamp], end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load data from yfinance with retry logic."""
        if not YFINANCE_AVAILABLE:
            raise DataLoadError("yfinance not available")

        # Default to 2 years of data if no start date
        if start_date is None:
            start_date = end_date - pd.Timedelta(days=730)

        for attempt in range(3):  # Simple retry logic
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=start_date, end=end_date, auto_adjust=True, prepost=False
                )

                if data.empty:
                    raise DataLoadError(f"No data found for ticker {ticker}")

                return data

            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (2**attempt))  # Exponential backoff
                    continue
                else:
                    raise DataLoadError(f"Failed to load data for {ticker}: {e}")

    def _process_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw yfinance data into standardized format with mandatory financial pipeline.

        This method implements the core financial data pipeline:
        OHLCV → price → pct_change → log_return

        These calculations are mandatory and automatic for all financial data.
        """
        # Start with a copy to preserve the original data structure
        if isinstance(raw_data.index, pd.DatetimeIndex):
            data = raw_data.reset_index()
        else:
            data = raw_data.copy()

        # Create result DataFrame with proper index
        result = pd.DataFrame(index=raw_data.index)

        # Add standard OHLCV columns
        if "Open" in data.columns:
            result["open"] = data["Open"].values
        if "High" in data.columns:
            result["high"] = data["High"].values
        if "Low" in data.columns:
            result["low"] = data["Low"].values
        if "Close" in data.columns:
            result["close"] = data["Close"].values
        if "Volume" in data.columns:
            result["volume"] = data["Volume"].values

        # MANDATORY FINANCIAL DATA PIPELINE
        # Step 1: Calculate price (OHLC average as specified in improvements.md)
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            # Use OHLC average: 0.25*(open + close + high + low)
            result["price"] = 0.25 * (
                data["Open"].values
                + data["Close"].values
                + data["High"].values
                + data["Low"].values
            )
        elif "Close" in data.columns:
            # Fallback to close price if OHLC not available
            result["price"] = data["Close"].values
        else:
            raise ValueError("No price data available - need Close or OHLC columns")

        # Step 2: Calculate percentage change (MANDATORY)
        result["pct_change"] = result["price"].pct_change(fill_method=None)

        # Step 3: Calculate log return (MANDATORY)
        # Using the formula from improvements.md: log_return = np.log(pct_change + 1)
        result["log_return"] = np.log(result["pct_change"] + 1.0)

        # Remove any rows with missing essential data (first row will have NaN pct_change/log_return)
        essential_columns = ["price", "pct_change", "log_return"]
        result = result.dropna(subset=essential_columns, how="any")

        return result

    def _validate_inputs(
        self, ticker: str, start_date: Optional[str], end_date: Optional[str]
    ) -> None:
        """Validate input parameters."""
        if not ticker or not isinstance(ticker, str):
            raise ValidationError("Ticker must be a non-empty string")

        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                raise ValidationError(f"Invalid date format: {e}")

            if start_dt >= end_dt:
                raise ValidationError("Start date must be before end date")

            if end_dt > datetime.now():
                raise ValidationError("End date cannot be in the future")

            # Check minimum time period
            if (end_dt - start_dt).days < 7:
                raise ValidationError("Minimum 7-day time period required")

    def _validate_data_quality(self, data: pd.DataFrame, ticker: str) -> None:
        """Validate loaded data quality including mandatory financial pipeline columns."""
        if data.empty:
            raise DataLoadError(f"No data loaded for {ticker}")

        if len(data) < 10:  # Minimum observations (restored to 10 as expected by tests)
            raise DataLoadError(f"Insufficient data for {ticker}: {len(data)} < 10")

        # Check for reasonable price values first (before checking mandatory columns)
        price_column = None
        for col in ["price", "close", "Close"]:
            if col in data.columns:
                price_column = col
                break

        if price_column and (data[price_column] <= 0).any():
            raise DataLoadError("Invalid price values")

        # Validate mandatory financial pipeline columns exist
        mandatory_columns = ["price", "pct_change", "log_return"]
        missing_columns = [col for col in mandatory_columns if col not in data.columns]
        if missing_columns:
            raise DataLoadError(
                f"Missing mandatory financial pipeline columns for {ticker}: {missing_columns}. "
                "These should be automatically calculated during data processing."
            )

        # Check for reasonable percentage change values (should be finite)
        if not np.isfinite(data["pct_change"]).all():
            # Allow for first row NaN (which was removed), but not others
            finite_pct_change = np.isfinite(data["pct_change"])
            if not finite_pct_change.all():
                raise DataLoadError(
                    f"Invalid percentage change values found for {ticker}"
                )

        # Check for reasonable log return values (should be finite)
        if not np.isfinite(data["log_return"]).all():
            finite_log_return = np.isfinite(data["log_return"])
            if not finite_log_return.all():
                raise DataLoadError(f"Invalid log return values found for {ticker}")

        # Additional validation for financial data sanity
        if "close" in data.columns and (data["close"] <= 0).any():
            raise DataLoadError(f"Invalid close price values found for {ticker}")

    def plot(self, ax=None, **kwargs) -> plt.Figure:
        """
        Generate visualization of financial data.

        Args:
            ax: Optional matplotlib axes to plot into for pipeline integration
            **kwargs: Additional plotting arguments

        Returns:
            matplotlib Figure with financial data plots
        """
        if self._last_data is None:
            if ax is not None:
                ax.text(
                    0.5,
                    0.5,
                    "No data loaded yet",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return ax.figure
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No data loaded yet",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return fig

        data = self._last_data

        # If ax is provided, create compact plot for pipeline integration
        if ax is not None:
            return self._plot_compact(ax, data, **kwargs)

        # Otherwise, create full standalone plot
        return self._plot_full(data, **kwargs)

    def _plot_compact(self, ax, data, **kwargs):
        """Create compact plot for pipeline integration."""
        # Check if regime overlay data is provided
        regime_data = kwargs.get("regime_data", None)

        # Plot price data only for compact view
        price_col = "close" if "close" in data.columns else "price"
        if price_col in data.columns:
            ax.plot(
                data.index, data[price_col], label="Price", linewidth=1.5, color="black"
            )

        # Add regime overlay if available
        if regime_data is not None:
            self._add_regime_overlay(ax, data, regime_data)

        ax.set_title(f"Price Data with Regime Overlay - {self.config.ticker}")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax.figure

    def _add_regime_overlay(self, ax, price_data, regime_data):
        """Add regime overlay to price chart."""
        # Color map for regimes
        regime_colors = {
            "Crisis": "red",
            "Bear": "orange",
            "Sideways": "gray",
            "Bull": "green",
            "Euphoric": "purple",
        }

        # Create regime background using date ranges
        current_regime = None
        regime_start = None

        # Get date range that matches both datasets
        common_dates = price_data.index.intersection(regime_data.index)
        if len(common_dates) == 0:
            return  # No common dates to plot

        for date in common_dates:
            if date in regime_data.index:
                row = regime_data.loc[date]
                regime_type = row.get(
                    "regime_name",
                    row.get("regime_type", f"State_{row.get('predicted_state', 0)}"),
                )
                confidence = row.get("confidence", 0.5)

                # If regime changed or this is the first point, draw previous regime and start new one
                if current_regime != regime_type:
                    # Draw the previous regime background
                    if current_regime is not None and regime_start is not None:
                        color = regime_colors.get(current_regime, "blue")
                        alpha = 0.2  # Fixed alpha for visibility
                        ax.axvspan(
                            regime_start, date, color=color, alpha=alpha, zorder=0
                        )

                    # Start new regime
                    current_regime = regime_type
                    regime_start = date

        # Draw the final regime
        if current_regime is not None and regime_start is not None:
            color = regime_colors.get(current_regime, "blue")
            alpha = 0.2
            ax.axvspan(
                regime_start, common_dates[-1], color=color, alpha=alpha, zorder=0
            )

        # Add regime legend
        if "regime_name" in regime_data.columns:
            unique_regimes = regime_data["regime_name"].unique()
        elif "regime_type" in regime_data.columns:
            unique_regimes = regime_data["regime_type"].unique()
        else:
            unique_regimes = []

        if len(unique_regimes) > 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=regime_colors.get(regime, "blue"), alpha=0.3, label=regime
                )
                for regime in unique_regimes
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    def _plot_full(self, data, **kwargs):
        """Create full standalone plot with subplots."""
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Price data
        ax1 = axes[0]
        if "close" in data.columns:
            ax1.plot(data.index, data["close"], label="Close Price", linewidth=1.5)
        elif "price" in data.columns:
            ax1.plot(data.index, data["price"], label="Price", linewidth=1.5)

        ax1.set_title(f"Price Data for {self.config.ticker}")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume (if available)
        ax2 = axes[1]
        if "volume" in data.columns:
            ax2.bar(data.index, data["volume"], alpha=0.6, label="Volume")
            ax2.set_title("Trading Volume")
            ax2.set_ylabel("Volume")
            ax2.legend()
        else:
            ax2.text(
                0.5,
                0.5,
                "Volume data not available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Trading Volume")

        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self._cache),
            "ticker": self.config.ticker,
            "source": self.config.source,
            "last_data_shape": (
                self._last_data.shape if self._last_data is not None else None
            ),
        }

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
