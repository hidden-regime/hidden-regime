"""
Data loading functionality for hidden-regime package.

Provides DataLoader class for loading stock market data from various sources
with robust error handling, caching, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time
import warnings

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")

from ..config.settings import DataConfig
from ..utils.exceptions import DataLoadError, ValidationError


class DataLoader:
    """
    Load stock market data from multiple sources.

    Supports yfinance as the primary data source with extensible architecture
    for adding additional sources in the future.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize DataLoader with configuration.

        Args:
            config: DataConfig instance. Uses defaults if None.
        """
        self.config = config or DataConfig()
        self._cache = {}  # Simple in-memory cache

        if not YFINANCE_AVAILABLE and self.config.default_source == "yfinance":
            raise DataLoadError(
                "yfinance is not installed but set as default source. "
                "Install with: pip install yfinance"
            )

    def load_stock_data(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: Optional[str] = None,
        use_ohlc_avg: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Load stock data for a single ticker.

        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date as string 'YYYY-MM-DD' or datetime
            end_date: End date as string 'YYYY-MM-DD' or datetime
            source: Data source to use. Defaults to config.default_source
            use_ohlc_avg: Use OHLC average vs close price. Defaults to config setting

        Returns:
            DataFrame with columns: date, price, log_return, volume (optional)

        Raises:
            DataLoadError: If data loading fails
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(ticker, start_date, end_date)

        # Convert dates to datetime if needed
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Use defaults from config if not specified
        source = source or self.config.default_source
        use_ohlc_avg = (
            use_ohlc_avg if use_ohlc_avg is not None else self.config.use_ohlc_average
        )

        # Check cache first
        cache_key = (
            f"{ticker}_{start_dt.date()}_{end_dt.date()}_{source}_{use_ohlc_avg}"
        )
        if self.config.cache_enabled and cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            if (
                datetime.now() - cache_time
            ).total_seconds() < self.config.cache_expiry_hours * 3600:
                return cached_data.copy()

        # Load data based on source
        if source == "yfinance":
            raw_data = self._load_from_yfinance(ticker, start_dt, end_dt)
        else:
            raise DataLoadError(f"Unsupported data source: {source}")

        # Process raw data into standardized format
        processed_data = self._process_raw_data(raw_data, use_ohlc_avg)

        # Validate data quality
        self._validate_data_quality(processed_data, ticker)

        # Cache the result
        if self.config.cache_enabled:
            self._cache[cache_key] = (processed_data.copy(), datetime.now())

        return processed_data

    def load_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers.

        Args:
            tickers: List of stock symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments passed to load_stock_data()

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                results[ticker] = self.load_stock_data(
                    ticker, start_date, end_date, **kwargs
                )

                # Rate limiting
                time.sleep(60 / self.config.requests_per_minute)

            except Exception as e:
                failed_tickers.append((ticker, str(e)))
                warnings.warn(f"Failed to load data for {ticker}: {e}")

        if failed_tickers:
            warnings.warn(
                f"Failed to load {len(failed_tickers)} tickers: {failed_tickers}"
            )

        return results

    def _load_from_yfinance(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Load data from yfinance with retry logic."""
        if not YFINANCE_AVAILABLE:
            raise DataLoadError("yfinance not available")

        for attempt in range(self.config.retry_attempts):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=start_date, end=end_date, auto_adjust=True, prepost=False
                )

                if data.empty:
                    raise DataLoadError(f"No data found for ticker {ticker}")

                return data

            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (2**attempt))
                    continue
                else:
                    raise DataLoadError(f"Failed to load data for {ticker}: {e}")

    def _process_raw_data(
        self, raw_data: pd.DataFrame, use_ohlc_avg: bool
    ) -> pd.DataFrame:
        """Process raw yfinance data into standardized format."""
        result = pd.DataFrame()

        # Reset index to make date a column
        data = raw_data.reset_index()

        # Calculate price based on preference
        if use_ohlc_avg and all(
            col in data.columns for col in ["Open", "High", "Low", "Close"]
        ):
            result["price"] = (
                data["Open"] + data["High"] + data["Low"] + data["Close"]
            ) / 4
        else:
            result["price"] = data["Close"]

        # Add date column - handle both 'Date' and 'index' column names
        if "Date" in data.columns:
            result["date"] = data["Date"]
        elif "index" in data.columns:
            result["date"] = data["index"]
        else:
            # Fallback: use the index if no date column found after reset_index
            result["date"] = raw_data.index

        # Calculate log returns
        result["log_return"] = np.log(result["price"] / result["price"].shift(1))

        # Add volume if requested and available
        if self.config.include_volume and "Volume" in data.columns:
            result["volume"] = data["Volume"]

        # Remove first row (NaN log return)
        result = result.dropna().reset_index(drop=True)

        return result

    def _validate_inputs(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> None:
        """Validate input parameters."""
        if not ticker or not isinstance(ticker, str):
            raise ValidationError("Ticker must be a non-empty string")

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
        """Validate loaded data quality."""
        if data.empty:
            raise DataLoadError(f"No data loaded for {ticker}")

        if len(data) < self.config.min_observations:
            raise DataLoadError(
                f"Insufficient data for {ticker}: {len(data)} < {self.config.min_observations}"
            )

        # Check for excessive missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > self.config.max_missing_data_pct:
            raise DataLoadError(
                f"Excessive missing data for {ticker}: {missing_pct:.2%} > "
                f"{self.config.max_missing_data_pct:.2%}"
            )

        # Check for reasonable price values
        if (data["price"] <= 0).any():
            raise DataLoadError(f"Invalid price values found for {ticker}")

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self._cache),
            "cache_enabled": self.config.cache_enabled,
            "cache_expiry_hours": self.config.cache_expiry_hours,
        }

    def plot(
        self,
        data: pd.DataFrame,
        plot_type: str = "all",
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ) -> "matplotlib.Figure":
        """
        Create comprehensive visualizations of market data.

        Args:
            data: DataFrame with columns 'date', 'price', 'log_return', and optionally 'volume'
            plot_type: Type of plot ('all', 'price', 'returns', 'distribution', 'volume')
            figsize: Figure size as (width, height)
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib/seaborn not available
            ValueError: If required columns missing from data
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import seaborn as sns
            from ..visualization.plotting import (
                setup_financial_plot_style,
                format_financial_axis,
                save_plot,
                get_regime_colors,
                REGIME_COLORS,
            )
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn"
            )

        # Validate required columns
        required_columns = ["date", "price", "log_return"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing from data: {missing_columns}")

        # Setup styling
        setup_financial_plot_style()

        # Prepare data with validation
        dates, valid_indices = self._validate_and_clean_dates(data["date"])

        # Filter data to match cleaned dates
        if valid_indices is not None:
            filtered_data = data.iloc[valid_indices].copy()
            prices = filtered_data["price"].values
            returns = filtered_data["log_return"].values
            has_volume = "volume" in filtered_data.columns
            if has_volume:
                volume_data = filtered_data["volume"].values
            else:
                volume_data = None
        else:
            prices = data["price"].values
            returns = data["log_return"].values
            has_volume = "volume" in data.columns
            volume_data = data["volume"].values if has_volume else None

        # Determine subplot configuration based on plot_type
        if plot_type == "all":
            n_subplots = 4 if has_volume else 3
            subplot_configs = [
                ("Price", "price"),
                ("Returns", "returns"),
                ("Distribution", "distribution"),
            ]
            if has_volume:
                subplot_configs.append(("Volume", "volume"))
        elif plot_type in ["price", "returns", "distribution", "volume"]:
            n_subplots = 1
            subplot_configs = [(plot_type.title(), plot_type)]
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        # Create subplots
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, (title, plot_subtype) in enumerate(subplot_configs):
            ax = axes[i]

            if plot_subtype == "price":
                # Price time series
                ax.plot(dates, prices, color="#1f77b4", linewidth=1.2, alpha=0.8)
                ax.set_title(f"Price Evolution", fontweight="bold")
                ax.set_ylabel("Price ($)")
                ax.grid(True, alpha=0.3)
                format_financial_axis(ax, dates)

                # Add simple moving averages
                if len(prices) >= 20:
                    ma_20 = pd.Series(prices).rolling(20, min_periods=1).mean()
                    ax.plot(
                        dates,
                        ma_20,
                        color="orange",
                        linewidth=1,
                        alpha=0.7,
                        linestyle="--",
                        label="20-day MA",
                    )

                if len(prices) >= 50:
                    ma_50 = pd.Series(prices).rolling(50, min_periods=1).mean()
                    ax.plot(
                        dates,
                        ma_50,
                        color=REGIME_COLORS["Crisis"],
                        linewidth=1,
                        alpha=0.7,
                        linestyle="--",
                        label="50-day MA",
                    )

                if len(prices) >= 20 or len(prices) >= 50:
                    ax.legend(loc="upper left")

            elif plot_subtype == "returns":
                # Returns time series with zero line - colorblind-friendly version
                # Use Bear (orange) for negative, Bull (blue) for positive returns
                bear_color = REGIME_COLORS["Bear"]  # Dark orange
                bull_color = REGIME_COLORS["Bull"]  # Blue

                # Separate positive and negative returns for different plotting
                positive_mask = returns >= 0
                negative_mask = returns < 0

                # Plot negative returns with down triangles
                if negative_mask.any():
                    ax.scatter(
                        dates[negative_mask],
                        returns[negative_mask],
                        c=bear_color,
                        marker="v",
                        alpha=0.7,
                        s=20,
                        edgecolors="black",
                        linewidth=0.5,
                        label="Negative Returns",
                    )

                # Plot positive returns with up triangles
                if positive_mask.any():
                    ax.scatter(
                        dates[positive_mask],
                        returns[positive_mask],
                        c=bull_color,
                        marker="^",
                        alpha=0.7,
                        s=20,
                        edgecolors="black",
                        linewidth=0.5,
                        label="Positive Returns",
                    )

                ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
                ax.set_title("Daily Log Returns", fontweight="bold")
                ax.set_ylabel("Log Returns")
                ax.grid(True, alpha=0.3)
                format_financial_axis(ax, dates)

                # Add legend for accessibility
                ax.legend(loc="upper right")

                # Add rolling volatility
                if len(returns) >= 20:
                    rolling_vol = pd.Series(returns).rolling(20, min_periods=1).std()
                    ax_vol = ax.twinx()
                    ax_vol.plot(
                        dates,
                        rolling_vol,
                        color="purple",
                        alpha=0.5,
                        linewidth=1,
                        label="20-day Vol",
                    )
                    ax_vol.set_ylabel("Rolling Volatility", color="purple")
                    ax_vol.tick_params(axis="y", labelcolor="purple")

            elif plot_subtype == "distribution":
                # Returns distribution analysis
                ax.hist(
                    returns,
                    bins=50,
                    density=True,
                    alpha=0.7,
                    color="lightblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add normal distribution overlay
                mu, sigma = np.mean(returns), np.std(returns)
                x = np.linspace(returns.min(), returns.max(), 100)
                normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x - mu) / sigma) ** 2
                )
                ax.plot(
                    x, normal_dist, "r-", linewidth=2, alpha=0.8, label="Normal Dist"
                )

                # Add KDE
                try:
                    from scipy import stats

                    kde = stats.gaussian_kde(returns)
                    ax.plot(x, kde(x), "g-", linewidth=2, alpha=0.8, label="KDE")
                except ImportError:
                    pass

                ax.axvline(
                    mu,
                    color=REGIME_COLORS["Bull"],
                    linestyle="--",
                    alpha=0.8,
                    label=f"Mean: {mu:.4f}",
                )
                ax.axvline(
                    mu + 2 * sigma,
                    color=REGIME_COLORS["Crisis"],
                    linestyle=":",
                    alpha=0.6,
                    label=f"+2σ: {mu + 2*sigma:.4f}",
                )
                ax.axvline(
                    mu - 2 * sigma,
                    color=REGIME_COLORS["Crisis"],
                    linestyle=":",
                    alpha=0.6,
                    label=f"-2σ: {mu - 2*sigma:.4f}",
                )

                ax.set_title("Returns Distribution Analysis", fontweight="bold")
                ax.set_xlabel("Log Returns")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)

            elif plot_subtype == "volume" and has_volume:
                # Volume analysis - colorblind-friendly
                volume = volume_data
                ax.bar(
                    dates,
                    volume,
                    color=REGIME_COLORS["Sideways"],
                    alpha=0.6,
                    width=1,
                    edgecolor="black",
                    linewidth=0.1,
                )
                ax.set_title("Trading Volume", fontweight="bold")
                ax.set_ylabel("Volume")
                ax.grid(True, alpha=0.3)
                format_financial_axis(ax, dates)

                # Add volume moving average
                if len(volume) >= 20:
                    vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean()
                    ax.plot(
                        dates,
                        vol_ma,
                        color=REGIME_COLORS["Bull"],
                        linewidth=2,
                        alpha=0.8,
                        label="20-day Vol MA",
                    )
                    ax.legend()

        # Add overall statistics text box
        if plot_type == "all":
            stats_text = self._create_summary_stats_text(data)
            fig.text(
                0.02,
                0.98,
                stats_text,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_plot(fig, save_path)

        return fig

    def _create_summary_stats_text(self, data: pd.DataFrame) -> str:
        """Create summary statistics text for the plot."""
        returns = data["log_return"].values
        prices = data["price"].values

        stats = {
            "Observations": len(data),
            "Date Range": f"{data['date'].min().date()} to {data['date'].max().date()}",
            "Price Range": f"${prices.min():.2f} - ${prices.max():.2f}",
            "Total Return": f"{((prices[-1]/prices[0]) - 1) * 100:.1f}%",
            "Mean Daily Return": f"{np.mean(returns) * 100:.3f}%",
            "Daily Volatility": f"{np.std(returns) * 100:.3f}%",
            "Annualized Return": f"{np.mean(returns) * 252 * 100:.1f}%",
            "Annualized Volatility": f"{np.std(returns) * np.sqrt(252) * 100:.1f}%",
            "Sharpe Ratio": f"{(np.mean(returns) / np.std(returns)) * np.sqrt(252):.2f}",
            "Max Daily Return": f"{np.max(returns) * 100:.2f}%",
            "Min Daily Return": f"{np.min(returns) * 100:.2f}%",
        }

        return "\n".join([f"{k}: {v}" for k, v in stats.items()])

    def _validate_and_clean_dates(
        self, dates: pd.Series
    ) -> Tuple[Union[pd.DatetimeIndex, pd.RangeIndex], Optional[np.ndarray]]:
        """
        Validate and clean date values for matplotlib compatibility.

        Args:
            dates: Series of date values

        Returns:
            Tuple of (cleaned_dates, valid_indices)
            - cleaned_dates: Clean DatetimeIndex or RangeIndex suitable for matplotlib
            - valid_indices: Array of valid indices if filtering occurred, None otherwise

        Raises:
            ValueError: If dates cannot be converted to valid matplotlib range
        """
        try:
            # Convert to datetime
            dt_index = pd.to_datetime(dates)

            # Check for valid matplotlib date range (years 0001-9999)
            # Handle timezone-aware vs timezone-naive comparison
            tz = getattr(dt_index, "tz", None) or (
                dt_index.dtype.tz if hasattr(dt_index.dtype, "tz") else None
            )

            if tz is not None:
                # If dt_index is timezone-aware, convert bounds to same timezone
                min_date = pd.Timestamp("0001-01-01").tz_localize(tz)
                max_date = pd.Timestamp("9999-12-31").tz_localize(tz)
            else:
                # If dt_index is timezone-naive, use timezone-naive bounds
                min_date = pd.Timestamp("0001-01-01")
                max_date = pd.Timestamp("9999-12-31")

            # Filter out invalid dates
            valid_mask = (dt_index >= min_date) & (dt_index <= max_date)

            # Also filter out NaT values
            valid_mask = valid_mask & (~dt_index.isna())

            if not valid_mask.any():
                raise ValueError("No valid dates found in matplotlib range (0001-9999)")

            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                warnings.warn(
                    f"Filtered out {invalid_count} dates outside matplotlib range (0001-9999) or NaT values"
                )
                valid_indices = np.where(valid_mask)[0]
                return dt_index[valid_mask], valid_indices

            # All dates are valid
            return dt_index, None

        except Exception as e:
            # Fallback to simple integer index if dates are problematic
            warnings.warn(f"Date conversion failed ({e}), using integer index instead")
            return pd.RangeIndex(len(dates)), None
