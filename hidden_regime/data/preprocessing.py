"""
Data preprocessing functionality for hidden-regime package.

Provides DataPreprocessor class for cleaning, transforming, and preparing
market data for regime detection and analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings
from scipy import stats
from scipy.interpolate import interp1d

from ..config.settings import PreprocessingConfig, ValidationConfig
from ..utils.exceptions import DataQualityError, ValidationError


class DataPreprocessor:
    """
    Preprocess market data for analysis.

    Handles outlier detection, missing value imputation, return calculations,
    volatility estimation, and feature engineering.
    """

    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
    ):
        """
        Initialize DataPreprocessor with configuration.

        Args:
            preprocessing_config: PreprocessingConfig instance
            validation_config: ValidationConfig instance for outlier detection
        """
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.validation_config = validation_config or ValidationConfig()

    def process_data(
        self, data: pd.DataFrame, ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to data.

        Args:
            data: DataFrame with columns: date, price, (volume)
            ticker: Optional ticker symbol for logging/debugging

        Returns:
            Preprocessed DataFrame with additional calculated features

        Raises:
            DataQualityError: If data quality issues prevent processing
        """
        if data.empty:
            raise DataQualityError("Cannot process empty DataFrame")

        # Make a copy to avoid modifying original
        result = data.copy()

        # Ensure date column is datetime
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])
            result = result.sort_values("date").reset_index(drop=True)

        # 1. Handle missing values
        result = self._handle_missing_values(result, ticker)

        # 2. Detect and handle outliers
        result = self._handle_outliers(result, ticker)

        # 3. Calculate returns if not already present
        if "log_return" not in result.columns:
            result = self._calculate_returns(result)

        # 4. Calculate volatility features
        if self.preprocessing_config.calculate_volatility:
            result = self._calculate_volatility_features(result)

        # 5. Apply smoothing if requested
        if self.preprocessing_config.apply_smoothing:
            result = self._apply_smoothing(result)

        # 6. Final validation
        self._validate_processed_data(result, ticker)

        return result

    def process_multiple_series(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple time series and align them.

        Args:
            data_dict: Dictionary mapping ticker -> DataFrame

        Returns:
            Dictionary of processed and aligned DataFrames
        """
        # Process each series individually
        processed = {}
        for ticker, data in data_dict.items():
            try:
                processed[ticker] = self.process_data(data, ticker)
            except Exception as e:
                warnings.warn(f"Failed to process {ticker}: {e}")

        # Align timestamps if requested
        if self.preprocessing_config.align_timestamps and len(processed) > 1:
            processed = self._align_timestamps(processed)

        return processed

    def _handle_missing_values(
        self, data: pd.DataFrame, ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """Handle missing values in the data."""
        result = data.copy()

        # Check for excessive missing values
        missing_counts = result.isnull().sum()
        if missing_counts.any():
            total_missing = missing_counts.sum()
            missing_pct = total_missing / (len(result) * len(result.columns))

            if missing_pct > 0.1:  # More than 10% missing
                warnings.warn(
                    f"High percentage of missing values in {ticker or 'data'}: {missing_pct:.1%}"
                )

        # Check for consecutive missing values BEFORE interpolation
        for column in ["price"]:
            if column in result.columns:
                consecutive_missing = self._find_consecutive_missing(result[column])
                if consecutive_missing > self.validation_config.max_consecutive_missing:
                    raise DataQualityError(
                        f"Too many consecutive missing values in {column}: "
                        f"{consecutive_missing} > {self.validation_config.max_consecutive_missing}"
                    )

        # Handle price missing values
        if "price" in result.columns and result["price"].isnull().any():
            if self.validation_config.interpolation_method == "linear":
                result["price"] = result["price"].interpolate(method="linear")
            elif self.validation_config.interpolation_method == "forward":
                result["price"] = result["price"].ffill()
            elif self.validation_config.interpolation_method == "backward":
                result["price"] = result["price"].bfill()

        # Handle volume missing values (less critical)
        if "volume" in result.columns and result["volume"].isnull().any():
            # Use median volume for missing values
            median_volume = result["volume"].median()
            result["volume"] = result["volume"].fillna(median_volume)

        return result

    def _handle_outliers(
        self, data: pd.DataFrame, ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """Detect and handle outliers in price data."""
        result = data.copy()

        if "price" not in result.columns:
            return result

        # Calculate returns for outlier detection
        if "log_return" in result.columns:
            returns = result["log_return"].dropna()
        else:
            returns = np.log(result["price"] / result["price"].shift(1)).dropna()

        # Detect outliers based on method
        outlier_mask = self._detect_outliers(returns)

        if outlier_mask.any():
            n_outliers = outlier_mask.sum()
            outlier_pct = n_outliers / len(returns)

            if outlier_pct > 0.05:  # More than 5% outliers
                warnings.warn(
                    f"High percentage of outliers in {ticker or 'data'}: "
                    f"{n_outliers} ({outlier_pct:.1%})"
                )

            # Handle outliers by winsorization (cap at percentiles)
            result = self._winsorize_outliers(result, outlier_mask)

        return result

    def _detect_outliers(self, returns: pd.Series) -> pd.Series:
        """Detect outliers using configured method."""
        if self.validation_config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(returns, nan_policy="omit"))
            return z_scores > self.validation_config.outlier_threshold

        elif self.validation_config.outlier_method == "iqr":
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.validation_config.iqr_multiplier * IQR
            upper_bound = Q3 + self.validation_config.iqr_multiplier * IQR
            return (returns < lower_bound) | (returns > upper_bound)

        elif self.validation_config.outlier_method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(returns.values.reshape(-1, 1))
                return pd.Series(outlier_labels == -1, index=returns.index)
            except ImportError:
                warnings.warn("scikit-learn not available, falling back to IQR method")
                return self._detect_outliers_iqr(returns)

        else:
            raise ValueError(
                f"Unknown outlier method: {self.validation_config.outlier_method}"
            )

    def _winsorize_outliers(
        self, data: pd.DataFrame, outlier_mask: pd.Series
    ) -> pd.DataFrame:
        """Winsorize outliers by capping at percentiles."""
        result = data.copy()

        if "log_return" in result.columns:
            returns = result["log_return"]
            # Cap outliers at 1st and 99th percentiles
            lower_percentile = returns.quantile(0.01)
            upper_percentile = returns.quantile(0.99)

            # Ensure outlier_mask aligns with the returns index
            if len(outlier_mask) == len(returns):
                # Handle both numpy arrays and pandas Series
                if hasattr(outlier_mask, "reindex"):
                    # pandas Series
                    aligned_mask = outlier_mask.reindex(returns.index, fill_value=False)
                else:
                    # numpy array - create Series with matching index
                    aligned_mask = pd.Series(outlier_mask, index=returns.index)

                result.loc[aligned_mask, "log_return"] = np.clip(
                    returns[aligned_mask], lower_percentile, upper_percentile
                )
            else:
                # If lengths don't match, apply winsorization to all outliers in returns
                result["log_return"] = np.clip(
                    returns, lower_percentile, upper_percentile
                )

        return result

    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price returns."""
        result = data.copy()

        if "price" not in result.columns:
            raise ValueError("Price column required for return calculation")

        if self.preprocessing_config.return_method == "log":
            result["log_return"] = np.log(result["price"] / result["price"].shift(1))
        elif self.preprocessing_config.return_method == "simple":
            result["simple_return"] = result["price"].pct_change()
            # Also calculate log returns for consistency
            result["log_return"] = np.log(1 + result["simple_return"])
        else:
            raise ValueError(
                f"Unknown return method: {self.preprocessing_config.return_method}"
            )

        return result

    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features."""
        result = data.copy()

        if "log_return" not in result.columns:
            return result

        window = self.preprocessing_config.volatility_window

        # Rolling volatility (standard deviation of returns)
        result["volatility"] = result["log_return"].rolling(window=window).std()

        # Rolling absolute returns (alternative volatility measure)
        result["abs_return"] = result["log_return"].abs()
        result["avg_abs_return"] = result["abs_return"].rolling(window=window).mean()

        return result

    def _apply_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to price or return series."""
        result = data.copy()
        window = self.preprocessing_config.smoothing_window

        # Smooth price series
        if "price" in result.columns:
            result["price_smoothed"] = (
                result["price"].rolling(window=window, center=True).mean()
            )

        # Smooth returns
        if "log_return" in result.columns:
            result["log_return_smoothed"] = (
                result["log_return"].rolling(window=window, center=True).mean()
            )

        return result

    def _align_timestamps(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align multiple time series to common timestamps."""
        if len(data_dict) <= 1:
            return data_dict

        # Find common date range
        all_dates = []
        for df in data_dict.values():
            if "date" in df.columns:
                all_dates.extend(df["date"].tolist())

        if not all_dates:
            return data_dict  # No date columns to align

        # Create common date index
        common_dates = pd.date_range(
            start=min(all_dates), end=max(all_dates), freq="B"  # Business days
        )

        # Reindex each series to common dates
        aligned = {}
        for ticker, df in data_dict.items():
            if "date" in df.columns:
                df_indexed = df.set_index("date")
                df_reindexed = df_indexed.reindex(common_dates)

                # Fill missing values based on config
                if self.preprocessing_config.fill_method == "forward":
                    df_reindexed = df_reindexed.ffill()
                elif self.preprocessing_config.fill_method == "backward":
                    df_reindexed = df_reindexed.bfill()
                elif self.preprocessing_config.fill_method == "interpolate":
                    df_reindexed = df_reindexed.interpolate()

                aligned[ticker] = df_reindexed.reset_index().rename(
                    columns={"index": "date"}
                )
            else:
                aligned[ticker] = df

        return aligned

    def _find_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum number of consecutive missing values."""
        is_null = series.isnull()
        groups = (is_null != is_null.shift()).cumsum()
        consecutive_counts = is_null.groupby(groups).sum()
        return consecutive_counts.max() if consecutive_counts.any() else 0

    def _validate_processed_data(
        self, data: pd.DataFrame, ticker: Optional[str] = None
    ) -> None:
        """Validate processed data quality."""
        if data.empty:
            raise DataQualityError("Processed data is empty")

        # Check for infinite values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if np.isinf(data[column]).any():
                raise DataQualityError(f"Infinite values found in {column}")

        # Check log returns if present
        if "log_return" in data.columns:
            returns = data["log_return"].dropna()
            if len(returns) == 0:
                raise DataQualityError("No valid returns calculated")

            # Check for extreme returns
            if (returns.abs() > self.validation_config.max_daily_return).any():
                warnings.warn(
                    f"Extreme returns detected in {ticker or 'data'} "
                    f"(>{self.validation_config.max_daily_return:.1%})"
                )

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for processed data."""
        summary = {
            "n_observations": len(data),
            "date_range": None,
            "columns": list(data.columns),
        }

        if "date" in data.columns:
            summary["date_range"] = {
                "start": data["date"].min(),
                "end": data["date"].max(),
                "days": (data["date"].max() - data["date"].min()).days,
            }

        if "log_return" in data.columns:
            returns = data["log_return"].dropna()
            summary["return_stats"] = {
                "mean": returns.mean(),
                "std": returns.std(),
                "min": returns.min(),
                "max": returns.max(),
                "skewness": stats.skew(returns),
                "kurtosis": stats.kurtosis(returns),
            }

        return summary
