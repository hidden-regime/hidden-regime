"""
Unit tests for DataPreprocessor class.

Tests data preprocessing functionality including outlier detection,
missing value handling, return calculations, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hidden_regime.data.preprocessing import DataPreprocessor
from hidden_regime.config.settings import PreprocessingConfig, ValidationConfig
from hidden_regime.utils.exceptions import DataQualityError
from tests.fixtures.sample_data import (
    create_sample_stock_data,
    create_invalid_stock_data,
    get_test_config_variations,
)


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.sample_data = create_sample_stock_data(n_days=100, add_volume=True)

    def test_init_default_config(self):
        """Test DataPreprocessor initialization with default configuration."""
        preprocessor = DataPreprocessor()
        assert preprocessor.preprocessing_config.return_method == "log"
        assert preprocessor.preprocessing_config.calculate_volatility == True
        assert preprocessor.validation_config.outlier_method == "iqr"

    def test_init_custom_config(self):
        """Test DataPreprocessor initialization with custom configuration."""
        preprocessing_config = PreprocessingConfig(
            return_method="simple", calculate_volatility=False, apply_smoothing=True
        )
        validation_config = ValidationConfig(
            outlier_method="zscore", outlier_threshold=2.0
        )

        preprocessor = DataPreprocessor(
            preprocessing_config=preprocessing_config,
            validation_config=validation_config,
        )

        assert preprocessor.preprocessing_config.return_method == "simple"
        assert preprocessor.preprocessing_config.calculate_volatility == False
        assert preprocessor.validation_config.outlier_method == "zscore"

    def test_process_data_basic(self):
        """Test basic data processing."""
        result = self.preprocessor.process_data(self.sample_data.copy())

        # Should have all original columns plus any new ones
        assert "date" in result.columns
        assert "price" in result.columns
        assert "log_return" in result.columns

        # Should have volatility features
        assert "volatility" in result.columns
        assert "abs_return" in result.columns

        # Data should not be empty
        assert not result.empty
        assert len(result) <= len(self.sample_data)  # May be shorter due to windows

    def test_process_data_empty_input(self):
        """Test processing empty DataFrame."""
        empty_data = pd.DataFrame()

        with pytest.raises(DataQualityError, match="Cannot process empty DataFrame"):
            self.preprocessor.process_data(empty_data)

    def test_handle_missing_values_none(self):
        """Test handling data with no missing values."""
        clean_data = self.sample_data.copy()
        result = self.preprocessor._handle_missing_values(clean_data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(result, clean_data)

    def test_handle_missing_values_interpolation(self):
        """Test missing value interpolation."""
        data_with_missing = self.sample_data.copy()

        # Add some missing values
        data_with_missing.loc[10:12, "price"] = np.nan

        result = self.preprocessor._handle_missing_values(data_with_missing)

        # Missing values should be filled
        assert not result["price"].isnull().any()

        # Values should be reasonable (between neighboring values)
        assert result.loc[11, "price"] > 0

    def test_handle_missing_values_excessive(self):
        """Test handling excessive consecutive missing values."""
        data_with_excessive_missing = self.sample_data.copy()

        # Add too many consecutive missing values
        data_with_excessive_missing.loc[10:25, "price"] = np.nan  # 16 consecutive

        with pytest.raises(
            DataQualityError, match="Too many consecutive missing values"
        ):
            self.preprocessor._handle_missing_values(data_with_excessive_missing)

    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        returns = pd.Series(
            [0.01, 0.02, -0.01, 0.15, 0.01, -0.02, 0.01]
        )  # 0.15 is outlier

        outliers = self.preprocessor._detect_outliers(returns)

        # Should detect the 0.15 return as outlier
        assert outliers.sum() == 1
        assert outliers.iloc[3] == True  # The 0.15 return

    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        validation_config = ValidationConfig(
            outlier_method="zscore", outlier_threshold=2.0
        )
        preprocessor = DataPreprocessor(validation_config=validation_config)

        returns = pd.Series([0.01, 0.02, -0.01, 0.15, 0.01, -0.02, 0.01])
        outliers = preprocessor._detect_outliers(returns)

        # Should detect outliers based on Z-score
        assert outliers.sum() >= 1

    def test_winsorize_outliers(self):
        """Test outlier winsorization."""
        data_with_outliers = self.sample_data.copy()

        # Add extreme outliers
        data_with_outliers.loc[10, "log_return"] = 0.3  # 30% return
        data_with_outliers.loc[20, "log_return"] = -0.25  # -25% return

        outlier_mask = pd.Series([False] * len(data_with_outliers))
        outlier_mask.iloc[10] = True
        outlier_mask.iloc[20] = True

        result = self.preprocessor._winsorize_outliers(data_with_outliers, outlier_mask)

        # Outliers should be capped
        assert abs(result.loc[10, "log_return"]) < 0.3
        assert abs(result.loc[20, "log_return"]) < 0.25

    def test_calculate_returns_log(self):
        """Test log return calculation."""
        # Remove returns to test calculation
        data_no_returns = self.sample_data[["date", "price", "volume"]].copy()

        result = self.preprocessor._calculate_returns(data_no_returns)

        # Should have log returns
        assert "log_return" in result.columns

        # First return should be NaN (no previous price)
        assert pd.isna(result["log_return"].iloc[0])

        # Subsequent returns should be calculated correctly
        expected_return_1 = np.log(result["price"].iloc[1] / result["price"].iloc[0])
        assert np.isclose(result["log_return"].iloc[1], expected_return_1, rtol=1e-10)

    def test_calculate_returns_simple(self):
        """Test simple return calculation."""
        preprocessing_config = PreprocessingConfig(return_method="simple")
        preprocessor = DataPreprocessor(preprocessing_config=preprocessing_config)

        data_no_returns = self.sample_data[["date", "price", "volume"]].copy()

        result = preprocessor._calculate_returns(data_no_returns)

        # Should have both simple and log returns
        assert "simple_return" in result.columns
        assert "log_return" in result.columns

        # Test calculation
        expected_simple_return_1 = (
            result["price"].iloc[1] / result["price"].iloc[0]
        ) - 1
        assert np.isclose(
            result["simple_return"].iloc[1], expected_simple_return_1, rtol=1e-10
        )

    def test_calculate_volatility_features(self):
        """Test volatility feature calculation."""
        result = self.preprocessor._calculate_volatility_features(
            self.sample_data.copy()
        )

        # Should have volatility columns
        assert "volatility" in result.columns
        assert "abs_return" in result.columns
        assert "avg_abs_return" in result.columns

        # First few values should be NaN due to rolling window
        window = self.preprocessor.preprocessing_config.volatility_window
        assert pd.isna(
            result["volatility"].iloc[window - 2]
        )  # Just before window is full
        assert not pd.isna(result["volatility"].iloc[window])  # When window is full

    def test_apply_smoothing(self):
        """Test data smoothing."""
        preprocessing_config = PreprocessingConfig(
            apply_smoothing=True, smoothing_window=5
        )
        preprocessor = DataPreprocessor(preprocessing_config=preprocessing_config)

        result = preprocessor._apply_smoothing(self.sample_data.copy())

        # Should have smoothed columns
        assert "price_smoothed" in result.columns
        assert "log_return_smoothed" in result.columns

        # Smoothed values should be less volatile than original
        original_std = self.sample_data["log_return"].std()
        smoothed_std = result["log_return_smoothed"].dropna().std()
        assert smoothed_std < original_std

    def test_process_multiple_series(self):
        """Test processing multiple time series."""
        multi_data = {
            "AAPL": create_sample_stock_data(n_days=50, price_start=150),
            "GOOGL": create_sample_stock_data(n_days=50, price_start=2500),
            "MSFT": create_sample_stock_data(n_days=50, price_start=300),
        }

        result = self.preprocessor.process_multiple_series(multi_data)

        # Should have same keys
        assert set(result.keys()) == set(multi_data.keys())

        # Each should be processed
        for ticker, data in result.items():
            assert not data.empty
            assert "volatility" in data.columns
            assert "log_return" in data.columns

    def test_align_timestamps(self):
        """Test timestamp alignment."""
        preprocessing_config = PreprocessingConfig(align_timestamps=True)
        preprocessor = DataPreprocessor(preprocessing_config=preprocessing_config)

        # Create data with different date ranges
        data1 = create_sample_stock_data(
            n_days=20, start_date=datetime(2023, 1, 1), price_start=100
        )
        data2 = create_sample_stock_data(
            n_days=15, start_date=datetime(2023, 1, 8), price_start=200  # Start later
        )

        multi_data = {"STOCK1": data1, "STOCK2": data2}

        result = preprocessor._align_timestamps(multi_data)

        # Should have same number of rows (aligned to common dates)
        lengths = [len(df) for df in result.values()]
        assert len(set(lengths)) == 1  # All same length

        # Should have overlapping date range
        for data in result.values():
            assert "date" in data.columns

    def test_find_consecutive_missing(self):
        """Test finding consecutive missing values."""
        series = pd.Series([1, 2, np.nan, np.nan, np.nan, 6, 7, np.nan, 9])

        consecutive = self.preprocessor._find_consecutive_missing(series)

        # Should find 3 consecutive missing values
        assert consecutive == 3

    def test_validate_processed_data_success(self):
        """Test validation of good processed data."""
        good_data = create_sample_stock_data(n_days=100)

        # Should not raise any exceptions
        self.preprocessor._validate_processed_data(good_data)

    def test_validate_processed_data_empty(self):
        """Test validation of empty processed data."""
        empty_data = pd.DataFrame()

        with pytest.raises(DataQualityError, match="Processed data is empty"):
            self.preprocessor._validate_processed_data(empty_data)

    def test_validate_processed_data_infinite_values(self):
        """Test validation with infinite values."""
        data_with_inf = self.sample_data.copy()
        data_with_inf.loc[10, "log_return"] = np.inf

        with pytest.raises(DataQualityError, match="Infinite values found"):
            self.preprocessor._validate_processed_data(data_with_inf)

    def test_validate_processed_data_no_returns(self):
        """Test validation with no valid returns."""
        data_no_returns = self.sample_data[["date", "price"]].copy()
        data_no_returns["log_return"] = np.nan

        with pytest.raises(DataQualityError, match="No valid returns calculated"):
            self.preprocessor._validate_processed_data(data_no_returns)

    def test_get_data_summary(self):
        """Test data summary generation."""
        summary = self.preprocessor.get_data_summary(self.sample_data)

        # Should have basic info
        assert "n_observations" in summary
        assert "columns" in summary
        assert "date_range" in summary
        assert "return_stats" in summary

        # Check values
        assert summary["n_observations"] == len(self.sample_data)
        assert "date" in summary["columns"]
        assert "mean" in summary["return_stats"]
        assert "std" in summary["return_stats"]

    def test_different_config_scenarios(self):
        """Test preprocessing with different configuration scenarios."""
        config_variations = get_test_config_variations()

        for scenario_name, configs in config_variations.items():
            preprocessor = DataPreprocessor(
                preprocessing_config=configs["preprocessing_config"],
                validation_config=configs["validation_config"],
            )

            # Should be able to process data with any valid config
            result = preprocessor.process_data(self.sample_data.copy())
            assert not result.empty, f"Failed with {scenario_name} config"

    def test_process_data_with_volume(self):
        """Test processing data that includes volume."""
        data_with_volume = create_sample_stock_data(n_days=50, add_volume=True)

        result = self.preprocessor.process_data(data_with_volume)

        # Volume should be preserved and processed
        assert "volume" in result.columns
        assert not result["volume"].isnull().all()

    def test_process_data_missing_date_column(self):
        """Test processing data without date column."""
        data_no_date = self.sample_data[["price", "log_return", "volume"]].copy()

        # Should still work, just with warning about limited temporal analysis
        result = self.preprocessor.process_data(data_no_date)

        assert not result.empty
        assert "volatility" in result.columns

    def test_outlier_detection_edge_cases(self):
        """Test outlier detection with edge cases."""
        # All same values (no variance)
        constant_returns = pd.Series([0.01] * 100)
        outliers = self.preprocessor._detect_outliers(constant_returns)
        assert outliers.sum() == 0  # No outliers in constant series

        # Very small series
        small_returns = pd.Series([0.01, 0.02])
        outliers = self.preprocessor._detect_outliers(small_returns)
        # Should handle gracefully without errors
        assert len(outliers) == 2
