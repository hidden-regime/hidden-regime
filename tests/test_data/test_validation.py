"""
Unit tests for DataValidator class.

Tests data validation functionality including quality checks,
anomaly detection, and validation reporting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hidden_regime.data.validation import DataValidator, ValidationResult
from hidden_regime.config.settings import ValidationConfig
from hidden_regime.utils.exceptions import ValidationError
from tests.fixtures.sample_data import (
    create_sample_stock_data,
    create_invalid_stock_data,
    get_test_config_variations,
)


class TestDataValidator:
    """Test suite for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.good_data = create_sample_stock_data(n_days=100, add_volume=True)
        self.invalid_data_scenarios = create_invalid_stock_data()

    def test_init_default_config(self):
        """Test DataValidator initialization with default configuration."""
        validator = DataValidator()
        assert validator.config.outlier_method == "iqr"
        assert validator.config.max_daily_return == 0.5
        assert validator.config.min_price == 0.01

    def test_init_custom_config(self):
        """Test DataValidator initialization with custom configuration."""
        custom_config = ValidationConfig(
            outlier_method="zscore",
            outlier_threshold=2.5,
            max_daily_return=0.2,
            min_price=1.0,
        )
        validator = DataValidator(config=custom_config)
        assert validator.config.outlier_method == "zscore"
        assert validator.config.outlier_threshold == 2.5
        assert validator.config.max_daily_return == 0.2
        assert validator.config.min_price == 1.0

    def test_validate_data_success(self):
        """Test validation of good data."""
        result = self.validator.validate_data(self.good_data, ticker="AAPL")

        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert len(result.issues) == 0
        assert result.quality_score > 0.8  # Should have high quality score
        assert "n_observations" in result.metrics

    def test_validate_ticker_format_valid(self):
        """Test ticker format validation with valid tickers."""
        valid_tickers = ["AAPL", "GOOGL", "SPY", "BRK-B", "TSM", "NVDA"]

        for ticker in valid_tickers:
            assert self.validator.validate_ticker_format(ticker) == True

    def test_validate_ticker_format_invalid(self):
        """Test ticker format validation with invalid tickers."""
        invalid_tickers = ["", "VERYLONGTICKERNAME", "ticker with spaces", 123, None]

        for ticker in invalid_tickers:
            assert self.validator.validate_ticker_format(ticker) == False

    def test_validate_date_range_valid(self):
        """Test date range validation with valid ranges."""
        is_valid, issues = self.validator.validate_date_range(
            "2023-01-01", "2023-12-31"
        )
        assert is_valid == True
        assert len(issues) == 0

        # Test with datetime objects
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2023, 12, 31)
        is_valid, issues = self.validator.validate_date_range(start_dt, end_dt)
        assert is_valid == True
        assert len(issues) == 0

    def test_validate_date_range_invalid(self):
        """Test date range validation with invalid ranges."""
        # Start after end
        is_valid, issues = self.validator.validate_date_range(
            "2023-12-31", "2023-01-01"
        )
        assert is_valid == False
        assert any("Start date must be before end date" in issue for issue in issues)

        # Future end date
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        is_valid, issues = self.validator.validate_date_range("2023-01-01", future_date)
        assert is_valid == False
        assert any("cannot be in the future" in issue for issue in issues)

        # Very short range
        is_valid, issues = self.validator.validate_date_range(
            "2023-01-01", "2023-01-03"
        )
        assert is_valid == False
        assert any("Date range too short" in issue for issue in issues)

    def test_validate_structure_good(self):
        """Test structure validation with good data."""
        issues = self.validator._validate_structure(self.good_data)
        assert len(issues) == 0

    def test_validate_structure_empty(self):
        """Test structure validation with empty data."""
        empty_data = pd.DataFrame()
        issues = self.validator._validate_structure(empty_data)
        assert len(issues) > 0
        assert any("DataFrame is empty" in issue for issue in issues)

    def test_validate_structure_missing_required_columns(self):
        """Test structure validation with missing required columns."""
        data_missing_price = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=10), "volume": [1000] * 10}
        )
        issues = self.validator._validate_structure(data_missing_price)
        assert len(issues) > 0
        assert any("Required column 'price' is missing" in issue for issue in issues)

    def test_validate_dates_good(self):
        """Test date validation with good dates."""
        issues, warnings, metrics = self.validator._validate_dates(self.good_data)

        assert len(issues) == 0
        assert "n_dates" in metrics
        assert "date_range_days" in metrics
        assert metrics["n_dates"] > 0

    def test_validate_dates_missing_column(self):
        """Test date validation without date column."""
        data_no_date = self.good_data.drop(columns=["date"])
        issues, warnings, metrics = self.validator._validate_dates(data_no_date)

        assert len(issues) == 0  # Not an error, just a warning
        assert len(warnings) > 0
        assert any("No date column found" in warning for warning in warnings)

    def test_validate_dates_duplicates(self):
        """Test date validation with duplicate dates."""
        data_with_duplicates = self.good_data.copy()
        data_with_duplicates.loc[5, "date"] = data_with_duplicates.loc[4, "date"]

        issues, warnings, metrics = self.validator._validate_dates(data_with_duplicates)

        assert len(issues) > 0
        assert any("duplicate dates" in issue for issue in issues)
        assert metrics["duplicate_dates"] > 0

    def test_validate_prices_good(self):
        """Test price validation with good prices."""
        issues, warnings, metrics = self.validator._validate_prices(self.good_data)

        assert len(issues) == 0
        assert "price_min" in metrics
        assert "price_max" in metrics
        assert metrics["price_min"] > 0
        assert metrics["invalid_prices"] == 0

    def test_validate_prices_negative(self):
        """Test price validation with negative prices."""
        data_with_negative = self.invalid_data_scenarios["negative_prices"]
        issues, warnings, metrics = self.validator._validate_prices(data_with_negative)

        assert len(issues) > 0
        assert any("non-positive prices" in issue for issue in issues)
        assert metrics["invalid_prices"] > 0

    def test_validate_prices_missing_column(self):
        """Test price validation without price column."""
        data_no_price = self.good_data.drop(columns=["price"])
        issues, warnings, metrics = self.validator._validate_prices(data_no_price)

        assert len(issues) > 0
        assert any("Price column is missing" in issue for issue in issues)

    def test_validate_returns_good(self):
        """Test return validation with good returns."""
        issues, warnings, metrics = self.validator._validate_returns(self.good_data)

        assert len(issues) == 0
        assert "return_mean" in metrics
        assert "return_std" in metrics
        assert "return_skewness" in metrics
        assert metrics["infinite_returns"] == 0

    def test_validate_returns_extreme(self):
        """Test return validation with extreme returns."""
        data_with_extreme = self.invalid_data_scenarios["extreme_outliers"]
        issues, warnings, metrics = self.validator._validate_returns(data_with_extreme)

        # Extreme returns should generate warnings
        assert len(warnings) > 0
        assert metrics["extreme_returns"] > 0

    def test_validate_returns_infinite(self):
        """Test return validation with infinite returns."""
        data_with_infinite = self.invalid_data_scenarios["infinite_values"]
        issues, warnings, metrics = self.validator._validate_returns(data_with_infinite)

        assert len(issues) > 0
        assert any("infinite returns" in issue for issue in issues)
        assert metrics["infinite_returns"] > 0

    def test_validate_missing_data_good(self):
        """Test missing data validation with good data."""
        issues, warnings, metrics = self.validator._validate_missing_data(
            self.good_data
        )

        assert len(issues) == 0
        assert metrics["missing_percentage"] < 0.01  # Should be very low
        assert metrics["total_missing"] >= 0

    def test_validate_missing_data_excessive(self):
        """Test missing data validation with excessive missing data."""
        data_with_missing = self.invalid_data_scenarios["all_missing"]
        issues, warnings, metrics = self.validator._validate_missing_data(
            data_with_missing
        )

        assert len(issues) > 0
        assert any("High percentage of missing data" in issue for issue in issues)
        assert metrics["missing_percentage"] > 0.5

    def test_detect_outliers_good_data(self):
        """Test outlier detection with normal data."""
        warnings, metrics = self.validator._detect_outliers(self.good_data)

        # Should have some outliers but not excessive
        assert metrics["outlier_percentage"] < 0.15  # Less than 15%
        assert "n_outliers" in metrics

    def test_detect_outliers_many_outliers(self):
        """Test outlier detection with many outliers."""
        data_with_outliers = self.good_data.copy()

        # Add many extreme returns
        extreme_indices = np.random.choice(len(data_with_outliers), 20, replace=False)
        data_with_outliers.loc[extreme_indices, "log_return"] = np.random.uniform(
            -0.2, 0.2, 20
        )

        warnings, metrics = self.validator._detect_outliers(data_with_outliers)

        assert len(warnings) > 0
        assert metrics["outlier_percentage"] > 0.1  # Should detect high percentage

    def test_validate_volume_good(self):
        """Test volume validation with good volume data."""
        warnings, metrics = self.validator._validate_volume(self.good_data)

        assert len(warnings) == 0 or len(warnings) <= 1  # Might have minor warnings
        assert metrics["negative_volume_days"] == 0
        assert metrics["volume_median"] > 0

    def test_validate_volume_negative(self):
        """Test volume validation with negative volume."""
        data_with_negative_volume = self.good_data.copy()
        data_with_negative_volume.loc[5:7, "volume"] = [-1000, -500, -2000]

        warnings, metrics = self.validator._validate_volume(data_with_negative_volume)

        assert len(warnings) > 0
        assert any("negative volume" in warning for warning in warnings)
        assert metrics["negative_volume_days"] > 0

    def test_validate_volume_zero_volume(self):
        """Test volume validation with many zero volume days."""
        data_with_zero_volume = self.good_data.copy()
        data_with_zero_volume.loc[:20, "volume"] = 0  # 21% zero volume

        warnings, metrics = self.validator._validate_volume(data_with_zero_volume)

        assert len(warnings) > 0
        assert any("zero volume" in warning for warning in warnings)
        assert metrics["zero_volume_days"] > 15

    def test_generate_recommendations_good_data(self):
        """Test recommendation generation for good data."""
        recommendations = self.validator._generate_recommendations(
            self.good_data, [], [], {"return_kurtosis": 2.0}
        )

        # Should have few or no recommendations for good data
        assert len(recommendations) <= 2

    def test_generate_recommendations_problematic_data(self):
        """Test recommendation generation for problematic data."""
        issues = ["High percentage of missing data: 15%", "Invalid price values found"]
        warnings = ["High percentage of outliers: 12%", "High volatility detected"]
        metrics = {"return_kurtosis": 5.0, "missing_percentage": 0.15}

        recommendations = self.validator._generate_recommendations(
            self.good_data, issues, warnings, metrics
        )

        # Should have multiple recommendations
        assert len(recommendations) > 2
        assert any("missing" in rec.lower() for rec in recommendations)
        assert any("outlier" in rec.lower() for rec in recommendations)
        assert any("kurtosis" in rec.lower() for rec in recommendations)

    def test_calculate_quality_score_good_data(self):
        """Test quality score calculation for good data."""
        score = self.validator._calculate_quality_score(
            [], [], {"missing_percentage": 0.01, "outlier_percentage": 0.02}
        )

        assert score > 0.8  # Should have high score
        assert score <= 1.0

    def test_calculate_quality_score_bad_data(self):
        """Test quality score calculation for bad data."""
        issues = ["Missing data", "Invalid prices", "Extreme returns"]
        warnings = ["Outliers", "High volatility", "Date gaps"]
        metrics = {"missing_percentage": 0.2, "outlier_percentage": 0.15}

        score = self.validator._calculate_quality_score(issues, warnings, metrics)

        assert score < 0.5  # Should have low score
        assert score >= 0.0

    def test_find_max_consecutive_missing(self):
        """Test finding maximum consecutive missing values."""
        series_with_gaps = pd.Series(
            [1, 2, np.nan, np.nan, np.nan, 6, 7, np.nan, 9, np.nan]
        )

        max_consecutive = self.validator._find_max_consecutive_missing(series_with_gaps)

        assert max_consecutive == 3  # Three consecutive NaNs

    def test_find_max_consecutive_missing_no_missing(self):
        """Test consecutive missing detection with no missing values."""
        series_no_missing = pd.Series([1, 2, 3, 4, 5])

        max_consecutive = self.validator._find_max_consecutive_missing(
            series_no_missing
        )

        assert max_consecutive == 0

    def test_validate_data_comprehensive_invalid_scenarios(self):
        """Test validation with all invalid data scenarios."""
        for scenario_name, invalid_data in self.invalid_data_scenarios.items():
            if scenario_name == "empty":
                continue  # Skip empty data as it's handled separately

            result = self.validator.validate_data(
                invalid_data, ticker=f"TEST_{scenario_name}"
            )

            # All invalid scenarios should have issues or low quality scores
            assert (not result.is_valid) or (
                result.quality_score < 0.7
            ), f"Scenario {scenario_name} should have been flagged as problematic"

    def test_validation_with_different_configs(self):
        """Test validation with different configuration scenarios."""
        config_variations = get_test_config_variations()

        for scenario_name, configs in config_variations.items():
            validator = DataValidator(config=configs["validation_config"])

            result = validator.validate_data(self.good_data, ticker="AAPL")

            # All configs should be able to validate good data
            assert isinstance(result, ValidationResult)
            assert result.quality_score > 0.5, f"Failed with {scenario_name} config"

    def test_validation_result_structure(self):
        """Test that ValidationResult has all expected fields."""
        result = self.validator.validate_data(self.good_data, ticker="AAPL")

        # Check all required fields are present
        assert hasattr(result, "is_valid")
        assert hasattr(result, "issues")
        assert hasattr(result, "warnings")
        assert hasattr(result, "recommendations")
        assert hasattr(result, "quality_score")
        assert hasattr(result, "metrics")

        # Check types
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.issues, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.quality_score, float)
        assert isinstance(result.metrics, dict)

    def test_edge_case_very_small_dataset(self):
        """Test validation with very small dataset."""
        tiny_data = create_sample_stock_data(n_days=5)

        result = self.validator.validate_data(tiny_data, ticker="TINY")

        # Should handle small datasets gracefully
        assert isinstance(result, ValidationResult)
        # Might have warnings about insufficient data but shouldn't crash

    def test_edge_case_single_row(self):
        """Test validation with single row dataset."""
        single_row = self.good_data.iloc[:1].copy()

        result = self.validator.validate_data(single_row, ticker="SINGLE")

        # Should handle single row gracefully
        assert isinstance(result, ValidationResult)
        # Will likely have issues but shouldn't crash
