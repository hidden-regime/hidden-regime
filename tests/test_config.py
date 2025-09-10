"""
Enhanced configuration testing for hidden-regime package.

Tests configuration classes, validation, serialization, inheritance,
edge cases, and interactions between different configuration types.
"""

import json
import os
import tempfile
from dataclasses import asdict, fields
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from hidden_regime import DataLoader, DataPreprocessor, DataValidator
from hidden_regime.config.settings import (
    DataConfig,
    PreprocessingConfig,
    ValidationConfig,
)


class TestDataConfig:
    """Test DataConfig class and its behavior."""

    def test_default_initialization(self):
        """Test DataConfig with default values."""
        config = DataConfig()

        # Check default values
        assert config.default_source == "yfinance"
        assert config.use_ohlc_average == True
        assert config.include_volume == True
        assert config.max_missing_data_pct == 0.05
        assert config.min_observations == 30
        assert config.cache_enabled == True
        assert config.cache_expiry_hours == 24
        assert config.requests_per_minute == 60
        assert config.retry_attempts == 3
        assert config.retry_delay_seconds == 1.0

    def test_custom_initialization(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            default_source="custom",
            use_ohlc_average=False,
            include_volume=False,
            max_missing_data_pct=0.1,
            min_observations=50,
            cache_enabled=False,
            cache_expiry_hours=12,
            requests_per_minute=30,
            retry_attempts=5,
            retry_delay_seconds=2.0,
        )

        assert config.default_source == "custom"
        assert config.use_ohlc_average == False
        assert config.include_volume == False
        assert config.max_missing_data_pct == 0.1
        assert config.min_observations == 50
        assert config.cache_enabled == False
        assert config.cache_expiry_hours == 12
        assert config.requests_per_minute == 30
        assert config.retry_attempts == 5
        assert config.retry_delay_seconds == 2.0

    def test_edge_case_values(self):
        """Test DataConfig with edge case values."""
        # Test minimum values
        min_config = DataConfig(
            max_missing_data_pct=0.0,
            min_observations=1,
            cache_expiry_hours=0,
            requests_per_minute=1,
            retry_attempts=0,
            retry_delay_seconds=0.0,
        )

        assert min_config.max_missing_data_pct == 0.0
        assert min_config.min_observations == 1
        assert min_config.retry_attempts == 0

        # Test maximum/extreme values
        max_config = DataConfig(
            max_missing_data_pct=1.0,  # 100% missing allowed
            min_observations=10000,
            cache_expiry_hours=8760,  # 1 year
            requests_per_minute=10000,
            retry_attempts=100,
            retry_delay_seconds=3600.0,  # 1 hour
        )

        assert max_config.max_missing_data_pct == 1.0
        assert max_config.min_observations == 10000
        assert max_config.cache_expiry_hours == 8760

    def test_serialization(self):
        """Test DataConfig serialization to dict/JSON."""
        config = DataConfig(
            use_ohlc_average=False, max_missing_data_pct=0.02, min_observations=100
        )

        # Test to dict
        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict["use_ohlc_average"] == False
        assert config_dict["max_missing_data_pct"] == 0.02
        assert config_dict["min_observations"] == 100

        # Test JSON serialization
        config_json = json.dumps(config_dict)
        assert isinstance(config_json, str)

        # Test deserialization
        restored_dict = json.loads(config_json)
        restored_config = DataConfig(**restored_dict)

        assert restored_config.use_ohlc_average == config.use_ohlc_average
        assert restored_config.max_missing_data_pct == config.max_missing_data_pct
        assert restored_config.min_observations == config.min_observations


class TestValidationConfig:
    """Test ValidationConfig class and its behavior."""

    def test_default_initialization(self):
        """Test ValidationConfig with default values."""
        config = ValidationConfig()

        assert config.outlier_method == "iqr"
        assert config.outlier_threshold == 3.0
        assert config.iqr_multiplier == 1.5
        assert config.min_price == 0.01
        assert config.max_daily_return == 0.5
        assert config.min_trading_days_per_month == 15
        assert config.max_consecutive_missing == 5
        assert config.interpolation_method == "linear"

    def test_outlier_method_validation(self):
        """Test validation of outlier detection methods."""
        valid_methods = ["iqr", "zscore", "isolation_forest"]

        for method in valid_methods:
            config = ValidationConfig(outlier_method=method)
            assert config.outlier_method == method

        # Test with custom method (should still accept, validation happens at runtime)
        custom_config = ValidationConfig(outlier_method="custom_method")
        assert custom_config.outlier_method == "custom_method"

    def test_threshold_edge_cases(self):
        """Test ValidationConfig with extreme threshold values."""
        # Very strict configuration
        strict_config = ValidationConfig(
            outlier_threshold=1.0,  # Very sensitive
            iqr_multiplier=0.5,  # Very tight IQR bounds
            min_price=1.0,  # High minimum price
            max_daily_return=0.01,  # 1% max daily return
            max_consecutive_missing=1,  # No consecutive missing allowed
        )

        assert strict_config.outlier_threshold == 1.0
        assert strict_config.max_daily_return == 0.01

        # Very lenient configuration
        lenient_config = ValidationConfig(
            outlier_threshold=10.0,  # Very insensitive
            iqr_multiplier=5.0,  # Very wide IQR bounds
            min_price=0.001,  # Very low minimum price
            max_daily_return=2.0,  # 200% max daily return
            max_consecutive_missing=100,  # Allow many consecutive missing
        )

        assert lenient_config.outlier_threshold == 10.0
        assert lenient_config.max_daily_return == 2.0

    def test_interpolation_method_options(self):
        """Test different interpolation method configurations."""
        methods = ["linear", "forward", "backward"]

        for method in methods:
            config = ValidationConfig(interpolation_method=method)
            assert config.interpolation_method == method

        # Test custom interpolation method
        custom_config = ValidationConfig(interpolation_method="spline")
        assert custom_config.interpolation_method == "spline"


class TestPreprocessingConfig:
    """Test PreprocessingConfig class and its behavior."""

    def test_default_initialization(self):
        """Test PreprocessingConfig with default values."""
        config = PreprocessingConfig()

        assert config.return_method == "log"
        assert config.apply_smoothing == False
        assert config.smoothing_window == 5
        assert config.calculate_volatility == True
        assert config.volatility_window == 20
        assert config.align_timestamps == True
        assert config.fill_method == "forward"

    def test_return_method_options(self):
        """Test different return calculation methods."""
        methods = ["log", "simple"]

        for method in methods:
            config = PreprocessingConfig(return_method=method)
            assert config.return_method == method

    def test_feature_engineering_combinations(self):
        """Test different feature engineering combinations."""
        # No features
        minimal_config = PreprocessingConfig(
            calculate_volatility=False, apply_smoothing=False
        )
        assert minimal_config.calculate_volatility == False
        assert minimal_config.apply_smoothing == False

        # All features
        full_config = PreprocessingConfig(
            calculate_volatility=True,
            apply_smoothing=True,
            volatility_window=30,
            smoothing_window=10,
        )
        assert full_config.calculate_volatility == True
        assert full_config.apply_smoothing == True
        assert full_config.volatility_window == 30
        assert full_config.smoothing_window == 10

    def test_window_size_edge_cases(self):
        """Test window size edge cases."""
        # Minimum window sizes
        min_config = PreprocessingConfig(volatility_window=1, smoothing_window=1)
        assert min_config.volatility_window == 1
        assert min_config.smoothing_window == 1

        # Large window sizes
        large_config = PreprocessingConfig(
            volatility_window=252,  # 1 year of daily data
            smoothing_window=50,  # ~2.5 months
        )
        assert large_config.volatility_window == 252
        assert large_config.smoothing_window == 50

    def test_alignment_options(self):
        """Test timestamp alignment and fill method options."""
        fill_methods = ["forward", "backward", "interpolate"]

        for method in fill_methods:
            config = PreprocessingConfig(fill_method=method)
            assert config.fill_method == method

        # Test alignment disabled
        no_align_config = PreprocessingConfig(align_timestamps=False)
        assert no_align_config.align_timestamps == False


class TestConfigurationInteractions:
    """Test interactions between different configuration types."""

    def test_data_loader_config_integration(self):
        """Test DataLoader with various DataConfig options."""
        from unittest.mock import patch

        from tests.fixtures.sample_data import MockYFinanceTicker

        # Test strict data quality requirements
        strict_config = DataConfig(
            max_missing_data_pct=0.01,  # 1% max missing
            min_observations=100,  # High minimum requirement
        )

        loader = DataLoader(strict_config)
        assert loader.config.max_missing_data_pct == 0.01
        assert loader.config.min_observations == 100

        # Test lenient data quality requirements
        lenient_config = DataConfig(
            max_missing_data_pct=0.2,  # 20% max missing
            min_observations=5,  # Low minimum requirement
        )

        lenient_loader = DataLoader(lenient_config)
        assert lenient_loader.config.max_missing_data_pct == 0.2
        assert lenient_loader.config.min_observations == 5

    def test_validator_config_combinations(self):
        """Test DataValidator with different ValidationConfig combinations."""
        # High-sensitivity configuration
        sensitive_config = ValidationConfig(
            outlier_method="zscore",
            outlier_threshold=2.0,  # More sensitive
            max_daily_return=0.1,  # Stricter return limits
            max_consecutive_missing=2,  # Less missing data allowed
        )

        validator = DataValidator(sensitive_config)
        assert validator.config.outlier_threshold == 2.0
        assert validator.config.max_daily_return == 0.1

        # Low-sensitivity configuration
        tolerant_config = ValidationConfig(
            outlier_method="iqr",
            outlier_threshold=5.0,  # Less sensitive
            max_daily_return=1.0,  # More lenient return limits
            max_consecutive_missing=20,  # More missing data allowed
        )

        tolerant_validator = DataValidator(tolerant_config)
        assert tolerant_validator.config.outlier_threshold == 5.0
        assert tolerant_validator.config.max_daily_return == 1.0

    def test_preprocessor_dual_config(self):
        """Test DataPreprocessor with both preprocessing and validation configs."""
        preprocessing_config = PreprocessingConfig(
            return_method="simple",
            calculate_volatility=True,
            volatility_window=15,
            apply_smoothing=True,
        )

        validation_config = ValidationConfig(
            outlier_method="iqr", outlier_threshold=2.5, interpolation_method="forward"
        )

        preprocessor = DataPreprocessor(
            preprocessing_config=preprocessing_config,
            validation_config=validation_config,
        )

        assert preprocessor.preprocessing_config.return_method == "simple"
        assert preprocessor.preprocessing_config.volatility_window == 15
        assert preprocessor.validation_config.outlier_method == "iqr"
        assert preprocessor.validation_config.interpolation_method == "forward"

    def test_configuration_override_behavior(self):
        """Test how method parameters override configuration defaults."""
        from unittest.mock import patch

        from tests.fixtures.sample_data import MockYFinanceTicker

        # Create loader with OHLC average disabled
        config = DataConfig(use_ohlc_average=False)
        loader = DataLoader(config)

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("CONFIG_OVERRIDE_TEST")
            mock_ticker_class.return_value = mock_ticker

            # Load with method-level override (should use OHLC average)
            data = loader.load_stock_data(
                "CONFIG_OVERRIDE_TEST",
                "2024-01-01",
                "2024-03-31",
                use_ohlc_avg=True,  # Override config setting
            )

            # Should work without errors
            assert isinstance(data, pd.DataFrame)
            assert not data.empty


class TestConfigurationPersistence:
    """Test configuration serialization, persistence, and restoration."""

    def test_config_to_file_persistence(self):
        """Test saving and loading configuration to/from files."""
        # Create configurations
        data_config = DataConfig(
            use_ohlc_average=False, max_missing_data_pct=0.03, min_observations=75
        )

        validation_config = ValidationConfig(
            outlier_method="zscore", outlier_threshold=2.5, max_daily_return=0.3
        )

        preprocessing_config = PreprocessingConfig(
            return_method="simple", calculate_volatility=True, volatility_window=25
        )

        # Save to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each config
            data_config_path = os.path.join(temp_dir, "data_config.json")
            validation_config_path = os.path.join(temp_dir, "validation_config.json")
            preprocessing_config_path = os.path.join(
                temp_dir, "preprocessing_config.json"
            )

            # Write configs
            with open(data_config_path, "w") as f:
                json.dump(asdict(data_config), f)

            with open(validation_config_path, "w") as f:
                json.dump(asdict(validation_config), f)

            with open(preprocessing_config_path, "w") as f:
                json.dump(asdict(preprocessing_config), f)

            # Read and restore configs
            with open(data_config_path, "r") as f:
                restored_data_config = DataConfig(**json.load(f))

            with open(validation_config_path, "r") as f:
                restored_validation_config = ValidationConfig(**json.load(f))

            with open(preprocessing_config_path, "r") as f:
                restored_preprocessing_config = PreprocessingConfig(**json.load(f))

            # Verify restoration
            assert restored_data_config.use_ohlc_average == data_config.use_ohlc_average
            assert (
                restored_data_config.max_missing_data_pct
                == data_config.max_missing_data_pct
            )

            assert (
                restored_validation_config.outlier_method
                == validation_config.outlier_method
            )
            assert (
                restored_validation_config.outlier_threshold
                == validation_config.outlier_threshold
            )

            assert (
                restored_preprocessing_config.return_method
                == preprocessing_config.return_method
            )
            assert (
                restored_preprocessing_config.volatility_window
                == preprocessing_config.volatility_window
            )

    def test_config_inheritance_patterns(self):
        """Test configuration inheritance and modification patterns."""
        # Base configuration
        base_config = ValidationConfig(
            outlier_method="iqr", outlier_threshold=3.0, max_daily_return=0.5
        )

        # Create derived configurations by modifying specific parameters
        conservative_config = ValidationConfig(
            **{**asdict(base_config), "outlier_threshold": 2.0, "max_daily_return": 0.2}
        )

        aggressive_config = ValidationConfig(
            **{**asdict(base_config), "outlier_threshold": 5.0, "max_daily_return": 1.0}
        )

        # Verify inheritance worked correctly
        assert (
            conservative_config.outlier_method == base_config.outlier_method
        )  # Inherited
        assert conservative_config.outlier_threshold == 2.0  # Modified
        assert conservative_config.max_daily_return == 0.2  # Modified

        assert (
            aggressive_config.outlier_method == base_config.outlier_method
        )  # Inherited
        assert aggressive_config.outlier_threshold == 5.0  # Modified
        assert aggressive_config.max_daily_return == 1.0  # Modified

    def test_config_validation_edge_cases(self):
        """Test configuration validation with invalid or unusual values."""
        # Test negative values where they don't make sense
        unusual_config = DataConfig(
            max_missing_data_pct=-0.1,  # Negative percentage
            min_observations=-5,  # Negative minimum
            cache_expiry_hours=-1,  # Negative expiry
            requests_per_minute=-10,  # Negative rate limit
            retry_attempts=-1,  # Negative retries
        )

        # These should be accepted by the dataclass (validation happens at runtime)
        assert unusual_config.max_missing_data_pct == -0.1
        assert unusual_config.min_observations == -5

        # Test extremely large values
        extreme_config = ValidationConfig(
            outlier_threshold=1000.0,  # Extremely high threshold
            max_daily_return=100.0,  # 10000% daily return
            max_consecutive_missing=1000000,  # Million consecutive missing
        )

        assert extreme_config.outlier_threshold == 1000.0
        assert extreme_config.max_daily_return == 100.0


class TestConfigurationDocumentation:
    """Test configuration field documentation and introspection."""

    def test_dataclass_field_introspection(self):
        """Test that configuration fields can be introspected properly."""
        # Get fields from each config class
        data_fields = fields(DataConfig)
        validation_fields = fields(ValidationConfig)
        preprocessing_fields = fields(PreprocessingConfig)

        # Verify expected fields exist
        data_field_names = {f.name for f in data_fields}
        expected_data_fields = {
            "default_source",
            "use_ohlc_average",
            "include_volume",
            "max_missing_data_pct",
            "min_observations",
            "cache_enabled",
            "cache_expiry_hours",
            "requests_per_minute",
            "retry_attempts",
            "retry_delay_seconds",
        }
        assert expected_data_fields.issubset(data_field_names)

        validation_field_names = {f.name for f in validation_fields}
        expected_validation_fields = {
            "outlier_method",
            "outlier_threshold",
            "iqr_multiplier",
            "min_price",
            "max_daily_return",
            "min_trading_days_per_month",
            "max_consecutive_missing",
            "interpolation_method",
        }
        assert expected_validation_fields.issubset(validation_field_names)

        preprocessing_field_names = {f.name for f in preprocessing_fields}
        expected_preprocessing_fields = {
            "return_method",
            "apply_smoothing",
            "smoothing_window",
            "calculate_volatility",
            "volatility_window",
            "align_timestamps",
            "fill_method",
        }
        assert expected_preprocessing_fields.issubset(preprocessing_field_names)

    def test_config_type_annotations(self):
        """Test that configuration fields have proper type annotations."""
        # Check DataConfig field types
        data_fields = {f.name: f.type for f in fields(DataConfig)}

        assert data_fields["default_source"] == str
        assert data_fields["use_ohlc_average"] == bool
        assert data_fields["max_missing_data_pct"] == float
        assert data_fields["min_observations"] == int

        # Check ValidationConfig field types
        validation_fields = {f.name: f.type for f in fields(ValidationConfig)}

        assert validation_fields["outlier_method"] == str
        assert validation_fields["outlier_threshold"] == float
        assert validation_fields["max_consecutive_missing"] == int

        # Check PreprocessingConfig field types
        preprocessing_fields = {f.name: f.type for f in fields(PreprocessingConfig)}

        assert preprocessing_fields["return_method"] == str
        assert preprocessing_fields["apply_smoothing"] == bool
        assert preprocessing_fields["volatility_window"] == int

    def test_config_default_values_consistency(self):
        """Test that default values are consistent and reasonable."""
        # Test that default configurations can be created without issues
        data_config = DataConfig()
        validation_config = ValidationConfig()
        preprocessing_config = PreprocessingConfig()

        # Test that defaults make sense
        assert 0.0 < data_config.max_missing_data_pct < 1.0
        assert data_config.min_observations > 0
        assert data_config.cache_expiry_hours > 0
        assert data_config.requests_per_minute > 0
        assert data_config.retry_attempts >= 0

        assert validation_config.outlier_threshold > 0
        assert validation_config.min_price > 0
        assert validation_config.max_daily_return > 0
        assert validation_config.max_consecutive_missing > 0

        assert preprocessing_config.volatility_window > 0
        assert preprocessing_config.smoothing_window > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
