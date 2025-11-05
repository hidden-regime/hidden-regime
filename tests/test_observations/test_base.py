"""
Unit tests for base observation components.

Tests the observation generation functionality that transforms raw financial
data into features suitable for hidden Markov model training and inference.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.observations.base import BaseObservationGenerator
from hidden_regime.utils.exceptions import ValidationError


class TestBaseObservationGenerator:
    """Test cases for BaseObservationGenerator."""

    def create_observation_config(self, **kwargs):
        """Create observation configuration for testing."""
        from hidden_regime.config.observation import ObservationConfig

        defaults = {"generators": ["log_return"]}
        defaults.update(kwargs)
        return ObservationConfig(**defaults)

    @pytest.mark.unit
    def test_base_observation_generator_initialization(self):
        """Test BaseObservationGenerator initialization."""
        from hidden_regime.config.observation import ObservationConfig

        config = ObservationConfig(generators=["log_return"])
        obs_gen = BaseObservationGenerator(config)

        assert obs_gen.config is not None
        assert len(obs_gen.generators) == 1
        assert obs_gen.last_data is None
        assert obs_gen.last_observations is None

    @pytest.mark.unit
    def test_base_observation_generator_default_window_size(self):
        """Test default generator configuration."""
        from hidden_regime.config.observation import ObservationConfig

        config = ObservationConfig(generators=["log_return"])
        obs_gen = BaseObservationGenerator(config)
        assert obs_gen.config is not None
        assert len(obs_gen.generators) == 1

    @pytest.mark.unit
    def test_invalid_generator_config(self):
        """Test invalid generator configuration handling."""
        from hidden_regime.config.observation import ObservationConfig
        from hidden_regime.utils.exceptions import ConfigurationError

        # Test empty generators list
        with pytest.raises(
            ConfigurationError,
            match="At least one observation generator must be specified",
        ):
            config = ObservationConfig(generators=[])
            config.validate()

        # Test invalid generator type
        with pytest.raises(
            ConfigurationError, match="Generator 0 must be string or callable"
        ):
            config = ObservationConfig(generators=[123])  # Invalid type
            config.validate()

    @pytest.mark.unit
    def test_update_with_insufficient_data(self):
        """Test update with empty data."""
        from hidden_regime.config.observation import ObservationConfig
        from hidden_regime.utils.exceptions import ValidationError

        config = ObservationConfig(generators=["log_return"])
        obs_gen = BaseObservationGenerator(config)

        # Create empty data
        data = pd.DataFrame()

        with pytest.raises(ValidationError, match="Input data cannot be empty"):
            obs_gen.update(data)

    @pytest.mark.integration
    def test_update_with_valid_data(self):
        """Test update with valid data."""
        from hidden_regime.config.observation import ObservationConfig

        config = ObservationConfig(generators=["log_return"])
        obs_gen = BaseObservationGenerator(config)

        # Create sufficient data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 10),
                "high": np.random.uniform(100, 110, 10),
                "low": np.random.uniform(90, 100, 10),
                "close": np.random.uniform(95, 105, 10),
                "volume": np.random.randint(1000000, 5000000, 10),
            },
            index=dates,
        )

        result = obs_gen.update(data)

        # Should return basic log returns
        assert isinstance(result, pd.DataFrame)
        assert "log_return" in result.columns
        assert len(result) == len(data)  # Same length, first value will be NaN

        # Verify log return calculation (skip NaN values)
        expected_returns = np.log(data["close"] / data["close"].shift(1))
        # Compare non-NaN values
        mask = ~np.isnan(result["log_return"])
        np.testing.assert_array_almost_equal(
            result["log_return"][mask].values, expected_returns[mask].values
        )

    @pytest.mark.integration
    def test_calculate_log_returns(self):
        """Test log return calculation."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        prices = pd.Series([100, 101, 99, 102, 98], name="close")
        log_returns = obs_gen._calculate_log_returns(prices)

        expected = np.log(prices / prices.shift(1)).dropna()
        np.testing.assert_array_almost_equal(log_returns.values, expected.values)
        assert log_returns.name == "log_return"

    @pytest.mark.integration
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, 0.02])
        volatility = obs_gen._calculate_volatility(returns, window=3)

        assert isinstance(volatility, pd.Series)
        assert volatility.name == "volatility"
        assert len(volatility) == len(returns)

        # First few values should be NaN due to window
        assert pd.isna(volatility.iloc[0])
        assert pd.isna(volatility.iloc[1])

        # Later values should be calculated
        assert not pd.isna(volatility.iloc[-1])

    @pytest.mark.unit
    def test_add_feature_validation(self):
        """Test feature addition validation."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        # Valid feature
        obs_gen.add_feature("volatility", window=10)
        assert "volatility" in obs_gen.features

        # Duplicate feature
        with pytest.raises(ValueError, match="Feature 'volatility' already exists"):
            obs_gen.add_feature("volatility", window=5)

        # Invalid feature name
        with pytest.raises(ValueError, match="Unknown feature type"):
            obs_gen.add_feature("invalid_feature")

    @pytest.mark.unit
    def test_remove_feature(self):
        """Test feature removal."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        obs_gen.add_feature("volatility", window=10)
        assert "volatility" in obs_gen.features

        obs_gen.remove_feature("volatility")
        assert "volatility" not in obs_gen.features

        # Removing non-existent feature should not raise error
        obs_gen.remove_feature("non_existent")

    @pytest.mark.integration
    def test_generate_features_with_multiple_features(self):
        """Test feature generation with multiple features."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())
        obs_gen.add_feature("volatility", window=5)
        obs_gen.add_feature("momentum", window=3)

        # Create test data
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {"close": 100 + np.cumsum(np.random.normal(0, 1, 20))}, index=dates
        )

        result = obs_gen.update(data)

        assert isinstance(result, pd.DataFrame)
        assert "log_return" in result.columns
        assert "volatility" in result.columns
        assert "momentum" in result.columns

        # Check that all features have reasonable values
        assert not result["log_return"].isna().all()
        assert not result["volatility"].isna().all()
        assert not result["momentum"].isna().all()

    @pytest.mark.integration
    def test_caching_behavior(self):
        """Test caching of intermediate calculations."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        # Create test data
        data = pd.DataFrame(
            {"close": [100, 101, 99, 102, 98, 105, 103]},
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )

        # First call should populate cache
        result1 = obs_gen.update(data)
        assert len(obs_gen._cache) > 0

        # Second call with same data should use cache
        result2 = obs_gen.update(data)
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.integration
    def test_update_incremental_data(self):
        """Test incremental data updates."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        # Initial data
        initial_data = pd.DataFrame(
            {"close": [100, 101, 99, 102, 98, 105]},
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )

        result1 = obs_gen.update(initial_data)

        # Add new data point
        new_data = pd.DataFrame(
            {"close": [100, 101, 99, 102, 98, 105, 103]},
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )

        result2 = obs_gen.update(new_data)

        # Results should be consistent for overlapping period
        common_index = result1.index.intersection(result2.index)
        pd.testing.assert_frame_equal(
            result1.loc[common_index], result2.loc[common_index]
        )

        # New result should have one more row
        assert len(result2) == len(result1) + 1

    @pytest.mark.unit
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        # Create data with missing values
        data = pd.DataFrame(
            {"close": [100, np.nan, 99, 102, np.nan, 105]},
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )

        with pytest.raises(ValidationError, match="Data contains missing values"):
            obs_gen.update(data)

    @pytest.mark.unit
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())

        empty_data = pd.DataFrame()

        with pytest.raises(ValidationError, match="Input data cannot be empty"):
            obs_gen.update(empty_data)

    @pytest.mark.integration
    def test_plot_functionality(self):
        """Test plotting functionality."""
        obs_gen = BaseObservationGenerator(self.create_observation_config())
        obs_gen.add_feature("volatility", window=3)

        # Create and process data
        data = pd.DataFrame(
            {"close": 100 + np.cumsum(np.random.normal(0, 1, 10))},
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )

        observations = obs_gen.update(data)

        # Test plotting
        fig = obs_gen.plot(observations=observations)

        assert fig is not None
        assert len(fig.axes) >= 1  # Should have at least one subplot

        # Test with specific features
        fig = obs_gen.plot(observations=observations, features=["log_return"])
        assert fig is not None

        # Test with invalid features
        with pytest.raises(ValueError, match="Feature 'invalid' not found"):
            obs_gen.plot(observations=observations, features=["invalid"])

    @pytest.mark.integration
    def test_serialization_support(self):
        """Test pickle serialization support."""
        import pickle

        obs_gen = BaseObservationGenerator(self.create_observation_config())
        obs_gen.add_feature("volatility", window=5)

        # Serialize and deserialize
        serialized = pickle.dumps(obs_gen)
        deserialized = pickle.loads(serialized)

        assert deserialized.window_size == obs_gen.window_size
        assert deserialized.features == obs_gen.features

        # Test functionality after deserialization
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    99,
                    102,
                    98,
                    105,
                    103,
                    101,
                    99,
                    104,
                    102,
                    100,
                    98,
                    105,
                    107,
                    106,
                ]
            },
            index=pd.date_range("2024-01-01", periods=16, freq="D"),
        )

        result = deserialized.update(data)
        assert isinstance(result, pd.DataFrame)
        assert "log_return" in result.columns
        assert "volatility" in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
