"""
Unit tests for HMMConfig class.

Tests configuration validation, factory methods, and integration with data validation.
"""

import numpy as np
import pytest

from hidden_regime.models.config import HMMConfig


class TestHMMConfig:
    """Test cases for HMMConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = HMMConfig()

        assert config.n_states == 3
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.regularization == 1e-6
        assert config.initialization_method == "random"
        assert config.random_seed is None
        assert config.min_regime_duration == 2
        assert config.min_variance == 1e-8
        assert config.check_convergence_every == 5
        assert config.early_stopping is True
        assert config.log_likelihood_threshold == -1e10

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = HMMConfig(
            n_states=4,
            max_iterations=200,
            tolerance=1e-7,
            regularization=1e-5,
            initialization_method="kmeans",
            random_seed=42,
            min_regime_duration=3,
            early_stopping=False,
        )

        assert config.n_states == 4
        assert config.max_iterations == 200
        assert config.tolerance == 1e-7
        assert config.regularization == 1e-5
        assert config.initialization_method == "kmeans"
        assert config.random_seed == 42
        assert config.min_regime_duration == 3
        assert config.early_stopping is False

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test n_states validation
        with pytest.raises(ValueError, match="Number of states must be at least 2"):
            HMMConfig(n_states=1)

        with pytest.raises(ValueError, match="Number of states should not exceed 10"):
            HMMConfig(n_states=11)

        # Test max_iterations validation
        with pytest.raises(ValueError, match="Maximum iterations must be positive"):
            HMMConfig(max_iterations=0)

        # Test tolerance validation
        with pytest.raises(ValueError, match="Tolerance must be between 0 and 1"):
            HMMConfig(tolerance=0)

        with pytest.raises(ValueError, match="Tolerance must be between 0 and 1"):
            HMMConfig(tolerance=1.5)

        # Test regularization validation
        with pytest.raises(ValueError, match="Regularization must be non-negative"):
            HMMConfig(regularization=-0.1)

        # Test min_regime_duration validation
        with pytest.raises(
            ValueError, match="Minimum regime duration must be at least 1"
        ):
            HMMConfig(min_regime_duration=0)

        # Test min_variance validation
        with pytest.raises(ValueError, match="Minimum variance must be positive"):
            HMMConfig(min_variance=0)

        # Test check_convergence_every validation
        with pytest.raises(
            ValueError, match="Convergence check frequency must be positive"
        ):
            HMMConfig(check_convergence_every=0)

    def test_for_market_data_factory(self):
        """Test for_market_data factory method."""
        # Default market data config
        config = HMMConfig.for_market_data()

        assert config.n_states == 3
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.regularization == 1e-6
        assert config.initialization_method == "random"
        assert config.min_regime_duration == 2
        assert config.early_stopping is True

        # Conservative market data config
        conservative_config = HMMConfig.for_market_data(conservative=True)

        assert conservative_config.n_states == 3
        assert conservative_config.max_iterations == 200
        assert conservative_config.tolerance == 1e-7
        assert conservative_config.regularization == 1e-5
        assert conservative_config.initialization_method == "kmeans"
        assert conservative_config.min_regime_duration == 3
        assert conservative_config.early_stopping is True

    def test_for_high_frequency_factory(self):
        """Test for_high_frequency factory method."""
        config = HMMConfig.for_high_frequency()

        assert config.n_states == 4
        assert config.max_iterations == 50
        assert config.tolerance == 1e-5
        assert config.regularization == 1e-5
        assert config.initialization_method == "kmeans"
        assert config.min_regime_duration == 1
        assert config.early_stopping is True

    def test_validate_for_data(self):
        """Test data validation method."""
        config = HMMConfig(n_states=3)

        # Test sufficient data
        config.validate_for_data(100)  # Should not raise

        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data for 3 states"):
            config.validate_for_data(20)  # 3 states * 10 min per state = 30 required

        # Test warning for limited data (should not raise but trigger warning)
        with pytest.warns(UserWarning, match="Limited data"):
            config.validate_for_data(50)  # 3 states * 20 warning threshold = 60

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = HMMConfig(n_states=3, max_iterations=100, tolerance=1e-6)
        config2 = HMMConfig(n_states=3, max_iterations=100, tolerance=1e-6)
        config3 = HMMConfig(n_states=4, max_iterations=100, tolerance=1e-6)

        assert config1.__dict__ == config2.__dict__
        assert config1.__dict__ != config3.__dict__

    def test_config_serialization(self):
        """Test that configuration can be serialized to dict."""
        config = HMMConfig(
            n_states=4, max_iterations=200, tolerance=1e-7, random_seed=42
        )

        config_dict = config.__dict__

        # Verify all expected keys are present
        expected_keys = {
            "n_states",
            "max_iterations",
            "tolerance",
            "regularization",
            "initialization_method",
            "random_seed",
            "min_regime_duration",
            "min_variance",
            "check_convergence_every",
            "early_stopping",
            "log_likelihood_threshold",
            "regime_type",
            "auto_select_states",
            "state_validation_threshold",
            "force_state_ordering",
            "validate_regime_economics",
        }

        assert set(config_dict.keys()) == expected_keys
        assert config_dict["n_states"] == 4
        assert config_dict["max_iterations"] == 200
        assert config_dict["tolerance"] == 1e-7
        assert config_dict["random_seed"] == 42

    def test_config_reconstruction(self):
        """Test that configuration can be reconstructed from dict."""
        original_config = HMMConfig(
            n_states=5,
            max_iterations=150,
            tolerance=1e-8,
            initialization_method="kmeans",
            random_seed=123,
        )

        config_dict = original_config.__dict__
        reconstructed_config = HMMConfig(**config_dict)

        assert reconstructed_config.__dict__ == original_config.__dict__

    def test_initialization_method_validation(self):
        """Test initialization method validation."""
        # Valid methods should work
        HMMConfig(initialization_method="random")
        HMMConfig(initialization_method="kmeans")
        HMMConfig(initialization_method="custom")

        # Invalid method should be caught by typing (if using strict type checking)
        # This test documents the expected behavior
        config = HMMConfig(initialization_method="invalid")
        assert (
            config.initialization_method == "invalid"
        )  # Type hint but no runtime validation
