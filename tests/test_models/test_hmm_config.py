"""
Unit tests for HMMConfig class.

Tests configuration validation, factory methods, and integration with data validation.
"""

import numpy as np
import pytest

from hidden_regime.config.model import HMMConfig


class TestHMMConfig:
    """Test cases for HMMConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = HMMConfig()

        assert config.n_states == 3
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.regularization == 1e-6
        assert config.initialization_method == "quantile"  # Updated default
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

    def test_initialization_methods(self):
        """Test all supported initialization methods."""
        # All valid methods should work
        for method in ["quantile", "kmeans", "gmm", "random", "custom"]:
            if method == "custom":
                config = HMMConfig(
                    initialization_method=method,
                    custom_emission_means=[0.0, 0.001, -0.001],
                    custom_emission_stds=[0.01, 0.02, 0.015],
                )
            else:
                config = HMMConfig(initialization_method=method)
            assert config.initialization_method == method

    def test_create_conservative_factory(self):
        """Test create_conservative factory method."""
        config = HMMConfig.create_conservative()

        assert config.n_states == 3
        assert config.max_iterations == 200
        assert config.tolerance == 1e-8
        assert config.initialization_method == "quantile"

    def test_create_aggressive_factory(self):
        """Test create_aggressive factory method."""
        config = HMMConfig.create_aggressive()

        assert config.n_states == 4
        assert config.max_iterations == 50
        assert config.tolerance == 1e-4
        assert config.initialization_method == "quantile"

    def test_create_balanced_factory(self):
        """Test create_balanced factory method."""
        config = HMMConfig.create_balanced()

        assert config.n_states == 3
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.initialization_method == "quantile"

    def test_from_regime_specs_factory(self):
        """Test from_regime_specs factory method."""
        regime_specs = [
            {'mean': -0.015, 'std': 0.025},  # Bear
            {'mean': 0.0, 'std': 0.015},     # Sideways
            {'mean': 0.012, 'std': 0.020},   # Bull
        ]

        config = HMMConfig.from_regime_specs(regime_specs)

        assert config.n_states == 3
        assert config.initialization_method == "custom"
        assert config.custom_emission_means == [-0.015, 0.0, 0.012]
        assert config.custom_emission_stds == [0.025, 0.015, 0.020]

    def test_custom_initialization_validation(self):
        """Test custom initialization parameter validation."""
        from hidden_regime.utils.exceptions import ConfigurationError

        # Missing emission means
        with pytest.raises(ConfigurationError, match="custom_emission_means is required"):
            config = HMMConfig(
                initialization_method="custom",
                custom_emission_stds=[0.01, 0.02, 0.015],
            )
            config.validate()

        # Missing emission stds
        with pytest.raises(ConfigurationError, match="custom_emission_stds is required"):
            config = HMMConfig(
                initialization_method="custom",
                custom_emission_means=[0.0, 0.001, -0.001],
            )
            config.validate()

        # Wrong number of means
        with pytest.raises(ConfigurationError, match="custom_emission_means must have 3 values"):
            config = HMMConfig(
                n_states=3,
                initialization_method="custom",
                custom_emission_means=[0.0, 0.001],  # Only 2, need 3
                custom_emission_stds=[0.01, 0.02, 0.015],
            )
            config.validate()

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        from hidden_regime.utils.exceptions import ConfigurationError

        # Test n_states validation
        with pytest.raises(ConfigurationError, match="n_states must be at least 2"):
            config = HMMConfig(n_states=1)
            config.validate()

        with pytest.raises(ConfigurationError, match="n_states should not exceed 10"):
            config = HMMConfig(n_states=11)
            config.validate()

        # Test max_iterations validation
        with pytest.raises(ConfigurationError, match="max_iterations must be positive"):
            config = HMMConfig(max_iterations=0)
            config.validate()

        # Test tolerance validation
        with pytest.raises(ConfigurationError, match="tolerance must be positive"):
            config = HMMConfig(tolerance=0)
            config.validate()

        # Test regularization validation
        with pytest.raises(ConfigurationError, match="regularization must be non-negative"):
            config = HMMConfig(regularization=-0.1)
            config.validate()

        # Test min_regime_duration validation
        with pytest.raises(ConfigurationError, match="min_regime_duration must be at least 1"):
            config = HMMConfig(min_regime_duration=0)
            config.validate()

        # Test min_variance validation
        with pytest.raises(ConfigurationError, match="min_variance must be positive"):
            config = HMMConfig(min_variance=0)
            config.validate()

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = HMMConfig(n_states=3, max_iterations=100, tolerance=1e-6)
        config2 = HMMConfig(n_states=3, max_iterations=100, tolerance=1e-6)
        config3 = HMMConfig(n_states=4, max_iterations=100, tolerance=1e-6)

        assert config1 == config2
        assert config1 != config3

    def test_create_component(self):
        """Test that config can create an HMM component."""
        config = HMMConfig(n_states=3)
        model = config.create_component()

        from hidden_regime.models.hmm import HiddenMarkovModel
        assert isinstance(model, HiddenMarkovModel)

    def test_get_cache_key(self):
        """Test cache key generation."""
        config = HMMConfig(n_states=3, initialization_method="quantile")
        cache_key = config.get_cache_key()

        assert "hmm" in cache_key
        assert "3" in cache_key
        assert "quantile" in cache_key
