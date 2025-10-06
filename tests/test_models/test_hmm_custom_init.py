"""
Tests for custom HMM initialization.

Tests the ability to specify custom initial parameters for HMMs,
including validation, warnings, and diagnostics.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.utils.exceptions import ConfigurationError


def create_test_data(n_obs: int = 100) -> pd.DataFrame:
    """Create simple test data."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_obs)
    dates = pd.date_range("2023-01-01", periods=n_obs, freq="D")
    return pd.DataFrame({'log_return': returns}, index=dates)


class TestCustomInitializationConfig:
    """Test custom initialization configuration and validation."""

    def test_custom_config_requires_means(self):
        """Custom initialization requires emission means."""
        with pytest.raises(ConfigurationError) as exc_info:
            config = HMMConfig(
                n_states=3,
                initialization_method='custom',
                # Missing custom_emission_means
                custom_emission_stds=[0.01, 0.02, 0.015],
            )
            config.validate()

        assert 'custom_emission_means' in str(exc_info.value)

    def test_custom_config_requires_stds(self):
        """Custom initialization requires emission stds."""
        with pytest.raises(ConfigurationError) as exc_info:
            config = HMMConfig(
                n_states=3,
                initialization_method='custom',
                custom_emission_means=[-0.01, 0.0, 0.01],
                # Missing custom_emission_stds
            )
            config.validate()

        assert 'custom_emission_stds' in str(exc_info.value)

    def test_custom_config_validates_dimensions(self):
        """Custom parameters must match n_states."""
        # Wrong number of means
        with pytest.raises(ConfigurationError) as exc_info:
            config = HMMConfig(
                n_states=3,
                initialization_method='custom',
                custom_emission_means=[-0.01, 0.01],  # Only 2 values for 3 states
                custom_emission_stds=[0.01, 0.02, 0.015],
            )
            config.validate()

        assert 'must have 3 values' in str(exc_info.value)

    def test_custom_config_validates_transition_matrix_dimensions(self):
        """Custom transition matrix must be NxN."""
        with pytest.raises(ConfigurationError) as exc_info:
            config = HMMConfig(
                n_states=3,
                initialization_method='custom',
                custom_emission_means=[-0.01, 0.0, 0.01],
                custom_emission_stds=[0.01, 0.02, 0.015],
                custom_transition_matrix=[
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],  # Only 2 rows for 3 states
                ],
            )
            config.validate()

        assert '3x3' in str(exc_info.value)

    def test_custom_config_valid(self):
        """Valid custom config should pass validation."""
        config = HMMConfig(
            n_states=3,
            initialization_method='custom',
            custom_emission_means=[-0.015, 0.0, 0.012],
            custom_emission_stds=[0.025, 0.015, 0.020],
        )
        config.validate()  # Should not raise


class TestCustomInitializationBasic:
    """Test basic custom initialization functionality."""

    def test_custom_init_with_minimal_params(self):
        """Custom initialization with only means and stds."""
        config = HMMConfig(
            n_states=3,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.0, 0.01],
            custom_emission_stds=[0.02, 0.015, 0.02],
            max_iterations=5,
            random_seed=42,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)
        hmm.fit(data)

        # Check model is fitted
        assert hmm.is_fitted

        # Note: After Baum-Welch training, parameters WILL have changed
        # from their initial values. Custom init sets STARTING parameters,
        # not final parameters. Check that they are in reasonable range.
        assert len(hmm.emission_means_) == 3
        assert len(hmm.emission_stds_) == 3
        assert np.all(hmm.emission_stds_ > 0)  # All positive

        # Check diagnostics show custom initialization
        diag = hmm.get_initialization_diagnostics()
        assert diag['method'] == 'custom'
        assert diag['n_states'] == 3

    def test_custom_init_with_all_params(self):
        """Custom initialization with all optional parameters."""
        transition_matrix = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ]

        initial_probs = [0.5, 0.3, 0.2]

        config = HMMConfig(
            n_states=3,
            initialization_method='custom',
            custom_emission_means=[-0.02, 0.0, 0.015],
            custom_emission_stds=[0.025, 0.015, 0.020],
            custom_transition_matrix=transition_matrix,
            custom_initial_probs=initial_probs,
            max_iterations=5,
            random_seed=42,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)
        hmm.fit(data)

        # Check model is fitted and has correct structure
        assert hmm.is_fitted
        assert len(hmm.emission_means_) == 3
        assert hmm.transition_matrix_.shape == (3, 3)
        assert len(hmm.initial_probs_) == 3

        # Note: Baum-Welch will update parameters from initial values
        # Check they're still valid stochastic matrices
        assert np.allclose(np.sum(hmm.transition_matrix_, axis=1), 1.0)
        assert np.isclose(np.sum(hmm.initial_probs_), 1.0)

    def test_custom_init_produces_diagnostics(self):
        """Custom initialization produces comprehensive diagnostics."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.02, 0.02],
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)
        hmm.fit(data)

        diag = hmm.get_initialization_diagnostics()

        # Check required fields
        assert 'method' in diag
        assert 'regime_characteristics' in diag
        assert 'transition_matrix' in diag
        assert 'initial_probs' in diag

        # Check regime characteristics
        assert len(diag['regime_characteristics']) == 2
        for regime_info in diag['regime_characteristics']:
            assert 'mean_return_pct' in regime_info
            assert 'volatility_pct' in regime_info
            assert 'persistence' in regime_info
            assert 'expected_duration_days' in regime_info


class TestCustomInitializationValidation:
    """Test parameter validation and warnings."""

    def test_validation_warns_extreme_negative_returns(self):
        """Validation warns about extremely negative returns."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.15, 0.01],  # -15% daily is extreme
            custom_emission_stds=[0.02, 0.02],
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hmm.fit(data)

            # Should have warning about extreme negative
            warning_messages = [str(warning.message) for warning in w]
            assert any('extremely negative' in msg.lower() for msg in warning_messages)

    def test_validation_warns_extreme_positive_returns(self):
        """Validation warns about extremely positive returns."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.15],  # +15% daily is extreme
            custom_emission_stds=[0.02, 0.02],
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hmm.fit(data)

            warning_messages = [str(warning.message) for warning in w]
            assert any('extremely positive' in msg.lower() for msg in warning_messages)

    def test_validation_warns_extreme_low_volatility(self):
        """Validation warns about extremely low volatility."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.0001, 0.02],  # 0.01% daily is very low
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hmm.fit(data)

            warning_messages = [str(warning.message) for warning in w]
            assert any('extremely low volatility' in msg.lower() for msg in warning_messages)

    def test_validation_rejects_negative_volatility(self):
        """Validation rejects negative volatility."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[-0.02, 0.02],  # Negative volatility is invalid
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with pytest.raises(ValueError) as exc_info:
            hmm.fit(data)

        assert 'positive' in str(exc_info.value).lower()


class TestCustomInitializationNormalization:
    """Test automatic normalization of probabilities."""

    def test_normalizes_transition_matrix_rows(self):
        """Transition matrix rows that don't sum to 1.0 are normalized."""
        transition_matrix = [
            [0.8, 0.3],  # Sums to 1.1
            [0.1, 0.9],  # OK
        ]

        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.02, 0.02],
            custom_transition_matrix=transition_matrix,
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hmm.fit(data)

            # Should warn about normalization
            warning_messages = [str(warning.message) for warning in w]
            assert any('normalizing' in msg.lower() for msg in warning_messages)

        # Rows should sum to 1.0 after normalization
        assert np.allclose(np.sum(hmm.transition_matrix_, axis=1), 1.0)

    def test_normalizes_initial_probs(self):
        """Initial probabilities that don't sum to 1.0 are normalized."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.02, 0.02],
            custom_initial_probs=[0.7, 0.5],  # Sums to 1.2
            max_iterations=5,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=50)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hmm.fit(data)

            warning_messages = [str(warning.message) for warning in w]
            assert any('normalizing' in msg.lower() for msg in warning_messages)

        # Should sum to 1.0 after normalization
        assert np.isclose(np.sum(hmm.initial_probs_), 1.0)


class TestRegimeSpecsFactoryMethod:
    """Test the from_regime_specs factory method."""

    def test_from_regime_specs_basic(self):
        """from_regime_specs creates valid config."""
        config = HMMConfig.from_regime_specs([
            {'mean': -0.015, 'std': 0.025},
            {'mean': 0.0, 'std': 0.015},
            {'mean': 0.012, 'std': 0.020},
        ])

        # Check config is valid
        assert config.n_states == 3
        assert config.initialization_method == 'custom'
        assert config.custom_emission_means == [-0.015, 0.0, 0.012]
        assert config.custom_emission_stds == [0.025, 0.015, 0.020]

        config.validate()  # Should not raise

    def test_from_regime_specs_with_overrides(self):
        """from_regime_specs accepts additional config overrides."""
        config = HMMConfig.from_regime_specs(
            regime_specs=[
                {'mean': -0.01, 'std': 0.02},
                {'mean': 0.01, 'std': 0.02},
            ],
            max_iterations=200,
            tolerance=1e-8,
        )

        assert config.n_states == 2
        assert config.max_iterations == 200
        assert config.tolerance == 1e-8

    def test_from_regime_specs_end_to_end(self):
        """from_regime_specs config works end-to-end."""
        config = HMMConfig.from_regime_specs([
            {'mean': -0.02, 'std': 0.025},  # Bear
            {'mean': 0.0, 'std': 0.015},    # Sideways
            {'mean': 0.015, 'std': 0.020},  # Bull
        ])

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=100)
        hmm.fit(data)

        assert hmm.is_fitted
        assert hmm.n_states == 3

        # Can make predictions
        predictions = hmm.predict(data)
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 100


class TestCustomInitializationIntegration:
    """Integration tests for custom initialization."""

    def test_custom_init_allows_prediction(self):
        """Model with custom init can make predictions."""
        config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.02, 0.02],
            max_iterations=10,
        )

        hmm = HiddenMarkovModel(config)
        data = create_test_data(n_obs=100)
        hmm.fit(data)

        predictions = hmm.predict(data)

        assert 'predicted_state' in predictions.columns
        assert 'confidence' in predictions.columns
        assert len(predictions) == 100

    def test_custom_vs_kmeans_initialization(self):
        """Custom and KMeans initialization both work and differ."""
        data = create_test_data(n_obs=100)

        # Custom initialization
        custom_config = HMMConfig(
            n_states=2,
            initialization_method='custom',
            custom_emission_means=[-0.01, 0.01],
            custom_emission_stds=[0.02, 0.02],
            max_iterations=5,
            random_seed=42,
        )
        hmm_custom = HiddenMarkovModel(custom_config)
        hmm_custom.fit(data)

        # KMeans initialization
        kmeans_config = HMMConfig(
            n_states=2,
            initialization_method='kmeans',
            max_iterations=5,
            random_seed=42,
        )
        hmm_kmeans = HiddenMarkovModel(kmeans_config)
        hmm_kmeans.fit(data)

        # Both should be fitted
        assert hmm_custom.is_fitted
        assert hmm_kmeans.is_fitted

        # Diagnostics should differ
        diag_custom = hmm_custom.get_initialization_diagnostics()
        diag_kmeans = hmm_kmeans.get_initialization_diagnostics()

        assert diag_custom['method'] == 'custom'
        assert diag_kmeans['method'] == 'kmeans'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
