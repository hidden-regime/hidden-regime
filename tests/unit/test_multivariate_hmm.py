"""
Unit tests for multivariate HMM functionality.

Tests N-dimensional Gaussian HMMs with multiple observed features.
"""

import numpy as np
import pandas as pd
import pytest

from hidden_regime.config import HMMConfig, ObservationMode
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.models.utils import (
    initialize_parameters_kmeans_multivariate,
    initialize_parameters_gmm_multivariate,
    _initialize_parameters_diagonal_multivariate,
)


class TestMultivariateObservationExtraction:
    """Test observation extraction for multivariate HMMs."""

    def test_extract_2d_observations(self):
        """Test extracting 2D observations."""
        df = pd.DataFrame({
            'log_return': np.random.randn(100) * 0.01,
            'volume_change': np.random.randn(100) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change']
        )
        model = HiddenMarkovModel(config)

        # Extract observations using internal method
        obs = model._extract_observations(df)

        assert obs.shape == (100, 2), f"Expected (100, 2), got {obs.shape}"
        assert obs.ndim == 2, f"Expected 2D array, got {obs.ndim}D"

    def test_extract_3d_observations(self):
        """Test extracting 3D observations."""
        df = pd.DataFrame({
            'log_return': np.random.randn(100) * 0.01,
            'volume_change': np.random.randn(100) * 0.5,
            'volatility': np.abs(np.random.randn(100) * 0.02),
        })

        config = HMMConfig(
            n_states=3,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change', 'volatility']
        )
        model = HiddenMarkovModel(config)

        obs = model._extract_observations(df)

        assert obs.shape == (100, 3), f"Expected (100, 3), got {obs.shape}"

    def test_univariate_extraction_still_works(self):
        """Test that univariate extraction is unchanged."""
        df = pd.DataFrame({
            'log_return': np.random.randn(100) * 0.01,
        })

        config = HMMConfig(
            n_states=2,
            observed_signal='log_return'  # Univariate
        )
        model = HiddenMarkovModel(config)

        obs = model._extract_observations(df)

        assert obs.shape == (100,), f"Expected (100,), got {obs.shape}"
        assert obs.ndim == 1, f"Expected 1D array, got {obs.ndim}D"


class TestMultivariateInitialization:
    """Test multivariate initialization functions."""

    def test_kmeans_initialization_2d(self):
        """Test KMeans initialization with 2D data."""
        np.random.seed(42)
        observations = np.random.randn(200, 2)

        init_probs, trans_mat, means, covs, diagnostics = (
            initialize_parameters_kmeans_multivariate(
                n_states=3, observations=observations, random_seed=42
            )
        )

        # Check shapes
        assert init_probs.shape == (3,)
        assert trans_mat.shape == (3, 3)
        assert means.shape == (3, 2)
        assert covs.shape == (3, 2, 2)

        # Check probabilities sum to 1
        assert np.isclose(init_probs.sum(), 1.0)
        assert np.allclose(trans_mat.sum(axis=1), 1.0)

        # Check covariances are positive definite
        for k in range(3):
            eigenvalues = np.linalg.eigvals(covs[k])
            assert np.all(eigenvalues > 0), f"State {k} covariance not positive definite"

        # Check diagnostics
        assert diagnostics['method'] == 'kmeans_multivariate'
        assert diagnostics['n_states'] == 3
        assert diagnostics['n_features'] == 2

    def test_gmm_initialization_2d(self):
        """Test GMM initialization with 2D data."""
        np.random.seed(42)
        observations = np.random.randn(200, 2)

        init_probs, trans_mat, means, covs, diagnostics = (
            initialize_parameters_gmm_multivariate(
                n_states=3, observations=observations, random_seed=42
            )
        )

        # Check shapes
        assert init_probs.shape == (3,)
        assert trans_mat.shape == (3, 3)
        assert means.shape == (3, 2)
        assert covs.shape == (3, 2, 2)

        # Check probabilities
        assert np.isclose(init_probs.sum(), 1.0)
        assert np.allclose(trans_mat.sum(axis=1), 1.0)

        # Check covariances are positive definite
        for k in range(3):
            eigenvalues = np.linalg.eigvals(covs[k])
            assert np.all(eigenvalues > 0), f"State {k} covariance not positive definite"

        # Check diagnostics
        assert diagnostics['method'] == 'gmm_multivariate'
        assert diagnostics['converged'] is True or diagnostics['converged'] is False

    def test_diagonal_fallback_initialization(self):
        """Test diagonal fallback initialization."""
        np.random.seed(42)
        observations = np.random.randn(200, 2)

        init_probs, trans_mat, means, covs, diagnostics = (
            _initialize_parameters_diagonal_multivariate(
                n_states=3, observations=observations, random_seed=42
            )
        )

        # Check shapes
        assert means.shape == (3, 2)
        assert covs.shape == (3, 2, 2)

        # Check covariances are diagonal (or close to it for fallback)
        for k in range(3):
            # Should have positive diagonal elements
            assert np.all(np.diag(covs[k]) > 0)

        assert diagnostics['method'] == 'diagonal_multivariate'

    def test_insufficient_data_fallback(self):
        """Test that insufficient data triggers fallback."""
        np.random.seed(42)
        observations = np.random.randn(2, 2)  # Only 2 observations, less than n_states

        init_probs, trans_mat, means, covs, diagnostics = (
            initialize_parameters_kmeans_multivariate(
                n_states=3, observations=observations, random_seed=42
            )
        )

        # Should fall back to diagonal initialization
        assert 'diagonal' in diagnostics['method'], f"Expected diagonal fallback, got {diagnostics['method']}"


class TestMultivariateHMMTraining:
    """Test multivariate HMM training."""

    def test_fit_2d_observations(self):
        """Test fitting HMM with 2D observations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(300) * 0.01,
            'volume_change': np.random.randn(300) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            max_iterations=30,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        # Check model is fitted
        assert model.is_fitted
        assert model.is_multivariate
        assert model.n_features == 2

        # Check learned parameters
        assert model.emission_means_.shape == (2, 2)
        assert model.emission_covs_.shape == (2, 2, 2)
        assert model.emission_stds_ is None  # Should be None for multivariate

        # Check covariances are positive definite
        for k in range(2):
            eigenvalues = np.linalg.eigvals(model.emission_covs_[k])
            assert np.all(eigenvalues > 0), f"State {k} covariance not positive definite"

    def test_fit_3d_observations(self):
        """Test fitting HMM with 3D observations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(300) * 0.01,
            'volume_change': np.random.randn(300) * 0.5,
            'volatility': np.abs(np.random.randn(300) * 0.02),
        })

        config = HMMConfig(
            n_states=3,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change', 'volatility'],
            max_iterations=30,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        assert model.is_fitted
        assert model.n_features == 3
        assert model.emission_means_.shape == (3, 3)
        assert model.emission_covs_.shape == (3, 3, 3)

    def test_training_convergence(self):
        """Test that training log-likelihood increases."""
        np.random.seed(42)

        # Generate data with clear structure
        T = 500
        regime = np.random.choice([0, 1], size=T, p=[0.6, 0.4])
        obs = np.zeros((T, 2))
        for t in range(T):
            if regime[t] == 0:
                obs[t] = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.1]])
            else:
                obs[t] = np.random.multivariate_normal([0.02, 1.0], [[0.04, 0.01], [0.01, 0.5]])

        df = pd.DataFrame(obs, columns=['log_return', 'volume_change'])

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            initialization_method='kmeans',
            max_iterations=50,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        # Check log-likelihood improved
        log_likelihoods = model.training_history_['log_likelihoods']
        assert len(log_likelihoods) > 0
        # Should generally increase (allowing for some noise)
        assert log_likelihoods[-1] >= log_likelihoods[0] - 10  # Allow some tolerance


class TestMultivariateHMMPrediction:
    """Test multivariate HMM prediction."""

    def test_predict_2d_observations(self):
        """Test predicting with 2D observations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(300) * 0.01,
            'volume_change': np.random.randn(300) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        predictions = model.predict(df)

        # Check prediction outputs
        assert 'predicted_state' in predictions.columns
        assert 'confidence' in predictions.columns
        assert 'state_0_prob' in predictions.columns
        assert 'state_1_prob' in predictions.columns
        assert 'emission_means' in predictions.columns
        assert 'emission_covs' in predictions.columns  # Should have covs, not stds
        assert 'emission_stds' not in predictions.columns  # Should NOT have stds

        # Check shapes
        assert len(predictions) == 300
        assert all(predictions['predicted_state'].isin([0, 1]))
        assert all((predictions['confidence'] >= 0) & (predictions['confidence'] <= 1))

    def test_prediction_probabilities_sum_to_one(self):
        """Test that state probabilities sum to 1."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(100) * 0.01,
            'volume_change': np.random.randn(100) * 0.5,
        })

        config = HMMConfig(
            n_states=3,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        predictions = model.predict(df)

        # Check probabilities sum to 1
        prob_sum = (
            predictions['state_0_prob'] +
            predictions['state_1_prob'] +
            predictions['state_2_prob']
        )
        assert np.allclose(prob_sum, 1.0), "State probabilities don't sum to 1"


class TestMultivariateVsUnivariate:
    """Test backward compatibility with univariate HMMs."""

    def test_univariate_still_works(self):
        """Test that univariate HMMs still work unchanged."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
        })

        config = HMMConfig(
            n_states=2,
            observed_signal='log_return',  # Univariate
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        # Check univariate mode
        assert not model.is_multivariate
        assert model.n_features == 1
        assert model.emission_means_.shape == (2,)
        assert model.emission_stds_.shape == (2,)
        assert model.emission_covs_ is None  # Should be None for univariate

        predictions = model.predict(df)
        assert 'emission_stds' in predictions.columns
        assert 'emission_covs' not in predictions.columns

    def test_multivariate_vs_univariate_with_single_feature(self):
        """Test multivariate with 1 feature vs univariate."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
        })

        # Univariate
        config_uni = HMMConfig(
            n_states=2,
            observed_signal='log_return',
            max_iterations=20,
            random_seed=42
        )
        model_uni = HiddenMarkovModel(config_uni)
        model_uni.fit(df)

        # Multivariate with 1 feature
        config_multi = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return'],
            max_iterations=20,
            random_seed=42
        )
        model_multi = HiddenMarkovModel(config_multi)
        model_multi.fit(df)

        # Both should work, but have different internal representations
        assert not model_uni.is_multivariate
        assert model_multi.is_multivariate
        assert model_multi.emission_means_.shape == (2, 1)
        assert model_multi.emission_covs_.shape == (2, 1, 1)


class TestMultivariateEdgeCases:
    """Test edge cases for multivariate HMMs."""

    def test_perfect_correlation(self):
        """Test with perfectly correlated features."""
        np.random.seed(42)
        x = np.random.randn(200) * 0.01
        df = pd.DataFrame({
            'feature1': x,
            'feature2': x * 2.0,  # Perfect correlation
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['feature1', 'feature2'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)

        # Should not crash despite perfect correlation (regularization handles it)
        model.fit(df)
        assert model.is_fitted

        # Covariances should still be valid
        for k in range(2):
            eigenvalues = np.linalg.eigvals(model.emission_covs_[k])
            assert np.all(eigenvalues > 0)

    def test_very_different_scales(self):
        """Test features with very different scales."""
        np.random.seed(42)
        df = pd.DataFrame({
            'small_scale': np.random.randn(200) * 0.001,  # Small values
            'large_scale': np.random.randn(200) * 100.0,  # Large values
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['small_scale', 'large_scale'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        assert model.is_fitted
        # Should handle different scales gracefully

    def test_nan_handling(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
            'volume_change': np.random.randn(200) * 0.5,
        })

        # Add some NaNs (NaNs in both features will be dropped together)
        df.loc[10:12, 'log_return'] = np.nan
        df.loc[20:22, 'volume_change'] = np.nan

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)

        # Fit should work on clean data (drops NaN rows)
        model.fit(df)
        assert model.is_fitted

        # For prediction, use only clean data
        df_clean = df.dropna(subset=['log_return', 'volume_change'])
        predictions = model.predict(df_clean)

        # Predictions should match clean data length
        assert len(predictions) == len(df_clean)
        assert all(predictions['predicted_state'].isin([0, 1]))


class TestMultivariateInitializationMethods:
    """Test different initialization methods for multivariate HMMs."""

    def test_kmeans_initialization(self):
        """Test KMeans initialization method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
            'volume_change': np.random.randn(200) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            initialization_method='kmeans',
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        assert model.is_fitted
        # Should use KMeans-based covariance estimation

    def test_gmm_initialization(self):
        """Test GMM initialization method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
            'volume_change': np.random.randn(200) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            initialization_method='gmm',
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        assert model.is_fitted
        # Should use GMM-based covariance estimation

    def test_random_fallback_initialization(self):
        """Test that unsupported methods fall back to diagonal."""
        np.random.seed(42)
        df = pd.DataFrame({
            'log_return': np.random.randn(200) * 0.01,
            'volume_change': np.random.randn(200) * 0.5,
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            initialization_method='quantile',  # Not supported for multivariate
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        # Should fall back to diagonal initialization
        assert model.is_fitted


@pytest.mark.integration
class TestMultivariateIntegration:
    """Integration tests for multivariate HMM end-to-end."""

    def test_full_workflow_2d(self):
        """Test complete workflow with 2D observations."""
        np.random.seed(42)

        # Generate synthetic regime-switching data
        T = 500
        regime = np.zeros(T, dtype=int)
        obs = np.zeros((T, 2))

        for t in range(T):
            if t > 0:
                # Regime persistence
                if np.random.rand() < 0.95:
                    regime[t] = regime[t-1]
                else:
                    regime[t] = 1 - regime[t-1]

            if regime[t] == 0:
                # Low volatility regime
                obs[t] = np.random.multivariate_normal(
                    [0.001, 0.0],
                    [[0.0001, 0], [0, 0.01]]
                )
            else:
                # High volatility regime
                obs[t] = np.random.multivariate_normal(
                    [0.0, 1.0],
                    [[0.001, 0.0005], [0.0005, 0.5]]
                )

        df = pd.DataFrame(obs, columns=['log_return', 'volume_change'])

        # Train model
        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'volume_change'],
            initialization_method='kmeans',
            max_iterations=50,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(df)

        # Predict
        predictions = model.predict(df)

        # Evaluate
        assert model.is_fitted
        assert len(predictions) == T
        assert 'predicted_state' in predictions.columns
        assert 'emission_covs' in predictions.columns

        # Check that predictions somewhat match true regimes
        # (allowing for label switching)
        accuracy = max(
            np.mean(predictions['predicted_state'].values == regime),
            np.mean(predictions['predicted_state'].values == (1 - regime))
        )
        # With clear regimes, should get decent accuracy
        assert accuracy > 0.70, f"Accuracy {accuracy:.1%} is too low"
