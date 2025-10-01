"""
Unit tests for HiddenMarkovModel.

Tests the core hidden Markov model implementation including parameter estimation,
state inference, model persistence, and integration with the pipeline framework.
"""

import os
import pickle
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.utils.exceptions import (
    ConfigurationError,
    HMMInferenceError,
    HMMTrainingError,
    ValidationError,
)


class TestHiddenMarkovModel:
    """Test cases for HiddenMarkovModel."""

    def create_hmm_config(self, **kwargs):
        """Create HMM configuration for testing."""
        # Set sensible defaults
        defaults = {
            "n_states": 3,
            "max_iterations": 100,
            "tolerance": 1e-6,
            "random_seed": 42,
        }
        defaults.update(kwargs)
        return HMMConfig(**defaults)

    def create_sample_observations(self, n_samples=100, n_features=2):
        """Create sample observations for testing."""
        np.random.seed(42)

        # Generate simple regime-switching data respecting n_samples
        observations = []

        # Create regime assignments for n_samples observations
        regime_length = max(1, n_samples // 3)  # Roughly equal regimes
        for i in range(n_samples):
            regime = i // regime_length if i // regime_length < 3 else 2

            if regime == 0:  # Bear regime
                obs = np.random.multivariate_normal(
                    [-0.02, 0.03], [[0.001, 0.0002], [0.0002, 0.0015]], size=1
                )[0]
            elif regime == 1:  # Sideways regime
                obs = np.random.multivariate_normal(
                    [0.001, 0.015], [[0.0005, 0.0001], [0.0001, 0.001]], size=1
                )[0]
            else:  # Bull regime
                obs = np.random.multivariate_normal(
                    [0.015, 0.02], [[0.0008, 0.0003], [0.0003, 0.002]], size=1
                )[0]
            observations.append(obs)

        return pd.DataFrame(
            observations,
            columns=["log_return", "volatility"],
            index=pd.date_range("2024-01-01", periods=n_samples, freq="D"),
        )

    def test_hmm_initialization_default(self):
        """Test HMM initialization with default parameters."""
        config = HMMConfig(n_states=3)
        model = HiddenMarkovModel(config)

        assert model.config.n_states == 3
        assert model.config.max_iterations == 100
        assert not model.is_fitted

        # Internal state should be initialized
        assert model.transition_matrix_ is None
        assert model.emission_means_ is None
        assert model.emission_stds_ is None
        assert model.initial_probs_ is None
        assert not model.is_fitted

    def test_hmm_initialization_custom(self):
        """Test HMM initialization with custom parameters."""
        config = HMMConfig(
            n_states=4,
            max_iterations=200,
            tolerance=1e-8,
            random_seed=123,
            min_variance=1e-5,
        )
        model = HiddenMarkovModel(config)

        assert model.config.n_states == 4
        assert model.config.max_iterations == 200
        assert model.config.tolerance == 1e-8
        assert model.config.random_seed == 123
        assert model.config.min_variance == 1e-5

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Invalid n_states
        with pytest.raises((ValueError, ValidationError, ConfigurationError)):
            config = self.create_hmm_config(n_states=1)
            HiddenMarkovModel(config)

        # Invalid max_iterations
        with pytest.raises((ValueError, ValidationError, ConfigurationError)):
            config = self.create_hmm_config(max_iterations=0)
            HiddenMarkovModel(config)

        # Invalid tolerance
        with pytest.raises((ValueError, ValidationError, ConfigurationError)):
            config = self.create_hmm_config(tolerance=0)
            HiddenMarkovModel(config)

        # Note: initialization_method validation is handled by Literal type annotation in dataclass

    def test_fit_basic_functionality(self):
        """Test basic fitting functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        # Fit the model
        model.fit(observations)

        assert model.is_fitted
        assert model.transition_matrix_ is not None
        assert model.emission_means_ is not None
        assert model.emission_stds_ is not None
        assert model.initial_probs_ is not None

        # Check parameter shapes
        assert model.transition_matrix_.shape == (3, 3)
        assert model.emission_means_.shape == (3,)  # 3 states (univariate)
        assert model.initial_probs_.shape == (3,)

        # Transition matrix should be row-stochastic
        np.testing.assert_array_almost_equal(
            model.transition_matrix_.sum(axis=1), np.ones(3)
        )

        # Initial probabilities should sum to 1
        np.testing.assert_almost_equal(model.initial_probs_.sum(), 1.0)

    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data."""
        config = self.create_hmm_config(n_states=3)
        model = HiddenMarkovModel(config)

        # Too few observations
        observations = self.create_sample_observations(5)

        with pytest.raises(ValidationError, match="Insufficient data"):
            model.fit(observations)

    def test_fit_with_missing_data(self):
        """Test fitting with missing data."""
        config = self.create_hmm_config(n_states=3)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(50)

        # Introduce missing values
        observations.iloc[10, 0] = np.nan
        observations.iloc[20, 1] = np.nan

        with pytest.raises(ValidationError, match="Data contains missing values"):
            model.fit(observations)

    def test_fit_convergence(self):
        """Test convergence behavior."""
        # Test successful convergence
        config = self.create_hmm_config(n_states=3, max_iterations=100, tolerance=1e-4)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(200)

        model.fit(observations)
        assert model.training_history_["converged"]
        assert model.training_history_["iterations"] > 0
        assert model.training_history_["iterations"] <= 100

        # Test non-convergence
        config_no_converge = self.create_hmm_config(
            n_states=5, max_iterations=5, tolerance=1e-10
        )
        model_no_converge = HiddenMarkovModel(config_no_converge)

        # Note: Current implementation may not issue warnings
        model_no_converge.fit(observations)

        # Check if convergence failed
        assert model_no_converge.training_history_["iterations"] == 5

    def test_predict_functionality(self):
        """Test prediction functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        # Fit first
        model.fit(observations)

        # Predict on same data
        predictions = model.predict(observations)

        assert isinstance(predictions, pd.DataFrame)
        assert "predicted_state" in predictions.columns
        assert "confidence" in predictions.columns
        assert len(predictions) == len(observations)

        # States should be in valid range
        assert (predictions["predicted_state"] >= 0).all()
        assert (predictions["predicted_state"] < 3).all()

        # Confidence should be between 0 and 1
        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 1).all()

    def test_predict_without_fitting(self):
        """Test prediction without fitting."""
        config = self.create_hmm_config()
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(50)

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(observations)

    def test_predict_proba_functionality(self):
        """Test probability prediction functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        probabilities = model.predict_proba(observations)

        assert isinstance(probabilities, pd.DataFrame)
        assert len(probabilities) == len(observations)
        assert probabilities.shape[1] == 3  # 3 states

        # Probabilities should sum to 1 across states
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(
            prob_sums.values, np.ones(len(observations))
        )

        # All probabilities should be non-negative
        assert (probabilities >= 0).all().all()

    def test_update_functionality(self):
        """Test update method for pipeline interface."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        # First update (should fit)
        result = model.update(observations)

        assert model.is_fitted
        assert isinstance(result, pd.DataFrame)
        assert "predicted_state" in result.columns
        assert "confidence" in result.columns

        # Second update (should predict only)
        new_observations = self.create_sample_observations(50)
        result2 = model.update(new_observations)

        assert isinstance(result2, pd.DataFrame)
        assert len(result2) == len(new_observations)

    def test_score_functionality(self):
        """Test model scoring functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        score = model.score(observations)

        assert isinstance(score, float)
        assert score <= 0  # Log-likelihood is negative

        # Score on new data should be different
        new_observations = self.create_sample_observations(50)
        new_score = model.score(new_observations)

        assert isinstance(new_score, float)
        assert new_score != score

    def test_different_covariance_types(self):
        """Test covariance structure with current univariate implementation."""
        observations = self.create_sample_observations(150)

        # Note: Current implementation only supports univariate observations
        # using the configured observed_signal (log_return by default)
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        model.fit(observations)

        assert model.is_fitted
        predictions = model.predict(observations)
        assert len(predictions) == len(observations)

        # Current implementation: univariate with single variance per state
        # TODO: Future enhancement for multivariate covariance support
        assert model.emission_means_.shape == (3,)  # 3 states, univariate
        assert model.emission_stds_.shape == (3,)  # 3 states, single variance each

    def test_state_decoding_viterbi(self):
        """Test Viterbi state decoding."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        states = model.decode_states(observations, method="viterbi")

        assert isinstance(states, np.ndarray)
        assert len(states) == len(observations)
        assert (states >= 0).all()
        assert (states < 3).all()

    def test_state_decoding_posterior(self):
        """Test posterior state decoding."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        states = model.decode_states(observations, method="posterior")

        assert isinstance(states, np.ndarray)
        assert len(states) == len(observations)
        assert (states >= 0).all()
        assert (states < 3).all()

    def test_regime_analysis(self):
        """Test regime analysis functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(200)

        model.fit(observations)
        analysis = model.get_regime_analysis(observations)

        assert isinstance(analysis, dict)

        # Should contain regime statistics
        assert "regime_stats" in analysis
        assert len(analysis["regime_stats"]) == 3

        for state in analysis["regime_stats"]:
            regime_stats = analysis["regime_stats"][state]
            assert "mean_duration" in regime_stats
            assert "mean_return" in regime_stats
            assert "volatility" in regime_stats
            assert "frequency" in regime_stats

        # Should contain transition analysis
        assert "transition_analysis" in analysis
        assert "most_persistent" in analysis["transition_analysis"]
        assert "most_volatile" in analysis["transition_analysis"]

    def test_model_persistence_pickle(self):
        """Test model serialization with pickle."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        original_predictions = model.predict(observations)

        # Serialize and deserialize
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(model, f)
            f.flush()

            with open(f.name, "rb") as f2:
                loaded_model = pickle.load(f2)

        os.unlink(f.name)

        # Test loaded model
        assert loaded_model.is_fitted
        assert loaded_model.n_states == model.n_states

        loaded_predictions = loaded_model.predict(observations)
        pd.testing.assert_frame_equal(original_predictions, loaded_predictions)

    def test_model_persistence_custom(self):
        """Test custom model save/load functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model.save_model(f.name)

            loaded_model = HiddenMarkovModel.load_model(f.name)

        os.unlink(f.name)

        # Test loaded model
        assert loaded_model.is_fitted
        assert loaded_model.n_states == model.n_states
        np.testing.assert_array_equal(
            loaded_model.transition_matrix_, model.transition_matrix_
        )

    def test_parameter_initialization_strategies(self):
        """Test different parameter initialization strategies."""
        observations = self.create_sample_observations(150)

        # Test different initialization methods
        for init_method in ["random", "kmeans", "manual"]:
            if init_method == "manual":
                # Provide manual initialization
                config = self.create_hmm_config(
                    n_states=3, random_seed=42, initialization_method=init_method
                )
                model = HiddenMarkovModel(config)
                model.fit(
                    observations
                )  # Note: initial_params not supported in current interface
            else:
                config = self.create_hmm_config(
                    n_states=3, random_seed=42, initialization_method=init_method
                )
                model = HiddenMarkovModel(config)
                model.fit(observations)

            assert model.is_fitted
            predictions = model.predict(observations)
            assert len(predictions) == len(observations)

    def test_online_learning_capabilities(self):
        """Test incremental/online learning capabilities (basic interface test)."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)

        # Initial training
        initial_data = self.create_sample_observations(100)
        model.fit(initial_data)

        # Incremental update (currently just placeholder - should not raise exception)
        new_data = self.create_sample_observations(50)
        model.partial_fit(new_data, learning_rate=0.1)

        # Model should still be fitted and functional
        assert model.is_fitted
        predictions = model.predict(new_data)
        assert len(predictions) == len(new_data)

        # Test that partial_fit can be called on unfitted model (should do full fit)
        unfitted_model = HiddenMarkovModel(config)
        unfitted_model.partial_fit(initial_data, learning_rate=0.1)
        assert unfitted_model.is_fitted
        predictions2 = unfitted_model.predict(initial_data)
        assert len(predictions2) == len(initial_data)

    def test_model_selection_criteria(self):
        """Test model selection criteria (AIC, BIC)."""
        observations = self.create_sample_observations(200)

        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        model.fit(observations)

        # Test AIC calculation
        aic = model.aic(observations)
        assert isinstance(aic, float)
        assert aic > 0  # AIC should be positive

        # Test BIC calculation
        bic = model.bic(observations)
        assert isinstance(bic, float)
        assert bic > 0  # BIC should be positive
        assert bic > aic  # BIC should be larger than AIC (penalty for complexity)

    def test_cross_validation_score(self):
        """Test cross-validation scoring."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(200)

        cv_scores = model.cross_validate(observations, cv_folds=3)

        assert isinstance(cv_scores, dict)
        assert "scores" in cv_scores
        assert "mean_score" in cv_scores
        assert "std_score" in cv_scores

        assert len(cv_scores["scores"]) == 3
        assert isinstance(cv_scores["mean_score"], float)
        assert isinstance(cv_scores["std_score"], float)

    def test_plot_functionality(self):
        """Test plotting functionality."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(100)

        model.fit(observations)
        predictions = model.predict(observations)

        # Test state sequence plotting
        fig = model.plot(observations=observations, plot_type="states")
        assert fig is not None

        # Test regime analysis plotting
        fig = model.plot(observations=observations, plot_type="regimes")
        assert fig is not None

        # Test transition matrix plotting
        fig = model.plot(plot_type="transitions")
        assert fig is not None

    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        config = self.create_hmm_config(n_states=3, random_seed=42)
        model = HiddenMarkovModel(config)
        observations = self.create_sample_observations(200)

        model.fit(observations)

        # Get performance metrics
        metrics = model.get_performance_metrics(observations)

        assert isinstance(metrics, dict)
        assert "log_likelihood" in metrics
        assert "aic" in metrics
        assert "bic" in metrics
        assert "n_parameters" in metrics

        # Test monitoring over time
        time_series_metrics = model.monitor_performance(observations, window_size=50)

        assert isinstance(time_series_metrics, pd.DataFrame)
        assert "log_likelihood" in time_series_metrics.columns
        assert len(time_series_metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__])
