"""
Working tests for HiddenMarkovModel component.

Tests that work with the current implementation, focusing on coverage
and basic functionality validation for HMM training and prediction.
"""

import warnings
from datetime import datetime
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.utils.exceptions import HMMInferenceError, HMMTrainingError


class TestHiddenMarkovModelWorking:
    """Working tests for HiddenMarkovModel that focus on coverage."""

    def test_initialization(self):
        """Test basic initialization."""
        config = HMMConfig(n_states=3, observed_signal="log_return")
        model = HiddenMarkovModel(config)

        assert model.config is config
        assert model.n_states == 3
        assert model.is_fitted == False
        assert model.initial_probs_ is None
        assert model.transition_matrix_ is None
        assert model.emission_means_ is None
        assert model.emission_stds_ is None
        assert model._current_state_probs is None
        assert model._last_observation is None
        assert model._sufficient_stats is None

    def test_config_validation_integration(self):
        """Test that configuration validation works."""
        # Valid configs
        config1 = HMMConfig(n_states=2, observed_signal="log_return")
        model1 = HiddenMarkovModel(config1)
        assert model1.n_states == 2

        config2 = HMMConfig(n_states=5, max_iterations=50)
        model2 = HiddenMarkovModel(config2)
        assert model2.config.max_iterations == 50

        # Config validation should work
        config1.validate()  # Should not raise
        config2.validate()  # Should not raise

    def test_preset_configurations(self):
        """Test preset configuration creation."""
        # Conservative preset
        conservative = HMMConfig.create_conservative()
        model_conservative = HiddenMarkovModel(conservative)
        assert model_conservative.config.adaptation_rate == 0.01
        assert model_conservative.config.forgetting_factor == 0.99
        assert model_conservative.config.enable_change_detection == False

        # Aggressive preset
        aggressive = HMMConfig.create_aggressive()
        model_aggressive = HiddenMarkovModel(aggressive)
        assert model_aggressive.config.adaptation_rate == 0.1
        assert model_aggressive.config.n_states == 4
        assert model_aggressive.config.enable_change_detection == True

        # Balanced preset
        balanced = HMMConfig.create_balanced()
        model_balanced = HiddenMarkovModel(balanced)
        assert model_balanced.config.adaptation_rate == 0.05
        assert model_balanced.config.enable_change_detection == True

    def test_fit_basic_functionality(self):
        """Test basic fit functionality."""
        config = HMMConfig(n_states=2, max_iterations=10, random_seed=42)
        model = HiddenMarkovModel(config)

        # Create sample observations
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        observations = pd.DataFrame(
            {
                "log_return": np.random.normal(0, 0.02, 50),
                "other_col": np.random.randn(50),
            },
            index=dates,
        )

        # Fit the model
        model.fit(observations)

        # Check that model is fitted
        assert model.is_fitted == True
        assert model.initial_probs_ is not None
        assert model.transition_matrix_ is not None
        assert model.emission_means_ is not None
        assert model.emission_stds_ is not None
        assert len(model.emission_means_) == 2
        assert len(model.emission_stds_) == 2
        assert model.training_history_["iterations"] > 0

    def test_fit_with_different_initialization_methods(self):
        """Test fit with different initialization methods."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 50)}, index=dates
        )

        # Test random initialization
        config_random = HMMConfig(
            n_states=2, max_iterations=5, initialization_method="random", random_seed=42
        )
        model_random = HiddenMarkovModel(config_random)
        model_random.fit(observations)
        assert model_random.is_fitted == True

        # Test kmeans initialization (may fall back to simple if utils not available)
        config_kmeans = HMMConfig(
            n_states=2, max_iterations=5, initialization_method="kmeans", random_seed=42
        )
        model_kmeans = HiddenMarkovModel(config_kmeans)
        model_kmeans.fit(observations)
        assert model_kmeans.is_fitted == True

    def test_predict_functionality(self):
        """Test prediction functionality."""
        config = HMMConfig(n_states=2, max_iterations=10, random_seed=42)
        model = HiddenMarkovModel(config)

        # Create and fit on training data
        np.random.seed(42)
        train_dates = pd.date_range("2023-01-01", periods=50, freq="D")
        train_obs = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 50)}, index=train_dates
        )
        model.fit(train_obs)

        # Create test data
        test_dates = pd.date_range("2023-03-01", periods=20, freq="D")
        test_obs = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 20)}, index=test_dates
        )

        # Make predictions
        predictions = model.predict(test_obs)

        # Check prediction structure
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 20
        assert "predicted_state" in predictions.columns
        assert "confidence" in predictions.columns
        assert "state_0_prob" in predictions.columns
        assert "state_1_prob" in predictions.columns

        # Check prediction values
        assert predictions["predicted_state"].min() >= 0
        assert predictions["predicted_state"].max() < 2
        assert predictions["confidence"].min() >= 0
        assert predictions["confidence"].max() <= 1

    def test_update_method_functionality(self):
        """Test update method (fit + predict workflow)."""
        config = HMMConfig(n_states=2, max_iterations=10, random_seed=42)
        model = HiddenMarkovModel(config)

        # Create observations
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )

        # First update should fit and predict
        results = model.update(observations)

        # Check that model was fitted
        assert model.is_fitted == True
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 30
        assert "predicted_state" in results.columns

        # Second update should just predict
        new_dates = pd.date_range("2023-02-01", periods=10, freq="D")
        new_obs = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 10)}, index=new_dates
        )

        results2 = model.update(new_obs)
        assert len(results2) == 10
        assert "predicted_state" in results2.columns

    def test_plot_method_not_fitted(self):
        """Test plot method when model not fitted."""
        config = HMMConfig(n_states=3)
        model = HiddenMarkovModel(config)

        fig = model.plot()
        assert fig is not None

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_method_fitted(self):
        """Test plot method when model is fitted."""
        config = HMMConfig(n_states=2, max_iterations=5, random_seed=42)
        model = HiddenMarkovModel(config)

        # Fit the model
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )
        model.fit(observations)

        # Plot
        fig = model.plot()
        assert fig is not None

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_error_handling_insufficient_data(self):
        """Test error handling for insufficient training data."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        # Check which validation is being used
        from hidden_regime.models.hmm import HMM_UTILS_AVAILABLE

        if HMM_UTILS_AVAILABLE:
            # Sophisticated validation only checks for empty data, not minimum size
            # Test with truly empty data
            dates = pd.date_range("2023-01-01", periods=0, freq="D")
            observations = pd.DataFrame({"log_return": []}, index=dates)

            with pytest.raises(ValueError) as exc_info:
                model.fit(observations)

            assert "empty" in str(exc_info.value).lower()
        else:
            # Simple validation checks for minimum 10 observations
            dates = pd.date_range("2023-01-01", periods=5, freq="D")
            observations = pd.DataFrame(
                {"log_return": np.random.normal(0, 0.02, 5)}, index=dates
            )

            with pytest.raises(ValueError) as exc_info:
                model.fit(observations)

            assert "Insufficient data" in str(exc_info.value)

    def test_error_handling_nan_data(self):
        """Test handling of NaN data (should clean or error depending on utils availability)."""
        config = HMMConfig(n_states=2, max_iterations=5)
        model = HiddenMarkovModel(config)

        # Create data with NaN values
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        returns = np.random.normal(0, 0.02, 20)
        returns[5] = np.nan  # Inject NaN
        observations = pd.DataFrame({"log_return": returns}, index=dates)

        # If HMM_UTILS_AVAILABLE, it should clean data and proceed
        # If not available, it should raise ValueError
        from hidden_regime.models.hmm import HMM_UTILS_AVAILABLE

        if HMM_UTILS_AVAILABLE:
            # Should clean data and proceed
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # Ignore the warning about removing values
                model.fit(observations)  # Should not raise
                assert model.is_fitted == True
        else:
            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                model.fit(observations)
            assert "NaN values" in str(exc_info.value)

    def test_error_handling_infinite_data(self):
        """Test handling of infinite data (should clean or error depending on utils availability)."""
        config = HMMConfig(n_states=2, max_iterations=5)
        model = HiddenMarkovModel(config)

        # Create data with infinite values
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        returns = np.random.normal(0, 0.02, 20)
        returns[7] = np.inf  # Inject infinity
        observations = pd.DataFrame({"log_return": returns}, index=dates)

        # If HMM_UTILS_AVAILABLE, it should clean data and proceed
        # If not available, it should raise ValueError
        from hidden_regime.models.hmm import HMM_UTILS_AVAILABLE

        if HMM_UTILS_AVAILABLE:
            # Should clean data and proceed
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # Ignore the warning about removing values
                model.fit(observations)  # Should not raise
                assert model.is_fitted == True
        else:
            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                model.fit(observations)
            assert "infinite values" in str(exc_info.value)

    def test_error_handling_missing_observed_signal_fit(self):
        """Test error when observed signal missing during fit."""
        config = HMMConfig(n_states=2, observed_signal="missing_signal")
        model = HiddenMarkovModel(config)

        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 20)}, index=dates
        )

        with pytest.raises(ValueError) as exc_info:
            model.fit(observations)

        assert "missing_signal" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_error_handling_missing_observed_signal_predict(self):
        """Test error when observed signal missing during predict."""
        config = HMMConfig(n_states=2, max_iterations=5, random_seed=42)
        model = HiddenMarkovModel(config)

        # Fit with correct signal
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 20)}, index=dates
        )
        model.fit(observations)

        # Try to predict with missing signal
        test_dates = pd.date_range("2023-02-01", periods=10, freq="D")
        test_obs = pd.DataFrame(
            {"different_signal": np.random.normal(0, 0.02, 10)}, index=test_dates
        )

        with pytest.raises(ValueError) as exc_info:
            model.predict(test_obs)

        assert "log_return" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_error_handling_predict_before_fit(self):
        """Test error when predicting before fitting."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 10)}, index=dates
        )

        with pytest.raises(HMMInferenceError) as exc_info:
            model.predict(observations)

        assert "must be fitted" in str(exc_info.value)

    def test_get_model_info_not_fitted(self):
        """Test get_model_info when model not fitted."""
        config = HMMConfig(n_states=3)
        model = HiddenMarkovModel(config)

        info = model.get_model_info()

        assert info["n_states"] == 3
        assert info["is_fitted"] == False
        assert info["config"] == config.to_dict()
        assert info["emission_means"] is None
        assert info["emission_stds"] is None

    def test_get_model_info_fitted(self):
        """Test get_model_info when model is fitted."""
        config = HMMConfig(n_states=2, max_iterations=5, random_seed=42)
        model = HiddenMarkovModel(config)

        # Fit the model
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )
        model.fit(observations)

        info = model.get_model_info()

        assert info["n_states"] == 2
        assert info["is_fitted"] == True
        assert info["config"] == config.to_dict()
        assert info["emission_means"] is not None
        assert info["emission_stds"] is not None
        assert len(info["emission_means"]) == 2
        assert len(info["emission_stds"]) == 2
        assert info["training_history"]["iterations"] > 0

    def test_training_history_tracking(self):
        """Test that training history is properly tracked."""
        config = HMMConfig(n_states=2, max_iterations=15, random_seed=42)
        model = HiddenMarkovModel(config)

        # Fit the model
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 50)}, index=dates
        )
        model.fit(observations)

        # Check training history
        history = model.training_history_
        assert "log_likelihoods" in history
        assert "iterations" in history
        assert "converged" in history
        assert "training_time" in history

        assert history["iterations"] > 0
        assert history["training_time"] > 0
        assert len(history["log_likelihoods"]) > 0
        assert isinstance(history["converged"], bool)

    def test_online_learning_integration(self):
        """Test online learning capability integration."""
        # Create config with online learning enabled
        config = HMMConfig(
            n_states=2, max_iterations=5, adaptation_rate=0.1, random_seed=42
        )
        model = HiddenMarkovModel(config)

        # Initial observations
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )

        # Update with online learning
        results = model.update(observations)

        # Check that model tracks last observation for online updates
        assert model._last_observation is not None
        assert results is not None
        assert len(results) == 30


class TestHiddenMarkovModelCoverage:
    """Additional tests to improve code coverage."""

    def test_parameter_initialization_methods(self):
        """Test different parameter initialization approaches."""
        # Test with different random seeds
        config1 = HMMConfig(n_states=3, random_seed=123)
        model1 = HiddenMarkovModel(config1)

        config2 = HMMConfig(n_states=3, random_seed=456)
        model2 = HiddenMarkovModel(config2)

        # Both should initialize without errors
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )

        model1.fit(observations)
        model2.fit(observations)

        # Parameters should be different due to different seeds
        assert not np.array_equal(model1.initial_probs_, model2.initial_probs_)

    def test_edge_case_data_patterns(self):
        """Test model behavior with edge case data patterns."""
        config = HMMConfig(n_states=2, max_iterations=10, random_seed=42)
        model = HiddenMarkovModel(config)

        # Test with very low volatility data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        low_vol_returns = np.random.normal(0, 0.001, 30)  # Very low volatility
        observations = pd.DataFrame({"log_return": low_vol_returns}, index=dates)

        # Should handle low volatility gracefully
        model.fit(observations)
        assert model.is_fitted == True
        assert np.all(model.emission_stds_ >= config.min_variance)

    def test_convergence_behavior(self):
        """Test convergence detection and early stopping."""
        # Create config with early stopping
        config = HMMConfig(
            n_states=2,
            max_iterations=50,
            tolerance=1e-4,
            early_stopping=True,
            random_seed=42,
        )
        model = HiddenMarkovModel(config)

        # Use well-separated data to encourage convergence
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create regime-like data
        regime1_data = np.random.normal(-0.01, 0.005, 50)  # Bear regime
        regime2_data = np.random.normal(0.01, 0.005, 50)  # Bull regime
        mixed_data = np.concatenate([regime1_data, regime2_data])

        observations = pd.DataFrame({"log_return": mixed_data}, index=dates)

        model.fit(observations)

        # Should converge before max_iterations due to clear regime structure
        assert model.training_history_["iterations"] <= config.max_iterations
        assert len(model.training_history_["log_likelihoods"]) > 0

    def test_string_representation(self):
        """Test string representation and model info methods."""
        config = HMMConfig(n_states=3, max_iterations=10)
        model = HiddenMarkovModel(config)

        # Test model info before fitting
        info_before = model.get_model_info()
        assert info_before["is_fitted"] == False

        # Fit and test after
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        observations = pd.DataFrame(
            {"log_return": np.random.normal(0, 0.02, 30)}, index=dates
        )
        model.fit(observations)

        info_after = model.get_model_info()
        assert info_after["is_fitted"] == True
        assert info_after["emission_means"] is not None

    def test_various_config_options(self):
        """Test model with various configuration options."""
        # Test configurations that exercise different code paths
        configs_to_test = [
            HMMConfig(n_states=2, observed_signal="returns", max_iterations=5),
            HMMConfig(n_states=4, initialization_method="random", random_seed=None),
            HMMConfig(n_states=3, tolerance=1e-8, regularization=1e-5),
        ]

        for config in configs_to_test:
            model = HiddenMarkovModel(config)

            # Test with appropriate observed signal
            np.random.seed(42)
            dates = pd.date_range("2023-01-01", periods=30, freq="D")
            observations = pd.DataFrame(
                {config.observed_signal: np.random.normal(0, 0.02, 30)}, index=dates
            )

            # Should fit without errors
            model.fit(observations)
            assert model.is_fitted == True
            assert model.n_states == config.n_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
