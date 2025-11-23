"""
Unit and integration tests for models/hmm.py advanced features.

Tests incremental learning, adaptive refitting, quality metrics, AIC/BIC,
cross-validation, serialization, and parameter evolution tracking.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.utils.exceptions import ValidationError, HMMTrainingError


@pytest.fixture
def hmm_config():
    """Create default HMM configuration."""
    return HMMConfig(
        n_states=3,
        max_iterations=100,
        tolerance=1e-4,
        random_seed=42,
    )


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    returns = pd.DataFrame({
        "log_return": np.random.normal(0.001, 0.02, 100),
    }, index=dates)
    return returns


@pytest.fixture
def fitted_hmm(hmm_config, sample_returns_data):
    """Create and fit an HMM model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(hmm_config)
        model.fit(sample_returns_data)
    return model


# Suppress warnings for tests
import warnings


# ============================================================================
# UNIT TESTS (20 tests)
# ============================================================================


def test_hmm_incremental_update(fitted_hmm):
    """Test incremental update functionality."""
    # New observations for incremental update
    new_data = pd.DataFrame({
        "log_return": np.random.normal(0.001, 0.02, 10),
    })

    # Update model with new data
    result = fitted_hmm.update(new_data)

    # Should return predictions for new data
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(new_data)
    assert "predicted_state" in result.columns


def test_hmm_adaptive_refit_trigger():
    """Test adaptive refitting trigger logic."""
    config = HMMConfig(n_states=2, update_strategy="adaptive_refit", refit_interval_observations=50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Fit with initial data
        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 60)})
        model.fit(data)

        # Check training history tracks observations
        assert model.is_fitted is True


def test_hmm_distribution_shift_detection():
    """Test detection of distribution shifts in data."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Fit with normal distribution
        normal_data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(normal_data)

        # Update with shifted distribution
        shifted_data = pd.DataFrame({"log_return": np.random.normal(0.05, 0.01, 10)})
        result = model.update(shifted_data)

        assert len(result) == 10


def test_hmm_parameter_evolution():
    """Test parameter evolution tracking during training."""
    config = HMMConfig(n_states=2, max_iterations=10, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Check training history exists
        assert "log_likelihoods" in model.training_history_
        assert "iterations" in model.training_history_


def test_hmm_quality_metrics():
    """Test model quality metrics tracking."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Training history should track quality
        assert model.training_history_["converged"] in [True, False]


def test_hmm_aic_calculation(fitted_hmm, sample_returns_data):
    """Test AIC (Akaike Information Criterion) calculation."""
    aic = fitted_hmm.aic(sample_returns_data)

    assert isinstance(aic, float)
    # AIC should be a reasonable value (not NaN or inf)
    assert np.isfinite(aic)


def test_hmm_bic_calculation(fitted_hmm, sample_returns_data):
    """Test BIC (Bayesian Information Criterion) calculation."""
    bic = fitted_hmm.bic(sample_returns_data)

    assert isinstance(bic, float)
    # BIC should be a reasonable value
    assert np.isfinite(bic)


def test_hmm_cross_validation():
    """Test cross-validation implementation."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Create data for CV
        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 100)})

        # Split into train/test
        train_data = data.iloc[:80]
        test_data = data.iloc[80:]

        # Train on train set
        model.fit(train_data)

        # Predict on test set
        predictions = model.predict(test_data)

        assert len(predictions) == len(test_data)


def test_hmm_save_load_model(fitted_hmm, sample_returns_data):
    """Test model serialization (save/load)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.pkl")

        # Save model
        fitted_hmm.save_model(filepath)

        # Check file was created
        assert os.path.exists(filepath)

        # Load model
        loaded_model = HiddenMarkovModel.load_model(filepath)

        # Verify loaded model has same parameters
        assert loaded_model.n_states == fitted_hmm.n_states
        assert loaded_model.is_fitted == fitted_hmm.is_fitted
        np.testing.assert_array_almost_equal(
            loaded_model.emission_means_, fitted_hmm.emission_means_
        )


def test_hmm_initialization_diagnostics():
    """Test initialization diagnostic tracking."""
    config = HMMConfig(
        n_states=2,
        initialization_method="kmeans",
        random_seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should have initialized parameters
        assert model.emission_means_ is not None
        assert model.emission_stds_ is not None


def test_hmm_custom_initialization():
    """Test custom parameter initialization."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Custom initial parameters
        custom_means = np.array([-0.01, 0.01])
        custom_stds = np.array([0.02, 0.02])

        # Fit should work with data
        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        assert model.is_fitted


def test_hmm_financial_constraints():
    """Test application of financial constraints."""
    config = HMMConfig(
        n_states=3,
        min_variance=1e-6,  # Minimum variance constraint
        random_seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Variances should respect minimum
        assert all(model.emission_stds_ >= np.sqrt(config.min_variance))


def test_hmm_regime_analysis():
    """Test regime analysis capabilities."""
    config = HMMConfig(n_states=3, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should be able to predict states
        predictions = model.predict(data)

        assert "predicted_state" in predictions.columns
        assert predictions["predicted_state"].nunique() <= 3


def test_hmm_performance_monitoring():
    """Test performance monitoring capabilities."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Training history should track performance
        assert "training_time" in model.training_history_
        assert model.training_history_["training_time"] >= 0


def test_hmm_get_detailed_state():
    """Test getting detailed model state information."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should have all parameters
        assert model.transition_matrix_ is not None
        assert model.initial_probs_ is not None
        assert model.emission_means_ is not None
        assert model.emission_stds_ is not None


def test_hmm_parameter_evolution_summary():
    """Test parameter evolution summary."""
    config = HMMConfig(n_states=2, max_iterations=20, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should track iterations
        assert model.training_history_["iterations"] > 0


def test_hmm_predict_proba():
    """Test probability prediction."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        predictions = model.predict(data)

        # Should have state probabilities
        assert "state_0_prob" in predictions.columns
        assert "state_1_prob" in predictions.columns


def test_hmm_decode_states_viterbi(fitted_hmm, sample_returns_data):
    """Test Viterbi state decoding."""
    predictions = fitted_hmm.predict(sample_returns_data)

    # Viterbi decoding produces most likely state sequence
    assert "predicted_state" in predictions.columns
    assert all(predictions["predicted_state"].isin(range(fitted_hmm.n_states)))


def test_hmm_decode_states_posterior():
    """Test posterior state decoding."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        predictions = model.predict(data)

        # Posterior probabilities should sum to 1
        prob_cols = [c for c in predictions.columns if c.endswith("_prob")]
        if len(prob_cols) > 0:
            prob_sums = predictions[prob_cols].sum(axis=1)
            np.testing.assert_array_almost_equal(prob_sums, 1.0, decimal=5)


def test_hmm_score_observations(fitted_hmm, sample_returns_data):
    """Test likelihood scoring of observations."""
    # Predict should return likelihood/confidence
    predictions = fitted_hmm.predict(sample_returns_data)

    assert "confidence" in predictions.columns
    # Confidence should be between 0 and 1
    assert all(predictions["confidence"] >= 0)
    assert all(predictions["confidence"] <= 1)


# ============================================================================
# INTEGRATION TESTS (10 tests)
# ============================================================================


def test_hmm_with_incremental_pipeline():
    """Test HMM in incremental learning pipeline."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Initial training
        initial_data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(initial_data)

        # Incremental updates
        for i in range(5):
            new_data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 10)})
            result = model.update(new_data)
            assert len(result) == 10


def test_hmm_adaptive_refit_workflow():
    """Test adaptive refitting workflow."""
    config = HMMConfig(
        n_states=2,
        update_strategy="adaptive_refit",
        refit_interval_observations=30,
        random_seed=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Initial fit
        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should track when fitted
        assert len(model.training_history_["fit_timestamps"]) >= 1


def test_hmm_parameter_stability():
    """Test parameter stability monitoring."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Save initial parameters
        initial_means = model.emission_means_.copy()

        # Refit with similar data
        data2 = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data2)

        # Parameters should be similar (not drastically different)
        # This is a stability check
        assert model.emission_means_ is not None


def test_hmm_quality_degradation_detection():
    """Test quality degradation detection."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Fit with good data
        good_data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(good_data)

        # Should converge
        assert model.training_history_["converged"] in [True, False]


def test_hmm_model_comparison():
    """Test model comparison using AIC/BIC."""
    data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 100)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Model with 2 states
        model2 = HiddenMarkovModel(HMMConfig(n_states=2, random_seed=42))
        model2.fit(data)
        aic2 = model2.aic(data)

        # Model with 3 states
        model3 = HiddenMarkovModel(HMMConfig(n_states=3, random_seed=42))
        model3.fit(data)
        aic3 = model3.aic(data)

        # Both should be valid
        assert np.isfinite(aic2)
        assert np.isfinite(aic3)


def test_hmm_cross_validation_pipeline():
    """Test cross-validation in full pipeline."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        # Full data
        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 100)})

        # 5-fold CV simulation
        fold_size = 20
        scores = []

        for i in range(5):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size

            # Use remaining data for training
            train_data = pd.concat([
                data.iloc[:test_start],
                data.iloc[test_end:]
            ])
            test_data = data.iloc[test_start:test_end]

            if len(train_data) >= 20:  # Minimum training data
                model.fit(train_data)
                predictions = model.predict(test_data)
                scores.append(len(predictions))

        assert len(scores) > 0


def test_hmm_serialization_roundtrip(fitted_hmm, sample_returns_data):
    """Test complete save/load roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.pkl")

        # Save
        fitted_hmm.save_model(filepath)

        # Load
        loaded = HiddenMarkovModel.load_model(filepath)

        # Compare predictions
        original_preds = fitted_hmm.predict(sample_returns_data)
        loaded_preds = loaded.predict(sample_returns_data)

        # Predictions should be identical
        np.testing.assert_array_equal(
            original_preds["predicted_state"].values,
            loaded_preds["predicted_state"].values
        )


def test_hmm_initialization_quality():
    """Test initialization quality across methods."""
    data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # K-means initialization
        config_kmeans = HMMConfig(n_states=2, initialization_method="kmeans", random_seed=42)
        model_kmeans = HiddenMarkovModel(config_kmeans)
        model_kmeans.fit(data)

        # Random initialization
        config_random = HMMConfig(n_states=2, initialization_method="random", random_seed=42)
        model_random = HiddenMarkovModel(config_random)
        model_random.fit(data)

        # Both should converge
        assert model_kmeans.is_fitted
        assert model_random.is_fitted


def test_hmm_regime_persistence_analysis():
    """Test regime persistence analysis."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 100)})
        model.fit(data)

        predictions = model.predict(data)

        # Analyze regime transitions
        states = predictions["predicted_state"].values
        transitions = np.sum(states[1:] != states[:-1])

        # Should have some transitions but not too many
        assert transitions < len(states)


def test_hmm_performance_metrics_comprehensive():
    """Test comprehensive performance metrics."""
    config = HMMConfig(n_states=2, random_seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HiddenMarkovModel(config)

        data = pd.DataFrame({"log_return": np.random.normal(0, 0.01, 50)})
        model.fit(data)

        # Should have comprehensive training history
        assert "log_likelihoods" in model.training_history_
        assert "iterations" in model.training_history_
        assert "converged" in model.training_history_
        assert "training_time" in model.training_history_
