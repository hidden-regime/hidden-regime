"""
Unit tests for HiddenMarkovModel class from base_hmm.

Tests the core HMM implementation that provides regime detection
functionality within the pipeline framework.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from hidden_regime.models.base_hmm import HiddenMarkovModel, HMMTrainingError, HMMInferenceError
from hidden_regime.models.config import HMMConfig
from hidden_regime.utils.exceptions import ValidationError


class TestHiddenMarkovModel:
    """Test cases for HiddenMarkovModel class."""

    def create_sample_returns(self, n_samples=100):
        """Create sample return data for testing."""
        np.random.seed(42)
        # Create regime-switching returns
        regime1 = np.random.normal(-0.02, 0.03, 30)  # Bear regime
        regime2 = np.random.normal(0.00, 0.015, 40)  # Sideways regime
        regime3 = np.random.normal(0.015, 0.025, 30)  # Bull regime
        
        returns = np.concatenate([regime1, regime2, regime3])
        return returns[:n_samples]

    def create_sample_series(self, n_samples=100):
        """Create sample return data as pandas Series."""
        returns = self.create_sample_returns(n_samples)
        return pd.Series(returns, name='log_return')

    def test_hidden_markov_model_initialization_default(self):
        """Test default HiddenMarkovModel initialization."""
        hmm = HiddenMarkovModel(n_states=3)

        assert hmm.n_states == 3
        assert hmm.config.n_states == 3
        assert hmm.config.max_iterations == 100
        assert not hmm.is_fitted
        assert hmm.initial_probs_ is None
        assert hmm.transition_matrix_ is None
        assert hmm.emission_params_ is None

    def test_hidden_markov_model_initialization_with_config(self):
        """Test HiddenMarkovModel initialization with custom config."""
        config = HMMConfig(
            n_states=4,
            max_iterations=200,
            tolerance=1e-8,
            random_seed=123
        )
        hmm = HiddenMarkovModel(n_states=4, config=config)

        assert hmm.n_states == 4
        assert hmm.config.max_iterations == 200
        assert hmm.config.tolerance == 1e-8
        assert hmm.config.random_seed == 123

    def test_fit_basic_functionality(self):
        """Test basic fitting functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)

        # Fit should return self
        result = hmm.fit(returns, verbose=False)
        assert result is hmm

        # Check that model is fitted
        assert hmm.is_fitted
        assert hmm.initial_probs_ is not None
        assert hmm.transition_matrix_ is not None
        assert hmm.emission_params_ is not None

        # Check parameter shapes
        assert hmm.initial_probs_.shape == (3,)
        assert hmm.transition_matrix_.shape == (3, 3)
        assert hmm.emission_params_.shape == (3, 2)  # mean and std for each state

        # Check parameter constraints
        assert np.allclose(np.sum(hmm.initial_probs_), 1.0)
        assert np.allclose(np.sum(hmm.transition_matrix_, axis=1), 1.0)

    def test_fit_with_pandas_series(self):
        """Test fitting with pandas Series."""
        hmm = HiddenMarkovModel(n_states=3)
        returns_series = self.create_sample_series(100)

        hmm.fit(returns_series, verbose=False)

        assert hmm.is_fitted
        assert hmm.initial_probs_ is not None

    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        hmm = HiddenMarkovModel(n_states=3)
        
        # Too few observations
        returns = self.create_sample_returns(5)
        
        with pytest.raises(HMMTrainingError, match="Data validation failed"):
            hmm.fit(returns)

    def test_fit_with_missing_data(self):
        """Test fitting with missing data."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        # Introduce NaN values - but not too many
        returns[10] = np.nan
        returns[50] = np.nan
        
        # Should handle some NaN by removing them
        hmm.fit(returns, verbose=False)
        assert hmm.is_fitted

    def test_predict_functionality(self):
        """Test prediction functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        # Fit first
        hmm.fit(returns, verbose=False)
        
        # Predict on same data
        states = hmm.predict(returns)
        
        assert isinstance(states, np.ndarray)
        assert len(states) == len(returns)
        
        # States should be in valid range
        assert (states >= 0).all()
        assert (states < 3).all()
        assert states.dtype == int

    def test_predict_without_fitting(self):
        """Test prediction without fitting."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(50)
        
        with pytest.raises(HMMTrainingError, match="Model must be fitted"):
            hmm.predict(returns)

    def test_predict_proba_functionality(self):
        """Test probability prediction functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        probabilities = hmm.predict_proba(returns)
        
        assert probabilities.shape == (len(returns), 3)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_score_functionality(self):
        """Test model scoring functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        score = hmm.score(returns)
        
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_online_update_functionality(self):
        """Test online update functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        # Fit first
        hmm.fit(returns, verbose=False)
        
        # Test online update
        result = hmm.update_with_observation(0.01)
        
        assert isinstance(result, dict)
        assert "most_likely_regime" in result
        assert "regime_probabilities" in result
        assert "confidence" in result
        assert 0 <= result["most_likely_regime"] < 3
        assert 0 <= result["confidence"] <= 1

    def test_online_update_without_fitting(self):
        """Test online update without fitting."""
        hmm = HiddenMarkovModel(n_states=3)
        
        with pytest.raises(HMMTrainingError, match="Model must be fitted"):
            hmm.update_with_observation(0.01)

    def test_regime_analysis(self):
        """Test comprehensive regime analysis."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        analysis = hmm.analyze_regimes(returns)
        
        assert isinstance(analysis, dict)
        assert "model_info" in analysis
        assert "regime_parameters" in analysis
        assert "state_sequence" in analysis
        
        # Check model info
        assert analysis["model_info"]["n_states"] == 3
        assert analysis["model_info"]["n_observations"] == len(returns)

    def test_model_serialization_pickle(self):
        """Test model serialization with pickle."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            hmm.save(f.name)
            
            loaded_hmm = HiddenMarkovModel.load(f.name)
        
        os.unlink(f.name)
        
        # Test loaded model
        assert loaded_hmm.is_fitted
        assert loaded_hmm.n_states == hmm.n_states
        np.testing.assert_array_equal(
            loaded_hmm.initial_probs_, 
            hmm.initial_probs_
        )

    def test_convergence_behavior(self):
        """Test convergence behavior."""
        # Create data that should converge quickly
        np.random.seed(42)
        easy_data = np.concatenate([
            np.random.normal(-0.05, 0.01, 40),  # Clear bear regime
            np.random.normal(0.05, 0.01, 40),   # Clear bull regime
            np.random.normal(0.0, 0.01, 20)     # Clear sideways regime
        ])

        config = HMMConfig(
            n_states=3, 
            max_iterations=100, 
            tolerance=1e-6, 
            random_seed=42
        )
        hmm = HiddenMarkovModel(n_states=3, config=config)

        hmm.fit(easy_data, verbose=False)
        
        assert hmm.is_fitted
        # Should converge reasonably quickly
        assert hmm.training_history_["iterations"] < 50

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create data with extreme values
        extreme_returns = np.array([0.2, -0.2, 0.15, -0.15] * 25)

        hmm = HiddenMarkovModel(n_states=3)
        
        # Should handle extreme data without numerical issues
        hmm.fit(extreme_returns, verbose=False)
        
        assert hmm.is_fitted
        assert np.all(np.isfinite(hmm.initial_probs_))
        assert np.all(np.isfinite(hmm.transition_matrix_))
        assert np.all(np.isfinite(hmm.emission_params_))

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Invalid n_states
        with pytest.raises(ValueError):
            HiddenMarkovModel(n_states=1)

    def test_training_history(self):
        """Test training history tracking."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        
        assert hasattr(hmm, 'training_history_')
        assert 'log_likelihoods' in hmm.training_history_
        assert 'iterations' in hmm.training_history_
        assert 'training_time' in hmm.training_history_
        
        assert len(hmm.training_history_['log_likelihoods']) > 0
        assert hmm.training_history_['iterations'] > 0
        assert hmm.training_history_['training_time'] > 0

    def test_state_reset(self):
        """Test state reset functionality."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        hmm.update_with_observation(0.01)  # Set current state
        
        # Verify state is set
        assert hmm._current_state_probs is not None
        assert hmm._last_observation is not None
        
        # Reset state
        hmm.reset_state()
        
        # Verify state is reset
        assert hmm._current_state_probs is None
        assert hmm._last_observation is None

    def test_string_representation(self):
        """Test string representation."""
        hmm = HiddenMarkovModel(n_states=3)
        
        # Unfitted model
        repr_str = repr(hmm)
        assert "HiddenMarkovModel" in repr_str
        assert "n_states=3" in repr_str
        assert "fitted=False" in repr_str
        
        # Fitted model
        returns = self.create_sample_returns(50)
        hmm.fit(returns, verbose=False)
        
        repr_str = repr(hmm)
        assert "fitted=True" in repr_str
        assert "log_likelihood=" in repr_str

    def test_error_handling_invalid_observations(self):
        """Test error handling with invalid observations."""
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(100)
        
        hmm.fit(returns, verbose=False)
        
        # Test with invalid online observation
        with pytest.raises(HMMInferenceError):
            hmm.update_with_observation(np.nan)
        
        with pytest.raises(HMMInferenceError):
            hmm.update_with_observation(np.inf)

    def test_different_configurations(self):
        """Test different HMM configurations."""
        returns = self.create_sample_returns(150)
        
        # Test different numbers of states
        for n_states in [2, 3, 4]:
            hmm = HiddenMarkovModel(n_states=n_states)
            hmm.fit(returns, verbose=False)
            
            assert hmm.is_fitted
            assert hmm.n_states == n_states
            assert hmm.initial_probs_.shape == (n_states,)
            assert hmm.transition_matrix_.shape == (n_states, n_states)

    def test_performance_with_larger_dataset(self):
        """Test performance with larger dataset."""
        import time
        
        hmm = HiddenMarkovModel(n_states=3)
        returns = self.create_sample_returns(1000)  # Larger dataset
        
        start_time = time.time()
        hmm.fit(returns, verbose=False)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30.0  # Less than 30 seconds
        assert hmm.is_fitted


if __name__ == "__main__":
    pytest.main([__file__])